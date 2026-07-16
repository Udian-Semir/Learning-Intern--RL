"""
OFT1_0 训练脚本 - 支持多GPU分布式训练 (Qwen2B 版本)
使用 LeRobot Dataset Loader 加载真实数据

多GPU训练使用方法 (Multi-GPU Training):
  # 使用 torchrun 启动分布式训练 (推荐)
  torchrun --nproc_per_node=4 train_for_libero_qwen2b_multigpu.py \
    --data_path /path/to/dataset \
    --steps 10000

  # 指定特定GPU运行
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_for_libero_qwen2b_multigpu.py \
    --data_path /path/to/dataset

单GPU训练使用方法 (Single-GPU Training):
  python train_for_libero_qwen2b_multigpu.py \
    --data_path /path/to/dataset \
    --device cuda:0

参数说明:
  --gradient_accumulation_steps: 梯度累积步数 (默认: 1)
  --local_rank: 本地进程排名 (由 torchrun 自动设置，无需手动指定)
"""
import argparse
import json
import math
import os
import sys
import re
import atexit
import random
import signal
from pathlib import Path
from typing import List, Optional, Dict
import time
from datetime import timedelta
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler
from transformers.feature_extraction_utils import BatchFeature
from tqdm import tqdm
import wandb

# 添加当前目录和项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))  # 项目根目录

from vlm2oft_pipeline import VLM2OFTPipeline, create_vlm2oft_pipeline
from constants import (
    ACTION_DIM, NUM_ACTIONS_CHUNK, LLM_OUTPUT_DIM_MLP_INPUT_DIM,
    PROPRIO_DIM, NUM_VLM_HIDDEN_LAYERS
)
from utils.polyfit_chunk import polyfit_chunk
from utils.lerobot_dataset_loader import (
    create_lerobot_dataloader, 
    LeRobotDataset, 
    collate_fn as lerobot_default_collate_fn,
    SharedVLMCache,
    preload_vlm_cache_distributed,
    get_total_vlm_states,
)

# Multi-Dataset 支持 (按需加载, 失败时多源功能不可用但单源场景零回归)
try:
    from utils.multi_dataset_index import MultiDatasetIndex
    from utils.normalization_stats_merge import MultiDatasetNormalizers
    from utils.multi_lerobot_dataset import MultiLeRobotDataset
    from utils.multi_chunk_batch_cache import MultiChunkBatchCache
    _MULTI_DATASET_AVAILABLE = True
    _MULTI_DATASET_IMPORT_ERROR = ""
except ImportError as _multi_import_exc:
    MultiDatasetIndex = None  # type: ignore[assignment]
    MultiDatasetNormalizers = None  # type: ignore[assignment]
    MultiLeRobotDataset = None  # type: ignore[assignment]
    MultiChunkBatchCache = None  # type: ignore[assignment]
    _MULTI_DATASET_AVAILABLE = False
    _MULTI_DATASET_IMPORT_ERROR = str(_multi_import_exc)

# Disk-shuffle 模式 (按需加载; 不需要 SHM, 全局 shuffle, 训练时按需 mmap 读)
try:
    from utils.disk_vlm_cache import (
        DiskVLMCacheManager,
        build_single_source_disk_cache,
    )
    _DISK_SHUFFLE_AVAILABLE = True
    _DISK_SHUFFLE_IMPORT_ERROR = ""
except ImportError as _disk_import_exc:
    DiskVLMCacheManager = None  # type: ignore[assignment]
    build_single_source_disk_cache = None  # type: ignore[assignment]
    _DISK_SHUFFLE_AVAILABLE = False
    _DISK_SHUFFLE_IMPORT_ERROR = str(_disk_import_exc)


# ============================================================================
# 四元数转轴角函数 - Quaternion to Axis-Angle Conversion
# ============================================================================

def quat2axisangle_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    四元数转轴角 (PyTorch 批量版本)
    
    Args:
        quat: (batch, 4) 或 (batch, seq, 4)，四元数 (qx, qy, qz, qw)
    
    Returns:
        axis_angle: (batch, 3) 或 (batch, seq, 3)，轴角 (ax, ay, az)
    """
    original_shape = quat.shape
    
    if quat.dim() == 3:
        batch_size, seq_len, _ = quat.shape
        quat = quat.reshape(-1, 4)
    else:
        batch_size, seq_len = quat.shape[0], None
    
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    qw = torch.clamp(qw, -1.0, 1.0)
    den = torch.sqrt(1.0 - qw * qw)
    angle = 2.0 * torch.acos(qw)
    small_angle_mask = den < 1e-8
    
    axis_angle = torch.zeros(quat.shape[0], 3, dtype=quat.dtype, device=quat.device)
    
    if (~small_angle_mask).any():
        scale = angle[~small_angle_mask] / den[~small_angle_mask]
        axis_angle[~small_angle_mask, 0] = qx[~small_angle_mask] * scale
        axis_angle[~small_angle_mask, 1] = qy[~small_angle_mask] * scale
        axis_angle[~small_angle_mask, 2] = qz[~small_angle_mask] * scale
    
    if seq_len is not None:
        axis_angle = axis_angle.reshape(batch_size, seq_len, 3)
    
    return axis_angle


def convert_state_quat_to_axisangle(state: torch.Tensor) -> torch.Tensor:
    """
    将用户的 9 维 state 转换为 8 维 state
    
    用户 state 格式 (9维):
        [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
    
    转换后 state 格式 (8维):
        [gripper1, gripper2, x, y, z, ax, ay, az]
    """
    batch_size = state.shape[0]
    gripper = state[:, 0:2]
    position = state[:, 2:5]
    quat = state[:, 5:9]
    axis_angle = quat2axisangle_torch(quat)
    converted_state = torch.cat([gripper, position, axis_angle], dim=1)
    return converted_state


def euler_to_quat_torch(euler: torch.Tensor) -> torch.Tensor:
    """
    欧拉角转四元数 (PyTorch 批量版本)
    使用 ZYX (yaw-pitch-roll) 顺序
    
    Args:
        euler: (batch, 3) 欧拉角 (roll, pitch, yaw) in radians
    
    Returns:
        quat: (batch, 4) 四元数 (qx, qy, qz, qw)
    """
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return torch.stack([qx, qy, qz, qw], dim=1)


# ============================================================================
# 训练配置类 - Training Configuration Class
# ============================================================================

@dataclass
class TrainingConfig:
    """
    OFT Action Head 训练配置 (Qwen2B 多GPU版本)
    
    ⚠️ 参数已与原始 N1.5 LIBERO 训练配置对齐
    """
    
    # ========== 数据相关参数 ==========
    data_path: str = ""
    batch_size: int = 32
    num_workers: int = 4
    val_split: float = 0.0
    
    # ========== 训练超参数 ==========
    epochs: int = 100000
    steps: int = 10000
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"  # "cosine", "constant", "warmup_step_decay"
    lr_decay_step: int = 10000  # warmup_step_decay 模式: 从第几个 step 开始改变学习率
    lr_decay_value: float = 5e-5  # warmup_step_decay 模式: 改变后的学习率值
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    
    # ========== 模型参数 ==========
    num_transformer_blocks: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.1
    action_head_hidden_dim: int = 4096
    num_vlm_layers: Optional[int] = None
    vlm_output_dim: Optional[int] = None
    
    # ========== 权重与保存 ==========
    out_dir: str = "./experiments/oft_qwen2b_finetuning/checkpoints"
    log_dir: str = "./experiments/oft_qwen2b_finetuning/logs"
    save_every: int = 1
    save_every_steps: int = 0
    
    # ========== 系统配置 ==========
    device: str = "cuda:0"
    gradient_accumulation_steps: int = 1
    
    # ========== 内存优化参数 ==========
    use_amp: bool = True
    amp_dtype: str = "float16"
    empty_cache_freq: int = 0
    
    # ========== Weights & Biases 配置 ==========
    use_wandb: bool = True
    wandb_project: str = "gr00t_oft_training"
    wandb_run_name: str = ""
    wandb_log_freq: int = 10
    """Wandb 基本日志记录频率 (每 N 步)
    Frequency for basic wandb logging (every N steps)
    - 记录: loss, learning_rate, batch_time, grad_norm 等
    """


# ============================================================================
# 分布式训练辅助函数
# ============================================================================

def setup_distributed():
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        
        # 设置超时时间 (2小时，用于支持长时间的 VLM states 预加载)
        timeout_seconds = int(os.environ.get('TORCH_DISTRIBUTED_TIMEOUT_SEC', 7200))
        timeout = timedelta(seconds=timeout_seconds)
        
        # 尝试使用 NCCL 后端，如果失败则回退到 gloo
        try:
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
            if rank == 0:
                print(f"✓ 使用 NCCL 后端初始化分布式训练 (timeout={timeout_seconds}s)")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ NCCL 初始化失败: {e}")
                print("  尝试使用 gloo 后端...")
            dist.init_process_group(backend='gloo', init_method='env://', timeout=timeout)
            if rank == 0:
                print(f"✓ 使用 gloo 后端初始化分布式训练 (timeout={timeout_seconds}s, 性能可能较低)")
        
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


# 全局变量用于存储共享内存缓存（用于清理）
_shared_vlm_cache_global = None
_shared_vlm_cache_rank = None

def cleanup_shared_memory_cache():
    """清理共享内存缓存的全局清理函数"""
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    if _shared_vlm_cache_global is not None:
        try:
            _shared_vlm_cache_global.close()
            if _shared_vlm_cache_rank == 0:
                _shared_vlm_cache_global.unlink()
        except Exception as e:
            pass  # 忽略清理时的错误

def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """检查是否为主进程"""
    return rank == 0


def print_rank0(msg: str, rank: int = 0):
    """只在主进程打印"""
    if is_main_process(rank):
        print(msg)


def format_time_seconds(seconds: float) -> str:
    """将秒数格式化为可读的时间字符串，如 '1h 23m 45s' 或 '45m 30s'"""
    if seconds < 0:
        return "0s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


TRAINING_STATE_FILENAME = "training_state.pt"


def get_checkpoint_paths(checkpoint_path: str) -> Dict[str, Path]:
    """标准化 checkpoint 路径，支持目录或目录内文件。"""
    path = Path(checkpoint_path).expanduser()
    checkpoint_dir = path if path.is_dir() else path.parent
    return {
        "dir": checkpoint_dir,
        "model": checkpoint_dir / "action_head.pt",
        "config": checkpoint_dir / "config.json",
        "training_state": checkpoint_dir / TRAINING_STATE_FILENAME,
    }


def get_rng_state() -> Dict[str, object]:
    """收集随机数状态，尽量让 resume 后的数据顺序与训练行为保持一致。"""
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def restore_rng_state(rng_state: Optional[Dict[str, object]], rank: int = 0):
    """恢复随机数状态。"""
    if not rng_state:
        return
    try:
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
        print_rank0("✓ 已恢复随机数状态", rank)
    except Exception as e:
        print_rank0(f"⚠️ 恢复随机数状态失败，将继续训练: {e}", rank)


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device):
    """确保 optimizer state tensors 位于当前设备。"""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_training_checkpoint(
    checkpoint_dir: Path,
    model_state_dict: Dict[str, torch.Tensor],
    model_config: Dict[str, object],
    training_state: Dict[str, object],
):
    """保存推理权重、配置和完整训练状态。"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model_state_dict, checkpoint_dir / "action_head.pt")
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(model_config, f, indent=2, default=str)

    tmp_state_path = checkpoint_dir / f"{TRAINING_STATE_FILENAME}.tmp"
    torch.save(training_state, tmp_state_path)
    os.replace(tmp_state_path, checkpoint_dir / TRAINING_STATE_FILENAME)


def load_resume_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    rank: int = 0,
    reset_lr: bool = False,
) -> Dict[str, object]:
    """加载 resume checkpoint，优先加载完整训练状态，兼容旧版仅权重 checkpoint。

    Args:
        checkpoint_path: checkpoint 路径 (目录或目录内文件)
        device: 加载到的设备
        rank: 当前进程 rank (仅用于日志)
        reset_lr: 若为 True，即使存在 training_state.pt 也只加载模型权重，
            退化为 weights_only 模式 → step / optimizer / scheduler / scaler / RNG
            全部重置，学习率调度器从 step 0 重新走 warmup → decay。
            适用场景: "用某个 ckpt 当作起点重新调度学习率 / 切换数据集 fine-tune"。
    """
    paths = get_checkpoint_paths(checkpoint_path)
    checkpoint_dir = paths["dir"]

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint 路径不存在: {checkpoint_dir}")

    if paths["training_state"].exists() and not reset_lr:
        # weights_only=False: 我们的 training_state.pt 里包含 numpy scalar / Python
        # 对象 (optimizer state, scheduler state, etc.), PyTorch 2.6+ 默认
        # weights_only=True 会拒绝加载. 这是我们自己训练保存的 ckpt, 可信.
        checkpoint = torch.load(
            paths["training_state"], map_location=device, weights_only=False
        )
        checkpoint["resume_mode"] = "full_state"
        checkpoint["resolved_checkpoint_dir"] = str(checkpoint_dir)
        print_rank0(f"✓ 加载完整训练状态: {paths['training_state']}", rank)
        return checkpoint

    if paths["training_state"].exists() and reset_lr:
        full_state = torch.load(
            paths["training_state"], map_location=device, weights_only=False
        )
        checkpoint = {
            "model_state_dict": full_state.get("model_state_dict"),
            "resume_mode": "weights_only",
            "resolved_checkpoint_dir": str(checkpoint_dir),
            "reset_lr": True,
        }
        if checkpoint["model_state_dict"] is None and paths["model"].exists():
            checkpoint["model_state_dict"] = torch.load(
                paths["model"], map_location=device, weights_only=False
            )
        if checkpoint["model_state_dict"] is None:
            raise RuntimeError(
                f"--resume_reset_lr 启用，但 {paths['training_state']} 中没有 model_state_dict，"
                f"且找不到 {paths['model']}。无法只加载权重。"
            )
        print_rank0(
            f"🔁 --resume_reset_lr 已启用：仅加载模型权重，optim/scheduler/step/RNG 全部重置 "
            f"(checkpoint: {checkpoint_dir})",
            rank,
        )
        return checkpoint

    if paths["model"].exists():
        checkpoint = {
            "model_state_dict": torch.load(
                paths["model"], map_location=device, weights_only=False
            ),
            "resume_mode": "weights_only",
            "resolved_checkpoint_dir": str(checkpoint_dir),
        }
        print_rank0(
            f"⚠️ 未找到 {TRAINING_STATE_FILENAME}，将仅加载模型权重并从 step 0 继续: {paths['model']}",
            rank,
        )
        return checkpoint

    raise FileNotFoundError(
        f"在 {checkpoint_dir} 下未找到可用 checkpoint。需要 `action_head.pt` 或 `{TRAINING_STATE_FILENAME}`。"
    )


# ============================================================================
# 归一化处理类
# ============================================================================

class MinMaxNormalizer:
    """Min-Max 归一化器"""
    
    def __init__(self, min_vals: torch.Tensor, max_vals: torch.Tensor):
        self.min_vals = min_vals
        self.max_vals = max_vals
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        mask = min_vals != max_vals
        normalized = torch.zeros_like(x)
        if mask.any():
            normalized[..., mask] = (x[..., mask] - min_vals[mask]) / (max_vals[mask] - min_vals[mask])
            normalized[..., mask] = 2.0 * normalized[..., mask] - 1.0
        return normalized
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        return (x + 1.0) / 2.0 * (max_vals - min_vals) + min_vals


def load_normalization_stats(dataset_path: str, convert_quat_to_axisangle: bool = True, load_state_stats: bool = False) -> dict:
    """从数据集加载归一化统计信息
    
    Args:
        dataset_path: 数据集路径
        convert_quat_to_axisangle: 是否将四元数转换为轴角
        load_state_stats: 是否加载 observation.state 的原始统计信息（用于 state mapper）
    
    Returns:
        包含归一化器和统计信息的字典
    """
    import math
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"[WARN] stats.json 不存在于 {stats_path}，将不进行归一化")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    normalizers = {}
    
    if 'observation.state' in stats:
        state_stats = stats['observation.state']
        original_min = state_stats['min']
        original_max = state_stats['max']
        original_dim = len(original_min)
        
        if convert_quat_to_axisangle and original_dim == 9:
            state_min = torch.tensor([
                original_min[0], original_min[1],
                original_min[2], original_min[3], original_min[4],
                -math.pi, -math.pi, -math.pi,
            ], dtype=torch.float32)
            state_max = torch.tensor([
                original_max[0], original_max[1],
                original_max[2], original_max[3], original_max[4],
                math.pi, math.pi, math.pi,
            ], dtype=torch.float32)
            print(f"✓ State 归一化统计量: {original_dim}维 → 8维 (四元数→轴角转换)")
        else:
            state_min = torch.tensor(original_min, dtype=torch.float32)
            state_max = torch.tensor(original_max, dtype=torch.float32)
            print(f"✓ 加载 state 归一化统计量，维度: {len(state_min)}")
        
        normalizers['state'] = MinMaxNormalizer(state_min, state_max)
    
    if 'action' in stats:
        action_stats = stats['action']
        action_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        action_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        print(f"✓ 加载 action 归一化统计量，维度: {len(action_min)}")
        normalizers['action'] = MinMaxNormalizer(action_min, action_max)
    
    return normalizers if normalizers else None


# ============================================================================
# 混合 Loss 计算
# ============================================================================

def compute_masked_l1_loss(predicted_actions, gt_actions, action_chunk_mask=None):
    """
    带 mask 的 L1 Loss，mask=0 的位置（零填充）不参与 loss 计算。
    
    Args:
        predicted_actions: (batch, 1, chunk_size * action_dim)
        gt_actions: (batch, 1, chunk_size * action_dim)
        action_chunk_mask: (batch, 1, chunk_size * action_dim) 或 None
                          1=有效位置, 0=填充位置
    """
    if action_chunk_mask is None:
        return torch.nn.functional.l1_loss(predicted_actions, gt_actions)
    
    elementwise_loss = torch.abs(predicted_actions - gt_actions)
    masked_loss = elementwise_loss * action_chunk_mask
    valid_count = action_chunk_mask.sum().clamp(min=1.0)
    return masked_loss.sum() / valid_count


def compute_mixed_loss(predicted_actions, gt_actions, 
                       mae_criterion, bce_criterion, 
                       bce_columns, action_dim, 
                       mae_weight=1.0, bce_weight=1.0,
                       action_chunk_mask=None):
    """
    混合 Loss: 指定列使用 BCEWithLogitsLoss，其余列使用 L1Loss
    支持 action_chunk_mask 屏蔽零填充位置。
    
    Args:
        predicted_actions: (batch, 1, chunk_size * action_dim)
        gt_actions: (batch, 1, chunk_size * action_dim)
        mae_criterion: nn.L1Loss 实例 (仅在无 mask 时使用)
        bce_criterion: nn.BCEWithLogitsLoss 实例 (仅在无 mask 时使用)
        bce_columns: 使用 BCE loss 的 action 列索引列表
        action_dim: action 维度数
        mae_weight: MAE loss 权重
        bce_weight: BCE loss 权重
        action_chunk_mask: (batch, 1, chunk_size * action_dim) 或 None
    
    Returns:
        加权混合 loss
    """
    batch_size = predicted_actions.shape[0]
    chunk_size = predicted_actions.shape[2] // action_dim
    
    # reshape: (batch, 1, C*D) -> (batch, C, D)
    pred = predicted_actions.view(batch_size, chunk_size, action_dim)
    gt = gt_actions.view(batch_size, chunk_size, action_dim)
    
    # reshape mask: (batch, 1, C*D) -> (batch, C, D), 取第一个 action_dim 的列即可代表每步
    chunk_mask = None  # (batch, C, 1) 每个 chunk 步的 mask
    if action_chunk_mask is not None:
        mask_reshaped = action_chunk_mask.view(batch_size, chunk_size, action_dim)
        chunk_mask = mask_reshaped[:, :, :1]  # (batch, C, 1)
    
    # 构建 MAE 列索引 (排除 BCE 列)
    bce_set = set(bce_columns)
    mae_columns = [d for d in range(action_dim) if d not in bce_set]
    
    total_loss = 0.0
    
    # MAE loss (连续维度)
    if mae_columns:
        pred_mae = pred[:, :, mae_columns]
        gt_mae = gt[:, :, mae_columns]
        if chunk_mask is not None:
            elementwise = torch.abs(pred_mae - gt_mae) * chunk_mask.expand_as(pred_mae)
            valid_count = (chunk_mask.expand_as(pred_mae)).sum().clamp(min=1.0)
            mae_loss = elementwise.sum() / valid_count
        else:
            mae_loss = mae_criterion(pred_mae, gt_mae)
        total_loss = total_loss + mae_weight * mae_loss
    
    # BCE loss (二值维度)
    if bce_columns:
        pred_bce = pred[:, :, bce_columns]
        gt_bce = gt[:, :, bce_columns]
        gt_bce_01 = (gt_bce + 1.0) / 2.0
        if chunk_mask is not None:
            elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_bce, gt_bce_01, reduction='none'
            ) * chunk_mask.expand_as(pred_bce)
            valid_count = (chunk_mask.expand_as(pred_bce)).sum().clamp(min=1.0)
            bce_loss = elementwise.sum() / valid_count
        else:
            bce_loss = bce_criterion(pred_bce, gt_bce_01)
        total_loss = total_loss + bce_weight * bce_loss
    
    return total_loss


# ============================================================================
# 数据处理函数
# ============================================================================

def oft_collate_fn(batch, normalizers=None, convert_quat_to_axisangle=True,
                   state_norm_columns_minmax=None,
                   use_action_polyfit=False, action_polyfit_degree=3,
                   use_action_from_state_diff=False, action_from_state_diff_degree=3,
                   state_diff_columns=None, action_diff_target_columns=None, action_keep_original_columns=None):
    """
    将 LeRobot batch 转换为 OFT 模型需要的格式
    
    OFT 模型输入:
    - vlm_hidden_states: List[Tensor], 每个形状 (batch, seq_len, hidden_dim)
    - proprioception: (batch, proprio_dim)
    - vlm_attention_mask: (batch, seq_len), 1=valid, 0=padding
    
    OFT 模型输出:
    - action_predictions: (batch, 1, NUM_ACTIONS_CHUNK * ACTION_DIM)
    
    Args:
        batch: LeRobotDataset 返回的 batch
        normalizers: 归一化器
        convert_quat_to_axisangle: 是否将 9 维 state (四元数) 转换为 8 维 state (轴角)
        state_norm_columns_minmax: State 归一化的列索引列表，用于从原始 stats 中提取对应列的 min/max
        use_action_polyfit: 是否对 action chunk 进行多项式拟合
        action_polyfit_degree: 多项式拟合阶数
        use_action_from_state_diff: 是否使用 state 差分替代 action (对 state chunk 进行多项式拟合后计算 state[t+1] - state[t])
        action_from_state_diff_degree: state 差分时多项式拟合阶数
        state_diff_columns: 参与差分计算的 state 列索引列表 (与 action_diff_target_columns 一一对应)
        action_diff_target_columns: 差分结果赋值的 action 列索引列表 (与 state_diff_columns 一一对应)
        action_keep_original_columns: 保持原始 action 值的列索引列表 (不被 state 差分替换)
    """
    vlm_tensor_raw = batch['vlm_hidden_states']  # (batch, num_layers, seq_len, hidden_dim)
    vlm_attention_mask_raw = batch.get('vlm_attention_mask', None)  # (batch, num_layers, seq_len)
    observation_state = batch['observation_state']  # (batch, state_dim)
    actions = batch['actions']  # (batch, num_chunks, action_dim)
    
    batch_size = vlm_tensor_raw.size(0)
    num_layers = vlm_tensor_raw.size(1)
    seq_len = vlm_tensor_raw.size(2)
    num_chunks = actions.size(1)
    action_dim = actions.size(2)
    
    # State 四元数转轴角
    if convert_quat_to_axisangle:
        original_state_dim = observation_state.size(1)
        if original_state_dim == 9:
            observation_state = convert_state_quat_to_axisangle(observation_state)
    
    actual_state_dim = observation_state.size(1)
    actual_action_dim = actions.size(2)
    
    # State 差分替代 Action: 使用 state chunk 拟合后的差分值替代 action
    # 计算方式: 对 state chunk 进行多项式拟合，然后计算 fitted_state[t+1] - fitted_state[t]
    if use_action_from_state_diff:
        observation_states_chunk = batch.get('observation_states_chunk', None)
        if observation_states_chunk is not None:
            # observation_states_chunk: (batch, num_chunks, state_dim)
            states_chunk = observation_states_chunk.clone()
            
            # 如果启用四元数转轴角，对 state chunk 也进行相同处理
            if convert_quat_to_axisangle:
                original_chunk_state_dim = states_chunk.size(2)
                if original_chunk_state_dim == 9:
                    # states_chunk: (batch, num_chunks, 9) -> (batch, num_chunks, 8)
                    states_chunk_reshaped = states_chunk.view(batch_size * num_chunks, 9)
                    states_chunk_converted = convert_state_quat_to_axisangle(states_chunk_reshaped)
                    states_chunk = states_chunk_converted.view(batch_size, num_chunks, -1)
            
            chunk_state_dim = states_chunk.size(2)
            frames = np.arange(num_chunks)
            device = actions.device
            dtype = actions.dtype
            
            # 对每个 batch 和每个 state 维度进行多项式拟合并计算差分
            states_np = states_chunk.cpu().numpy()
            actions_np = actions.cpu().numpy()
            
            # 确定要处理的列映射关系
            if state_diff_columns is not None and action_diff_target_columns is not None:
                # 使用自定义列映射: state_diff_columns[i] -> action_diff_target_columns[i]
                assert len(state_diff_columns) == len(action_diff_target_columns), \
                    f"state_diff_columns ({len(state_diff_columns)}) 和 action_diff_target_columns ({len(action_diff_target_columns)}) 长度必须相同"
                column_mapping = list(zip(state_diff_columns, action_diff_target_columns))
            else:
                # 默认映射: state[d] -> action[d] (d < min(state_dim, action_dim))
                max_dim = min(chunk_state_dim, actual_action_dim)
                column_mapping = [(d, d) for d in range(max_dim)]
            
            # 排除需要保持原始值的 action 列
            if action_keep_original_columns is not None:
                keep_set = set(action_keep_original_columns)
                column_mapping = [(s, a) for s, a in column_mapping if a not in keep_set]
            
            for b in range(batch_size):
                for state_col_idx, action_col_idx in column_mapping:
                    # 检查索引是否有效
                    if state_col_idx >= chunk_state_dim or action_col_idx >= actual_action_dim:
                        continue
                    
                    # 对 state 进行多项式拟合
                    state_col = states_np[b, :, state_col_idx]
                    fitted_state, _, _ = polyfit_chunk(frames, state_col, degree=action_from_state_diff_degree)
                    
                    # 计算差分: state[t+1] - state[t]
                    # fitted_state 有 num_chunks 个点，差分得到 num_chunks-1 个值
                    state_diff = np.diff(fitted_state)  # (num_chunks - 1,)
                    # 补齐到 num_chunks: 用最后一个差分值填充末尾
                    action_from_state = np.concatenate([state_diff, [state_diff[-1]]])  # (num_chunks,)
                    
                    # 替换 action 对应维度
                    actions_np[b, :, action_col_idx] = action_from_state
            
            actions = torch.from_numpy(actions_np).to(device=device, dtype=dtype)
    
    # 归一化
    if normalizers is not None:
        if 'state' in normalizers:
            # 检查 state 维度是否匹配（state 预处理可能改变维度）
            normalizer_dim = normalizers['state'].min_vals.shape[0]
            if normalizer_dim == actual_state_dim:
                # 维度匹配，直接归一化
                observation_state = normalizers['state'].normalize(observation_state)
            elif state_norm_columns_minmax is not None and len(state_norm_columns_minmax) > 0:
                # 维度不匹配但指定了归一化列，从原始 stats 中提取对应列的 min/max
                # state_norm_columns_minmax 指定了原始 stats 中的列索引
                selected_min = normalizers['state'].min_vals[state_norm_columns_minmax]
                selected_max = normalizers['state'].max_vals[state_norm_columns_minmax]
                # 创建临时归一化器，只对预处理后 state 的前 len(state_norm_columns_minmax) 列归一化
                temp_normalizer = MinMaxNormalizer(selected_min, selected_max)
                num_cols_to_normalize = len(state_norm_columns_minmax)
                if num_cols_to_normalize <= actual_state_dim:
                    # 只归一化前 num_cols_to_normalize 列
                    observation_state[:, :num_cols_to_normalize] = temp_normalizer.normalize(
                        observation_state[:, :num_cols_to_normalize]
                    )
            # 如果维度不匹配且没有指定列，跳过 state 归一化
        if 'action' in normalizers:
            actions_flat = actions.reshape(batch_size * num_chunks, actual_action_dim)
            actions_normalized = normalizers['action'].normalize(actions_flat)
            actions = actions_normalized.reshape(batch_size, num_chunks, actual_action_dim)
    
    # Action Polyfit: 对每个 action chunk 进行多项式拟合
    if use_action_polyfit:
        frames = np.arange(num_chunks)
        device = actions.device
        dtype = actions.dtype
        actions_np = actions.cpu().numpy()
        
        for b in range(batch_size):
            for d in range(actual_action_dim):
                action_col = actions_np[b, :, d]
                fitted_values, _, _ = polyfit_chunk(frames, action_col, degree=action_polyfit_degree)
                actions_np[b, :, d] = fitted_values
        
        actions = torch.from_numpy(actions_np).to(device=device, dtype=dtype)
    
    # VLM hidden states: 拆分为列表
    vlm_hidden_states = [vlm_tensor_raw[:, i, :, :] for i in range(num_layers)]
    
    # VLM attention mask: 处理为 (batch, seq_len * num_layers)
    # 因为 VLM hidden states 会沿 seq_len 维度拼接
    # mask: 1=valid, 0=padding
    if vlm_attention_mask_raw is not None:
        # 原始形状: (batch, num_layers, seq_len)
        # 拼接后形状: (batch, seq_len * num_layers)
        # 按照 vlm_hidden_states 拼接的顺序: layer0_seq + layer1_seq + ...
        vlm_attention_mask = vlm_attention_mask_raw.view(batch_size, num_layers * seq_len)
    else:
        # 如果没有 mask，创建全 1 的 mask（所有位置都有效）
        vlm_attention_mask = None
    
    # Proprioception: 直接使用转换后的 state
    proprioception = observation_state  # (batch, 8)
    
    # ========== 处理 ground truth actions ==========
    # 连续模式: reshape 为 (batch, 1, NUM_ACTIONS_CHUNK * ACTION_DIM)
    gt_actions = actions.view(batch_size, 1, num_chunks * actual_action_dim)
    
    # ========== 处理 action chunk mask ==========
    # mask: (batch, num_chunks) -> 展开为 (batch, 1, num_chunks * action_dim)
    # 每个 chunk 步的 mask 值重复 action_dim 次，与 gt_actions flat 形状对齐
    action_chunk_mask = batch.get('action_chunk_mask', None)
    action_chunk_mask_flat = None
    if action_chunk_mask is not None:
        # (batch, num_chunks) -> (batch, num_chunks, 1) -> (batch, num_chunks, action_dim) -> (batch, 1, num_chunks * action_dim)
        action_chunk_mask_flat = action_chunk_mask.unsqueeze(-1).expand(
            batch_size, num_chunks, actual_action_dim
        ).reshape(batch_size, 1, num_chunks * actual_action_dim)
    
    return {
        'vlm_hidden_states': vlm_hidden_states,
        'proprioception': proprioception,
        'gt_actions': gt_actions,
        'vlm_attention_mask': vlm_attention_mask,  # (batch, seq_len * num_layers) or None
        'action_chunk_mask': action_chunk_mask_flat,  # (batch, 1, num_chunks * action_dim) or None
    }


def safe_wandb_log(log_dict, step=None, rank=0, max_retries=2, silent=True):
    """
    安全的 wandb 日志记录 - 网络断开不会中断训练
    
    Args:
        log_dict: 要记录的字典
        step: 步数
        rank: 进程 rank
        max_retries: 最大重试次数
        silent: 是否静默失败（不打印警告）
    """
    if not is_main_process(rank):
        return
    
    for attempt in range(max_retries):
        try:
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)
            return  # 成功则返回
        except Exception as e:
            if attempt == max_retries - 1 and not silent:
                print(f"⚠️ Wandb log 失败 (网络问题?): {type(e).__name__}")
            continue


# ============================================================================
# GPU 实时监控
# ============================================================================

# 尝试导入 pynvml
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️ pynvml 未安装，GPU 温度/风扇监控不可用。安装: pip install pynvml")

_nvml_initialized = False

def init_nvml():
    """初始化 NVML"""
    global _nvml_initialized
    if not PYNVML_AVAILABLE:
        return False
    if _nvml_initialized:
        return True
    try:
        pynvml.nvmlInit()
        _nvml_initialized = True
        print("✓ NVML 初始化成功，GPU 监控已启用")
        return True
    except Exception as e:
        print(f"⚠️ NVML 初始化失败: {e}")
        print("  GPU 温度/利用率监控不可用，将只记录基本显存信息")
        return False

def shutdown_nvml():
    """关闭 NVML"""
    global _nvml_initialized
    if _nvml_initialized and PYNVML_AVAILABLE:
        try:
            pynvml.nvmlShutdown()
            _nvml_initialized = False
        except Exception:
            pass

def get_gpu_stats(device_index=0):
    """
    获取单个 GPU 实时信息
    
    Args:
        device_index: NVML 物理 GPU 索引
    
    Returns:
        dict: 包含温度、显存、功率、风扇等信息（key 不含 gpu_ 前缀）
    """
    stats = {}
    
    if not PYNVML_AVAILABLE or not init_nvml():
        return stats
    
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        
        # 温度 (摄氏度)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            stats['temperature_c'] = temp
        except Exception:
            pass
        
        # 显存使用
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats['memory_used_gb'] = round(mem_info.used / (1024**3), 3)
            stats['memory_total_gb'] = round(mem_info.total / (1024**3), 3)
            stats['memory_util_percent'] = round(mem_info.used / mem_info.total * 100, 1)
        except Exception:
            pass
        
        # GPU 利用率
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            stats['utilization_percent'] = util.gpu
            stats['memory_bandwidth_util_percent'] = util.memory
        except Exception:
            pass
        
        # 功率 (瓦特)
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle)  # 单位是毫瓦
            stats['power_w'] = round(power / 1000, 1)
        except Exception:
            pass
        
        # 功率限制
        try:
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
            stats['power_limit_w'] = round(power_limit / 1000, 1)
        except Exception:
            pass
        
        # 风扇转速 (百分比) - 部分 GPU 可能不支持
        try:
            fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            stats['fan_speed_percent'] = fan_speed
        except Exception:
            pass  # 很多服务器 GPU 没有可读取的风扇信息
        
        # GPU 时钟频率
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            stats['sm_clock_mhz'] = sm_clock
            stats['mem_clock_mhz'] = mem_clock
        except Exception:
            pass
        
    except Exception:
        pass
    
    return stats


def get_all_gpu_stats():
    """
    获取所有物理 GPU 的统计信息
    
    使用 NVML 获取所有物理 GPU（不受 CUDA_VISIBLE_DEVICES 限制）
    
    Returns:
        dict: 包含所有 GPU 的统计信息，格式：
            - system/gpu.0.temperature_c, system/gpu.1.temperature_c, ...（每个 GPU）
            - system/gpu_temperature_c, system/gpu_utilization_percent, ...（汇总/GPU 0）
    """
    all_stats = {}
    
    if not PYNVML_AVAILABLE or not init_nvml():
        # 退回使用 PyTorch 获取基本显存信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                all_stats[f'system/gpu.{i}.memory_used_gb'] = round(torch.cuda.memory_allocated(i) / (1024**3), 3)
                all_stats[f'system/gpu.{i}.memory_reserved_gb'] = round(torch.cuda.memory_reserved(i) / (1024**3), 3)
        return all_stats
    
    try:
        # 使用 NVML 获取所有物理 GPU 数量（不受 CUDA_VISIBLE_DEVICES 限制）
        num_gpus = pynvml.nvmlDeviceGetCount()
    except Exception:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # 记录每个 GPU 的信息
    for i in range(num_gpus):
        gpu_stats = get_gpu_stats(i)
        for key, value in gpu_stats.items():
            # 格式: system/gpu.0.temperature_c (使用点分隔，Wandb 会自动分组)
            all_stats[f'system/gpu.{i}.{key}'] = value
    
    # 添加汇总信息（Wandb System 面板期望的格式）
    # 使用 GPU 0 的数据作为汇总值
    if num_gpus > 0:
        first_gpu_stats = get_gpu_stats(0)
        for key, value in first_gpu_stats.items():
            # 格式: system/gpu_temperature_c (与 Wandb 默认 System 面板匹配)
            all_stats[f'system/gpu_{key}'] = value
    
    return all_stats


# ============================================================================
# 主训练函数
# ============================================================================

def main():
    # 声明全局变量（用于共享内存缓存清理）
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    
    parser = argparse.ArgumentParser(
        description='Train OFT Action Head with Multi-GPU Support (Qwen2B)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    config = TrainingConfig()
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, required=True, help="LeRobot 数据集路径")
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_workers", type=int, default=config.num_workers)
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="每个 worker 预取的 batch 数量 (仅在 num_workers>0 时生效) | Prefetch factor per worker (default: 4)")
    parser.add_argument("--val_split", type=float, default=config.val_split)
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--steps", type=int, default=config.steps)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=config.warmup_ratio)
    parser.add_argument("--lr_scheduler_type", type=str, default=config.lr_scheduler_type,
                        choices=["cosine", "constant", "warmup_step_decay"],
                        help="学习率调度类型: cosine (warmup + cosine decay), constant (warmup + 保持不变), warmup_step_decay (warmup + 保持 + step后衰减)")
    parser.add_argument("--lr_decay_step", type=int, default=config.lr_decay_step,
                        help="warmup_step_decay 模式: 从第几个 step 开始改变学习率")
    parser.add_argument("--lr_decay_value", type=float, default=config.lr_decay_value,
                        help="warmup_step_decay 模式: 改变后的学习率值")
    parser.add_argument("--adam_beta1", type=float, default=config.adam_beta1)
    parser.add_argument("--adam_beta2", type=float, default=config.adam_beta2)
    
    # 模型参数
    parser.add_argument("--num_transformer_blocks", type=int, default=config.num_transformer_blocks)
    parser.add_argument("--num_attention_heads", type=int, default=config.num_attention_heads)
    parser.add_argument("--dropout", type=float, default=config.dropout)
    parser.add_argument("--action_head_hidden_dim", type=int, default=config.action_head_hidden_dim)
    parser.add_argument("--num_vlm_layers", type=int, default=None)
    parser.add_argument("--vlm_output_dim", type=int, default=None)
    
    # 权重与保存
    parser.add_argument("--out_dir", type=str, default=config.out_dir)
    parser.add_argument("--log_dir", type=str, default=config.log_dir)
    parser.add_argument("--save_every", type=int, default=config.save_every)
    parser.add_argument("--save_every_steps", type=int, default=config.save_every_steps)
    parser.add_argument("--resume_from_checkpoint", type=str, default="",
                        help="从指定 checkpoint 恢复训练，支持传入 step_xxx/epoch_xxx 目录或其中的文件路径")
    parser.add_argument("--resume_reset_lr", action="store_true", default=False,
                        help="resume 时只加载模型权重，强制 step/optimizer/scheduler/scaler/RNG 全部重置 "
                             "→ 学习率从 step 0 重新走 warmup → decay (warmup_step_decay 也会从头计算). "
                             "适用场景: 用某个 ckpt 当作起点重新调度学习率 / 换数据集继续训练.")
    parser.add_argument("--no_save_best", action="store_true", default=False,
                        help="禁用保存最佳模型 | Disable saving best model")
    
    # 系统配置
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 内存优化参数
    parser.add_argument("--use_amp", action="store_true", default=config.use_amp)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--amp_dtype", type=str, default=config.amp_dtype, choices=["float16", "bfloat16"])
    parser.add_argument("--empty_cache_freq", type=int, default=config.empty_cache_freq)
    
    # 数据加载优化参数
    parser.add_argument("--vlm_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="加载 VLM hidden states 时的数据类型 (bfloat16 可减少约 50%% 显存)")
    parser.add_argument("--skip_images", action="store_true", default=False,
                        help="跳过加载图像 (仅使用预保存的 VLM hidden states 训练时使用)")
    parser.add_argument("--cache_vlm_states", action="store_true", default=False,
                        help="缓存 VLM hidden states 到 RAM (使用 mmap 模式)")
    parser.add_argument("--cache_max_samples", type=int, default=-1,
                        help="[已弃用] 最大缓存样本数")
    parser.add_argument("--use_shared_cache", action="store_true", default=False,
                        help="使用共享内存缓存 (多 GPU 共享，rank 0 预加载，自动检测形状，推荐)")
    parser.add_argument("--cache_dtype", type=str, default="float32", choices=["float32", "float16"],
                        help="共享内存缓存的数据类型 (float16 可减半内存占用)")
    
    # WebDataset 大规模数据训练优化
    parser.add_argument("--use_webdataset", action="store_true", default=False,
                        help="使用 WebDataset tar 分片格式加载数据 (适用于超大规模数据集)")
    parser.add_argument("--webdataset_shard_pattern", type=str, default="",
                        help="WebDataset 分片路径模式 (如: /path/shards/shard-{000000..000099}.tar)")
    parser.add_argument("--webdataset_shuffle_buffer", type=int, default=10000,
                        help="WebDataset shuffle 缓冲区大小 (越大随机性越好)")
    
    # WebDataset 分批缓存模式 (将分片分批加载到内存，加速数据读取)
    parser.add_argument("--use_webdataset_cached", action="store_true", default=False,
                        help="使用 WebDataset 分批缓存模式 (分批加载分片到内存)")
    parser.add_argument("--webdataset_cache_shards", type=int, default=4,
                        help="每批缓存的分片数量 (默认: 4)")
    parser.add_argument("--webdataset_cache_dtype", type=str, default="float32",
                        choices=["float32", "float16"],
                        help="分批缓存的数据类型 (float16 可减半内存)")

    # Chunk 分批缓存模式 (基于 lerobot chunk-XXX.npz，原生格式无需转 WebDataset)
    parser.add_argument("--use_chunk_batch_cache", action="store_true", default=False,
                        help="使用 ChunkBatchCache 分批共享内存缓存 "
                             "(直接读 lerobot vlm_hidden_states/chunk-XXX.npz, 内存>数据时仍可单批装下；"
                             "内存不足时自动按 chunk 切批装入 /dev/shm)")
    parser.add_argument("--chunk_batch_safety_ratio", type=float, default=0.65,
                        help="ChunkBatchCache 内存预算安全系数, "
                             "预算 = min(RAM_avail, /dev/shm_avail) × ratio (默认 0.65)")
    parser.add_argument("--chunk_batch_cache_dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="ChunkBatchCache 共享内存中 VLM 数据类型 (float16 可减半占用)")
    parser.add_argument("--chunk_batch_min_chunks", type=int, default=1,
                        help="ChunkBatchCache 每批最少 chunk 数 (默认 1)")
    parser.add_argument("--chunk_batch_max_chunks", type=int, default=-1,
                        help="ChunkBatchCache 每批最多 chunk 数, -1 表示不限")
    parser.add_argument("--chunk_batch_manual_chunks", type=int, default=-1,
                        help="ChunkBatchCache 直接指定每批 chunk 数, -1 表示自动按内存预算估算")
    parser.add_argument("--chunk_batch_inflation", type=float, default=1.05,
                        help="ChunkBatchCache 估算每 chunk RAM 占用时的安全放大倍数 (默认 1.05)")
    parser.add_argument("--chunk_batch_seed", type=int, default=42,
                        help="ChunkBatchCache 跨批 chunk 顺序的随机种子 (epoch 内 shuffle 用 seed+epoch)")
    # ---- Multi-Dataset (跨子集) 训练模式 ----
    # 开启后，会扫描 --multi_dataset_root 下所有 LeRobot 子集，自动跨子集 shuffle vlm chunks
    # 并自动 force --use_chunk_batch_cache=True (多源场景一定走 SHM 分批模式)。
    parser.add_argument("--use_multi_dataset", action="store_true", default=False,
                        help="开启多子集 LeRobot 训练; data_path 当作 root, 下面是 N 个独立子集")
    parser.add_argument("--multi_dataset_root", type=str, default="",
                        help="多子集 root 目录; 留空时回退到 --data_path")
    parser.add_argument("--multi_dataset_include", type=str, default="",
                        help="逗号分隔的白名单子集名; 留空表示自动扫所有子目录")
    parser.add_argument("--multi_dataset_exclude", type=str, default="",
                        help="逗号分隔的黑名单子集名")
    parser.add_argument("--multi_dataset_stats_strategy", type=str, default="per_subset",
                        choices=["per_subset", "minmax_union"],
                        help="跨子集归一化策略 (per_subset 默认)")
    parser.add_argument("--multi_dataset_require_complete_vlm", action="store_true", default=True,
                        help="vlm 不完整的子集就跳过 (默认开启)")
    parser.add_argument("--multi_dataset_no_require_complete_vlm",
                        dest="multi_dataset_require_complete_vlm", action="store_false",
                        help="即使 vlm 不完整也保留 (仅排查用)")
    parser.add_argument("--multi_dataset_strict_state_dim", action="store_true", default=False,
                        help="严格要求各子集 state_dim 一致; 默认允许并 right-pad 到 PROPRIO_DIM")
    parser.add_argument("--multi_dataset_target_action_dim", type=int, default=-1,
                        help="强制 action_dim 等于这个值; <=0 时取多数派")
    parser.add_argument("--multi_dataset_save_manifest", type=str, default="",
                        help="把扫描结果落盘到这个 JSON 路径, 便于审计 / resume")
    parser.add_argument("--multi_dataset_dry_run", action="store_true", default=False,
                        help="只扫子集 + 打印计划 + 写 manifest, 然后退出, 不真训")
    # Step-Segments 模式：把整训按 step 切成 num_batches 段，每段绑定一个 ChunkBatch
    # 整训只装载 num_batches 次 chunk → SHM（vs 旧逻辑每 epoch × num_batches 次）。
    # 默认开启；通过 --no_step_based_segments 可回退到旧逻辑（每 epoch 切批）。
    parser.add_argument("--use_step_based_segments", dest="use_step_based_segments",
                        action="store_true",
                        help="ChunkBatchCache 模式下按 step 分段加载 ChunkBatch (默认: 开启)")
    parser.add_argument("--no_step_based_segments", dest="use_step_based_segments",
                        action="store_false",
                        help="回退到旧的 per-epoch 切批逻辑")
    parser.set_defaults(use_step_based_segments=True)

    # ---------------------------------------------------------------------- #
    # Disk-Shuffle 模式 (无 SHM, 全局 shuffle, 训练时按需 mmap 读单帧)
    # ---------------------------------------------------------------------- #
    # 设计目标:
    #   彻底绕开 ChunkBatchCache 的"分段进 SHM"机制 → 全局 sample 池 shuffle,
    #   切段不再发生, loss 突刺消失。代价是每个 sample 都要从磁盘 mmap 读单帧,
    #   性能强依赖磁盘随机 IO (NVMe 上可用, HDD 上慢 50-200 倍, 启动时会告警)。
    # 互斥性:
    #   --use_disk_shuffle 优先级高于 ChunkBatchCache / WebDataset / SharedCache,
    #   开启时会自动关闭其它 cache 路径。多源模式 (--use_multi_dataset) 下原本会
    #   force --use_chunk_batch_cache, 这里改为只在没开 disk_shuffle 时 force。
    parser.add_argument("--use_disk_shuffle", action="store_true", default=False,
                        help="启用 Disk-Shuffle 模式: 不预加载到 SHM, 训练时按需 mmap 读单帧, "
                             "支持全局 shuffle 不受 SHM 大小限制。注意: 需要 NVMe/SSD 才能跑出合理速度.")
    parser.add_argument("--disk_shuffle_npz_lru_per_worker", type=int, default=-1,
                        help="每个 DataLoader worker 缓存的 npz mmap 句柄数; -1 表示不限 "
                             "(每个 worker 持有所有 chunk mmap, 推荐). 想严格控制虚拟内存才设正数.")
    parser.add_argument("--disk_shuffle_parquet_cache_max", type=int, default=4096,
                        help="LeRobotDataset 内 parquet_cache 上限 (LRU 驱逐); "
                             "disk-shuffle 模式下 worker 会接触到全部 episode, "
                             "parquet 文件多时需要限制大小防止 OOM. 0 表示不限.")

    # State 四元数转轴角配置
    parser.add_argument("--convert_quat_to_axisangle", action="store_true", default=True,
                        help="将 9 维 state (四元数) 转换为 8 维 state (轴角)")
    parser.add_argument("--no_convert_quat_to_axisangle", action="store_false", 
                        dest="convert_quat_to_axisangle",
                        help="禁用四元数转轴角转换")
    
    # State 归一化参数
    parser.add_argument("--state_norm_columns_minmax", type=int, nargs="+", default=None,
                        help="使用 minmax 归一化的列索引，如: --state_norm_columns_minmax 0 1 2 3 4")
    
    # State 预处理参数
    parser.add_argument("--state_process_order", type=str, nargs="+", default=None,
                        help="State 预处理执行顺序，如: hand_binary")
    parser.add_argument("--hand_binary_columns", type=int, nargs="+", default=None,
                        help="原始 state 中手部数据列范围，每组2个数 [start, end)，如: 6 12 18 24")
    parser.add_argument("--hand_binary_threshold", type=float, default=442.0,
                        help="手部二值化阈值")
    
    # Action Polyfit 参数
    parser.add_argument("--use_action_polyfit", action="store_true", default=False,
                        help="对 action chunk 进行多项式拟合")
    parser.add_argument("--action_polyfit_degree", type=int, default=3,
                        help="多项式拟合阶数 (默认 3)")
    
    # Action from State Diff 参数 (使用 state 差分替代 action)
    parser.add_argument("--use_action_from_state_diff", action="store_true", default=False,
                        help="使用 state 差分替代 action (对 state chunk 进行多项式拟合后计算 state[t+1] - state[t])")
    parser.add_argument("--action_from_state_diff_degree", type=int, default=3,
                        help="state 差分时多项式拟合阶数 (默认 3)")
    parser.add_argument("--state_diff_columns", type=int, nargs="+", default=None,
                        help="参与差分计算的 state 列索引列表 (与 --action_diff_target_columns 一一对应)，如: 0 1 2 3 4 5 6 7")
    parser.add_argument("--action_diff_target_columns", type=int, nargs="+", default=None,
                        help="差分结果赋值的 action 列索引列表 (与 --state_diff_columns 一一对应)，如: 0 1 2 3 4 5 6 7")
    parser.add_argument("--action_keep_original_columns", type=int, nargs="+", default=None,
                        help="保持原始 action 值的列索引列表 (不被 state 差分替换)，如: 8 9 10 11 12 13")
    
    # 混合 Loss 参数
    parser.add_argument("--use_mixed_loss", action="store_true", default=False,
                        help="启用混合 loss (MAE + BCE)")
    parser.add_argument("--bce_action_columns", type=int, nargs="+", default=None,
                        help="使用 BCE loss 的 action 列索引，如: --bce_action_columns 6")
    parser.add_argument("--mae_loss_weight", type=float, default=1.0,
                        help="MAE loss 权重 (默认 1.0)")
    parser.add_argument("--bce_loss_weight", type=float, default=1.0,
                        help="BCE loss 权重 (默认 1.0)")
    
    # Wandb 配置
    parser.add_argument("--use_wandb", action="store_true", default=config.use_wandb)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb_project", type=str, default=config.wandb_project)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_log_freq", type=int, default=config.wandb_log_freq,
                        help=f"Wandb 基本日志记录频率 (每 N 步) | Basic wandb logging frequency (default: {config.wandb_log_freq})")
    
    args = parser.parse_args()

    # ========== Disk-Shuffle 模式与其它 cache 的互斥处理 ==========
    if getattr(args, "use_disk_shuffle", False):
        if not _DISK_SHUFFLE_AVAILABLE:
            raise SystemExit(
                f"--use_disk_shuffle 已开启但 disk_vlm_cache 模块导入失败: {_DISK_SHUFFLE_IMPORT_ERROR}"
            )
        # disk-shuffle 优先级最高, 关闭其它 cache 路径
        if args.use_chunk_batch_cache:
            print("⚠️ --use_disk_shuffle 已开启, 自动关闭 --use_chunk_batch_cache")
            args.use_chunk_batch_cache = False
        if args.use_shared_cache:
            print("⚠️ --use_disk_shuffle 已开启, 自动关闭 --use_shared_cache")
            args.use_shared_cache = False
        if args.use_webdataset:
            print("⚠️ --use_disk_shuffle 已开启, 自动关闭 --use_webdataset")
            args.use_webdataset = False
        if args.use_webdataset_cached:
            print("⚠️ --use_disk_shuffle 已开启, 自动关闭 --use_webdataset_cached")
            args.use_webdataset_cached = False

    # ========== Multi-Dataset 模式早期处理 ==========
    if getattr(args, "use_multi_dataset", False):
        if not _MULTI_DATASET_AVAILABLE:
            raise SystemExit(
                f"--use_multi_dataset 已开启但多源模块导入失败: {_MULTI_DATASET_IMPORT_ERROR}"
            )
        if not args.multi_dataset_root:
            args.multi_dataset_root = args.data_path
        if not args.multi_dataset_root:
            raise SystemExit(
                "--use_multi_dataset 需要 --multi_dataset_root 或 --data_path 之一"
            )
        # 多源场景需要一个支持跨子集索引的 cache 后端:
        #   - 默认: force ChunkBatchCache (历史路径, 走 SHM 分段)
        #   - 已开 disk_shuffle: 直接走 DiskVLMCacheManager (跨子集全局 shuffle)
        if not args.use_chunk_batch_cache and not getattr(args, "use_disk_shuffle", False):
            print("⚠️ --use_multi_dataset 已开启, 自动 force --use_chunk_batch_cache=True")
            args.use_chunk_batch_cache = True
        # 让下游 wandb 命名 / log 命名仍能用 data_path
        args.data_path = args.multi_dataset_root

    # ========== 初始化分布式训练环境 ==========
    rank, local_rank, world_size, is_distributed = setup_distributed()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        args.device = str(device)
    else:
        device = torch.device(args.device)
    
    # 打印分布式训练信息
    if is_distributed:
        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"分布式训练模式 Distributed Training Mode", rank)
        print_rank0(f"{'='*60}", rank)
        print_rank0(f"  World Size (总GPU数): {world_size}", rank)
        print_rank0(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}", rank)
        print_rank0(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps * world_size}", rank)
        print_rank0(f"{'='*60}\n", rank)
        print(f"[Rank {rank}] Using device: {device}")
    else:
        print(f"\n单GPU训练模式 Single-GPU Training Mode")
        print(f"  Device: {device}")
        print(f"  Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        print(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}\n")

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        # 创建 custom 文件夹用于记录每个 step 的 loss
        custom_log_dir = log_dir / "custom"
        custom_log_dir.mkdir(parents=True, exist_ok=True)
    
    if is_distributed:
        dist.barrier()
    
    # 所有进程都设置 custom_log_dir 路径（但只有主进程会写入）
    custom_log_dir = log_dir / "custom"
    
    # 生成 run name
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.wandb_run_name is None:
        run_name = f'oft_qwen2b_lr{args.lr}_bs{args.batch_size}_gpu{world_size}_{timestamp}'
    else:
        run_name = args.wandb_run_name
    
    # 初始化 wandb
    # 设置离线模式备选，网络断开时自动切换到离线模式，不会中断训练
    wandb_initialized = False
    if args.use_wandb and is_main_process(rank):
        try:
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
            # 设置 wandb 在网络不稳定时自动重试，而非崩溃
            os.environ.setdefault("WANDB_HTTP_TIMEOUT", "30")
            os.environ.setdefault("WANDB_RESUME", "allow")  # 允许从断点恢复
            
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=str(log_dir),
                tags=["oft", "action_head", "qwen2b", f"gpu{world_size}"],
                notes=f"Training OFT Action Head on {Path(args.data_path).name} with {world_size} GPUs",
                # 注意: 移除了 _stats_sample_rate_seconds 和 _stats_samples_to_average 设置
                # 这些参数在新版本的 wandb 中已不再支持
            )
            wandb_initialized = True
            print("✓ Wandb 初始化成功 (网络断开时自动缓存，恢复后同步)")
            
            # 记录系统信息
            import platform
            import socket
            system_info = {
                "system/hostname": socket.gethostname(),
                "system/platform": platform.platform(),
                "system/python_version": platform.python_version(),
                "system/pytorch_version": torch.__version__,
                "system/cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "system/cudnn_version": str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A",
                "system/num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }
            
            # 记录每个 GPU 的信息
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    system_info[f"system/gpu_{i}_name"] = props.name
                    system_info[f"system/gpu_{i}_memory_gb"] = round(props.total_memory / (1024**3), 2)
                    system_info[f"system/gpu_{i}_compute_capability"] = f"{props.major}.{props.minor}"
            
            wandb.config.update(system_info, allow_val_change=True)
            print(f"✓ 系统信息已记录到 Wandb")
            print(f"  - Hostname: {system_info['system/hostname']}")
            print(f"  - PyTorch: {system_info['system/pytorch_version']}")
            print(f"  - CUDA: {system_info['system/cuda_version']}")
            print(f"  - GPU 数量: {system_info['system/num_gpus']}")
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                print(f"  - GPU 0: {system_info.get('system/gpu_0_name', 'N/A')}")
        except Exception as e:
            print(f"⚠️ Wandb 初始化失败: {e}")
            wandb_initialized = False

    # ========== 自动检测参数 ==========
    # Qwen2B hidden dimension = 1536
    backbone_dim = 1536
    
    # 从 data_path 自动检测参数
    match = re.search(r'hidden_dim_(\d+)_(\d+)_(\d+)', args.data_path)
    
    if args.num_vlm_layers is not None:
        print_rank0(f"Using command-line specified num_vlm_layers={args.num_vlm_layers}", rank)
    elif match:
        args.num_vlm_layers = int(match.group(1))
        print_rank0(f"Auto-detected num_vlm_layers={args.num_vlm_layers} from data_path", rank)
    else:
        args.num_vlm_layers = NUM_VLM_HIDDEN_LAYERS
        print_rank0(f"Using default num_vlm_layers={args.num_vlm_layers} from constants.py", rank)
    
    if args.vlm_output_dim is not None:
        print_rank0(f"Using command-line specified vlm_output_dim={args.vlm_output_dim}", rank)
    elif match:
        args.vlm_output_dim = int(match.group(3))
        print_rank0(f"Auto-detected vlm_output_dim={args.vlm_output_dim} from data_path", rank)
    else:
        args.vlm_output_dim = backbone_dim  # 默认使用 Qwen2B 的维度
        print_rank0(f"Using default vlm_output_dim={args.vlm_output_dim}", rank)

    # 加载数据集信息
    use_multi_dataset_mode = bool(getattr(args, "use_multi_dataset", False))
    multi_dataset_index = None
    multi_dataset_normalizers = None
    convert_quat_to_axisangle = args.convert_quat_to_axisangle

    if use_multi_dataset_mode:
        # ----- Multi-Dataset 分支 -----
        print_rank0(f"\nLoading multi-subset dataset root: {args.multi_dataset_root}", rank)
        include = (
            [s.strip() for s in args.multi_dataset_include.split(",") if s.strip()]
            if args.multi_dataset_include
            else None
        )
        exclude = (
            [s.strip() for s in args.multi_dataset_exclude.split(",") if s.strip()]
            if args.multi_dataset_exclude
            else None
        )
        target_action_dim_arg = (
            args.multi_dataset_target_action_dim
            if args.multi_dataset_target_action_dim > 0
            else None
        )
        multi_dataset_index = MultiDatasetIndex.scan(
            root_dir=args.multi_dataset_root,
            include=include,
            exclude=exclude,
            require_complete_vlm=args.multi_dataset_require_complete_vlm,
            require_uniform_action_dim=True,
            target_action_dim=target_action_dim_arg,
            allow_state_dim_mismatch=(not args.multi_dataset_strict_state_dim),
            verbose=is_main_process(rank),
        )
        if args.multi_dataset_save_manifest and is_main_process(rank):
            mp = multi_dataset_index.save_manifest(args.multi_dataset_save_manifest)
            print_rank0(f"✓ multi-dataset manifest saved -> {mp}", rank)

        if args.multi_dataset_dry_run:
            print_rank0("\n[multi_dataset_dry_run] 仅扫描子集 + 打印计划, 正常退出。", rank)
            if is_distributed and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            return

        actual_action_dim = int(multi_dataset_index.max_action_dim)
        actual_state_dim_max = int(multi_dataset_index.max_state_dim)
        # 注意: total_episodes 保留为 ep_uid 全量范围 (subset_episode_offset 用), 仅日志用 eligible
        total_episodes = int(multi_dataset_index.total_global_episodes)
        eligible_eps = int(multi_dataset_index.eligible_global_episodes)
        eligible_frames = int(multi_dataset_index.eligible_global_frames)

        print_rank0(f"Multi-dataset summary:", rank)
        print_rank0(f"  Subsets:           {len(multi_dataset_index.subsets)}", rank)
        print_rank0(f"  Skipped:           {len(multi_dataset_index.skipped)}", rank)
        print_rank0(f"  Eligible episodes: {eligible_eps}  (子集声明 {total_episodes})", rank)
        print_rank0(f"  Eligible frames:   {eligible_frames}  (子集声明 {multi_dataset_index.total_global_frames})", rank)
        print_rank0(f"  Total chunks:      {len(multi_dataset_index.global_chunks)}", rank)
        print_rank0(f"  max action_dim:    {actual_action_dim}", rank)
        print_rank0(f"  max state_dim:     {actual_state_dim_max}", rank)
        print_rank0(f"  Action chunks:     {NUM_ACTIONS_CHUNK}", rank)

        if actual_action_dim != ACTION_DIM:
            print_rank0(
                f"\n[FATAL] (multi-dataset) ACTION_DIM 不匹配:\n"
                f"   constants.ACTION_DIM = {ACTION_DIM}\n"
                f"   multi-dataset uniform action_dim = {actual_action_dim}\n"
                f"   修复: export ACTION_DIM={actual_action_dim}\n",
                rank,
            )
            raise SystemExit(2)
        if actual_state_dim_max > PROPRIO_DIM:
            print_rank0(
                f"\n[FATAL] (multi-dataset) PROPRIO_DIM 容不下 max state_dim:\n"
                f"   constants.PROPRIO_DIM = {PROPRIO_DIM}\n"
                f"   multi-dataset max state_dim = {actual_state_dim_max}\n"
                f"   修复: export PROPRIO_DIM={actual_state_dim_max}\n",
                rank,
            )
            raise SystemExit(2)
        # 多源 sample 一律 right-pad 到 PROPRIO_DIM
        actual_state_dim = PROPRIO_DIM
        if actual_state_dim_max < PROPRIO_DIM:
            print_rank0(
                f"ℹ️ multi-dataset max state_dim={actual_state_dim_max} < PROPRIO_DIM="
                f"{PROPRIO_DIM}, 全部 right-pad 到 {PROPRIO_DIM}",
                rank,
            )

        print_rank0("\n构造多子集归一化器...", rank)
        print_rank0(
            f"  策略: {args.multi_dataset_stats_strategy}  四元数→轴角: "
            f"{'启用' if convert_quat_to_axisangle else '禁用'}",
            rank,
        )
        multi_dataset_normalizers = MultiDatasetNormalizers.build(
            multi_dataset_index,
            strategy=args.multi_dataset_stats_strategy,
            target_state_dim=actual_state_dim,
            target_action_dim=actual_action_dim,
            convert_quat_to_axisangle=convert_quat_to_axisangle,
            verbose=is_main_process(rank),
        )

        # 多源场景下, state/action 归一化 + pad 在 MultiLeRobotDataset.__getitem__ 已完成,
        # collate_fn (oft_collate_fn) 收到 normalizers=None 即可跳过二次归一化。
        normalizers = None
        print_rank0(
            "✓ 多子集模式: state/action 归一化由 MultiLeRobotDataset 自身完成, "
            "collate_fn 不再二次归一化",
            rank,
        )
    else:
        # ----- 单源分支 (原有逻辑, 零改动) -----
        print_rank0(f"Loading dataset from: {args.data_path}", rank)
        info_path = Path(args.data_path) / "meta" / "info.json"
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        total_episodes = info['total_episodes']
        actual_action_dim = info['features']['action']['shape'][0]
        actual_state_dim = info['features']['observation.state']['shape'][0]
        
        print_rank0(f"Dataset info:", rank)
        print_rank0(f"  Total episodes: {total_episodes}", rank)
        print_rank0(f"  Action dim: {actual_action_dim}", rank)
        print_rank0(f"  State dim: {actual_state_dim}", rank)
        print_rank0(f"  Action chunks: {NUM_ACTIONS_CHUNK}", rank)

        if actual_action_dim != ACTION_DIM:
            msg = (
                f"\n[FATAL] ACTION_DIM 不匹配:\n"
                f"   constants.ACTION_DIM = {ACTION_DIM}  (model head 输出维度)\n"
                f"   dataset action dim   = {actual_action_dim}  "
                f"(meta/info.json features.action.shape[0])\n"
                f"   shape mismatch 会让 loss 计算时报 "
                f"'tensor a ({ACTION_DIM*NUM_ACTIONS_CHUNK}) vs b "
                f"({actual_action_dim*NUM_ACTIONS_CHUNK})'。\n\n"
                f"   修复方法（任选一）：\n"
                f"     1) 在训练脚本里加: export ACTION_DIM={actual_action_dim}\n"
                f"     2) 直接改 Action_Heads/OFT1_0/constants.py 的 ACTION_DIM\n"
            )
            print_rank0(msg, rank)
            raise SystemExit(2)
        if actual_state_dim != PROPRIO_DIM:
            msg = (
                f"\n[FATAL] PROPRIO_DIM 不匹配:\n"
                f"   constants.PROPRIO_DIM = {PROPRIO_DIM}\n"
                f"   dataset state dim     = {actual_state_dim}\n"
                f"   修复方法（任选一）：\n"
                f"     1) 在训练脚本里加: export PROPRIO_DIM={actual_state_dim}\n"
                f"     2) 直接改 Action_Heads/OFT1_0/constants.py 的 PROPRIO_DIM\n"
            )
            print_rank0(msg, rank)
            raise SystemExit(2)
        
        print_rank0("\n加载归一化统计量...", rank)
        print_rank0(f"  四元数→轴角转换: {'启用' if convert_quat_to_axisangle else '禁用'}", rank)
        normalizers = load_normalization_stats(
            args.data_path, 
            convert_quat_to_axisangle=convert_quat_to_axisangle,
        )
        if normalizers:
            print_rank0("✓ 将对 state 和 action 进行 min_max 归一化到 [-1, 1]", rank)
        else:
            print_rank0("⚠️ 未加载归一化统计量，state 和 action 将不进行归一化", rank)
    
    # 记录数据集信息到 wandb
    if wandb_initialized and is_main_process(rank):
        try:
            wandb.config.update({
                "dataset/name": Path(args.data_path).name,
                "dataset/total_episodes": total_episodes,
                "dataset/actual_action_dim": actual_action_dim,
                "dataset/actual_state_dim": actual_state_dim,
                "distributed/world_size": world_size,
                "distributed/gradient_accumulation_steps": args.gradient_accumulation_steps,
                "distributed/effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                "memory/use_amp": args.use_amp,
                "memory/amp_dtype": args.amp_dtype if args.use_amp else "disabled",
            }, allow_val_change=True)
        except Exception:
            pass
    
    # 划分训练集和验证集
    if use_multi_dataset_mode:
        # 多源: 按 ep_uid 简单切分; val 在多源下不太常用, 默认行为 = 全部用作 train
        if args.val_split > 0:
            print_rank0(
                f"⚠️ multi-dataset 模式下不支持 val_split, 强制设为 0",
                rank,
            )
            args.val_split = 0
        train_episode_indices = list(range(total_episodes))
        val_episode_indices = None
        print_rank0(
            f"Multi-dataset train ep_uids: {len(train_episode_indices)} (val 已禁用)",
            rank,
        )
    else:
        val_episodes = max(1, int(total_episodes * args.val_split)) if args.val_split > 0 else 0
        train_episodes = total_episodes - val_episodes
        
        if train_episodes < 1:
            print_rank0(f"Warning: Not enough episodes for train/val split. Using all {total_episodes} episodes for training.", rank)
            train_episode_indices = list(range(total_episodes))
            val_episode_indices = None
        else:
            train_episode_indices = list(range(train_episodes))
            val_episode_indices = list(range(train_episodes, total_episodes)) if val_episodes > 0 else None
            print_rank0(f"Train episodes: {len(train_episode_indices)}, Val episodes: {val_episodes}", rank)
    
    # 打印数据加载优化配置
    print_rank0(f"\n✓ 数据加载优化配置:", rank)
    print_rank0(f"  - VLM hidden states 数据类型: {args.vlm_dtype}", rank)
    print_rank0(f"  - 跳过加载图像: {args.skip_images}", rank)
    if args.vlm_dtype in ["float16", "bfloat16"]:
        print_rank0(f"  - 预计 VLM 显存节省: ~50%", rank)
    if args.skip_images:
        print_rank0(f"  - 预计数据加载速度提升: ~2-3x", rank)
    
    # ========== 数据加载模式选择 ==========
    # 优先级:
    #   ChunkBatchCache (lerobot 原生分批 SHM) >
    #   WebDataset 分批缓存 >
    #   WebDataset >
    #   共享内存缓存（一次性全装） >
    #   mmap 模式 >
    #   普通加载
    shared_vlm_cache = None
    use_webdataset_mode = args.use_webdataset and args.webdataset_shard_pattern
    use_webdataset_cached_mode = args.use_webdataset_cached and args.webdataset_shard_pattern
    use_chunk_batch_cache_mode = bool(args.use_chunk_batch_cache)
    use_disk_shuffle_mode = bool(getattr(args, "use_disk_shuffle", False))

    # Disk-Shuffle 模式优先级最高: 关闭其它 cache (早期 args 处理时已禁用, 这里
    # 再兜底一次)。此模式不预加载到 SHM, 训练时全局 shuffle + mmap 按需读单帧.
    if use_disk_shuffle_mode:
        use_webdataset_mode = False
        use_webdataset_cached_mode = False
        use_chunk_batch_cache_mode = False
        print_rank0(f"\n{'=' * 60}", rank)
        print_rank0(f"🚀 启用 Disk-Shuffle 模式 (无 SHM, 全局 shuffle, mmap 按需读)", rank)
        print_rank0(f"  - 数据集: {args.data_path}", rank)
        print_rank0(
            f"  - npz mmap LRU/worker: "
            f"{'unlimited' if args.disk_shuffle_npz_lru_per_worker <= 0 else args.disk_shuffle_npz_lru_per_worker}",
            rank,
        )
        print_rank0(f"  - parquet_cache 上限: {args.disk_shuffle_parquet_cache_max}", rank)
        print_rank0(f"  - 注意: HDD 上极慢, 建议数据放在 NVMe/SSD", rank)
        print_rank0(f"{'=' * 60}", rank)

    # ChunkBatchCache 模式 (次优先): 禁用其他 cache 路径
    if use_chunk_batch_cache_mode:
        use_webdataset_mode = False
        use_webdataset_cached_mode = False
        if args.use_shared_cache:
            print_rank0(
                "⚠️ 已启用 --use_chunk_batch_cache，自动忽略 --use_shared_cache",
                rank,
            )
            args.use_shared_cache = False
        print_rank0(f"\n{'=' * 60}", rank)
        print_rank0(f"🚀 启用 ChunkBatchCache 分批共享内存模式", rank)
        print_rank0(f"  - 数据集: {args.data_path}", rank)
        print_rank0(f"  - 安全系数 (RAM/SHM): {args.chunk_batch_safety_ratio}", rank)
        print_rank0(f"  - 缓存数据类型: {args.chunk_batch_cache_dtype}", rank)
        print_rank0(f"  - 每批最少 chunk: {args.chunk_batch_min_chunks}", rank)
        print_rank0(
            f"  - 每批最多 chunk: {args.chunk_batch_max_chunks if args.chunk_batch_max_chunks > 0 else '不限'}",
            rank,
        )
        print_rank0(
            f"  - 手动指定每批 chunk: {args.chunk_batch_manual_chunks if args.chunk_batch_manual_chunks > 0 else '自动'}",
            rank,
        )
        print_rank0(f"{'=' * 60}", rank)

    # WebDataset 分批缓存模式优先
    if use_webdataset_cached_mode:
        use_webdataset_mode = False  # 禁用普通 WebDataset 模式
        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"🚀 启用 WebDataset 分批缓存模式", rank)
        print_rank0(f"  - 分片路径: {args.webdataset_shard_pattern}", rank)
        print_rank0(f"  - 每批分片数: {args.webdataset_cache_shards}", rank)
        print_rank0(f"  - 缓存数据类型: {args.webdataset_cache_dtype}", rank)
        print_rank0(f"{'='*60}", rank)
    elif use_webdataset_mode:
        # WebDataset 模式 - 适用于超大规模数据集
        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"🚀 使用 WebDataset 模式 (超大规模数据优化)", rank)
        print_rank0(f"  - 分片路径: {args.webdataset_shard_pattern}", rank)
        print_rank0(f"  - Shuffle Buffer: {args.webdataset_shuffle_buffer}", rank)
        print_rank0(f"  - 优势: 减少小文件 I/O，原生分布式支持", rank)
        print_rank0(f"{'='*60}", rank)
    
    elif args.use_shared_cache:
        # 共享内存缓存模式 - 获取数据集中所有 VLM hidden states 的总数
        cache_size = get_total_vlm_states(args.data_path)
        
        # 解析缓存数据类型
        cache_dtype_str = getattr(args, 'cache_dtype', 'float32')
        if cache_dtype_str == 'float16':
            cache_dtype = np.float16
            memory_reduction = "50% (float16)"
        else:
            cache_dtype = np.float32
            memory_reduction = "0% (float32)"
        
        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"🚀 启用共享内存缓存 (加载所有轨迹的 VLM hidden states)", rank)
        print_rank0(f"  - 数据集总帧数: {cache_size}", rank)
        print_rank0(f"  - VLM 索引范围: 0 - {cache_size - 1}", rank)
        print_rank0(f"  - 缓存数据类型: {cache_dtype_str}", rank)
        print_rank0(f"  - 内存节省: {memory_reduction}", rank)
        print_rank0(f"  - 训练集和验证集将共享此缓存", rank)
        print_rank0(f"{'='*60}", rank)
        
        # 预加载 VLM 缓存到共享内存（自动检测 seq_len）
        shared_vlm_cache = preload_vlm_cache_distributed(
            dataset_path=args.data_path,
            num_samples=cache_size,
            sample_shape=None,  # 自动检测
            rank=rank,
            world_size=world_size,
            dtype=cache_dtype,  # 使用指定的缓存数据类型
            verbose=is_main_process(rank),
            auto_detect_shape=True,
            cache_dtype=cache_dtype_str,  # 传入字符串类型
        )
        
        # 注册清理函数（确保异常退出时也能清理）
        _shared_vlm_cache_global = shared_vlm_cache
        _shared_vlm_cache_rank = rank
        atexit.register(cleanup_shared_memory_cache)
        
        # 注册信号处理器（处理 SIGTERM/SIGINT）
        def signal_handler(signum, frame):
            cleanup_shared_memory_cache()
            sys.exit(0)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        print_rank0(f"✅ 共享内存缓存预加载完成", rank)
    
    # ========== 创建数据加载器 ==========
    train_dataset = None
    train_sampler = None
    shard_batch_cache = None  # 分批缓存管理器
    chunk_batch_cache = None  # ChunkBatchCache 管理器（新）
    chunk_batch_current_eps = None  # 当前批次涵盖的 ep idx 列表
    disk_shuffle_cache = None  # DiskVLMCacheManager (无 SHM, 全局 shuffle)

    if use_disk_shuffle_mode:
        # ---------------------------------------------------------------- #
        # Disk-Shuffle 模式: 完全不创建 SHM, ConcatDataset 一次性激活全部 episode,
        # DistributedSampler 在整个数据集上 shuffle。下游训练循环不需要切段。
        # ---------------------------------------------------------------- #
        if use_multi_dataset_mode:
            disk_shuffle_cache = DiskVLMCacheManager(
                index=multi_dataset_index,
                npz_lru_per_worker=args.disk_shuffle_npz_lru_per_worker,
                rank=rank,
                world_size=world_size,
                verbose=is_main_process(rank),
                seed=args.chunk_batch_seed,  # 复用 chunk_batch_seed 概念
            )
            ep_uids_all, _vlm_remap_unused, n_frames_total, _ = (
                disk_shuffle_cache.load_next_batch()
            )
            chunk_batch_current_eps = ep_uids_all  # 占位 (multi 路径不直接用)
            shared_vlm_cache = None  # 多源走 train_dataset.set_active_episodes 注入 per-subset view
        else:
            if not _DISK_SHUFFLE_AVAILABLE:
                raise SystemExit(
                    f"--use_disk_shuffle 单源路径需要 disk_vlm_cache 模块: "
                    f"{_DISK_SHUFFLE_IMPORT_ERROR}"
                )
            # 单源: 复用 build_single_source_disk_cache,
            # 拿到 (view, ep_locals, n_frames, manager)
            disk_view, ep_locals, n_frames_total, disk_shuffle_cache = (
                build_single_source_disk_cache(
                    dataset_path=args.data_path,
                    episode_indices=train_episode_indices,
                    npz_lru_per_worker=args.disk_shuffle_npz_lru_per_worker,
                    rank=rank,
                    world_size=world_size,
                    verbose=is_main_process(rank),
                )
            )
            shared_vlm_cache = disk_view
            ep_uids_all = ep_locals  # 单源场景下下游需要的是子集内 ep_idx_local
            chunk_batch_current_eps = ep_locals

        # 清理钩子: 训练结束/异常退出时关闭 mmap 句柄
        def _cleanup_disk_shuffle():
            try:
                if disk_shuffle_cache is not None:
                    disk_shuffle_cache.cleanup()
                if shared_vlm_cache is not None and hasattr(shared_vlm_cache, "close"):
                    shared_vlm_cache.close()
            except Exception:
                pass

        atexit.register(_cleanup_disk_shuffle)

        def _ds_signal_handler(signum, frame):
            _cleanup_disk_shuffle()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _ds_signal_handler)
        signal.signal(signal.SIGINT, _ds_signal_handler)

        print_rank0(
            f"✅ Disk-Shuffle 缓存就绪: {len(ep_uids_all)} 个 episode, {n_frames_total} 帧 "
            f"(无 SHM 占用, mmap lazy)",
            rank,
        )

        # 创建 train_dataset: 多源/单源分支
        if use_multi_dataset_mode:
            train_dataset = MultiLeRobotDataset(
                index=multi_dataset_index,
                normalizers=multi_dataset_normalizers,
                target_state_dim=actual_state_dim,
                target_action_dim=actual_action_dim,
                num_action_chunks=NUM_ACTIONS_CHUNK,
                enable_chunking=True,
                skip_images=args.skip_images,
                vlm_cache_dtype=args.vlm_dtype,
                verbose=is_main_process(rank),
            )
            train_dataset.set_active_episodes(
                ep_uids_all, disk_shuffle_cache.current_subset_caches
            )
        else:
            train_dataset = LeRobotDataset(
                dataset_path=args.data_path,
                num_action_chunks=NUM_ACTIONS_CHUNK,
                enable_chunking=True,
                episode_indices=ep_uids_all,  # 单源时这里是子集内 ep_idx_local
                cache_vlm_states=False,
                cache_max_samples=-1,
                verbose=is_main_process(rank),
                skip_images=args.skip_images,
                shared_vlm_cache=shared_vlm_cache,
                state_process_order=args.state_process_order,
                hand_binary_columns=args.hand_binary_columns,
                hand_binary_threshold=args.hand_binary_threshold,
            )

        # 用户配的 parquet / episode mmap 缓存 LRU 上限
        # (disk-shuffle 模式下 worker 会接触所有 ep, 不限会无限增长)
        if args.disk_shuffle_parquet_cache_max > 0:
            _ds_parquet_max = int(args.disk_shuffle_parquet_cache_max)
            # 经验值: 给 episode_mmap_cache 留 1/4 的额度
            _ds_ep_mmap_max = max(128, _ds_parquet_max // 4)
            if use_multi_dataset_mode:
                # MultiLeRobotDataset 内部的 LeRobotDataset 实例
                for _sub_ds in train_dataset._active_subsets.values():  # type: ignore[attr-defined]
                    if hasattr(_sub_ds, "set_cache_limits"):
                        _sub_ds.set_cache_limits(
                            parquet_cache_max=_ds_parquet_max,
                            episode_mmap_cache_max=_ds_ep_mmap_max,
                            chunk_npz_cache_max=0,
                        )
            else:
                train_dataset.set_cache_limits(
                    parquet_cache_max=_ds_parquet_max,
                    episode_mmap_cache_max=_ds_ep_mmap_max,
                    chunk_npz_cache_max=0,
                )

        lerobot_collate_fn_with_dtype = partial(
            lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype
        )

        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            shuffle = False
        else:
            shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            sampler=train_sampler,
            collate_fn=lerobot_collate_fn_with_dtype,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            # 整训只有一个"虚拟段", dataset 不需要重建 → persistent_workers=True
            # 让 worker 持有 mmap 句柄横跨 epoch, 避免反复重建
            persistent_workers=bool(args.num_workers > 0),
        )

        print_rank0(
            f"✓ Disk-Shuffle 训练数据加载器创建完成 ({len(train_dataset)} 个样本, "
            f"全局 shuffle 半径 = 整数据集)",
            rank,
        )

    elif use_chunk_batch_cache_mode:
        max_chunks = args.chunk_batch_max_chunks if args.chunk_batch_max_chunks > 0 else None
        manual_chunks = args.chunk_batch_manual_chunks if args.chunk_batch_manual_chunks > 0 else None

        if use_multi_dataset_mode:
            # 多源: MultiChunkBatchCache 鸭子类型为 ChunkBatchCache (load_*() 4-tuple)
            chunk_batch_cache = MultiChunkBatchCache(
                index=multi_dataset_index,
                cache_dtype=args.chunk_batch_cache_dtype,
                safety_ratio=args.chunk_batch_safety_ratio,
                min_chunks_per_batch=args.chunk_batch_min_chunks,
                max_chunks_per_batch=max_chunks,
                manual_chunks_per_batch=manual_chunks,
                ram_inflation_factor=args.chunk_batch_inflation,
                rank=rank,
                world_size=world_size,
                verbose=is_main_process(rank),
                seed=args.chunk_batch_seed,
            )
        else:
            try:
                from chunk_batch_cache import ChunkBatchCache
            except ImportError:
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'))
                from chunk_batch_cache import ChunkBatchCache

            chunk_batch_cache = ChunkBatchCache(
                dataset_path=args.data_path,
                episode_indices=train_episode_indices,
                cache_dtype=args.chunk_batch_cache_dtype,
                safety_ratio=args.chunk_batch_safety_ratio,
                min_chunks_per_batch=args.chunk_batch_min_chunks,
                max_chunks_per_batch=max_chunks,
                manual_chunks_per_batch=manual_chunks,
                ram_inflation_factor=args.chunk_batch_inflation,
                rank=rank,
                world_size=world_size,
                verbose=is_main_process(rank),
                seed=args.chunk_batch_seed,
            )

        # step-segments 模式：冻结 chunk 顺序，整训过程 segment→ChunkBatch 映射稳定。
        # 旧逻辑（每 epoch 切批）保留，通过 --no_step_based_segments 启用。
        use_step_segments_mode = bool(args.use_step_based_segments)
        if use_step_segments_mode:
            chunk_batch_cache.freeze_packing(args.chunk_batch_seed)
            num_segments_global = chunk_batch_cache.num_batches
            steps_per_segment_global = max(
                1, math.ceil(args.steps / max(1, num_segments_global))
            )
            print_rank0(
                f"\n🪜 [Step Segments] 启用按 step 分段加载: "
                f"num_segments={num_segments_global}, "
                f"steps_per_segment={steps_per_segment_global} "
                f"(total_steps={args.steps})",
                rank,
            )
            # 装第 0 段
            ep_indices_batch, _vlm_remap, n_frames_batch, batch_shared_cache = (
                chunk_batch_cache.load_segment(0)
            )
            loaded_segment_idx = 0
        else:
            num_segments_global = chunk_batch_cache.num_batches
            steps_per_segment_global = 0  # 旧模式下无意义
            loaded_segment_idx = -1
            # epoch 0 -> 装第一批（旧逻辑）
            chunk_batch_cache.set_epoch(0)
            ep_indices_batch, _vlm_remap, n_frames_batch, batch_shared_cache = (
                chunk_batch_cache.load_next_batch()
            )
        chunk_batch_current_eps = ep_indices_batch
        shared_vlm_cache = batch_shared_cache  # 让 LeRobotDataset 走 shared_vlm_cache 路径

        # 注册全局清理（确保 SIGINT/SIGTERM 时也能清理 SHM）
        _shared_vlm_cache_global = shared_vlm_cache
        _shared_vlm_cache_rank = rank

        def _cleanup_chunk_batch_cache():
            try:
                if chunk_batch_cache is not None:
                    chunk_batch_cache.cleanup()
            except Exception:
                pass

        atexit.register(_cleanup_chunk_batch_cache)

        def _cb_signal_handler(signum, frame):
            _cleanup_chunk_batch_cache()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _cb_signal_handler)
        signal.signal(signal.SIGINT, _cb_signal_handler)

        print_rank0(
            f"✅ ChunkBatchCache 第 1/{chunk_batch_cache.num_batches} 批装载完成: "
            f"{len(ep_indices_batch)} 个 episode, {n_frames_batch} 帧",
            rank,
        )

        # 创建 train_dataset: 多源/单源分支
        if use_multi_dataset_mode:
            train_dataset = MultiLeRobotDataset(
                index=multi_dataset_index,
                normalizers=multi_dataset_normalizers,
                target_state_dim=actual_state_dim,
                target_action_dim=actual_action_dim,
                num_action_chunks=NUM_ACTIONS_CHUNK,
                enable_chunking=True,
                skip_images=args.skip_images,
                vlm_cache_dtype=args.vlm_dtype,
                verbose=is_main_process(rank),
            )
            train_dataset.set_active_episodes(
                ep_indices_batch, chunk_batch_cache.current_subset_caches
            )
        else:
            train_dataset = LeRobotDataset(
                dataset_path=args.data_path,
                num_action_chunks=NUM_ACTIONS_CHUNK,
                enable_chunking=True,
                episode_indices=ep_indices_batch,
                cache_vlm_states=False,
                cache_max_samples=-1,
                verbose=is_main_process(rank),
                skip_images=args.skip_images,
                shared_vlm_cache=shared_vlm_cache,
                state_process_order=args.state_process_order,
                hand_binary_columns=args.hand_binary_columns,
                hand_binary_threshold=args.hand_binary_threshold,
            )

        lerobot_collate_fn_with_dtype = partial(lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype)

        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            shuffle = False
        else:
            shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            sampler=train_sampler,
            collate_fn=lerobot_collate_fn_with_dtype,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
            persistent_workers=False,  # 切批时要重建 dataset，不能 persistent
        )

        print_rank0(
            f"✓ ChunkBatchCache 训练数据加载器创建完成（首批 {len(train_dataset)} 个样本）",
            rank,
        )

    elif use_webdataset_cached_mode:
        # WebDataset 分批缓存模式 - 分批加载分片到内存
        try:
            from webdataset_utils import ShardBatchCache, CachedSamplesDataset, cached_samples_collate_fn
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'))
            from webdataset_utils import ShardBatchCache, CachedSamplesDataset, cached_samples_collate_fn
        
        shard_batch_cache = ShardBatchCache(
            shard_pattern=args.webdataset_shard_pattern,
            shards_per_batch=args.webdataset_cache_shards,
            cache_dtype=args.webdataset_cache_dtype,
            rank=rank,
            world_size=world_size,
            verbose=is_main_process(rank),
        )
        
        # 加载第一批分片
        num_samples, train_dataset = shard_batch_cache.load_next_batch()
        
        # 创建 DataLoader
        cached_collate_fn = partial(cached_samples_collate_fn, vlm_dtype=args.vlm_dtype)
        
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            shuffle = False
        else:
            shuffle = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,  # 数据已在内存，不需要多进程
            sampler=train_sampler,
            collate_fn=cached_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        print_rank0(f"✓ WebDataset 分批缓存模式数据加载器创建完成，当前批次 {num_samples} 个样本", rank)
        
    elif use_webdataset_mode:
        # WebDataset 模式 - 使用 tar 分片格式
        try:
            from lerobot_dataset_loader import create_webdataset_dataloader
        except ImportError:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'utils'))
            from lerobot_dataset_loader import create_webdataset_dataloader
        
        train_loader = create_webdataset_dataloader(
            shard_pattern=args.webdataset_shard_pattern,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            shuffle_buffer_size=args.webdataset_shuffle_buffer,
            vlm_dtype=args.vlm_dtype,
            epoch_length=None,  # 使用所有数据
            distributed=is_distributed,
            rank=rank,
            world_size=world_size,
            verbose=is_main_process(rank),
        )
        print_rank0(f"✓ WebDataset 训练数据加载器创建完成", rank)
    else:
        # 标准 LeRobot 数据集模式
        train_dataset = LeRobotDataset(
            dataset_path=args.data_path,
            num_action_chunks=NUM_ACTIONS_CHUNK,
            enable_chunking=True,
            episode_indices=train_episode_indices,
            cache_vlm_states=args.cache_vlm_states,
            cache_max_samples=args.cache_max_samples,
            verbose=is_main_process(rank),
            skip_images=args.skip_images,
            shared_vlm_cache=shared_vlm_cache,  # 传入共享内存缓存
            # State 预处理参数
            state_process_order=args.state_process_order,
            hand_binary_columns=args.hand_binary_columns,
            hand_binary_threshold=args.hand_binary_threshold,
        )
        
        # 创建带 vlm_dtype 参数的 collate_fn
        lerobot_collate_fn_with_dtype = partial(lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype)
        
        # 创建 sampler
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            shuffle = False
        else:
            shuffle = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            sampler=train_sampler,
            collate_fn=lerobot_collate_fn_with_dtype,  # 使用带 vlm_dtype 的 collate 函数
            pin_memory=True,
            drop_last=True,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,  # 预取队列深度
            persistent_workers=True if args.num_workers > 0 else False,  # 保持 workers 存活，避免每 epoch 重启
        )
        
        print_rank0(f"✓ 训练数据加载器创建完成，共 {len(train_dataset)} 个样本", rank)
    
    # 创建验证数据加载器
    val_loader = None
    val_sampler = None
    if val_episode_indices is not None:
        # 在 ChunkBatchCache 模式下，shared_vlm_cache 只装了当前训练 batch 的 chunk，
        # 验证 episode 不在其中 → 必须走 mmap 路径
        val_shared_cache = None if use_chunk_batch_cache_mode else shared_vlm_cache
        val_cache_vlm_states = True if use_chunk_batch_cache_mode else args.cache_vlm_states
        # 验证集也使用共享缓存（因为已加载所有轨迹的 VLM hidden states）
        val_dataset = LeRobotDataset(
            dataset_path=args.data_path,
            num_action_chunks=NUM_ACTIONS_CHUNK,
            enable_chunking=True,
            episode_indices=val_episode_indices,
            cache_vlm_states=val_cache_vlm_states,  # 如果没有共享缓存则使用 mmap 模式
            cache_max_samples=args.cache_max_samples,
            verbose=False,
            skip_images=args.skip_images,
            shared_vlm_cache=val_shared_cache,  # ChunkBatchCache 模式下走 mmap
            # State 预处理参数
            state_process_order=args.state_process_order,
            hand_binary_columns=args.hand_binary_columns,
            hand_binary_threshold=args.hand_binary_threshold,
        )
        
        if is_distributed:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            sampler=val_sampler,
            collate_fn=lerobot_collate_fn_with_dtype,  # 使用带 vlm_dtype 的 collate 函数
            pin_memory=True,
            drop_last=False,
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,  # 预取队列深度
            persistent_workers=True if args.num_workers > 0 else False,  # 保持 workers 存活
        )

    # ========== 创建模型 ==========
    model = create_vlm2oft_pipeline(
        num_transformer_blocks=args.num_transformer_blocks,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        action_head_hidden_dim=args.action_head_hidden_dim,
        num_vlm_layers=args.num_vlm_layers,
        vlm_output_dim=args.vlm_output_dim
    ).to(device)

    resume_checkpoint = None
    if args.resume_from_checkpoint:
        resume_checkpoint = load_resume_checkpoint(
            args.resume_from_checkpoint,
            device,
            rank,
            reset_lr=args.resume_reset_lr,
        )
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        print_rank0(
            f"✓ 已从 checkpoint 加载模型参数: {resume_checkpoint['resolved_checkpoint_dir']}",
            rank,
        )
        if args.resume_reset_lr:
            print_rank0(
                "🔁 --resume_reset_lr=True：仅加载权重，学习率 / step / optimizer / RNG 全部重置 "
                "(等价于从该权重 fine-tune 重新走 warmup → decay)",
                rank,
            )
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f'\n{"="*60}', rank)
    print_rank0(f'模型参数量 Model Parameters', rank)
    print_rank0(f'{"="*60}', rank)
    print_rank0(f'  总参数量 (Total):     {total_params:,} ({total_params/1e6:.2f}M)', rank)
    print_rank0(f'  可训练参数 (Trainable): {trainable_params:,} ({trainable_params/1e6:.2f}M)', rank)
    print_rank0(f'{"="*60}\n', rank)
    
    # 分布式训练：包装模型为 DDP
    # 设置 find_unused_parameters=True 以处理模型中可能未使用的参数
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        print_rank0(f"✓ 模型已包装为 DistributedDataParallel (find_unused_parameters=True)", rank)
    
    if wandb_initialized and is_main_process(rank):
        try:
            model_to_watch = model.module if is_distributed else model
            wandb.watch(model_to_watch, log='all', log_freq=100)
            
            total_params = sum(p.numel() for p in model_to_watch.parameters())
            trainable_params = sum(p.numel() for p in model_to_watch.parameters() if p.requires_grad)
            wandb.config.update({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/vlm_output_dim": args.vlm_output_dim,
                "model/num_vlm_layers": args.num_vlm_layers,
                "model/num_transformer_blocks": args.num_transformer_blocks,
                "model/action_head_hidden_dim": args.action_head_hidden_dim,
            }, allow_val_change=True)
        except Exception:
            pass
    
    model.train()
    
    # Loss 函数
    mae_criterion = nn.L1Loss()
    bce_criterion = None
    use_mixed_loss = args.use_mixed_loss and args.bce_action_columns is not None and len(args.bce_action_columns) > 0
    if use_mixed_loss:
        bce_criterion = nn.BCEWithLogitsLoss()
        print_rank0(f"✓ 使用混合 Loss (MAE + BCE)", rank)
        print_rank0(f"  BCE 列: {args.bce_action_columns}", rank)
        print_rank0(f"  MAE 权重: {args.mae_loss_weight}, BCE 权重: {args.bce_loss_weight}", rank)
    else:
        print_rank0(f"✓ 使用 L1Loss (连续模式)", rank)
    
    # 优化器配置
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=1e-8
    )
    print_rank0(f"\n✓ 优化器配置:", rank)
    print_rank0(f"  - 学习率: {args.lr}", rank)
    print_rank0(f"  - 权重衰减: {args.weight_decay}", rank)
    print_rank0(f"  - Adam beta1: {args.adam_beta1}", rank)
    print_rank0(f"  - Adam beta2: {args.adam_beta2}", rank)
    
    # 学习率调度器
    total_steps = args.steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # warmup 阶段: 线性增长
            return float(current_step) / float(max(1, warmup_steps))
        else:
            if args.lr_scheduler_type == "constant":
                # constant 模式: warmup 后保持不变
                return 1.0
            elif args.lr_scheduler_type == "warmup_step_decay":
                # warmup_step_decay 模式: warmup 后保持，到达指定 step 后切换到新学习率
                if current_step < args.lr_decay_step:
                    return 1.0
                else:
                    return args.lr_decay_value / args.lr
            else:
                # cosine 模式: warmup 后 cosine decay
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optim, lr_lambda)
    if args.lr_scheduler_type == "constant":
        scheduler_type_desc = "warmup + constant"
    elif args.lr_scheduler_type == "warmup_step_decay":
        scheduler_type_desc = f"warmup + step decay (step={args.lr_decay_step}, lr={args.lr_decay_value})"
    else:
        scheduler_type_desc = "warmup + cosine decay"
    print_rank0(f"\n✓ 学习率调度器配置:", rank)
    print_rank0(f"  - 预热比例: {args.warmup_ratio}", rank)
    print_rank0(f"  - 预热步数: {warmup_steps}", rank)
    print_rank0(f"  - 总步数: {total_steps}", rank)
    print_rank0(f"  - 调度类型: {scheduler_type_desc}", rank)
    
    # 混合精度训练配置
    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        scaler = GradScaler('cuda', enabled=True)
        print_rank0(f"\n✓ 混合精度训练 (AMP):", rank)
        print_rank0(f"  - 启用: True", rank)
        print_rank0(f"  - 数据类型: {args.amp_dtype}", rank)
    else:
        amp_dtype = torch.float32
        scaler = GradScaler('cuda', enabled=False)
        print_rank0(f"\n✓ 混合精度训练 (AMP): 禁用 (使用 FP32)", rank)
    
    empty_cache_freq = args.empty_cache_freq
    
    step = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = -1
    best_step = -1
    start_epoch = 0
    resume_shard_batch_idx = 0
    resume_batch_idx = 0
    training_start_time = time.time()

    if resume_checkpoint is not None and resume_checkpoint.get("resume_mode") == "full_state":
        if resume_checkpoint.get("optimizer_state_dict") is not None:
            optim.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optim, device)
        if resume_checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if resume_checkpoint.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            except Exception as e:
                print_rank0(f"⚠️ 恢复 GradScaler 失败，将继续训练: {e}", rank)

        step = int(resume_checkpoint.get("step", 0))
        best_val_loss = float(resume_checkpoint["best_val_loss"]) if resume_checkpoint.get("best_val_loss") is not None else float('inf')
        best_train_loss = float(resume_checkpoint["best_train_loss"]) if resume_checkpoint.get("best_train_loss") is not None else float('inf')
        best_epoch = int(resume_checkpoint.get("best_epoch", -1))
        best_step = int(resume_checkpoint.get("best_step", -1))
        start_epoch = int(resume_checkpoint.get("epoch", 0))
        resume_shard_batch_idx = int(resume_checkpoint.get("shard_batch_idx", 0))
        resume_batch_idx = int(resume_checkpoint.get("next_batch_idx", 0))

        if resume_checkpoint.get("completed_epoch", False):
            start_epoch += 1
            resume_shard_batch_idx = 0
            resume_batch_idx = 0

        elapsed_training_time = float(resume_checkpoint.get("elapsed_training_time_sec", 0.0))
        training_start_time = time.time() - max(0.0, elapsed_training_time)
        restore_rng_state(resume_checkpoint.get("rng_state"), rank)

        print_rank0(
            f"✓ 恢复训练状态: step={step}, start_epoch={start_epoch}, "
            f"resume_shard_batch_idx={resume_shard_batch_idx}, resume_batch_idx={resume_batch_idx}",
            rank,
        )
    
    # 创建 custom 日志文件，记录每个 step 的 loss（只在主进程）
    step_loss_log_file = None
    if is_main_process(rank):
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        step_loss_log_path = custom_log_dir / f"step_loss_{timestamp}.csv"
        step_loss_log_file = open(step_loss_log_path, 'w')
        step_loss_log_file.write("step,epoch,loss,learning_rate,grad_norm,batch_time\n")
        step_loss_log_file.flush()
        print(f"✓ Step loss 日志文件: {step_loss_log_path}")
    
    # 创建 collate_fn
    collate_fn_with_normalizers = partial(
        oft_collate_fn,
        normalizers=normalizers,
        convert_quat_to_axisangle=convert_quat_to_axisangle,
        state_norm_columns_minmax=args.state_norm_columns_minmax,  # 用于 state 预处理后的归一化
        use_action_polyfit=args.use_action_polyfit,
        action_polyfit_degree=args.action_polyfit_degree,
        use_action_from_state_diff=args.use_action_from_state_diff,
        action_from_state_diff_degree=args.action_from_state_diff_degree,
        state_diff_columns=args.state_diff_columns,
        action_diff_target_columns=args.action_diff_target_columns,
        action_keep_original_columns=args.action_keep_original_columns,
    )
    
    gradient_accumulation_steps = args.gradient_accumulation_steps
    optim.zero_grad(set_to_none=True)
    
    # 模型配置 (用于保存)
    model_config = {
        'action_head_type': 'oft',
        'type': 'oft',
        'hidden_dim': args.action_head_hidden_dim,
        'llm_output_dim': args.vlm_output_dim,
        'proprio_dim': PROPRIO_DIM,
        'action_dim': ACTION_DIM,
        'num_actions_chunk': NUM_ACTIONS_CHUNK,
        'num_vlm_hidden_layers': args.num_vlm_layers,
        'num_blocks': args.num_transformer_blocks,
        'num_attention_heads': args.num_attention_heads,
        'action_head_hidden_dim': args.action_head_hidden_dim,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'resume_from_checkpoint': args.resume_from_checkpoint or None,
        'resume_reset_lr': bool(args.resume_reset_lr) if args.resume_from_checkpoint else None,
        # State 预处理配置
        'state_process_order': args.state_process_order,
        'hand_binary_columns': args.hand_binary_columns,
        'hand_binary_threshold': args.hand_binary_threshold,
        'state_norm_columns_minmax': args.state_norm_columns_minmax,
        # Action Polyfit 配置
        'use_action_polyfit': args.use_action_polyfit,
        'action_polyfit_degree': args.action_polyfit_degree if args.use_action_polyfit else None,
        # Action from State Diff 配置
        'use_action_from_state_diff': args.use_action_from_state_diff,
        'action_from_state_diff_degree': args.action_from_state_diff_degree if args.use_action_from_state_diff else None,
        'state_diff_columns': args.state_diff_columns if args.use_action_from_state_diff else None,
        'action_diff_target_columns': args.action_diff_target_columns if args.use_action_from_state_diff else None,
        'action_keep_original_columns': args.action_keep_original_columns if args.use_action_from_state_diff else None,
        # 混合 Loss 配置
        'use_mixed_loss': use_mixed_loss,
        'bce_action_columns': args.bce_action_columns if use_mixed_loss else None,
        'mae_loss_weight': args.mae_loss_weight if use_mixed_loss else None,
        'bce_loss_weight': args.bce_loss_weight if use_mixed_loss else None,
    }
    
    # 记录训练开始时间，用于计算已用时间和剩余时间
    if resume_checkpoint is None or resume_checkpoint.get("resume_mode") != "full_state":
        training_start_time = time.time()
    
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = []
        batch_times = []
        should_resume_epoch = (
            resume_checkpoint is not None
            and resume_checkpoint.get("resume_mode") == "full_state"
            and epoch == start_epoch
        )
        
        # ========== 确定分片批次数量 ==========
        # 分批缓存模式：一个 epoch 需要遍历所有分片批次
        # 其他模式：只有 1 个批次（直接使用 train_loader）
        if use_disk_shuffle_mode:
            # Disk-Shuffle 模式: 永远只有 1 个"虚拟段", DataLoader 直接遍历全数据集.
            # set_epoch 推给 DistributedSampler 做全局 shuffle。
            num_shard_batches = 1
        elif use_webdataset_cached_mode and shard_batch_cache is not None:
            num_shard_batches = shard_batch_cache.num_batches
            # 设置 epoch 并重新打乱分片顺序（确保每个 epoch 分片顺序不同）
            shard_batch_cache.set_epoch(epoch)
        elif use_chunk_batch_cache_mode and chunk_batch_cache is not None:
            if use_step_segments_mode:
                # Step-Segments 模式：epoch 不再驱动切批，由 step 边界决定。
                # 在 epoch 开头检查当前 step 应处于哪个 segment，必要时切换。
                target_segment = min(
                    step // steps_per_segment_global,
                    num_segments_global - 1,
                )
                if target_segment != loaded_segment_idx:
                    print_rank0(
                        f"\n📂 [Step Segments] 切换到 segment "
                        f"{target_segment + 1}/{num_segments_global} "
                        f"(step={step}, boundary={target_segment * steps_per_segment_global})",
                        rank,
                    )
                    ep_indices_batch, _vlm_remap_unused, n_frames_batch, batch_shared_cache = (
                        chunk_batch_cache.load_segment(target_segment)
                    )
                    chunk_batch_current_eps = ep_indices_batch
                    shared_vlm_cache = batch_shared_cache
                    _shared_vlm_cache_global = shared_vlm_cache

                    try:
                        if train_dataset is not None and hasattr(train_dataset, "close_video_readers"):
                            train_dataset.close_video_readers()
                    except Exception:
                        pass

                    if use_multi_dataset_mode:
                        # 多源: 复用 train_dataset (MultiLeRobotDataset), 仅切换 active eps
                        train_dataset.set_active_episodes(
                            ep_indices_batch, chunk_batch_cache.current_subset_caches
                        )
                    else:
                        train_dataset = LeRobotDataset(
                            dataset_path=args.data_path,
                            num_action_chunks=NUM_ACTIONS_CHUNK,
                            enable_chunking=True,
                            episode_indices=ep_indices_batch,
                            cache_vlm_states=False,
                            cache_max_samples=-1,
                            verbose=False,
                            skip_images=args.skip_images,
                            shared_vlm_cache=shared_vlm_cache,
                            state_process_order=args.state_process_order,
                            hand_binary_columns=args.hand_binary_columns,
                            hand_binary_threshold=args.hand_binary_threshold,
                        )

                    if is_distributed:
                        train_sampler = DistributedSampler(
                            train_dataset,
                            num_replicas=world_size,
                            rank=rank,
                            shuffle=True,
                        )
                    else:
                        train_sampler = None

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=not is_distributed,
                        num_workers=args.num_workers,
                        sampler=train_sampler,
                        collate_fn=partial(lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype),
                        pin_memory=True,
                        drop_last=True,
                        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                        persistent_workers=False,
                    )
                    print_rank0(
                        f"  ✓ Segment {target_segment + 1}/{num_segments_global} 装载: "
                        f"{len(ep_indices_batch)} ep / {n_frames_batch} frame",
                        rank,
                    )
                    loaded_segment_idx = target_segment
                # 段内只跑一个"虚拟分片"，实际由 inner-epoch（=外层 epoch 计数）驱动 sample shuffle
                num_shard_batches = 1
            else:
                # 旧逻辑：set_epoch 重新 shuffle + 重新打包
                chunk_batch_cache.set_epoch(epoch)
                num_shard_batches = chunk_batch_cache.num_batches
        else:
            num_shard_batches = 1
        
        # ========== 遍历所有分片批次 ==========
        for shard_batch_idx in range(num_shard_batches):
            if should_resume_epoch and shard_batch_idx < resume_shard_batch_idx:
                print_rank0(
                    f"⏭️ 跳过已完成的分片批次: epoch={epoch}, shard_batch_idx={shard_batch_idx}",
                    rank,
                )
                continue

            # 分批缓存模式：每个分片批次需要加载数据
            if use_webdataset_cached_mode and shard_batch_cache is not None:
                if shard_batch_idx > 0 or epoch > 0:
                    # 非第一个 epoch 的第一批，或者同一 epoch 的后续批次，需要加载
                    print_rank0(f"\n📂 Epoch {epoch}, 分片批次 {shard_batch_idx + 1}/{num_shard_batches}", rank)
                    num_samples, train_dataset = shard_batch_cache.load_next_batch()
                    
                    # 重新创建 sampler 和 DataLoader
                    if is_distributed:
                        train_sampler = DistributedSampler(
                            train_dataset,
                            num_replicas=world_size,
                            rank=rank,
                            shuffle=True
                        )
                    
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=not is_distributed,
                        num_workers=0,
                        sampler=train_sampler if is_distributed else None,
                        collate_fn=partial(cached_samples_collate_fn, vlm_dtype=args.vlm_dtype),
                        pin_memory=True,
                        drop_last=True,
                    )
                else:
                    # epoch 0, 第一批已在初始化时加载
                    print_rank0(f"\n📂 Epoch {epoch}, 分片批次 {shard_batch_idx + 1}/{num_shard_batches} (已加载)", rank)
            elif use_chunk_batch_cache_mode and chunk_batch_cache is not None:
                if use_step_segments_mode:
                    # Step-Segments 模式：本段已在外层切换/初始化时加载好，这里不再装载。
                    print_rank0(
                        f"\n📂 Epoch {epoch}, Segment "
                        f"{loaded_segment_idx + 1}/{num_segments_global} (已加载)",
                        rank,
                    )
                elif shard_batch_idx > 0 or epoch > 0:
                    print_rank0(
                        f"\n📂 Epoch {epoch}, ChunkBatch {shard_batch_idx + 1}/{num_shard_batches}",
                        rank,
                    )
                    ep_indices_batch, _vlm_remap_unused, n_frames_batch, batch_shared_cache = (
                        chunk_batch_cache.load_next_batch()
                    )
                    chunk_batch_current_eps = ep_indices_batch
                    shared_vlm_cache = batch_shared_cache  # 全新的 SHM cache
                    _shared_vlm_cache_global = shared_vlm_cache

                    # 关闭/释放旧 train_dataset 的 video reader 等资源
                    try:
                        if train_dataset is not None and hasattr(train_dataset, "close_video_readers"):
                            train_dataset.close_video_readers()
                    except Exception:
                        pass

                    if use_multi_dataset_mode:
                        train_dataset.set_active_episodes(
                            ep_indices_batch, chunk_batch_cache.current_subset_caches
                        )
                    else:
                        train_dataset = LeRobotDataset(
                            dataset_path=args.data_path,
                            num_action_chunks=NUM_ACTIONS_CHUNK,
                            enable_chunking=True,
                            episode_indices=ep_indices_batch,
                            cache_vlm_states=False,
                            cache_max_samples=-1,
                            verbose=False,
                            skip_images=args.skip_images,
                            shared_vlm_cache=shared_vlm_cache,
                            state_process_order=args.state_process_order,
                            hand_binary_columns=args.hand_binary_columns,
                            hand_binary_threshold=args.hand_binary_threshold,
                        )

                    if is_distributed:
                        train_sampler = DistributedSampler(
                            train_dataset,
                            num_replicas=world_size,
                            rank=rank,
                            shuffle=True,
                        )
                    else:
                        train_sampler = None

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=args.batch_size,
                        shuffle=not is_distributed,
                        num_workers=args.num_workers,
                        sampler=train_sampler,
                        collate_fn=partial(lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype),
                        pin_memory=True,
                        drop_last=True,
                        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                        persistent_workers=False,
                    )
                    print_rank0(
                        f"  ✓ ChunkBatch {shard_batch_idx + 1}/{num_shard_batches} 装载: "
                        f"{len(ep_indices_batch)} ep / {n_frames_batch} frame",
                        rank,
                    )
                else:
                    print_rank0(
                        f"\n📂 Epoch {epoch}, ChunkBatch {shard_batch_idx + 1}/{num_shard_batches} (已加载)",
                        rank,
                    )
            
            if is_distributed and train_sampler is not None:
                if use_disk_shuffle_mode:
                    # Disk-Shuffle 模式：epoch 直接驱动全局 shuffle (整数据集为 shuffle 半径)
                    train_sampler.set_epoch(epoch)
                elif use_chunk_batch_cache_mode and use_step_segments_mode:
                    # Step-Segments 模式：sampler seed 用 (segment_idx, epoch) 复合键，
                    # 段固定时 epoch 充当段内 inner-epoch 计数，保证每轮 sample 顺序不同。
                    train_sampler.set_epoch(loaded_segment_idx * 1_000_003 + epoch)
                else:
                    train_sampler.set_epoch(epoch * num_shard_batches + shard_batch_idx)
            
            if is_main_process(rank):
                if use_disk_shuffle_mode:
                    desc = f"Epoch {epoch} DiskShuf step{step}/{args.steps}"
                elif use_chunk_batch_cache_mode and use_step_segments_mode:
                    desc = (
                        f"Epoch {epoch} Seg {loaded_segment_idx + 1}/{num_segments_global} "
                        f"step{step}/{args.steps}"
                    )
                elif use_webdataset_cached_mode:
                    desc = f"Epoch {epoch} Shard {shard_batch_idx + 1}/{num_shard_batches}"
                elif use_chunk_batch_cache_mode:
                    desc = f"Epoch {epoch} CBC {shard_batch_idx + 1}/{num_shard_batches}"
                else:
                    desc = f"Epoch {epoch}/{args.epochs-1}"
                pbar = tqdm(train_loader, desc=desc, ncols=120)
            else:
                pbar = train_loader
            
            accumulated_loss = 0.0
            accumulation_count = 0
            skip_batches = resume_batch_idx if should_resume_epoch and shard_batch_idx == resume_shard_batch_idx else 0
            if skip_batches > 0:
                print_rank0(
                    f"⏭️ 将在 epoch={epoch}, shard_batch_idx={shard_batch_idx} 跳过前 {skip_batches} 个 batch",
                    rank,
                )
            
            # 数据加载时间监控
            prev_batch_end_time = time.time()
            
            for batch_idx, batch in enumerate(pbar):
                if skip_batches > 0 and batch_idx < skip_batches:
                    prev_batch_end_time = time.time()
                    continue

                # 记录batch开始时间（数据已从 DataLoader 获取完成）
                batch_start_time = time.time()
                
                # ========== 数据加载等待时间监控 ==========
                data_wait_time = batch_start_time - prev_batch_end_time
                if is_main_process(rank) and batch_idx > 0:
                    if data_wait_time > 0.1:  # 等待超过 100ms
                        print(f"⚠️ GPU 在等待数据! data_time={data_wait_time:.2f}s (batch {batch_idx})")
                    elif batch_idx % 50 == 0:  # 每 50 个 batch 打印一次正常状态
                        print(f"✅ 流水线正常 data_time={data_wait_time:.3f}s (batch {batch_idx})")
                
                # 转换数据格式
                processed = collate_fn_with_normalizers(batch)
                
                vlm_hidden_states = [v.to(device) for v in processed['vlm_hidden_states']]
                proprioception = processed['proprioception'].to(device)
                gt_actions = processed['gt_actions'].to(device)
                # VLM attention mask: (batch, seq_len * num_layers), 1=valid, 0=padding
                vlm_attention_mask = processed['vlm_attention_mask']
                if vlm_attention_mask is not None:
                    vlm_attention_mask = vlm_attention_mask.to(device)
                # Action chunk mask: (batch, 1, num_chunks * action_dim), 1=valid, 0=pad
                action_chunk_mask = processed.get('action_chunk_mask', None)
                if action_chunk_mask is not None:
                    action_chunk_mask = action_chunk_mask.to(device)
                
                is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0
                
                if is_distributed and is_accumulating:
                    with model.no_sync():
                        with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                            predicted_actions = model(vlm_hidden_states, proprioception, attention_mask=vlm_attention_mask)
                            if use_mixed_loss:
                                loss = compute_mixed_loss(
                                    predicted_actions, gt_actions,
                                    mae_criterion, bce_criterion,
                                    args.bce_action_columns, ACTION_DIM,
                                    args.mae_loss_weight, args.bce_loss_weight,
                                    action_chunk_mask=action_chunk_mask
                                ) / gradient_accumulation_steps
                            else:
                                loss = compute_masked_l1_loss(
                                    predicted_actions, gt_actions, action_chunk_mask
                                ) / gradient_accumulation_steps
                        scaler.scale(loss).backward()
                else:
                    with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        predicted_actions = model(vlm_hidden_states, proprioception, attention_mask=vlm_attention_mask)
                        if use_mixed_loss:
                            loss = compute_mixed_loss(
                                predicted_actions, gt_actions,
                                mae_criterion, bce_criterion,
                                args.bce_action_columns, ACTION_DIM,
                                args.mae_loss_weight, args.bce_loss_weight,
                                action_chunk_mask=action_chunk_mask
                            ) / gradient_accumulation_steps
                        else:
                            loss = compute_masked_l1_loss(
                                predicted_actions, gt_actions, action_chunk_mask
                            ) / gradient_accumulation_steps
                    scaler.scale(loss).backward()
                
                accumulated_loss += float(loss.item() * gradient_accumulation_steps)
                accumulation_count += 1
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optim)
                    
                    # 计算梯度范数 + 梯度裁剪（一步完成，更高效）
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=1.0  # 梯度裁剪阈值
                    ).item()
                    
                    # 原始方式（效率较低，每个参数都会触发一次 CPU-GPU 同步）：
                    # grad_norm = 0.0
                    # for p in model.parameters():
                    #     if p.grad is not None:
                    #         grad_norm += p.grad.data.norm(2).item() ** 2
                    # grad_norm = grad_norm ** 0.5
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optim)
                    scaler.update()
                    scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    
                    batch_end_time = time.time()
                    batch_time = batch_end_time - batch_start_time
                    batch_times.append(batch_time)
                    
                    avg_accumulated_loss = accumulated_loss / accumulation_count
                    epoch_losses.append(avg_accumulated_loss)
                    
                    accumulated_loss = 0.0
                    accumulation_count = 0
                    
                    # 先更新 step，确保所有记录和保存使用一致的 step 值
                    step += 1
                    
                    # 记录每个 step 的 loss 到 custom 日志文件（只在主进程）
                    if step_loss_log_file is not None and is_main_process(rank):
                        current_lr = optim.param_groups[0]['lr']
                        step_loss_log_file.write(f"{step},{epoch},{avg_accumulated_loss:.6f},{current_lr:.8f},{grad_norm:.6f},{batch_time:.4f}\n")
                        # 每 100 步 flush 一次，确保数据写入磁盘
                        if step % 100 == 0:
                            step_loss_log_file.flush()
                    
                    if empty_cache_freq > 0 and step % empty_cache_freq == 0:
                        torch.cuda.empty_cache()
                    
                    # Log to wandb (GPU 统计由 Wandb 自带系统监控自动记录)
                    if wandb_initialized and is_main_process(rank) and step % args.wandb_log_freq == 0:
                        log_dict = {
                            'train/loss': avg_accumulated_loss,
                            'train/step': step,
                            'train/batch_time': batch_time,
                            'train/grad_norm': grad_norm,
                            'train/learning_rate': optim.param_groups[0]['lr'],
                            'train/samples_per_sec': args.batch_size * gradient_accumulation_steps * world_size / batch_time if batch_time > 0 else 0,
                        }
                        
                        try:
                            wandb.log(log_dict)
                        except Exception as e:
                            print(f"⚠️ Wandb log 失败: {type(e).__name__}: {e}")
                    
                    if is_main_process(rank):
                        avg_loss = np.mean(epoch_losses[-20:]) if len(epoch_losses) >= 20 else np.mean(epoch_losses)
                        avg_batch_time = np.mean(batch_times[-20:]) if len(batch_times) >= 20 else np.mean(batch_times)
                        
                        # 计算已用时间和剩余时间
                        elapsed_time = time.time() - training_start_time
                        if step > 0:
                            avg_step_time = elapsed_time / step
                            remaining_steps = args.steps - step
                            remaining_time = avg_step_time * remaining_steps
                            time_str = f"{format_time_seconds(elapsed_time)}/{format_time_seconds(remaining_time)}"
                        else:
                            time_str = f"{format_time_seconds(elapsed_time)}/--"
                        
                        pbar.set_postfix({
                            'loss': f'{avg_accumulated_loss:.4f}',
                            'avg_loss': f'{avg_loss:.4f}',
                            'step': f'{step}/{args.steps}',
                            'time': time_str,
                        })
                
                    # 按步数保存 checkpoint
                    if is_main_process(rank) and args.save_every_steps > 0 and step % args.save_every_steps == 0:
                        step_dir = out_dir / f"step_{step}"
                        model_to_save = model.module if is_distributed else model
                        config_to_save = model_config.copy()
                        config_to_save['saved_at_step'] = step
                        config_to_save['saved_at_epoch'] = epoch
                        config_to_save['saved_at_shard_batch_idx'] = shard_batch_idx
                        config_to_save['saved_next_batch_idx'] = batch_idx + 1
                        training_state = {
                            "checkpoint_version": 2,
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optim.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "step": step,
                            "epoch": epoch,
                            "shard_batch_idx": shard_batch_idx,
                            "next_batch_idx": batch_idx + 1,
                            "completed_epoch": False,
                            "best_val_loss": None if best_val_loss == float('inf') else float(best_val_loss),
                            "best_train_loss": None if best_train_loss == float('inf') else float(best_train_loss),
                            "best_epoch": best_epoch,
                            "best_step": best_step,
                            "elapsed_training_time_sec": max(0.0, time.time() - training_start_time),
                            "rng_state": get_rng_state(),
                            "args_snapshot": vars(args).copy(),
                            "model_config": config_to_save.copy(),
                        }
                        save_training_checkpoint(step_dir, model_to_save.state_dict(), config_to_save, training_state)
                        print(f"\n✓ Step checkpoint saved to {step_dir}/")
                
                # 更新上一个 batch 结束时间（用于下一个 batch 的数据等待时间计算）
                prev_batch_end_time = time.time()
                
                if step >= args.steps:
                    break
                # Step-Segments 模式：跨过当前段尾立即跳出 batch loop，
                # 让外层 epoch loop 在下一轮检测到段切换并加载新 ChunkBatch。
                if (
                    use_chunk_batch_cache_mode
                    and use_step_segments_mode
                    and step >= (loaded_segment_idx + 1) * steps_per_segment_global
                    and loaded_segment_idx < num_segments_global - 1
                ):
                    break
            
            # ========== batch_idx 循环结束（一个分片批次训练完成） ==========
            if is_main_process(rank):
                pbar.close()
            
            # 分片批次结束后检查是否需要跳出
            if step >= args.steps:
                break
            if should_resume_epoch and shard_batch_idx == resume_shard_batch_idx:
                resume_batch_idx = 0
        
        # ========== shard_batch_idx 循环结束（所有分片批次遍历完成） ==========
        
        # ========== 一个 epoch 结束（在 shard_batch_idx 循环外） ==========
        if step >= args.steps:
            break
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            if is_distributed and val_sampler is not None:
                val_sampler.set_epoch(epoch)
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation", ncols=120) if is_main_process(rank) else val_loader
                for batch in val_pbar:
                    processed = collate_fn_with_normalizers(batch)
                    vlm_hidden_states = [v.to(device) for v in processed['vlm_hidden_states']]
                    proprioception = processed['proprioception'].to(device)
                    gt_actions = processed['gt_actions'].to(device)
                    vlm_attention_mask = processed['vlm_attention_mask']
                    if vlm_attention_mask is not None:
                        vlm_attention_mask = vlm_attention_mask.to(device)
                    action_chunk_mask = processed.get('action_chunk_mask', None)
                    if action_chunk_mask is not None:
                        action_chunk_mask = action_chunk_mask.to(device)
                    
                    with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        predicted_actions = model(vlm_hidden_states, proprioception, attention_mask=vlm_attention_mask)
                        if use_mixed_loss:
                            loss = compute_mixed_loss(
                                predicted_actions, gt_actions,
                                mae_criterion, bce_criterion,
                                args.bce_action_columns, ACTION_DIM,
                                args.mae_loss_weight, args.bce_loss_weight,
                                action_chunk_mask=action_chunk_mask
                            )
                        else:
                            loss = compute_masked_l1_loss(
                                predicted_actions, gt_actions, action_chunk_mask
                            )
                    val_losses.append(float(loss.item()))
            
            val_loss = np.mean(val_losses)
            train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_batch_time = np.mean(batch_times) if batch_times else 0.0
            print_rank0(f"\nEpoch {epoch} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Avg Batch Time: {avg_batch_time:.3f}s", rank)
            
            if wandb_initialized and is_main_process(rank):
                safe_wandb_log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss': val_loss,
                    'train/avg_batch_time': avg_batch_time,
                    'progress/epoch': epoch,
                    'progress/step': step,
                }, rank=rank)
            
            # 保存最佳模型
            if is_main_process(rank) and val_loss < best_val_loss and not args.no_save_best:
                best_val_loss = val_loss
                best_epoch = epoch
                best_step = step
                best_dir = out_dir.parent / "best"
                model_to_save = model.module if is_distributed else model
                config_to_save = model_config.copy()
                config_to_save['best_val_loss'] = best_val_loss
                config_to_save['best_epoch'] = best_epoch
                config_to_save['best_step'] = best_step
                training_state = {
                    "checkpoint_version": 2,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "shard_batch_idx": 0,
                    "next_batch_idx": 0,
                    "completed_epoch": True,
                    "best_val_loss": float(best_val_loss),
                    "best_train_loss": None if best_train_loss == float('inf') else float(best_train_loss),
                    "best_epoch": best_epoch,
                    "best_step": best_step,
                    "elapsed_training_time_sec": max(0.0, time.time() - training_start_time),
                    "rng_state": get_rng_state(),
                    "args_snapshot": vars(args).copy(),
                    "model_config": config_to_save.copy(),
                }
                save_training_checkpoint(best_dir, model_to_save.state_dict(), config_to_save, training_state)
                print(f"✓ Saved best model to {best_dir}/ with val_loss={val_loss:.6f}")
                
                if wandb_initialized:
                    try:
                        safe_wandb_log({'best/val_loss': best_val_loss, 'best/epoch': best_epoch, 'best/step': best_step}, rank=rank)
                        wandb.run.summary['best_val_loss'] = best_val_loss
                        wandb.run.summary['best_epoch'] = best_epoch
                        wandb.run.summary['best_step'] = best_step
                    except Exception:
                        pass
        else:
            train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_batch_time = np.mean(batch_times) if batch_times else 0.0
            print_rank0(f"\nEpoch {epoch} Summary - Train Loss: {train_loss:.6f}, Avg Batch Time: {avg_batch_time:.3f}s", rank)
            
            if wandb_initialized and is_main_process(rank):
                safe_wandb_log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/avg_batch_time': avg_batch_time,
                    'progress/epoch': epoch,
                    'progress/step': step,
                }, rank=rank)
            
            # 没有验证集时，使用训练损失保存最佳模型
            if is_main_process(rank) and train_loss < best_train_loss and not args.no_save_best:
                best_train_loss = train_loss
                best_epoch = epoch
                best_step = step
                best_dir = out_dir.parent / "best"
                model_to_save = model.module if is_distributed else model
                config_to_save = model_config.copy()
                config_to_save['best_train_loss'] = best_train_loss
                config_to_save['best_epoch'] = best_epoch
                config_to_save['best_step'] = best_step
                training_state = {
                    "checkpoint_version": 2,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "shard_batch_idx": 0,
                    "next_batch_idx": 0,
                    "completed_epoch": True,
                    "best_val_loss": None if best_val_loss == float('inf') else float(best_val_loss),
                    "best_train_loss": float(best_train_loss),
                    "best_epoch": best_epoch,
                    "best_step": best_step,
                    "elapsed_training_time_sec": max(0.0, time.time() - training_start_time),
                    "rng_state": get_rng_state(),
                    "args_snapshot": vars(args).copy(),
                    "model_config": config_to_save.copy(),
                }
                save_training_checkpoint(best_dir, model_to_save.state_dict(), config_to_save, training_state)
                print(f"✓ Saved best model to {best_dir}/ with train_loss={train_loss:.6f}")
                
                if wandb_initialized:
                    try:
                        safe_wandb_log({'best/train_loss': best_train_loss, 'best/epoch': best_epoch, 'best/step': best_step}, rank=rank)
                        wandb.run.summary['best_train_loss'] = best_train_loss
                        wandb.run.summary['best_epoch'] = best_epoch
                        wandb.run.summary['best_step'] = best_step
                    except Exception:
                        pass
        
        # 按 epoch 保存 checkpoint
        if is_main_process(rank) and epoch % args.save_every == 0:
            epoch_dir = out_dir / f"epoch_{epoch}"
            model_to_save = model.module if is_distributed else model
            config_to_save = model_config.copy()
            training_state = {
                "checkpoint_version": 2,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": step,
                "epoch": epoch,
                "shard_batch_idx": 0,
                "next_batch_idx": 0,
                "completed_epoch": True,
                "best_val_loss": None if best_val_loss == float('inf') else float(best_val_loss),
                "best_train_loss": None if best_train_loss == float('inf') else float(best_train_loss),
                "best_epoch": best_epoch,
                "best_step": best_step,
                "elapsed_training_time_sec": max(0.0, time.time() - training_start_time),
                "rng_state": get_rng_state(),
                "args_snapshot": vars(args).copy(),
                "model_config": config_to_save.copy(),
            }
            save_training_checkpoint(epoch_dir, model_to_save.state_dict(), config_to_save, training_state)
            print(f"✓ Checkpoint saved to {epoch_dir}/")
        
        if step >= args.steps:
            break
    
    # ========== epoch 循环结束 ==========

    # 保存最终 checkpoint
    if is_main_process(rank):
        final_epoch_dir = out_dir / f"epoch_{epoch}"
        if not final_epoch_dir.exists():
            model_to_save = model.module if is_distributed else model
            config_to_save = model_config.copy()
            training_state = {
                "checkpoint_version": 2,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": step,
                "epoch": epoch,
                "shard_batch_idx": 0,
                "next_batch_idx": 0,
                "completed_epoch": True,
                "best_val_loss": None if best_val_loss == float('inf') else float(best_val_loss),
                "best_train_loss": None if best_train_loss == float('inf') else float(best_train_loss),
                "best_epoch": best_epoch,
                "best_step": best_step,
                "elapsed_training_time_sec": max(0.0, time.time() - training_start_time),
                "rng_state": get_rng_state(),
                "args_snapshot": vars(args).copy(),
                "model_config": config_to_save.copy(),
            }
            save_training_checkpoint(final_epoch_dir, model_to_save.state_dict(), config_to_save, training_state)
            print(f"✓ Final checkpoint saved to {final_epoch_dir}/")
        print(f"\n✓ Training completed! All checkpoints saved to {out_dir}/")

    if wandb_initialized and is_main_process(rank):
        try:
            training_duration = time.time() - training_start_time
            wandb.run.summary['total_training_time_sec'] = training_duration
            wandb.run.summary['total_training_time_min'] = training_duration / 60
            wandb.run.summary['total_steps'] = step
            wandb.run.summary['total_epochs'] = epoch + 1
            wandb.run.summary['final_train_loss'] = train_loss
            wandb.run.summary['avg_samples_per_sec'] = (step * args.batch_size * gradient_accumulation_steps * world_size) / training_duration if training_duration > 0 else 0
            wandb.finish()
            print("✓ Wandb 日志已保存")
        except Exception as e:
            print(f"⚠️ Wandb 结束时出错: {e}")
    
    # 关闭 step loss 日志文件
    if step_loss_log_file is not None and is_main_process(rank):
        step_loss_log_file.flush()
        step_loss_log_file.close()
        print(f"✓ Step loss 日志已保存到 {custom_log_dir}/")

    # 清理 ChunkBatchCache（必须在 shared_vlm_cache 之前，因为 shared_vlm_cache 可能就是它包装出来的）
    if chunk_batch_cache is not None:
        try:
            chunk_batch_cache.cleanup()
            print_rank0("✓ ChunkBatchCache 共享内存已清理", rank)
        except Exception as e:
            print_rank0(f"⚠️ 清理 ChunkBatchCache 时出错: {e}", rank)
        # ChunkBatchCache.cleanup 已经处理了内部 SHM；
        # shared_vlm_cache 是它的 RemappedSharedVLMCache，下面再调一次 close 是安全的（幂等）
        _shared_vlm_cache_global = None
        _shared_vlm_cache_rank = None

    # 清理共享内存缓存
    if shared_vlm_cache is not None:
        try:
            shared_vlm_cache.close()
            if rank == 0 and chunk_batch_cache is None:
                # ChunkBatchCache 已 unlink，不要重复
                shared_vlm_cache.unlink()
            print_rank0("✓ 共享内存缓存已清理", rank)
        except Exception as e:
            print_rank0(f"⚠️ 清理共享内存缓存时出错: {e}", rank)
        
        # 清除全局变量
        _shared_vlm_cache_global = None
        _shared_vlm_cache_rank = None
    
    # 清理 NVML 资源
    shutdown_nvml()
    
    cleanup_distributed()


if __name__ == "__main__":
    # ========================================================================
    # 运行方式说明 Execution Methods
    # ========================================================================
    # 
    # 方式 1 (推荐用于正式训练): 命令行运行
    # Method 1 (Recommended for production): Command-line execution
    # 
    #   python train_for_libero_qwen2b_multigpu.py --data_path /path/to/dataset --batch_size 32 --epochs 300
    # 
    #   - 灵活: 可以随时改变参数而不修改代码
    #   - 规范: 适合在服务器上批量运行实验
    #   - 使用: 注释掉下面的 sys.argv 设置，直接调用 main()
    # 
    # 方式 2: 多GPU训练
    # Method 2: Multi-GPU Training
    # 
    #   torchrun --nproc_per_node=4 train_for_libero_qwen2b_multigpu.py --data_path /path/to/dataset
    # 
    #   - 多GPU: 自动分配数据到多个GPU
    #   - 高效: 线性加速训练
    # 
    # 方式 3 (推荐用于快速测试): 直接运行脚本
    # Method 3 (Recommended for quick testing): Direct script execution
    # 
    #   python train_for_libero_qwen2b_multigpu.py  (不带任何参数)
    # 
    #   - 方便: 在 IDE 中点击运行按钮即可
    #   - 快速: 不需要每次输入长长的命令行
    #   - 使用: 在下面的 sys.argv 中设置好参数
    # 
    # ⚠️ 注意: 如果同时设置了 sys.argv 和命令行参数，命令行参数优先级更高
    # ⚠️ Note: If both sys.argv and command-line args are provided, command-line takes precedence
    # 
    # ========================================================================
    # 
    # 终端运行命令 (单GPU) Terminal Command (Single GPU):
    # ========================================================================
    # 
    # python train_for_libero_qwen2b_multigpu.py \
    #     --data_path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_qwen2b-1layer-only-libero_spatial/libero_lerobot_spatial \
    #     --batch_size 32 \
    #     --num_workers 4 \
    #     --val_split 0 \
    #     --epochs 300 \
    #     --steps 10000 \
    #     --lr 1e-4 \
    #     --weight_decay 1e-5 \
    #     --warmup_ratio 0.05 \
    #     --adam_beta1 0.95 \
    #     --adam_beta2 0.999 \
    #     --num_transformer_blocks 2 \
    #     --num_attention_heads 8 \
    #     --dropout 0.1 \
    #     --action_head_hidden_dim 4096 \
    #     --num_vlm_layers 1 \
    #     --vlm_output_dim 1536 \
    #     --save_every 1 \
    #     --save_every_steps 500 \
    #     --out_dir /data/HuangWenlong/datasets/Sai0_action_head_weight_libero/Qwen2-VL-2B-Instruct-hidden_dim_1536-vlm_layer_num_1/OFT1_0/experiments/libero_lerobot_spatial/checkpoints \
    #     --log_dir /data/HuangWenlong/datasets/Sai0_action_head_weight_libero/Qwen2-VL-2B-Instruct-hidden_dim_1536-vlm_layer_num_1/OFT1_0/experiments/libero_lerobot_spatial/logs \
    #     --device cuda:0 \
    #     --gradient_accumulation_steps 1 \
    #     --use_amp \
    #     --amp_dtype float16 \
    #     --empty_cache_freq 0 \
    #     --use_wandb \
    #     --wandb_project Qwen2B_OFT_Training
    # 
    # ========================================================================
    # 
    # 终端运行命令 (多GPU, 4卡) Terminal Command (Multi-GPU, 4 GPUs):
    # ========================================================================
    # 
    # torchrun --nproc_per_node=4 train_for_libero_qwen2b_multigpu.py \
    #     --data_path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_qwen2b-1layer-only-libero_spatial/libero_lerobot_spatial \
    #     --batch_size 32 \
    #     --num_workers 4 \
    #     --val_split 0 \
    #     --epochs 300 \
    #     --steps 10000 \
    #     --lr 1e-4 \
    #     --weight_decay 1e-5 \
    #     --warmup_ratio 0.05 \
    #     --adam_beta1 0.95 \
    #     --adam_beta2 0.999 \
    #     --num_transformer_blocks 2 \
    #     --num_attention_heads 8 \
    #     --dropout 0.1 \
    #     --action_head_hidden_dim 4096 \
    #     --num_vlm_layers 1 \
    #     --vlm_output_dim 1536 \
    #     --save_every 1 \
    #     --save_every_steps 500 \
    #     --out_dir /data/HuangWenlong/datasets/Sai0_action_head_weight_libero/Qwen2-VL-2B-Instruct-hidden_dim_1536-vlm_layer_num_1/OFT1_0/experiments/libero_lerobot_spatial/checkpoints \
    #     --log_dir /data/HuangWenlong/datasets/Sai0_action_head_weight_libero/Qwen2-VL-2B-Instruct-hidden_dim_1536-vlm_layer_num_1/OFT1_0/experiments/libero_lerobot_spatial/logs \
    #     --gradient_accumulation_steps 1 \
    #     --use_amp \
    #     --amp_dtype float16 \
    #     --empty_cache_freq 0 \
    #     --use_wandb \
    #     --wandb_project Qwen2B_OFT_Training
    # 
    # ========================================================================
    # 
    # 指定特定GPU运行 (例如使用 GPU 4,5,6,7):
    # Run on specific GPUs (e.g., GPU 4,5,6,7):
    # 
    # CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 train_for_libero_qwen2b_multigpu.py \
    #     --data_path /path/to/dataset \
    #     ... (其他参数同上)
    # 
    # ========================================================================
    # 
    # 参数说明 Parameter Description:
    # 
    # 数据参数 Data Parameters:
    #   --data_path          : LeRobot 数据集路径 (必需)
    #   --batch_size         : 训练批次大小 (默认: 32, 原始N1.5: 32)
    #   --num_workers        : 数据加载线程数 (默认: 4)
    #   --val_split          : 验证集比例 (默认: 0, 不使用验证集)
    # 
    # 训练参数 Training Parameters:
    #   --epochs             : 训练轮数 (默认: 300, 原始N1.5: 300)
    #   --steps              : 最大训练步数 (默认: 10000, 原始N1.5: 10000)
    #   --lr                 : 学习率 (默认: 1e-4, 原始N1.5: 1e-4)
    #   --weight_decay       : 权重衰减 (默认: 1e-5, 原始N1.5: 1e-5)
    #   --warmup_ratio       : 预热比例 (默认: 0.05, 原始N1.5: 0.05)
    #   --adam_beta1         : AdamW beta1 (默认: 0.95, 原始N1.5: 0.95)
    #   --adam_beta2         : AdamW beta2 (默认: 0.999, 原始N1.5: 0.999)
    # 
    # 模型参数 Model Parameters:
    #   --num_transformer_blocks : Transformer 块数量 (默认: 2)
    #   --num_attention_heads    : 注意力头数量 (默认: 8)
    #   --dropout                : Dropout 比率 (默认: 0.1)
    #   --action_head_hidden_dim : Action head 隐藏层维度 (默认: 4096)
    #   --num_vlm_layers         : VLM 隐藏层数量 (可从data_path自动检测)
    #   --vlm_output_dim         : VLM 输出维度 (Qwen2B: 1536, Qwen4B: 2560)
    # 
    # 保存参数 Checkpoint Parameters:
    #   --save_every         : 每 N 个 epoch 保存 checkpoint (默认: 1)
    #   --save_every_steps   : 每 N 个 step 保存 checkpoint (默认: 0, 禁用)
    #   --out_dir            : Checkpoint 输出目录
    #   --log_dir            : 日志输出目录 (tensorboard)
    # 
    # 系统参数 System Parameters:
    #   --device             : 训练设备 (默认: cuda:0, 多GPU时自动分配)
    #   --gradient_accumulation_steps : 梯度累积步数 (默认: 1)
    # 
    # 内存优化 Memory Optimization:
    #   --use_amp / --no_amp : 启用/禁用混合精度训练 (默认: 启用)
    #   --amp_dtype          : AMP 数据类型 (float16 或 bfloat16)
    #   --empty_cache_freq   : CUDA 缓存清理频率 (默认: 0, 禁用)
    # 
    # Wandb 配置 W&B Configuration:
    #   --use_wandb / --no_wandb : 启用/禁用 W&B 日志 (默认: 启用)
    #   --wandb_project      : W&B 项目名称
    #   --wandb_run_name     : W&B 运行名称 (可选, 自动生成)
    # 
    # ========================================================================
    
    main()
