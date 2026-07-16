"""
训练脚本 - 支持从预训练权重初始化 (多GPU分布式训练版本)
使用 LeRobot Dataset Loader 加载真实数据

多GPU训练使用方法 (Multi-GPU Training):
  # 使用 torchrun 启动分布式训练 (推荐)
  torchrun --nproc_per_node=4 train_with_pretrained_action_head_weight_eagle25_multigpu.py \
    --data_path /path/to/dataset \
    --pretrained_weights ./pretrained_action_head.pt \
    --steps 1000

  # 指定特定GPU运行
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_with_pretrained_action_head_weight_eagle25_multigpu.py \
    --data_path /path/to/dataset

单GPU训练使用方法 (Single-GPU Training):
  python train_with_pretrained_action_head_weight_eagle25_multigpu.py \
    --data_path /path/to/dataset \
    --device cuda:0

参数说明:
  --gradient_accumulation_steps: 梯度累积步数 (默认: 1)
  --local_rank: 本地进程排名 (由 torchrun 自动设置，无需手动指定)
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import List, Optional
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
from transformers.feature_extraction_utils import BatchFeature
from tqdm import tqdm
import wandb

from models.action_head.flow_matching_action_head import FlowmatchingActionHead
from config import get_flowmatching_action_head_config_original

# 添加 utils 路径以导入 lerobot_dataset_loader
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'utils'))
from lerobot_dataset_loader import (
    create_lerobot_dataloader,
    preload_vlm_cache_distributed,
    get_total_vlm_states,
)
import atexit
import signal
DEFAULT_EMBODIMENT_ID = 31

# 全局变量用于清理共享内存
_shared_vlm_cache_global = None
_shared_vlm_cache_rank = None

def cleanup_shared_memory_cache():
    """清理共享内存缓存"""
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    if _shared_vlm_cache_global is not None:
        try:
            _shared_vlm_cache_global.close()
            if _shared_vlm_cache_rank == 0:
                _shared_vlm_cache_global.unlink()
                print("✓ 共享内存缓存已清理")
        except Exception as e:
            print(f"清理共享内存时出错: {e}")


# ============================================================================
# 四元数转轴角函数 - Quaternion to Axis-Angle Conversion
# ============================================================================
# 
# 参考: Isaac-GR00T/examples/Libero/eval/utils.py 第 68-92 行
# 
# 用户的 state 格式 (9维):
#   gripper1, gripper2, x, y, z, qx, qy, qz, qw
#   需要转换: 四元数 (qx, qy, qz, qw) → 轴角 (ax, ay, az)
# 
# 用户的 action 格式 (7维):
#   delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper
#   已经是 7 维，不需要转换
# 
# ============================================================================

def quat2axisangle_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    四元数转轴角 (PyTorch 批量版本)
    
    参考: Isaac-GR00T/examples/Libero/eval/utils.py quat2axisangle()
    
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.
    
    Args:
        quat: (batch, 4) 或 (batch, seq, 4)，四元数 (qx, qy, qz, qw)
    
    Returns:
        axis_angle: (batch, 3) 或 (batch, seq, 3)，轴角 (ax, ay, az)
    """
    original_shape = quat.shape
    
    # 确保是 2D: (N, 4)
    if quat.dim() == 3:
        batch_size, seq_len, _ = quat.shape
        quat = quat.reshape(-1, 4)  # (batch * seq, 4)
    else:
        batch_size, seq_len = quat.shape[0], None
    
    # quat: (N, 4) = (qx, qy, qz, qw)
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    
    # Clip qw to [-1, 1] to avoid numerical issues with acos
    qw = torch.clamp(qw, -1.0, 1.0)
    
    # Calculate denominator: sqrt(1 - qw^2)
    den = torch.sqrt(1.0 - qw * qw)
    
    # Calculate angle
    angle = 2.0 * torch.acos(qw)
    
    # Handle near-zero rotation (den ≈ 0)
    # 当 den 接近 0 时，返回零向量
    small_angle_mask = den < 1e-8
    
    # Compute axis-angle
    axis_angle = torch.zeros(quat.shape[0], 3, dtype=quat.dtype, device=quat.device)
    
    # For non-small angles: axis_angle = (qx, qy, qz) * angle / den
    if (~small_angle_mask).any():
        scale = angle[~small_angle_mask] / den[~small_angle_mask]
        axis_angle[~small_angle_mask, 0] = qx[~small_angle_mask] * scale
        axis_angle[~small_angle_mask, 1] = qy[~small_angle_mask] * scale
        axis_angle[~small_angle_mask, 2] = qz[~small_angle_mask] * scale
    
    # For small angles: axis_angle ≈ 0 (already initialized to zeros)
    
    # Reshape back if needed
    if seq_len is not None:
        axis_angle = axis_angle.reshape(batch_size, seq_len, 3)
    
    return axis_angle


def convert_state_quat_to_axisangle(state: torch.Tensor) -> torch.Tensor:
    """
    将用户的 9 维 state 转换为 8 维 state
    
    用户 state 格式 (9维):
        [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
            0         1      2  3  4   5   6   7   8
    
    转换后 state 格式 (8维):
        [gripper1, gripper2, x, y, z, ax, ay, az]
            0         1      2  3  4   5   6   7
    
    Args:
        state: (batch, 9) 用户原始 state
    
    Returns:
        converted_state: (batch, 8) 转换后的 state
    """
    batch_size = state.shape[0]
    
    # 提取各部分 (按用户数据顺序)
    gripper = state[:, 0:2]           # (batch, 2): gripper1, gripper2
    position = state[:, 2:5]          # (batch, 3): x, y, z
    quat = state[:, 5:9]              # (batch, 4): qx, qy, qz, qw
    
    # 四元数 → 轴角
    axis_angle = quat2axisangle_torch(quat)  # (batch, 3): ax, ay, az
    
    # 拼接: [gripper, position, axis_angle]
    converted_state = torch.cat([gripper, position, axis_angle], dim=1)  # (batch, 8)
    
    return converted_state


# convert_action_quat_to_axisangle 不再需要，因为 action 已经是 7 维
# Action 格式: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]


# ============================================================================
# 训练配置类 - Training Configuration Class
# ============================================================================
# 
# ⚠️ 重要说明 IMPORTANT NOTE:
# 
# TrainingConfig 类仅用于设置命令行参数的【默认值】
# TrainingConfig is ONLY used for setting DEFAULT VALUES for command-line arguments
# 
# 参数修改的三种方式 (Three ways to modify parameters):
# 
# 1️⃣ 修改 TrainingConfig 类中的值
#    - 作用: 改变所有参数的默认值
#    - 影响: 命令行不指定参数时使用这些值
#    - 示例: batch_size: int = 4  →  改为 8 后，默认 batch_size=8
# 
# 2️⃣ 在 if __name__ == "__main__": 中修改 sys.argv
#    - 作用: 直接运行脚本时使用的参数（推荐用于快速测试）
#    - 影响: 只在 Python 直接运行时生效，命令行运行时被忽略
#    - 示例: '--batch_size', '8'  →  直接运行脚本时 batch_size=8
# 
# 3️⃣ 命令行参数 (推荐用于正式训练)
#    - 作用: 通过命令行传入参数
#    - 影响: 优先级最高，会覆盖 TrainingConfig 中的默认值
#    - 示例: python train.py --batch_size 8  →  batch_size=8
# 
# 优先级 Priority: 命令行参数 > sys.argv > TrainingConfig 默认值
#                Command-line > sys.argv > TrainingConfig defaults
# 
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Flow Matching Action Head 训练配置
    
    该配置类包含所有训练超参数，便于查看和修改。
    所有参数都有详细的中英文说明。
    
    ⚠️ 注意: 此类中的值仅作为【默认值】，会被命令行参数或 sys.argv 覆盖
    ⚠️ Note: Values here are DEFAULT VALUES only, will be overridden by command-line args or sys.argv
    
    修改这里的值 → 改变默认值（不传参数时生效）
    Modify values here → Change defaults (takes effect when no args provided)
    
    ⚠️ 参数已与原始 N1.5 LIBERO 训练配置对齐
    ⚠️ Parameters aligned with original N1.5 LIBERO training config
    """
    
    # ========== 数据相关参数 Data Parameters ==========
    data_path: str = ""
    """数据集路径 (必需)
    Path to LeRobot dataset (Required)
    示例: '/path/to/pickupanapple_v1_hidden_dim_3_512_2048'
    """
    
    batch_size: int = 32  # 原始 N1.5: 32
    """训练批次大小 (原始 N1.5 使用 32)
    Training batch size (Original N1.5 uses 32)
    - 较大的 batch_size 训练更稳定但需要更多显存
    """
    
    num_workers: int = 4
    """数据加载线程数
    Number of data loading workers
    - 0: 主线程加载 (推荐调试时使用)
    - 4-12: 多线程加载 (推荐正式训练)
    """
    
    val_split: float = 0.0
    """验证集比例 (原始 N1.5 不使用验证集)
    Validation split ratio (Original N1.5 does not use validation)
    - 0.0: 不使用验证集 (推荐)
    """
    
    # ========== 训练超参数 Training Hyperparameters ==========
    epochs: int = 100000  # 原始 N1.5: 300
    """训练轮数 (原始 N1.5 使用 300)
    Number of training epochs (Original N1.5 uses 300)
    """
    
    steps: int = 20000  # 原始 N1.5: 10000
    """最大训练步数 (原始 N1.5 使用 10000)
    Maximum training steps (Original N1.5 uses 10000)
    - 达到此步数后自动停止训练
    """
    
    lr: float = 1e-4  # 原始 N1.5: 1e-4
    """学习率 (原始 N1.5 使用 1e-4)
    Learning rate (Original N1.5 uses 1e-4)
    """
    
    weight_decay: float = 1e-5  # 原始 N1.5: 1e-5
    """权重衰减 (原始 N1.5 使用 1e-5)
    Weight decay (Original N1.5 uses 1e-5)
    """
    
    warmup_ratio: float = 0.05  # 原始 N1.5: 0.05
    """预热比例 (原始 N1.5 使用 0.05)
    Warmup ratio (Original N1.5 uses 0.05)
    - 前 5% 的训练步数用于学习率预热
    """
    
    adam_beta1: float = 0.95  # 原始 N1.5: 0.95
    """AdamW beta1 参数 (原始 N1.5 使用 0.95，而非默认的 0.9)
    AdamW beta1 (Original N1.5 uses 0.95, not default 0.9)
    """
    
    adam_beta2: float = 0.999  # 原始 N1.5: 0.999
    """AdamW beta2 参数 (原始 N1.5 使用 0.999)
    AdamW beta2 (Original N1.5 uses 0.999)
    """
    
    # ========== 模型维度参数 Model Dimension Parameters ==========
    max_action_dim: int = 32
    """模型的最大动作维度 (padding 目标)
    Maximum action dimension (for padding)
    - 必须与预训练模型一致
    - 实际 action_dim 会自动 padding 到此维度
    - ⚠️ 不要修改此值，否则无法加载预训练权重
    """
    
    max_state_dim: int = 64
    """模型的最大状态维度 (padding 目标)
    Maximum state dimension (for padding)
    - 必须与预训练模型一致
    - 实际 state_dim 会自动 padding 到此维度
    - ⚠️ 不要修改此值，否则无法加载预训练权重
    """
    
    num_action_chunks: int = 16  # 原始 N1.5 LIBERO: 16
    """动作预测时间步数 (原始 N1.5 LIBERO 使用 16)
    Action horizon (Original N1.5 LIBERO uses 16)
    - 也称为 action_horizon
    - 表示一次预测未来多少步的动作
    """
    
    # ========== 权重与保存 Weights and Checkpoints ==========
    use_pretrain: bool = True
    """是否使用预训练权重
    Whether to use pretrained weights
    - True: 加载预训练权重进行微调
    - False: 从头开始训练
    """
    
    pretrained_weights: str = "./pretrained_action_head.pt"
    """预训练权重文件路径
    Path to pretrained weights file
    - 用于迁移学习/微调
    - 仅当 use_pretrain=True 时生效
    示例: './pretrained_action_head.pt'
    """
    
    out_dir: str = "./experiments/fm0_pretrained_finetuning/checkpoints"
    """checkpoint 输出目录
    Checkpoint output directory
    - 训练过程中会保存 epoch_0/, epoch_1/, best/ 等子目录
    - 每个子目录包含 action_head.pt 和 config.json
    """
    
    log_dir: str = "./experiments/fm0_pretrained_finetuning/logs"
    """日志输出目录
    Logs output directory
    - 保存 tensorboard 和 wandb 日志
    """
    
    save_every: int = 1
    """每 N 个 epoch 保存一次 checkpoint
    Save checkpoint every N epochs
    - 1: 每个 epoch 都保存
    - 5: 每 5 个 epoch 保存一次
    """
    
    save_every_steps: int = 0
    """每 N 个 step 保存一次 checkpoint
    Save checkpoint every N steps
    - 0: 禁用按步数保存 (默认)
    - 500: 每 500 步保存一次
    - 1000: 每 1000 步保存一次
    """
    
    # ========== 系统配置 System Configuration ==========
    device: str = "cuda:0"
    """训练设备 (单GPU模式使用)
    Training device (for single-GPU mode)
    - 'cuda:0', 'cuda:1', etc.: 指定 GPU
    - 'cpu': 使用 CPU (非常慢，不推荐)
    - 多GPU模式下，此参数会被 local_rank 覆盖
    """
    
    gradient_accumulation_steps: int = 1
    """梯度累积步数
    Gradient accumulation steps
    - 1: 不使用梯度累积 (默认)
    - N: 每 N 个 batch 更新一次参数
    - 有效 batch_size = batch_size * gradient_accumulation_steps * world_size
    """
    
    # ========== 内存优化参数 Memory Optimization ==========
    use_amp: bool = True
    """是否使用混合精度训练 (AMP)
    Whether to use Automatic Mixed Precision training
    - True: 使用 FP16/BF16 混合精度，显著减少显存占用 (推荐)
    - False: 使用 FP32 全精度训练
    """
    
    amp_dtype: str = "float16"
    """混合精度训练的数据类型
    Data type for AMP
    - 'float16': FP16 (适用于大多数 GPU)
    - 'bfloat16': BF16 (适用于 Ampere 及以上架构，如 A100, RTX 3090)
    """
    
    empty_cache_freq: int = 0
    """清理 CUDA 缓存的频率 (每 N 步)
    Frequency of emptying CUDA cache (every N steps)
    - 0: 不清理 (默认，推荐)
    - 50-100: 显存非常紧张时
    - 100-500: 显存较紧张时
    注意: 大多数情况下不需要开启，PyTorch 会自动管理显存
    """
    
    # ========== Weights & Biases 配置 W&B Configuration ==========
    use_wandb: bool = True
    """是否使用 Weights & Biases 记录训练
    Whether to use Weights & Biases for logging
    - True: 启用远程实验跟踪
    - False: 只使用本地日志
    """
    
    wandb_project: str = "gr00t_flowmatching_training"
    """W&B 项目名称
    Weights & Biases project name
    - 所有实验会分组到此项目下
    """
    
    wandb_run_name: str = ""
    """W&B 运行名称 (可选)
    Weights & Biases run name (optional)
    - "": 自动生成 (推荐)
    - 字符串: 自定义名称
    示例: 'fm0_pretrained_lr1e-4_bs8_20251127'
    """
    
    wandb_log_freq: int = 10
    """Wandb 基本日志记录频率 (每 N 步)
    Frequency for basic wandb logging (every N steps)
    - 记录: loss, learning_rate, batch_time, grad_norm 等
    """
    
    
    def to_dict(self):
        """转换为字典格式，用于保存配置"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def from_args(cls, args: argparse.Namespace):
        """从命令行参数创建配置"""
        return cls(
            data_path=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_split=args.val_split,
            epochs=args.epochs,
            steps=args.steps,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            max_action_dim=args.max_action_dim,
            max_state_dim=args.max_state_dim,
            num_action_chunks=args.num_action_chunks,
            pretrained_weights=args.pretrained_weights,
            out_dir=args.out_dir,
            log_dir=args.log_dir,
            save_every=args.save_every,
            device=args.device,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )


# ============================================================================
# 分布式训练辅助函数 Distributed Training Helper Functions
# ============================================================================

def setup_distributed():
    """
    初始化分布式训练环境
    
    Returns:
        rank: 全局进程排名
        local_rank: 本地进程排名 (用于确定 GPU)
        world_size: 总进程数
        is_distributed: 是否为分布式模式
    """
    # 检查是否由 torchrun 或 torch.distributed.launch 启动
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # 初始化进程组 (设置 4 小时超时，支持大规模数据预加载)
        dist.init_process_group(
            backend='nccl', 
            init_method='env://',
            timeout=datetime.timedelta(hours=4)
        )
        
        # 设置当前进程使用的 GPU
        torch.cuda.set_device(local_rank)
        
        return rank, local_rank, world_size, True
    else:
        # 单 GPU 模式
        return 0, 0, 1, False


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    """检查是否为主进程 (rank 0)"""
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


# ============================================================================
# 归一化处理类 - Normalization Class (与原始 N1.5 对齐)
# ============================================================================

class MinMaxNormalizer:
    """
    Min-Max 归一化器，与原始 GR00T N1.5 的归一化方式完全对齐
    
    参考: Isaac-GR00T/gr00t/data/transform/state_action.py 第 153-173 行
    
    归一化公式: 
        - 对于 min != max 的维度: normalized = 2 * (x - min) / (max - min) - 1
        - 对于 min == max 的维度: normalized = 0
    输出范围: [-1, 1]
    
    反归一化公式: x = (normalized + 1) / 2 * (max - min) + min
    """
    
    def __init__(self, min_vals: torch.Tensor, max_vals: torch.Tensor):
        """
        初始化归一化器
        
        Args:
            min_vals: 最小值张量
            max_vals: 最大值张量
        """
        self.min_vals = min_vals
        self.max_vals = max_vals
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        归一化到 [-1, 1]，与原始 GR00T 完全对齐
        
        参考: Isaac-GR00T/gr00t/data/transform/state_action.py Normalizer.forward()
        
        Args:
            x: 输入张量，形状可以是 (batch, dim) 或 (batch, seq, dim)
        
        Returns:
            归一化后的张量
        """
        # 确保统计量在正确的设备上
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        
        # 创建 mask: 只对 min != max 的维度进行归一化
        # 与原始 GR00T 一致: mask = min != max
        mask = min_vals != max_vals
        
        # 初始化输出为 0
        normalized = torch.zeros_like(x)
        
        # 对 min != max 的维度进行归一化
        # 公式: 2 * (x - min) / (max - min) - 1
        if mask.any():
            normalized[..., mask] = (x[..., mask] - min_vals[mask]) / (max_vals[mask] - min_vals[mask])
            normalized[..., mask] = 2.0 * normalized[..., mask] - 1.0
        
        # 对 min == max 的维度，保持为 0（与原始 GR00T 一致）
        # normalized[..., ~mask] 已经是 0
        
        return normalized
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        反归一化，从 [-1, 1] 恢复到原始范围
        
        参考: Isaac-GR00T/gr00t/data/transform/state_action.py Normalizer.inverse()
        
        Args:
            x: 归一化后的张量
        
        Returns:
            反归一化后的张量
        """
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        
        # 公式: (x + 1) / 2 * (max - min) + min
        return (x + 1.0) / 2.0 * (max_vals - min_vals) + min_vals


def load_normalization_stats(dataset_path: str, convert_quat_to_axisangle: bool = True) -> dict:
    """
    从数据集加载归一化统计信息
    
    ⚠️ 重要: 如果 convert_quat_to_axisangle=True，会自动处理 9维→8维 的统计量转换
    
    用户原始数据 (9维):
        [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
            0         1      2  3  4   5   6   7   8
    
    转换后 (8维):
        [gripper1, gripper2, x, y, z, ax, ay, az]
            0         1      2  3  4   5   6   7
    
    归一化范围:
        - gripper1, gripper2: 使用 stats.json 的第 0, 1 维 min/max
        - x, y, z: 使用 stats.json 的第 2, 3, 4 维 min/max
        - ax, ay, az: 固定范围 [-π, π]（轴角的数学范围）
    
    Args:
        dataset_path: 数据集路径
        convert_quat_to_axisangle: 是否进行 9维→8维 转换 (默认 True)
    
    Returns:
        包含 state_normalizer 和 action_normalizer 的字典
    """
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"[WARN] stats.json 不存在于 {stats_path}，将不进行归一化")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    normalizers = {}
    
    # 加载 state 归一化统计量
    if 'observation.state' in stats:
        state_stats = stats['observation.state']
        original_min = state_stats['min']
        original_max = state_stats['max']
        original_dim = len(original_min)
        
        if convert_quat_to_axisangle and original_dim == 9:
            # 9维 → 8维 转换
            # 用户数据顺序: [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
            #                  0         1      2  3  4   5   6   7   8
            # 转换后顺序:   [gripper1, gripper2, x, y, z, ax, ay, az]
            #                  0         1      2  3  4   5   6   7
            #
            # Gripper (0:2): 使用原始第 0, 1 维 min/max
            # 位置 (2:5): 使用原始第 2, 3, 4 维 min/max
            # 轴角 (5:8): 固定范围 [-π, π]
            #
            import math
            state_min = torch.tensor([
                original_min[0], original_min[1],                   # gripper1, gripper2
                original_min[2], original_min[3], original_min[4],  # x, y, z
                -math.pi, -math.pi, -math.pi,                       # ax, ay, az (固定范围)
            ], dtype=torch.float32)
            state_max = torch.tensor([
                original_max[0], original_max[1],                   # gripper1, gripper2
                original_max[2], original_max[3], original_max[4],  # x, y, z
                math.pi, math.pi, math.pi,                          # ax, ay, az (固定范围)
            ], dtype=torch.float32)
            print(f"✓ State 归一化统计量: {original_dim}维 → 8维 (四元数→轴角转换)")
            print(f"  - Gripper1 min/max: [{state_min[0].item():.4f}, {state_max[0].item():.4f}]")
            print(f"  - Gripper2 min/max: [{state_min[1].item():.4f}, {state_max[1].item():.4f}]")
            print(f"  - 位置 min: {state_min[2:5].tolist()}")
            print(f"  - 位置 max: {state_max[2:5].tolist()}")
            print(f"  - 轴角范围: [-π, π]")
        else:
            # 直接使用原始统计量
            state_min = torch.tensor(original_min, dtype=torch.float32)
            state_max = torch.tensor(original_max, dtype=torch.float32)
            print(f"✓ 加载 state 归一化统计量，维度: {len(state_min)}")
        
        normalizers['state'] = MinMaxNormalizer(state_min, state_max)
    
    # 加载 action 归一化统计量
    if 'action' in stats:
        action_stats = stats['action']
        original_min = action_stats['min']
        original_max = action_stats['max']
        original_dim = len(original_min)
        
        # 直接使用原始统计量 (动作已经是7维)
        action_min = torch.tensor(original_min, dtype=torch.float32)
        action_max = torch.tensor(original_max, dtype=torch.float32)
        print(f"✓ 加载 action 归一化统计量，维度: {len(action_min)}")
        
        normalizers['action'] = MinMaxNormalizer(action_min, action_max)
    
    return normalizers if normalizers else None


# ============================================================================
# 数据处理函数 Data Processing Functions
# ============================================================================


def lerobot_collate_fn(batch, max_state_dim=64, max_action_dim=32, normalizers=None, 
                        convert_quat_to_axisangle=True):
    """
    将 LeRobot batch 转换为 FlowMatching 模型需要的格式
    
    ⚠️ 与原始 GR00T N1.5 训练 LIBERO 的方式完全对齐
    
    参考:
    - Isaac-GR00T/gr00t/model/transforms.py: _prepare_state(), _prepare_action()
    - Isaac-GR00T/gr00t/data/transform/state_action.py: Normalizer.forward()
    - Isaac-GR00T/examples/Libero/custom_data_config.py: LiberoDataConfig
    
    用户数据格式:
    - state: 9维 (gripper1, gripper2, x, y, z, qx, qy, qz, qw) - 需要四元数转轴角
    - action: 7维 (delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper) - 已经是7维
    
    转换后数据格式:
    - state: 8维 (gripper1, gripper2, x, y, z, ax, ay, az)
    - action: 7维 (不变)
    
    处理流程:
    1. State: 四元数转轴角 (9维→8维)
    2. Action: 不需要转换 (已经是7维)
    3. 归一化: min_max 到 [-1, 1]
    4. Padding: 用 0 填充到 max_dim
    5. Mask: 真实维度为 True/1.0，padding 为 False/0.0
    
    Args:
        batch: 输入 batch 数据
        max_state_dim: 最大状态维度 (默认 64，与 GR00T 一致)
        max_action_dim: 最大动作维度 (默认 32，与 GR00T 一致)
        normalizers: 归一化器字典，包含 'state' 和 'action' 归一化器
                     注意: state 应该提供转换后 8 维的归一化器，action 提供 7 维的归一化器
        convert_quat_to_axisangle: 是否将 state 的四元数转换为轴角 (默认 True)
    """
    # 从 lerobot dataloader 获取的 batch 已经是字典格式
    # VLM hidden states 从 collate_fn 返回: (batch, num_layers, seq_len, hidden_dim)
    # 但我们只需要单层，所以要去掉 num_layers 维度
    vlm_tensor_raw = batch['vlm_hidden_states']  # (batch, num_layers, seq_len, hidden_dim)
    observation_state = batch['observation_state']  # (batch, state_dim)
    actions = batch['actions']  # (batch, num_chunks, action_dim)
    
    # 处理 VLM hidden states: 去掉 num_layers 维度
    # (batch, num_layers, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim)
    if vlm_tensor_raw.dim() == 4:
        # 取第一层（或唯一一层）
        vlm_tensor = vlm_tensor_raw[:, 0, :, :]  # (batch, seq_len, hidden_dim)
    else:
        vlm_tensor = vlm_tensor_raw  # 已经是 3D
    
    batch_size = vlm_tensor.size(0)
    seq_len = vlm_tensor.size(1)
    hidden_dim = vlm_tensor.size(2)
    num_chunks = actions.size(1)
    
    # ========== 0. State 四元数转轴角 ==========
    # 参考: Isaac-GR00T/examples/Libero/eval/utils.py quat2axisangle()
    #
    # State: [gripper1, gripper2, x, y, z, qx, qy, qz, qw] (9维) → [gripper1, gripper2, x, y, z, ax, ay, az] (8维)
    # Action: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper] (7维) - 不需要转换
    #
    if convert_quat_to_axisangle:
        original_state_dim = observation_state.size(1)
        
        if original_state_dim == 9:
            observation_state = convert_state_quat_to_axisangle(observation_state)
        
        # Action 已经是 7 维，不需要转换
    
    # 更新实际维度
    actual_state_dim = observation_state.size(1)
    actual_action_dim = actions.size(2)
    
    # ========== 1. 归一化 state 和 action (与原始 GR00T 对齐) ==========
    # 参考: Isaac-GR00T/gr00t/data/transform/state_action.py
    #
    # ⚠️ 注意: 如果使用了四元数转轴角，normalizers 应该是 7 维的
    #          或者不使用归一化器，使用固定范围归一化
    #
    if normalizers is not None:
        # 归一化 state (只归一化实际维度)
        if 'state' in normalizers:
            observation_state = normalizers['state'].normalize(observation_state)
        
        # 归一化 action (只归一化实际维度)
        if 'action' in normalizers:
            # actions 形状: (batch, num_chunks, action_dim)
            # 需要对每个时间步的 action 进行归一化
            actions_flat = actions.reshape(batch_size * num_chunks, actual_action_dim)
            actions_normalized = normalizers['action'].normalize(actions_flat)
            actions = actions_normalized.reshape(batch_size, num_chunks, actual_action_dim)
    
    # ========== 2. VLM Hidden States 处理 ==========
    # 2D hidden states: (batch, seq_len, hidden_dim)
    # 直接使用，不需要 reshape
    backbone_features = vlm_tensor  # (batch, seq_len, hidden_dim)
    backbone_attention_mask = torch.ones(
        batch_size, seq_len, dtype=torch.long, device=vlm_tensor.device
    )
    
    # ========== 3. State 处理 (与 GR00T _prepare_state 对齐) ==========
    # 参考: Isaac-GR00T/gr00t/model/transforms.py 第 240-270 行
    #
    # 原始代码:
    #   if n_state_dims > self.max_state_dim:
    #       state = state[:, :self.max_state_dim]
    #   else:
    #       state = np.pad(state, ((0, 0), (0, self.max_state_dim - n_state_dims)), "constant")
    #   state_mask = np.zeros_like(state).astype(bool)
    #   state_mask[:, :n_state_dims] = True
    
    n_state_dims = actual_state_dim
    
    if n_state_dims > max_state_dim:
        # 截断到 max_state_dim
        observation_state = observation_state[:, :max_state_dim]
        n_state_dims = max_state_dim
    else:
        # Padding 到 max_state_dim (用 0 填充)
        padding = torch.zeros(batch_size, max_state_dim - n_state_dims,
                             dtype=observation_state.dtype, device=observation_state.device)
        observation_state = torch.cat([observation_state, padding], dim=1)
    
    # 添加时间维度 T=1 (state_horizon=1)
    state = observation_state.unsqueeze(1)  # (batch, 1, max_state_dim)
    
    # 创建 state_mask (与 GR00T 一致: 真实维度为 True)
    state_mask = torch.zeros(batch_size, 1, max_state_dim, dtype=torch.bool, device=state.device)
    state_mask[:, :, :n_state_dims] = True
    
    # ========== 4. Action 处理 (与 GR00T _prepare_action 对齐) ==========
    # 参考: Isaac-GR00T/gr00t/model/transforms.py 第 272-299 行
    #
    # 原始代码:
    #   actions = np.pad(actions, ((0, 0), (0, self.max_action_dim - n_action_dims)), "constant")
    #   actions_mask = np.zeros((n_action_tokens, self.max_action_dim), dtype=bool)
    #   actions_mask[:, :n_action_dims] = True
    
    n_action_dims = actual_action_dim
    
    if n_action_dims > max_action_dim:
        # 截断到 max_action_dim
        actions = actions[:, :, :max_action_dim]
        n_action_dims = max_action_dim
    else:
        # Padding 到 max_action_dim (用 0 填充)
        padding = torch.zeros(batch_size, num_chunks, max_action_dim - n_action_dims,
                             dtype=actions.dtype, device=actions.device)
        actions = torch.cat([actions, padding], dim=2)
    
    action = actions  # (batch, num_chunks, max_action_dim)
    
    # 创建 action_mask (与 GR00T 一致: 真实维度为 1.0，padding 维度为 0.0)
    # 注意: GR00T 原始使用 bool 类型，但 FlowMatching 需要 float
    action_mask = torch.zeros_like(action)
    action_mask[:, :, :n_action_dims] = 1.0
    
    # ========== 5. Embodiment ID ==========
    embodiment_id = torch.full(
        (batch_size,), DEFAULT_EMBODIMENT_ID, dtype=torch.long, device=vlm_tensor.device
    )
    
    # ========== 6. 构建 BatchFeature ==========
    backbone_output = BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_attention_mask,
        }
    )
    
    action_head_inputs = BatchFeature(
        data={
            "state": state,
            "action": action,
            "action_mask": action_mask,
            "embodiment_id": embodiment_id,
        }
    )
    
    return backbone_output, action_head_inputs


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


def main():
    parser = argparse.ArgumentParser(
        description='Train Flow Matching Action Head with Pretrained Weights',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 Examples:
  # 基本用法 Basic usage:
  python train_with_pretrained.py --data_path /path/to/dataset
  
  # 完整参数 Full arguments:
  python train_with_pretrained.py \\
    --data_path /path/to/dataset \\
    --batch_size 8 \\
    --epochs 10 \\
    --steps 1000 \\
    --lr 1e-4
        """
    )
    
    # 使用 TrainingConfig 类作为默认值
    config = TrainingConfig()
    
    # 数据相关参数
    parser.add_argument("--data_path", type=str, required=True, 
                        help=f"LeRobot 数据集路径 (必需) | LeRobot dataset path (Required)")
    parser.add_argument("--batch_size", type=int, default=config.batch_size,
                        help=f"训练批次大小 | Training batch size (default: {config.batch_size})")
    parser.add_argument("--num_workers", type=int, default=config.num_workers,
                        help=f"数据加载线程数 | Number of data loading workers (default: {config.num_workers})")
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="每个 worker 预取的 batch 数量 (仅在 num_workers>0 时生效) | Prefetch factor per worker (default: 4)")
    parser.add_argument("--val_split", type=float, default=config.val_split,
                        help=f"验证集比例 (0.0-1.0) | Validation split ratio (default: {config.val_split})")
    
    # 训练超参数
    parser.add_argument("--epochs", type=int, default=config.epochs,
                        help=f"训练轮数 | Number of epochs (default: {config.epochs})")
    parser.add_argument("--steps", type=int, default=config.steps,
                        help=f"最大训练步数 | Maximum training steps (default: {config.steps})")
    parser.add_argument("--lr", type=float, default=config.lr,
                        help=f"学习率 | Learning rate (default: {config.lr})")
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay,
                        help=f"权重衰减 | Weight decay (default: {config.weight_decay})")
    parser.add_argument("--warmup_ratio", type=float, default=config.warmup_ratio,
                        help=f"预热比例 (原始N1.5使用0.05) | Warmup ratio (default: {config.warmup_ratio})")
    parser.add_argument("--adam_beta1", type=float, default=config.adam_beta1,
                        help=f"AdamW beta1 (原始N1.5使用0.95) | AdamW beta1 (default: {config.adam_beta1})")
    parser.add_argument("--adam_beta2", type=float, default=config.adam_beta2,
                        help=f"AdamW beta2 | AdamW beta2 (default: {config.adam_beta2})")
    
    # 模型维度参数
    parser.add_argument("--max_action_dim", type=int, default=config.max_action_dim,
                        help=f"最大动作维度 (必须与预训练模型一致) | Max action dim (default: {config.max_action_dim})")
    parser.add_argument("--max_state_dim", type=int, default=config.max_state_dim,
                        help=f"最大状态维度 (必须与预训练模型一致) | Max state dim (default: {config.max_state_dim})")
    parser.add_argument("--num_action_chunks", type=int, default=config.num_action_chunks,
                        help=f"动作预测时间步数 | Action chunks (default: {config.num_action_chunks})")
    
    # 权重与保存
    parser.add_argument("--no_pretrain", action="store_true", default=not config.use_pretrain,
                        help="禁用预训练权重加载，从头开始训练 | Disable pretrained weights loading, train from scratch")
    parser.add_argument("--pretrained_weights", type=str, default=config.pretrained_weights,
                        help=f"预训练权重文件路径 | Pretrained weights path (default: {config.pretrained_weights})")
    parser.add_argument("--out_dir", type=str, default=config.out_dir,
                        help=f"Checkpoint 输出目录 | Checkpoint output directory")
    parser.add_argument("--log_dir", type=str, default=config.log_dir,
                        help=f"日志输出目录 | Logs output directory")
    parser.add_argument("--save_every", type=int, default=config.save_every,
                        help=f"每 N 个 epoch 保存 checkpoint | Save checkpoint every N epochs (default: {config.save_every})")
    parser.add_argument("--save_every_steps", type=int, default=config.save_every_steps,
                        help=f"每 N 个 step 保存 checkpoint (0=禁用) | Save checkpoint every N steps, 0 to disable (default: {config.save_every_steps})")
    parser.add_argument("--no_save_best", action="store_true", default=False,
                        help="禁用保存最佳模型 | Disable saving best model")
    
    # 系统配置
    parser.add_argument("--device", type=str, default=config.device,
                        help=f"训练设备 (单GPU模式) | Training device for single-GPU mode (default: {config.device})")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps,
                        help=f"梯度累积步数 | Gradient accumulation steps (default: {config.gradient_accumulation_steps})")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的本地排名 (由 torchrun 自动设置) | Local rank for distributed training (set by torchrun)")
    
    # 内存优化参数
    parser.add_argument("--use_amp", action="store_true", default=config.use_amp,
                        help="启用混合精度训练 (AMP) | Enable Automatic Mixed Precision training")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="禁用混合精度训练 | Disable AMP")
    parser.add_argument("--amp_dtype", type=str, default=config.amp_dtype, choices=["float16", "bfloat16"],
                        help=f"AMP 数据类型 (float16/bfloat16) | AMP dtype (default: {config.amp_dtype})")
    parser.add_argument("--empty_cache_freq", type=int, default=config.empty_cache_freq,
                        help=f"清理 CUDA 缓存频率 (0=禁用) | CUDA cache clearing frequency (default: {config.empty_cache_freq})")
    
    # 数据加载优化参数
    parser.add_argument("--vlm_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                        help="加载 VLM hidden states 时的数据类型 (bfloat16 可减少约 50%% 显存) | VLM hidden states dtype (default: float32)")
    parser.add_argument("--skip_images", action="store_true", default=False,
                        help="跳过加载图像 (仅使用预保存的 VLM hidden states 训练时使用) | Skip loading images to reduce I/O")
    parser.add_argument("--cache_vlm_states", action="store_true", default=False,
                        help="缓存 VLM hidden states 到 RAM (首个 epoch 后加速) | Cache VLM states to RAM")
    parser.add_argument("--cache_max_samples", type=int, default=-1,
                        help="最大缓存样本数 (-1 表示缓存所有样本) | Max cached samples (-1 for unlimited)")
    parser.add_argument("--use_shared_cache", action="store_true", default=False,
                        help="使用共享内存缓存 (多 GPU 共享, 加载所有轨迹) | Use shared memory cache for multi-GPU")
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
    
    # State 四元数转轴角配置
    parser.add_argument("--convert_quat_to_axisangle", action="store_true", default=True,
                        help="将 9 维 state (四元数) 转换为 8 维 state (轴角)")
    parser.add_argument("--no_convert_quat_to_axisangle", action="store_false", 
                        dest="convert_quat_to_axisangle",
                        help="禁用四元数转轴角转换")
    
    # Weights & Biases 配置
    parser.add_argument("--use_wandb", action="store_true", default=config.use_wandb,
                        help="启用 Weights & Biases 日志 | Enable W&B logging")
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb",
                        help="禁用 Weights & Biases 日志 | Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default=config.wandb_project,
                        help=f"W&B 项目名称 | W&B project name (default: {config.wandb_project})")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B 运行名称 (可选，默认自动生成) | W&B run name (optional, auto-generated if not specified)")
    parser.add_argument("--wandb_log_freq", type=int, default=config.wandb_log_freq,
                        help=f"Wandb 基本日志记录频率 (每 N 步) | Basic wandb logging frequency (default: {config.wandb_log_freq})")
    
    args = parser.parse_args()

    # ========== 初始化分布式训练环境 ==========
    rank, local_rank, world_size, is_distributed = setup_distributed()
    
    # 设置训练设备
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
    
    # 只在主进程创建目录
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        # 创建 custom 文件夹用于记录每个 step 的 loss
        custom_log_dir = log_dir / "custom"
        custom_log_dir.mkdir(parents=True, exist_ok=True)
    
    # 等待主进程创建目录
    if is_distributed:
        dist.barrier()
    
    # 所有进程都设置 custom_log_dir 路径（但只有主进程会写入）
    custom_log_dir = log_dir / "custom"
    
    # Generate run name with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.wandb_run_name is None:
        run_name = f'fm0_pretrained_lr{args.lr}_bs{args.batch_size}_gpu{world_size}_{timestamp}'
    else:
        run_name = args.wandb_run_name
    
    # Initialize wandb (只在主进程初始化，网络问题不影响训练)
    wandb_initialized = False
    if args.use_wandb and is_main_process(rank):
        try:
            # 设置 wandb 超时，避免长时间等待
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "60")
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=str(log_dir),
                tags=["flow_matching", "action_head", "pretrained", "eagle25", f"gpu{world_size}"],
                notes=f"Training Flow Matching Action Head on {Path(args.data_path).name} with {world_size} GPUs",
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
            print(f"⚠️ Wandb 初始化失败，将继续训练但不记录到 wandb: {e}")
            print("  提示: 可以使用 --no_wandb 禁用 wandb 日志")
            wandb_initialized = False

    # 维度设置
    # Qwen2B hidden dimension = 1536
    backbone_dim = 1536
    max_action_dim = args.max_action_dim
    action_horizon = args.num_action_chunks
    max_state_dim = args.max_state_dim

    # 加载数据集信息以获取总 episode 数
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
    print_rank0(f"  Action chunks: {action_horizon}", rank)
    
    # ========== 加载归一化统计量 (与原始 N1.5 对齐) ==========
    # 
    # State (9维): [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
    # State 转换后 (8维): [gripper1, gripper2, x, y, z, ax, ay, az]
    # Action (7维): [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper] - 不需要转换
    #
    convert_quat_to_axisangle = args.convert_quat_to_axisangle  # 是否对 state 进行四元数→轴角转换
    
    print_rank0("\n加载归一化统计量...", rank)
    print_rank0(f"  四元数→轴角转换: {'启用' if convert_quat_to_axisangle else '禁用'}", rank)
    
    normalizers = load_normalization_stats(args.data_path, convert_quat_to_axisangle=convert_quat_to_axisangle) if is_main_process(rank) else None
    
    # 分布式模式下，需要在所有进程中同步加载 normalizers
    if is_distributed:
        # 在非主进程中也加载 normalizers (因为每个进程都需要用于数据预处理)
        normalizers = load_normalization_stats(args.data_path, convert_quat_to_axisangle=convert_quat_to_axisangle)
    
    if normalizers:
        print_rank0("✓ 将对 state 和 action 进行 min_max 归一化到 [-1, 1]", rank)
    else:
        print_rank0("⚠️ 未加载归一化统计量，state 和 action 将不进行归一化", rank)
    
    # 记录数据集信息到 wandb config (只在主进程)
    if wandb_initialized and is_main_process(rank):
        try:
            wandb.config.update({
                "dataset/name": Path(args.data_path).name,
                "dataset/total_episodes": total_episodes,
                "dataset/actual_action_dim": actual_action_dim,
                "dataset/actual_state_dim": actual_state_dim,
                "dataset/action_horizon": action_horizon,
                "distributed/world_size": world_size,
                "distributed/gradient_accumulation_steps": args.gradient_accumulation_steps,
                "distributed/effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
                "memory/use_amp": args.use_amp,
                "memory/amp_dtype": args.amp_dtype if args.use_amp else "disabled",
                "memory/empty_cache_freq": args.empty_cache_freq,
            }, allow_val_change=True)
        except Exception:
            pass
    
    # 划分训练集和验证集
    val_episodes = max(1, int(total_episodes * args.val_split))
    train_episodes = total_episodes - val_episodes
    
    if train_episodes < 1:
        print_rank0(f"Warning: Not enough episodes for train/val split. Using all {total_episodes} episodes for training.", rank)
        train_episode_indices = list(range(total_episodes))
        val_episode_indices = None
    else:
        train_episode_indices = list(range(train_episodes))
        val_episode_indices = list(range(train_episodes, total_episodes))
        print_rank0(f"Train episodes: {len(train_episode_indices)}, Val episodes: {len(val_episode_indices)}", rank)
    
    # 创建训练数据加载器
    # 注意：分布式训练时，需要使用 DistributedSampler
    # create_lerobot_dataloader 返回的是 DataLoader，我们需要获取 dataset 来创建 sampler
    
    # 先获取 dataset（不创建 dataloader）
    from lerobot_dataset_loader import LeRobotDataset, collate_fn as lerobot_default_collate_fn
    
    # 打印数据加载优化配置
    print_rank0(f"\n✓ 数据加载优化配置:", rank)
    print_rank0(f"  - VLM hidden states 数据类型: {args.vlm_dtype}", rank)
    print_rank0(f"  - 跳过加载图像: {args.skip_images}", rank)
    print_rank0(f"  - 使用共享内存缓存: {args.use_shared_cache}", rank)
    if args.vlm_dtype in ["float16", "bfloat16"]:
        print_rank0(f"  - 预计 VLM 显存节省: ~50%", rank)
    if args.skip_images:
        print_rank0(f"  - 预计数据加载速度提升: ~2-3x", rank)
    
    # ========== 数据加载模式选择 ==========
    # 优先级: WebDataset 分批缓存 > WebDataset > 共享内存缓存 > mmap 模式 > 普通加载
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    shared_vlm_cache = None
    use_webdataset_mode = args.use_webdataset and args.webdataset_shard_pattern
    use_webdataset_cached_mode = args.use_webdataset_cached and args.webdataset_shard_pattern
    
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
        # 获取数据集中所有 VLM hidden states 的总数（即所有轨迹的帧数总和）
        # 这样训练集和验证集都可以使用同一个共享缓存
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
            dtype=cache_dtype,
            verbose=is_main_process(rank),
            auto_detect_shape=True,
            cache_dtype=cache_dtype_str,
        )
        
        # 注册清理函数
        _shared_vlm_cache_global = shared_vlm_cache
        _shared_vlm_cache_rank = rank
        atexit.register(cleanup_shared_memory_cache)
        
        # 注册信号处理器
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
    
    if use_webdataset_cached_mode:
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
            num_action_chunks=action_horizon,
            enable_chunking=True,
            episode_indices=train_episode_indices,
            cache_vlm_states=args.cache_vlm_states,
            cache_max_samples=args.cache_max_samples,
            verbose=is_main_process(rank),  # 只在主进程打印
            skip_images=args.skip_images,  # 跳过加载图像以减少 I/O
            shared_vlm_cache=shared_vlm_cache,  # 传入共享内存缓存
        )
        
        # 创建带 vlm_dtype 参数的 collate_fn
        lerobot_collate_fn_with_dtype = partial(lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype)
        
        # 创建 sampler (分布式模式下使用 DistributedSampler)
        if is_distributed:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            shuffle = False  # 使用 sampler 时不能同时使用 shuffle
        else:
            shuffle = True
        
        start_time_train_loader = time.time()
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            sampler=train_sampler,
            collate_fn=lerobot_collate_fn_with_dtype,  # 使用带 vlm_dtype 的 collate 函数
            pin_memory=True,
            drop_last=True,  # 分布式训练时建议 drop_last=True 以保持 batch 大小一致
            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,  # 预取队列深度
            persistent_workers=True if args.num_workers > 0 else False,  # 保持 workers 存活，避免每 epoch 重启
        )
        end_time_train_loader = time.time()
        print_rank0(f"✓ 训练数据加载器创建完成，共 {len(train_dataset)} 个样本，耗时 {end_time_train_loader - start_time_train_loader:.2f} 秒", rank)
    
    # 创建验证数据加载器（如果有）
    val_loader = None
    val_sampler = None
    if val_episode_indices is not None:
        # 验证集也使用共享缓存（因为已加载所有轨迹的 VLM hidden states）
        val_dataset = LeRobotDataset(
            dataset_path=args.data_path,
            num_action_chunks=action_horizon,
            enable_chunking=True,
            episode_indices=val_episode_indices,
            cache_vlm_states=args.cache_vlm_states,
            cache_max_samples=args.cache_max_samples,
            verbose=False,  # 验证集不需要打印详细信息
            skip_images=args.skip_images,  # 跳过加载图像以减少 I/O
            shared_vlm_cache=shared_vlm_cache,  # 验证集也使用共享缓存
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


    start_time_cfg = time.time()
    # 配置与模型（使用 max_action_dim 作为模型的 action_dim）
    cfg = get_flowmatching_action_head_config_original(
        action_backbone_dim=backbone_dim,
        action_dim=max_action_dim,  # 模型配置使用 max_action_dim
        action_horizon=action_horizon,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
    )
    end_time_cfg = time.time()
    print_rank0(f"✓ 配置创建完成，耗时 {end_time_cfg - start_time_cfg} 秒!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", rank)
    model = FlowmatchingActionHead(cfg).to(device)
    end_time_model = time.time()
    print_rank0(f"✓ 模型创建完成，耗时 {end_time_model - end_time_cfg} 秒!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", rank)
    
    # 加载预训练权重 (根据 --no_pretrain 参数决定是否加载)
    use_pretrain = not args.no_pretrain
    if use_pretrain and args.pretrained_weights:
        print_rank0(f"Loading pretrained weights from {args.pretrained_weights}", rank)
        pretrained_dict = torch.load(args.pretrained_weights, map_location=device)
        model_dict = model.state_dict()
        
        # 找出匹配和不匹配的权重
        matched_keys = []
        mismatched_keys = []
        
        for k, v in pretrained_dict.items():
            if k not in model_dict:
                mismatched_keys.append(f"{k} (not in model)")
            elif v.shape != model_dict[k].shape:
                mismatched_keys.append(f"{k} (shape mismatch: pretrained {v.shape} vs model {model_dict[k].shape})")
            else:
                matched_keys.append(k)
        
        # 只加载匹配的权重
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in matched_keys}
        
        print_rank0(f"\nLoaded {len(pretrained_dict_filtered)}/{len(model_dict)} weights from pretrained model", rank)
        
        if mismatched_keys:
            print_rank0(f"\nSkipped {len(mismatched_keys)} weights due to mismatch:", rank)
            for key in mismatched_keys:
                print_rank0(f"  - {key}", rank)
        
        model_dict.update(pretrained_dict_filtered)
        model.load_state_dict(model_dict)
        print_rank0("\nPretrained weights loaded successfully!", rank)
    elif not use_pretrain:
        print_rank0("\n" + "="*60, rank)
        print_rank0("⚠️  Training from scratch (--no_pretrain specified)", rank)
        print_rank0("="*60 + "\n", rank)
    
    # ========== 分布式训练：包装模型为 DDP ==========
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        print_rank0(f"✓ 模型已包装为 DistributedDataParallel", rank)
    
    if wandb_initialized and is_main_process(rank):
        try:
            # 对于 DDP 模型，需要 watch 原始模型
            model_to_watch = model.module if is_distributed else model
            wandb.watch(model_to_watch, log='all', log_freq=100)
            
            # 记录模型信息
            total_params = sum(p.numel() for p in model_to_watch.parameters())
            trainable_params = sum(p.numel() for p in model_to_watch.parameters() if p.requires_grad)
            wandb.config.update({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/backbone_dim": backbone_dim,
                "model/max_action_dim": max_action_dim,
                "model/max_state_dim": max_state_dim,
                "model/action_horizon": action_horizon,
            }, allow_val_change=True)
            
            # 记录 GPU 信息
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(device)
                gpu_memory_total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                wandb.config.update({
                    "system/gpu_name": gpu_name,
                    "system/gpu_memory_total_gb": round(gpu_memory_total, 2),
                    "system/cuda_version": torch.version.cuda,
                    "system/pytorch_version": torch.__version__,
                }, allow_val_change=True)
        except Exception:
            pass
    
    model.train()
    
    # ========== 优化器配置 (与原始 N1.5 对齐) ==========
    # 原始 N1.5 使用: adam_beta1=0.95, adam_beta2=0.999, adam_epsilon=1e-8
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),  # 原始 N1.5: (0.95, 0.999)
        eps=1e-8
    )
    print_rank0(f"\n✓ 优化器配置 (与原始 N1.5 对齐):", rank)
    print_rank0(f"  - 学习率: {args.lr}", rank)
    print_rank0(f"  - 权重衰减: {args.weight_decay}", rank)
    print_rank0(f"  - Adam beta1: {args.adam_beta1} (原始 N1.5: 0.95)", rank)
    print_rank0(f"  - Adam beta2: {args.adam_beta2}", rank)
    
    # ========== 学习率调度器 (与原始 N1.5 对齐) ==========
    # 原始 N1.5 使用: warmup_ratio=0.05, cosine decay
    total_steps = args.steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(current_step):
        """
        学习率调度函数：
        - 前 warmup_steps 步：线性预热
        - 之后：余弦衰减
        """
        if current_step < warmup_steps:
            # 线性预热
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 余弦衰减
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optim, lr_lambda)
    print_rank0(f"\n✓ 学习率调度器配置 (与原始 N1.5 对齐):", rank)
    print_rank0(f"  - 预热比例: {args.warmup_ratio}", rank)
    print_rank0(f"  - 预热步数: {warmup_steps}", rank)
    print_rank0(f"  - 总步数: {total_steps}", rank)
    print_rank0(f"  - 调度类型: warmup + cosine decay", rank)
    
    # ========== 混合精度训练 (AMP) 配置 ==========
    use_amp = args.use_amp and torch.cuda.is_available()
    if use_amp:
        amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
        scaler = GradScaler(enabled=True)
        print_rank0(f"\n✓ 混合精度训练 (AMP) 配置:", rank)
        print_rank0(f"  - 启用: True", rank)
        print_rank0(f"  - 数据类型: {args.amp_dtype}", rank)
        print_rank0(f"  - 预期显存节省: ~40-50%", rank)
    else:
        amp_dtype = torch.float32
        scaler = GradScaler(enabled=False)
        print_rank0(f"\n✓ 混合精度训练 (AMP): 禁用 (使用 FP32)", rank)
    
    # 内存优化参数
    empty_cache_freq = args.empty_cache_freq
    if empty_cache_freq > 0:
        print_rank0(f"✓ CUDA 缓存清理: 每 {empty_cache_freq} 步", rank)

    step = 0
    best_val_loss = float('inf')
    best_train_loss = float('inf')  # 用于没有验证集时的 best model 判断
    best_epoch = -1
    best_step = -1
    training_start_time = time.time()
    
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
    
    # 创建带有 normalizers 和 convert_quat_to_axisangle 参数的 collate_fn
    collate_fn_with_normalizers = partial(
        lerobot_collate_fn,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        normalizers=normalizers,
        convert_quat_to_axisangle=convert_quat_to_axisangle
    )
    
    # 梯度累积相关变量
    gradient_accumulation_steps = args.gradient_accumulation_steps
    
    # 初始化梯度为零
    optim.zero_grad(set_to_none=True)
    
    # 记录训练开始时间，用于计算已用时间和剩余时间
    training_start_time = time.time()
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        epoch_losses = []
        batch_times = []
        
        # 分布式训练时，每个 epoch 设置 sampler 的 epoch (确保不同 epoch 的 shuffle 不同)
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # WebDataset 分批缓存模式：每个 epoch 重新打乱分片顺序
        if use_webdataset_cached_mode and shard_batch_cache is not None:
            shard_batch_cache.set_epoch(epoch)
        
        # 创建进度条 (只在主进程显示)
        if is_main_process(rank):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs-1}", ncols=120)
        else:
            pbar = train_loader
        
        # 梯度累积计数器
        accumulated_loss = 0.0
        accumulation_count = 0
        
        # 数据加载时间监控
        prev_batch_end_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            # 记录batch开始时间（数据已从 DataLoader 获取完成）
            batch_start_time = time.time()
            
            # ========== 数据加载等待时间监控 ==========
            data_wait_time = batch_start_time - prev_batch_end_time
            if batch_idx > 0 and is_main_process(rank):
                if data_wait_time > 0.1:  # 等待超过 100ms
                    print(f"⚠️ GPU 在等待数据! data_time={data_wait_time:.2f}s (batch {batch_idx})")
                elif batch_idx % 50 == 0:  # 每 50 个 batch 打印一次正常状态
                    print(f"✅ 流水线正常 data_time={data_wait_time:.3f}s (batch {batch_idx})")
            
            # ========== 1. Collate function (CPU操作) ==========
            bb, ah = collate_fn_with_normalizers(batch)
            
            # ========== 2. 移动到设备 ==========
            bb = BatchFeature(data={k: v.to(device) for k, v in bb.items()})
            ah = BatchFeature(data={k: v.to(device) for k, v in ah.items()})
            
            # 梯度累积时，使用 no_sync() 上下文管理器避免不必要的梯度同步
            is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0
            
            if is_distributed and is_accumulating:
                # ========== 3a. 梯度累积期间：前向+反向 (无通信) ==========
                with model.no_sync():
                    with autocast(dtype=amp_dtype, enabled=use_amp):
                        out = model(bb, ah)
                        loss = out["loss"] / gradient_accumulation_steps
                    scaler.scale(loss).backward()
            else:
                # ========== 3b. 最后一步：前向+反向+通信 ==========
                with autocast(dtype=amp_dtype, enabled=use_amp):
                    out = model(bb, ah)
                    loss = out["loss"] / gradient_accumulation_steps
                scaler.scale(loss).backward()
            
            # 累积 loss 用于显示
            accumulated_loss += float(out["loss"])
            accumulation_count += 1
            
            # 每 gradient_accumulation_steps 步更新一次参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # ========== 4. Unscale 梯度 ==========
                scaler.unscale_(optim)
                
                # ========== 5. 计算梯度范数 ==========
                # 使用更高效的方式计算梯度范数（避免多次 CPU-GPU 同步）
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=float('inf')  # 不裁剪，只计算范数
                ).item()
                
                # 原始方式（效率较低，每个参数都会触发一次 CPU-GPU 同步）：
                # grad_norm = 0.0
                # for p in model.parameters():
                #     if p.grad is not None:
                #         grad_norm += p.grad.data.norm(2).item() ** 2
                # grad_norm = grad_norm ** 0.5
                
                # ========== 6. Optimizer step ==========
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                optim.zero_grad(set_to_none=True)
                
                # ========== 7. 定期清理 CUDA 缓存 ==========
                if empty_cache_freq > 0 and step % empty_cache_freq == 0:
                    torch.cuda.empty_cache()
                
                # 记录 batch 结束时间
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_times.append(batch_time)
                
                # 计算平均 loss
                avg_accumulated_loss = accumulated_loss / accumulation_count
                epoch_losses.append(avg_accumulated_loss)
                
                # 重置累积变量
                accumulated_loss = 0.0
                accumulation_count = 0
                
                # 先更新 step，确保所有记录使用一致的 step 值
                step += 1
                
                # 记录每个 step 的 loss 到 custom 日志文件（只在主进程）
                if step_loss_log_file is not None and is_main_process(rank):
                    current_lr = optim.param_groups[0]['lr']
                    step_loss_log_file.write(f"{step},{epoch},{avg_accumulated_loss:.6f},{current_lr:.8f},{grad_norm:.6f},{batch_time:.4f}\n")
                    # 每 100 步 flush 一次，确保数据写入磁盘
                    if step % 100 == 0:
                        step_loss_log_file.flush()
                
                # Log to wandb (只在主进程，网络问题不影响训练)
                if wandb_initialized and is_main_process(rank) and step % args.wandb_log_freq == 0:
                    log_dict = {
                        # 基本训练指标
                        'train/loss': avg_accumulated_loss,
                        'train/step': step,
                        'train/grad_norm': grad_norm,
                        'train/learning_rate': optim.param_groups[0]['lr'],
                        'train/samples_per_sec': args.batch_size * gradient_accumulation_steps * world_size / batch_time if batch_time > 0 else 0,
                        # 总时间
                        'time/batch_total': batch_time,
                    }
                    
                    try:
                        wandb.log(log_dict)
                    except Exception as e:
                        print(f"⚠️ Wandb log 失败: {type(e).__name__}: {e}")
                
                # 更新进度条 (只在主进程)
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
            
                # 按步数保存 checkpoint (只在主进程保存)
                if is_main_process(rank) and args.save_every_steps > 0 and step % args.save_every_steps == 0:
                    step_dir = out_dir / f"step_{step}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    # 保存原始模型 (DDP 时需要用 model.module)
                    model_to_save = model.module if is_distributed else model
                    torch.save(model_to_save.state_dict(), step_dir / "action_head.pt")
                    config_to_save = cfg.__dict__.copy()
                    config_to_save['max_action_dim'] = max_action_dim
                    config_to_save['max_state_dim'] = max_state_dim
                    config_to_save['saved_at_step'] = step
                    config_to_save['saved_at_epoch'] = epoch
                    with open(step_dir / "config.json", "w") as f:
                        json.dump(config_to_save, f, indent=2, default=str)
                    print(f"\n✓ Step checkpoint saved to {step_dir}/")
            
            # 更新上一个 batch 结束时间（用于下一个 batch 的数据等待时间计算）
            prev_batch_end_time = time.time()
            
            if step >= args.steps:
                break
        
        if is_main_process(rank):
            pbar.close()
        
        # 验证阶段
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            # 分布式训练时，每个 epoch 设置 sampler 的 epoch
            if is_distributed and val_sampler is not None:
                val_sampler.set_epoch(epoch)
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation", ncols=120) if is_main_process(rank) else val_loader
                for batch in val_pbar:
                    bb, ah = collate_fn_with_normalizers(batch)
                    bb = BatchFeature(data={k: v.to(device) for k, v in bb.items()})
                    ah = BatchFeature(data={k: v.to(device) for k, v in ah.items()})
                    # 验证时也使用混合精度
                    with autocast(dtype=amp_dtype, enabled=use_amp):
                        out = model(bb, ah)
                    val_losses.append(float(out["loss"]))
            
            val_loss = np.mean(val_losses)
            train_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_batch_time = np.mean(batch_times) if batch_times else 0.0
            print_rank0(f"\nEpoch {epoch} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Avg Batch Time: {avg_batch_time:.3f}s", rank)
            
            # Log to wandb (只在主进程，网络问题不影响训练)
            if wandb_initialized and is_main_process(rank):
                safe_wandb_log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_loss_std': np.std(epoch_losses) if epoch_losses else 0.0,
                    'val/epoch_loss': val_loss,
                    'val/epoch_loss_std': np.std(val_losses),
                    'train/avg_batch_time': avg_batch_time,
                    'train/total_samples': step * args.batch_size * gradient_accumulation_steps * world_size,
                    'train/epoch_samples_per_sec': len(epoch_losses) * args.batch_size * gradient_accumulation_steps * world_size / sum(batch_times) if sum(batch_times) > 0 else 0,
                    'progress/epoch': epoch,
                    'progress/step': step,
                    'progress/epoch_progress_pct': (epoch + 1) / args.epochs * 100,
                }, rank=rank)
            
            # 保存最佳模型 (只在主进程)
            if is_main_process(rank) and val_loss < best_val_loss and not args.no_save_best:
                best_val_loss = val_loss
                best_epoch = epoch
                best_step = step
                best_dir = out_dir.parent / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), best_dir / "action_head.pt")
                config_to_save = cfg.__dict__.copy()
                config_to_save['max_action_dim'] = max_action_dim
                config_to_save['max_state_dim'] = max_state_dim
                config_to_save['best_val_loss'] = best_val_loss
                config_to_save['best_epoch'] = best_epoch
                config_to_save['best_step'] = best_step
                with open(best_dir / "config.json", "w") as f:
                    json.dump(config_to_save, f, indent=2, default=str)
                print(f"✓ Saved best model to {best_dir}/ with val_loss={val_loss:.6f}")
                
                # 记录最佳模型信息到 wandb (网络问题不影响训练)
                if wandb_initialized:
                    try:
                        safe_wandb_log({
                            'best/val_loss': best_val_loss,
                            'best/epoch': best_epoch,
                            'best/step': best_step,
                        }, rank=rank)
                        # 更新 summary（会显示在 wandb 表格中）
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
                    'train/epoch_loss_std': np.std(epoch_losses) if epoch_losses else 0.0,
                    'train/avg_batch_time': avg_batch_time,
                    'train/total_samples': step * args.batch_size * gradient_accumulation_steps * world_size,
                    'train/epoch_samples_per_sec': len(epoch_losses) * args.batch_size * gradient_accumulation_steps * world_size / sum(batch_times) if sum(batch_times) > 0 else 0,
                    'progress/epoch': epoch,
                    'progress/step': step,
                    'progress/epoch_progress_pct': (epoch + 1) / args.epochs * 100,
                }, rank=rank)
            
            # 没有验证集时，使用训练损失保存最佳模型 (只在主进程)
            if is_main_process(rank) and train_loss < best_train_loss and not args.no_save_best:
                best_train_loss = train_loss
                best_epoch = epoch
                best_step = step
                best_dir = out_dir.parent / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model_to_save = model.module if is_distributed else model
                torch.save(model_to_save.state_dict(), best_dir / "action_head.pt")
                config_to_save = cfg.__dict__.copy()
                config_to_save['max_action_dim'] = max_action_dim
                config_to_save['max_state_dim'] = max_state_dim
                config_to_save['best_train_loss'] = best_train_loss
                config_to_save['best_epoch'] = best_epoch
                config_to_save['best_step'] = best_step
                with open(best_dir / "config.json", "w") as f:
                    json.dump(config_to_save, f, indent=2, default=str)
                print(f"✓ Saved best model to {best_dir}/ with train_loss={train_loss:.6f}")
                
                # 记录最佳模型信息到 wandb
                if wandb_initialized:
                    try:
                        safe_wandb_log({
                            'best/train_loss': best_train_loss,
                            'best/epoch': best_epoch,
                            'best/step': best_step,
                        }, rank=rank)
                        wandb.run.summary['best_train_loss'] = best_train_loss
                        wandb.run.summary['best_epoch'] = best_epoch
                        wandb.run.summary['best_step'] = best_step
                    except Exception:
                        pass
        
        # Save checkpoint every N epochs (只在主进程)
        if is_main_process(rank) and epoch % args.save_every == 0:
            epoch_dir = out_dir / f"epoch_{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = model.module if is_distributed else model
            torch.save(model_to_save.state_dict(), epoch_dir / "action_head.pt")
            config_to_save = cfg.__dict__.copy()
            config_to_save['max_action_dim'] = max_action_dim
            config_to_save['max_state_dim'] = max_state_dim
            with open(epoch_dir / "config.json", "w") as f:
                json.dump(config_to_save, f, indent=2, default=str)
            print(f"✓ Checkpoint saved to {epoch_dir}/")
        
        if step >= args.steps:
            break
    
    # 保存最终 epoch 的权重与配置（如果还没有保存，只在主进程）
    if is_main_process(rank):
        final_epoch_dir = out_dir / f"epoch_{epoch}"
        if not final_epoch_dir.exists():
            final_epoch_dir.mkdir(parents=True, exist_ok=True)
            model_to_save = model.module if is_distributed else model
            torch.save(model_to_save.state_dict(), final_epoch_dir / "action_head.pt")
            config_to_save = cfg.__dict__.copy()
            config_to_save['max_action_dim'] = max_action_dim
            config_to_save['max_state_dim'] = max_state_dim
            with open(final_epoch_dir / "config.json", "w") as f:
                json.dump(config_to_save, f, indent=2, default=str)
            print(f"✓ Final checkpoint saved to {final_epoch_dir}/")
        print(f"\n✓ Training completed! All checkpoints saved to {out_dir}/")
    
    # 关闭 step loss 日志文件
    if step_loss_log_file is not None and is_main_process(rank):
        step_loss_log_file.flush()
        step_loss_log_file.close()
        print(f"✓ Step loss 日志已保存到 {custom_log_dir}/")
    
    if wandb_initialized and is_main_process(rank):
        try:
            # 记录训练总结信息
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
            print(f"⚠️ Wandb 结束时出错（不影响训练结果）: {e}")
    
    # 清理分布式训练环境
    cleanup_distributed()


if __name__ == "__main__":
    # ========================================================================
    # 运行方式说明 Execution Methods
    # ========================================================================
    # 
    # 方式 1 (推荐用于正式训练): 命令行运行
    # Method 1 (Recommended for production): Command-line execution
    # 
    #   python train_with_pretrained.py --data_path /path/to/dataset --batch_size 8 --epochs 10
    # 
    #   - 灵活: 可以随时改变参数而不修改代码
    #   - 规范: 适合在服务器上批量运行实验
    #   - 使用: 注释掉下面的 sys.argv 设置，直接调用 main()
    # 
    # 方式 2 (推荐用于快速测试): 直接运行脚本
    # Method 2 (Recommended for quick testing): Direct script execution
    # 
    #   python train_with_pretrained.py  (不带任何参数)
    # 
    #   - 方便: 在 IDE 中点击运行按钮即可
    #   - 快速: 不需要每次输入长长的命令行
    #   - 使用: 在下面的 sys.argv 中设置好参数
    # 
    # ⚠️ 注意: 如果同时设置了 sys.argv 和命令行参数，命令行参数优先级更高
    # ⚠️ Note: If both sys.argv and command-line args are provided, command-line takes precedence
    # 
    # ========================================================================
    
    # 方式 1: 使用命令行参数
    # main()
    
    # 方式 2: 直接设置参数运行（推荐用于快速测试）
    # import sys
    # from pathlib import Path
    
    # # 获取脚本所在目录（Flow_Matching_0 目录）
    # SCRIPT_DIR = Path(__file__).parent
    
    # ========== 参数配置 (与原始 N1.5 LIBERO 训练配置对齐) ==========
    # 
    # 原始 N1.5 LIBERO 训练参数:
    #   - batch_size: 32
    #   - max_steps: 10000
    #   - learning_rate: 1e-4
    #   - weight_decay: 1e-5
    #   - warmup_ratio: 0.05
    #   - adam_beta1: 0.95
    #   - adam_beta2: 0.999
    #   - lr_scheduler: cosine
    #   - action_horizon: 16
    #   - max_state_dim: 64
    #   - max_action_dim: 32
    #   - state/action 归一化: min_max → [-1, 1]
    #
    # sys.argv = [
    #     'train_with_pretrained.py',
    #     '--data_path', '/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_eagle25-only-libero_spatial/libero_lerobot_spatial',
    #     # ========== 与原始 N1.5 对齐的训练参数 ==========
    #     '--batch_size', '32',        # 原始 N1.5: 32
    #     '--epochs', '300',           # 原始 N1.5: 300
    #     '--steps', '10000',          # 原始 N1.5: 10000
    #     '--lr', '1e-4',              # 原始 N1.5: 1e-4
    #     '--weight_decay', '1e-5',    # 原始 N1.5: 1e-5
    #     '--warmup_ratio', '0.05',    # 原始 N1.5: 0.05
    #     '--adam_beta1', '0.95',      # 原始 N1.5: 0.95
    #     '--adam_beta2', '0.999',     # 原始 N1.5: 0.999
    #     '--val_split', '0',          # 原始 N1.5 不使用验证集
    #     '--num_workers', '4',        # 数据加载线程数
    #     '--device', 'cuda:7',
    #     '--num_action_chunks', '16', # 原始 N1.5 LIBERO: 16
    #     '--save_every', '1',
    #     '--max_action_dim', '32',    # ! 不要修改，必须与预训练模型一致
    #     '--max_state_dim', '64',     # ! 不要修改，必须与预训练模型一致
    #     '--out_dir', str(SCRIPT_DIR / 'experiments/libero_spatial_eagle25/checkpoints'),
    #     '--log_dir', str(SCRIPT_DIR / 'experiments/libero_spatial_eagle25/logs'),
    #     # 预训练权重
    #     '--pretrained_weights', str(SCRIPT_DIR / 'pretrained_action_head.pt'),
    #     # Wandb 配置
    #     '--use_wandb',
    #     '--wandb_project', 'eagle25_flowmatching_0_libero',
    # ]
    main()
