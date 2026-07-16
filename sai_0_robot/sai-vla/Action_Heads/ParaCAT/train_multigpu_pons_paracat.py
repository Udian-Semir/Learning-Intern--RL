"""
Pons + ParaCAT 联合训练脚本 - 支持多GPU分布式训练

端到端训练 Pons Adapter 和 ParaCAT Action Head:
  VLM Hidden States -> Pons -> ParaCAT -> Actions

多GPU训练使用方法:
  torchrun --nproc_per_node=4 train_multigpu_pons_paracat.py \
    --data_path /path/to/dataset \
    --steps 10000

单GPU训练使用方法:
  python train_multigpu_pons_paracat.py \
    --data_path /path/to/dataset \
    --device cuda:0

特点:
  - Pons 和 ParaCAT 联合训练
  - 支持 action 离散化 (ParaCAT 输出 3 类别) 或连续输出
  - Loss: L1Loss (连续) 或 CrossEntropyLoss (离散)
"""

import argparse
import json
import math
import os
import sys
import re
import atexit
import signal
from pathlib import Path
from typing import List, Optional, Dict
import time
from datetime import timedelta
from dataclasses import dataclass
from functools import partial
from contextlib import ExitStack

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# 添加路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

from Action_Heads.ParaCAT.model.action_head.paracat_action_head import (
    ParaCATActionHead, create_paracat_action_head
)
from Adapter.Pons.pons_adapter import PonsAdapter, create_pons_adapter
from utils.lerobot_dataset_loader import (
    LeRobotDataset,
    collate_fn as lerobot_default_collate_fn,
    SharedVLMCache,
    preload_vlm_cache_distributed,
    get_total_vlm_states,
)

# 可选 wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb 未安装")


# ============================================================================
# 训练配置类
# ============================================================================

@dataclass
class TrainingConfig:
    """Pons + ParaCAT 联合训练配置"""
    
    # 数据相关参数
    data_path: str = ""
    batch_size: int = 32
    num_workers: int = 4
    
    # 训练超参数
    steps: int = 10000
    lr: float = 1e-4
    pons_lr_scale: float = 1.0  # Pons 学习率缩放因子
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    
    # ParaCAT 参数
    chunk_size: int = 16
    action_dim: int = 7
    num_transformer_blocks: int = 2
    num_mlp_layers: int = 2
    mlp_expand_dim: int = 1024
    num_heads: int = 8
    
    # Pons 参数
    pons_q_seq_len: int = 64
    pons_num_blocks: int = 2
    pons_num_heads: int = 8
    pons_dropout: float = 0.1
    
    # VLM 参数
    num_vlm_layers: Optional[int] = None
    vlm_output_dim: Optional[int] = None
    
    # 离散化参数 (6个参数)
    discrete_actions: bool = False
    discrete_columns: Optional[List[int]] = None
    discrete_deltas: Optional[List[float]] = None
    undiscrete_actions: bool = False
    undiscrete_columns: Optional[List[int]] = None
    undiscrete_deltas: Optional[List[float]] = None
    
    # 保存参数
    out_dir: str = "./experiments/pons_paracat_training/checkpoints"
    log_dir: str = "./experiments/pons_paracat_training/logs"
    save_every_steps: int = 1000
    
    # 系统配置
    device: str = "cuda:0"
    gradient_accumulation_steps: int = 1
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "pons_paracat_training"


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


# ============================================================================
# 欧拉角转轴角函数 - Euler to Axis-Angle Conversion (for State Mapper)
# ============================================================================

def euler_to_quat_torch(euler: torch.Tensor) -> torch.Tensor:
    """
    欧拉角转四元数 (PyTorch 批量版本)
    
    Args:
        euler: (batch, 3) 欧拉角 (roll, pitch, yaw)
    
    Returns:
        quat: (batch, 4) 四元数 (qx, qy, qz, qw)
    """
    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]
    
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return torch.stack([qx, qy, qz, qw], dim=1)


def euler_to_axisangle_torch(euler: torch.Tensor) -> torch.Tensor:
    """
    欧拉角转轴角 (欧拉角 -> 四元数 -> 轴角)
    
    Args:
        euler: (batch, 3) 欧拉角 (roll, pitch, yaw)
    
    Returns:
        axis_angle: (batch, 3) 轴角 (ax, ay, az)，范围 [-pi, pi]
    """
    quat = euler_to_quat_torch(euler)
    return quat2axisangle_torch(quat)


# ============================================================================
# Observation State 映射器
# ============================================================================

class ObservationStateMapper(nn.Module):
    """
    将 observation_state 按列归一化并映射到 VLM 隐藏空间
    
    支持两种归一化方式:
    1. minmax: 从 stats.json 读取 min/max 进行 [0, 1] 归一化
    2. axisangle: 对已转换的轴角值 (范围 [-pi, pi]) 进行 [0, 1] 归一化
       注意: 欧拉角->轴角的转换应在数据加载时通过 state_axisangle_columns 完成
    
    数据流:
        state (batch, state_dim) 
        -> 按列归一化
        -> MLP (state_dim -> hidden_dim)
        -> (batch, 1, hidden_dim)
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        minmax_columns: List[int] = None,      # 使用 minmax 归一化的列索引
        axisangle_columns: List[int] = None,   # 使用 [-pi,pi]->[0,1] 归一化的列索引 (已转换为轴角)
        state_min: torch.Tensor = None,        # minmax 归一化的 min 值
        state_max: torch.Tensor = None,        # minmax 归一化的 max 值
    ):
        super().__init__()
        self.state_dim = state_dim
        self.minmax_columns = minmax_columns or []
        self.axisangle_columns = axisangle_columns or []
        
        # 注册 min/max 为 buffer
        if state_min is not None:
            self.register_buffer('state_min', state_min)
        else:
            self.register_buffer('state_min', torch.zeros(state_dim))
            
        if state_max is not None:
            self.register_buffer('state_max', state_max)
        else:
            self.register_buffer('state_max', torch.ones(state_dim))
        
        # MLP: state_dim -> hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim)
        Returns:
            (batch, 1, hidden_dim)
        """
        batch_size = state.shape[0]
        normalized = state.clone()
        
        # MinMax 归一化
        for col in self.minmax_columns:
            col_min = self.state_min[col]
            col_max = self.state_max[col]
            col_range = col_max - col_min + 1e-8
            normalized[:, col] = (state[:, col] - col_min) / col_range
        
        # AxisAngle 归一化 (输入已经是轴角值，范围 [-pi, pi])
        # 数据加载时已通过 state_axisangle_columns 完成欧拉角->轴角转换
        # 这里只做 [-pi, pi] -> [0, 1] 的归一化
        for col in self.axisangle_columns:
            # 轴角范围是 [-pi, pi]，归一化到 [0, 1]
            normalized[:, col] = (state[:, col] + math.pi) / (2 * math.pi)
        
        # MLP 映射
        state_embedding = self.mlp(normalized)
        
        # 变成 (batch, 1, hidden_dim)
        return state_embedding.unsqueeze(1)
    
    def print_model_summary(self, print_details: bool = False):
        """打印模型参数量统计"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n🔄 ObservationStateMapper 参数量:")
        print(f"   • State Dim: {self.state_dim}")
        print(f"   • MinMax Columns: {self.minmax_columns}")
        print(f"   • AxisAngle Columns: {self.axisangle_columns}")
        print(f"   • Total Params: {total_params:,}")
        print(f"   • Trainable Params: {trainable_params:,}")


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
        
        timeout_seconds = int(os.environ.get('TORCH_DISTRIBUTED_TIMEOUT_SEC', 7200))
        timeout = timedelta(seconds=timeout_seconds)
        
        try:
            dist.init_process_group(backend='nccl', init_method='env://', timeout=timeout)
        except Exception:
            dist.init_process_group(backend='gloo', init_method='env://', timeout=timeout)
        
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_rank0(msg: str, rank: int = 0):
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
# 共享内存缓存清理
# ============================================================================

_shared_vlm_cache_global = None
_shared_vlm_cache_rank = None

def cleanup_shared_memory_cache():
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    if _shared_vlm_cache_global is not None:
        try:
            _shared_vlm_cache_global.close()
            if _shared_vlm_cache_rank == 0:
                _shared_vlm_cache_global.unlink()
        except Exception:
            pass


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


def load_normalization_stats(dataset_path: str, load_state_stats: bool = False) -> dict:
    """从数据集加载归一化统计信息
    
    Args:
        dataset_path: 数据集路径
        load_state_stats: 是否加载 observation.state 的统计信息
    
    Returns:
        包含归一化器和统计信息的字典
    """
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"[WARN] stats.json 不存在于 {stats_path}")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    normalizers = {}
    
    # Action 归一化器
    if 'action' in stats:
        action_stats = stats['action']
        action_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        action_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        normalizers['action'] = MinMaxNormalizer(action_min, action_max)
    
    # State 归一化统计 (用于 ObservationStateMapper)
    if load_state_stats and 'observation.state' in stats:
        state_stats = stats['observation.state']
        normalizers['state_min'] = torch.tensor(state_stats['min'], dtype=torch.float32)
        normalizers['state_max'] = torch.tensor(state_stats['max'], dtype=torch.float32)
        print(f"✓ 已加载 observation.state 统计信息: min shape={normalizers['state_min'].shape}")
    
    return normalizers if normalizers else None


# ============================================================================
# 数据处理函数
# ============================================================================

def pons_paracat_collate_fn(batch, normalizers=None, num_vlm_layers=1, discrete_actions=False, gripper_columns=None, convert_quat_to_axisangle=True):
    """
    将 LeRobot batch 转换为 Pons + ParaCAT 训练需要的格式
    
    Args:
        batch: LeRobotDataset 返回的 batch
        normalizers: 归一化器 (仅连续模式使用)
        num_vlm_layers: VLM 层数
        discrete_actions: 是否启用离散化模式
        gripper_columns: Gripper 列索引列表 (LIBERO 专用)
        convert_quat_to_axisangle: 是否将 9 维 state (四元数) 转换为 8 维 state (轴角)
    """
    vlm_tensor_raw = batch['vlm_hidden_states']  # (batch, num_layers, seq_len, hidden_dim)
    vlm_attention_mask_raw = batch.get('vlm_attention_mask', None)
    actions = batch['actions']  # (batch, num_chunks, action_dim)
    
    # State 四元数转轴角 (如果需要)
    observation_state = batch.get('observation_state', None)
    if observation_state is not None and convert_quat_to_axisangle:
        original_state_dim = observation_state.size(1)
        if original_state_dim == 9:
            observation_state = convert_state_quat_to_axisangle(observation_state)
    
    batch_size = vlm_tensor_raw.size(0)
    num_layers = vlm_tensor_raw.size(1)
    seq_len = vlm_tensor_raw.size(2)
    num_chunks = actions.size(1)
    action_dim = actions.size(2)
    
    # VLM hidden states: 拆分为列表
    vlm_hidden_states = [vlm_tensor_raw[:, i, :, :] for i in range(num_layers)]
    
    # VLM attention mask
    if vlm_attention_mask_raw is not None:
        vlm_attention_mask = vlm_attention_mask_raw.view(batch_size, num_layers * seq_len)
    else:
        vlm_attention_mask = None
    
    # 归一化 actions (仅连续模式)
    if normalizers is not None and 'action' in normalizers and not discrete_actions:
        actions_flat = actions.reshape(batch_size * num_chunks, action_dim)
        actions_normalized = normalizers['action'].normalize(actions_flat)
        actions = actions_normalized.reshape(batch_size, num_chunks, action_dim)
    
    # ========== 处理 ground truth actions ==========
    # 离散化模式下，将离散值 {-1, 0, 1} 转换为类别索引 {0, 1, 2}
    # 映射关系: -1 -> 0, 0 -> 1, 1 -> 2
    # 公式: class_idx = discrete_val + 1
    #
    # 处理两类列:
    # 1. discrete_columns: 位置/旋转列 (已经过 discrete_constrain_delta 离散化)
    # 2. gripper_columns: Gripper 列 (LIBERO 原始值已是 {-1, 0, 1}，只需 +1)
    if discrete_actions:
        discrete_columns = batch.get('discrete_columns', None)
        
        # 初始化所有列为 0 (如果某列没有被处理，loss 会出问题)
        gt_actions_discrete = torch.zeros_like(actions, dtype=torch.long)
        
        # 处理普通离散化列 (位置/旋转，已经过 discrete_constrain_delta)
        if discrete_columns is not None:
            for col_idx in discrete_columns:
                if col_idx < action_dim:
                    # actions[:, :, col_idx] 已经是 {-1, 0, 1}
                    # 直接 +1 转为类别索引 {0, 1, 2}
                    gt_actions_discrete[:, :, col_idx] = (actions[:, :, col_idx] + 1).long()
        
        # 处理 Gripper 列 (LIBERO: 原始值已是 {-1, 0, 1}，直接 +1)
        if gripper_columns is not None:
            for col_idx in gripper_columns:
                if col_idx < action_dim:
                    # gripper 原始值已是 {-1, 0, 1}，直接 +1 转为类别索引
                    gt_actions_discrete[:, :, col_idx] = (actions[:, :, col_idx] + 1).long()
        
        # 兼容旧模式：如果没有指定任何列，假设所有列都已离散化
        if discrete_columns is None and gripper_columns is None:
            gt_actions_discrete = (actions + 1).long()
        
        gt_actions = gt_actions_discrete
    else:
        gt_actions = actions
    
    return {
        'vlm_hidden_states': vlm_hidden_states,
        'vlm_attention_mask': vlm_attention_mask,
        'observation_state': observation_state,
        'gt_actions': gt_actions,
        'discrete_columns': batch.get('discrete_columns', None),
        'discrete_deltas': batch.get('discrete_deltas', None),
        'gripper_columns': gripper_columns,
    }


# ============================================================================
# Pons + ParaCAT 联合模型
# ============================================================================

class PonsParaCATModel(nn.Module):
    """
    Pons + ParaCAT 联合模型
    
    数据流:
        VLM Hidden States -> Pons -> ParaCAT -> Action Predictions
        
    可选的 State Mapper:
        observation_state -> 归一化 -> MLP -> Concat 到 Pons 输出
    """
    
    def __init__(
        self,
        pons: PonsAdapter,
        paracat: ParaCATActionHead,
        state_mapper: Optional[ObservationStateMapper] = None,
    ):
        super().__init__()
        self.pons = pons
        self.paracat = paracat
        self.state_mapper = state_mapper
    
    def forward(
        self,
        vlm_hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        observation_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            vlm_hidden_states: List of (batch, seq_len, hidden_dim) tensors
            attention_mask: Optional (batch, total_seq_len) mask
            observation_state: Optional (batch, state_dim) tensor for state mapping
        
        Returns:
            predictions: (batch, chunk_size, action_dim, 3)
        """
        # Pons: VLM -> 压缩特征
        pons_output = self.pons(vlm_hidden_states, attention_mask=attention_mask)
        # pons_output: (batch, pons_q_seq_len, hidden_dim)
        
        # 如果启用 state_mapper，拼接 state embedding
        if self.state_mapper is not None and observation_state is not None:
            state_embedding = self.state_mapper(observation_state)
            # state_embedding: (batch, 1, hidden_dim)
            pons_output = torch.cat([pons_output, state_embedding], dim=1)
            # pons_output: (batch, pons_q_seq_len + 1, hidden_dim)
        
        # ParaCAT: 压缩特征 -> Actions
        predictions = self.paracat(pons_output)
        
        return predictions


# ============================================================================
# 主训练函数
# ============================================================================

def main():
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    
    parser = argparse.ArgumentParser(description='Train Pons + ParaCAT End-to-End')
    config = TrainingConfig()
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_workers", type=int, default=config.num_workers)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    
    # 训练参数
    parser.add_argument("--steps", type=int, default=config.steps)
    parser.add_argument("--lr", type=float, default=config.lr)
    parser.add_argument("--pons_lr_scale", type=float, default=config.pons_lr_scale,
                        help="Pons 学习率缩放因子 (pons_lr = lr * pons_lr_scale)")
    parser.add_argument("--weight_decay", type=float, default=config.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=config.warmup_ratio)
    
    # ParaCAT 参数
    parser.add_argument("--chunk_size", type=int, default=config.chunk_size)
    parser.add_argument("--action_dim", type=int, default=config.action_dim)
    parser.add_argument("--num_transformer_blocks", type=int, default=config.num_transformer_blocks)
    parser.add_argument("--num_mlp_layers", type=int, default=config.num_mlp_layers)
    parser.add_argument("--mlp_expand_dim", type=int, default=config.mlp_expand_dim)
    parser.add_argument("--num_heads", type=int, default=config.num_heads)
    
    # Pons 参数
    parser.add_argument("--pons_q_seq_len", type=int, default=config.pons_q_seq_len)
    parser.add_argument("--pons_num_blocks", type=int, default=config.pons_num_blocks)
    parser.add_argument("--pons_num_heads", type=int, default=config.pons_num_heads)
    parser.add_argument("--pons_dropout", type=float, default=config.pons_dropout)
    
    # VLM 参数
    parser.add_argument("--num_vlm_layers", type=int, default=None)
    parser.add_argument("--vlm_output_dim", type=int, default=None)
    
    # 离散化参数 (6个参数)
    parser.add_argument("--discrete_actions", action="store_true", default=False,
                        help="启用 action 离散化")
    parser.add_argument("--discrete_columns", type=int, nargs="+", default=None,
                        help="参与离散化的列索引，例如: --discrete_columns 0 1 2")
    parser.add_argument("--discrete_deltas", type=float, nargs="+", default=None,
                        help="对应列的 delta 值，例如: --discrete_deltas 0.01 0.02 0.01")
    parser.add_argument("--discrete_method", type=str, default="constrain_delta",
                        choices=["constrain_delta", "chunk_calculus"],
                        help="离散化方法: constrain_delta (默认) 或 chunk_calculus")
    # State 预处理参数
    parser.add_argument("--state_process_order", type=str, nargs="+", default=None,
                        help="State 预处理执行顺序，如: hand_binary euler_to_axisangle")
    parser.add_argument("--hand_binary_columns", type=int, nargs="+", default=None,
                        help="原始 state 中手部数据列范围，每组2个数 [start, end)，如: 6 12 18 24")
    parser.add_argument("--hand_binary_threshold", type=float, default=442.0,
                        help="手部二值化阈值")
    parser.add_argument("--state_euler_to_axisangle_columns", type=int, nargs="+", default=None,
                        help="原始 state 中欧拉角列索引 (连续3列)")
    parser.add_argument("--undiscrete_actions", action="store_true", default=False,
                        help="启用反离散化配置 (用于推理)")
    parser.add_argument("--undiscrete_columns", type=int, nargs="+", default=None,
                        help="参与反离散化的列索引")
    parser.add_argument("--undiscrete_deltas", type=float, nargs="+", default=None,
                        help="对应列的反离散化 delta 值")
    
    # Gripper 列参数 (LIBERO 专用)
    # Gripper 原始值已是 {-1, 0, 1}，只需 +1 转为类别索引，无需 discrete_constrain_delta
    parser.add_argument("--gripper_columns", type=int, nargs="+", default=None,
                        help="Gripper 列索引 (空格分隔)，如: --gripper_columns 6 7")
    parser.add_argument("--gripper_method", type=str, default="libero_gripper_to_class_idx",
                        help="Gripper 处理方法函数名 (默认: libero_gripper_to_class_idx)")
    
    # 预训练权重
    parser.add_argument("--pons_checkpoint", type=str, default="",
                        help="可选: 预训练的 Pons checkpoint")
    parser.add_argument("--paracat_checkpoint", type=str, default="",
                        help="可选: 预训练的 ParaCAT checkpoint")
    
    # 保存参数
    parser.add_argument("--out_dir", type=str, default=config.out_dir)
    parser.add_argument("--log_dir", type=str, default=config.log_dir)
    parser.add_argument("--save_every_steps", type=int, default=config.save_every_steps)
    
    # 系统参数
    parser.add_argument("--device", type=str, default=config.device)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=config.gradient_accumulation_steps)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # 混合精度
    parser.add_argument("--use_amp", action="store_true", default=config.use_amp)
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")
    parser.add_argument("--amp_dtype", type=str, default=config.amp_dtype)
    
    # 数据加载
    parser.add_argument("--vlm_dtype", type=str, default="float32")
    parser.add_argument("--skip_images", action="store_true", default=True)
    parser.add_argument("--cache_vlm_states", action="store_true", default=False)
    parser.add_argument("--use_shared_cache", action="store_true", default=False)
    parser.add_argument("--cache_dtype", type=str, default="float32", choices=["float32", "float16"])
    
    # WebDataset 大规模数据训练优化
    parser.add_argument("--use_webdataset", action="store_true", default=False)
    parser.add_argument("--webdataset_shard_pattern", type=str, default="")
    parser.add_argument("--webdataset_shuffle_buffer", type=int, default=10000)
    
    # WebDataset 分批缓存模式
    parser.add_argument("--use_webdataset_cached", action="store_true", default=False)
    parser.add_argument("--webdataset_cache_shards", type=int, default=4)
    parser.add_argument("--webdataset_cache_dtype", type=str, default="float32", choices=["float32", "float16"])
    
    # State 四元数转轴角配置
    parser.add_argument("--convert_quat_to_axisangle", action="store_true", default=True,
                        help="将 9 维 state (四元数) 转换为 8 维 state (轴角)")
    parser.add_argument("--no_convert_quat_to_axisangle", action="store_false", 
                        dest="convert_quat_to_axisangle",
                        help="禁用四元数转轴角转换")
    
    # Observation State 映射器参数
    parser.add_argument("--enable_state_mapper", action="store_true", default=False,
                        help="启用 observation_state 映射器")
    parser.add_argument("--state_dim", type=int, default=8,
                        help="State 维度 (默认为 8，四元数转轴角后的维度)")
    parser.add_argument("--state_norm_columns_minmax", type=int, nargs="+", default=None,
                        help="使用 minmax 归一化的列索引，如: --state_norm_columns_minmax 0 1 2 3 4")
    parser.add_argument("--state_norm_columns_axisangle", type=int, nargs="+", default=None,
                        help="使用 axisangle 归一化的列索引 (必须是连续3列的欧拉角)，如: --state_norm_columns_axisangle 5 6 7")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=config.use_wandb)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb_project", type=str, default=config.wandb_project)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_log_freq", type=int, default=10)
    
    args = parser.parse_args()

    # ========== 初始化分布式环境 ==========
    rank, local_rank, world_size, is_distributed = setup_distributed()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        args.device = str(device)
    else:
        device = torch.device(args.device)
    
    print_rank0(f"\n{'='*60}", rank)
    print_rank0(f"Pons + ParaCAT 联合训练", rank)
    print_rank0(f"{'='*60}", rank)
    if is_distributed:
        print_rank0(f"  World Size: {world_size}", rank)
        print_rank0(f"  Effective Batch: {args.batch_size * args.gradient_accumulation_steps * world_size}", rank)
    else:
        print_rank0(f"  Single GPU: {device}", rank)
    print_rank0(f"  Discrete Actions: {args.discrete_actions}", rank)
    print_rank0(f"{'='*60}\n", rank)

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
    
    # 初始化 wandb
    wandb_initialized = False
    if args.use_wandb and WANDB_AVAILABLE and is_main_process(rank):
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = args.wandb_run_name or f'pons_paracat_lr{args.lr}_{timestamp}'
            
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=str(log_dir),
                # 注意: 移除了 _stats_sample_rate_seconds 和 _stats_samples_to_average 设置
                # 这些参数在新版本的 wandb 中已不再支持
            )
            wandb_initialized = True
            print("✓ Wandb 初始化成功")
            
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

    # ========== 自动检测参数 ==========
    match = re.search(r'hidden_dim_(\d+)_(\d+)_(\d+)', args.data_path)
    
    if args.num_vlm_layers is None:
        args.num_vlm_layers = int(match.group(1)) if match else 1
    
    if args.vlm_output_dim is None:
        args.vlm_output_dim = int(match.group(3)) if match else 1536
    
    print_rank0(f"VLM 配置: layers={args.num_vlm_layers}, dim={args.vlm_output_dim}", rank)

    # ========== 加载数据集 ==========
    info_path = Path(args.data_path) / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    total_episodes = info['total_episodes']
    actual_action_dim = info['features']['action']['shape'][0]
    
    if args.action_dim != actual_action_dim:
        print_rank0(f"⚠️ 覆盖 action_dim: {args.action_dim} -> {actual_action_dim}", rank)
        args.action_dim = actual_action_dim
    
    # ========== 校验离散化配置 ==========
    # 确保 discrete_columns + gripper_columns = action_dim
    if args.discrete_actions:
        discrete_cols = set(args.discrete_columns or [])
        gripper_cols = set(args.gripper_columns or [])
        
        # 检查是否有重复列
        overlap = discrete_cols & gripper_cols
        if overlap:
            raise ValueError(
                f"❌ 离散化配置错误！discrete_columns 和 gripper_columns 有重复列: {overlap}\n"
                f"  discrete_columns: {args.discrete_columns}\n"
                f"  gripper_columns: {args.gripper_columns}"
            )
        
        # 检查总列数是否等于 action_dim
        total_cols = len(discrete_cols) + len(gripper_cols)
        if total_cols != args.action_dim:
            raise ValueError(
                f"❌ 离散化配置不完整！\n"
                f"  discrete_columns ({len(discrete_cols)} 列): {args.discrete_columns}\n"
                f"  gripper_columns ({len(gripper_cols)} 列): {args.gripper_columns}\n"
                f"  总计: {total_cols} 列 != action_dim ({args.action_dim} 列)\n"
                f"  提示: 所有 action 列必须被覆盖，请检查 DISCRETE_COLUMNS 和 GRIPPER_COLUMNS 配置"
            )
        
        print_rank0(f"✓ 离散化配置校验通过: discrete={len(discrete_cols)} + gripper={len(gripper_cols)} = {args.action_dim}", rank)
    
    print_rank0(f"\n数据集: {args.data_path}", rank)
    print_rank0(f"  Episodes: {total_episodes}, Action dim: {args.action_dim}", rank)
    
    # 归一化 (如果启用 state_mapper，也加载 state 统计信息)
    normalizers = load_normalization_stats(
        args.data_path, 
        load_state_stats=args.enable_state_mapper
    ) if not args.discrete_actions or args.enable_state_mapper else None
    
    train_episode_indices = list(range(total_episodes))
    
    # ========== 数据加载模式选择 ==========
    # 优先级: WebDataset 分批缓存 > WebDataset > 共享内存缓存 > 普通加载
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    shared_vlm_cache = None
    use_webdataset_mode = args.use_webdataset and args.webdataset_shard_pattern
    use_webdataset_cached_mode = args.use_webdataset_cached and args.webdataset_shard_pattern
    
    # WebDataset 分批缓存模式优先
    if use_webdataset_cached_mode:
        use_webdataset_mode = False
        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"🚀 启用 WebDataset 分批缓存模式", rank)
        print_rank0(f"  - 分片路径: {args.webdataset_shard_pattern}", rank)
        print_rank0(f"  - 每批分片数: {args.webdataset_cache_shards}", rank)
        print_rank0(f"  - 缓存数据类型: {args.webdataset_cache_dtype}", rank)
        print_rank0(f"{'='*60}", rank)
    elif use_webdataset_mode:
        print_rank0(f"\n{'='*60}", rank)
        print_rank0(f"🚀 使用 WebDataset 模式", rank)
        print_rank0(f"  - 分片路径: {args.webdataset_shard_pattern}", rank)
        print_rank0(f"  - Shuffle Buffer: {args.webdataset_shuffle_buffer}", rank)
        print_rank0(f"{'='*60}", rank)
    elif args.use_shared_cache:
        cache_size = get_total_vlm_states(args.data_path)
        cache_dtype_str = getattr(args, 'cache_dtype', 'float32')
        cache_dtype = np.float16 if cache_dtype_str == 'float16' else np.float32
        
        print_rank0(f"\n启用共享内存缓存，加载 {cache_size} 个样本...", rank)
        
        shared_vlm_cache = preload_vlm_cache_distributed(
            dataset_path=args.data_path,
            num_samples=cache_size,
            sample_shape=None,
            rank=rank,
            world_size=world_size,
            dtype=cache_dtype,
            verbose=is_main_process(rank),
            auto_detect_shape=True,
            cache_dtype=cache_dtype_str,
        )
        
        _shared_vlm_cache_global = shared_vlm_cache
        _shared_vlm_cache_rank = rank
        atexit.register(cleanup_shared_memory_cache)
    
    # ========== 创建数据加载器 ==========
    train_dataset = None
    train_sampler = None
    shard_batch_cache = None
    
    if use_webdataset_cached_mode:
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
        
        num_samples, train_dataset = shard_batch_cache.load_next_batch()
        cached_collate_fn = partial(cached_samples_collate_fn, vlm_dtype=args.vlm_dtype)
        
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            shuffle = False
        else:
            shuffle = True
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=0,
            sampler=train_sampler,
            collate_fn=cached_collate_fn,
            pin_memory=True,
            drop_last=True,
        )
        print_rank0(f"✓ WebDataset 分批缓存模式数据加载器创建完成，当前批次 {num_samples} 个样本", rank)
        
    elif use_webdataset_mode:
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
            epoch_length=None,
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
            num_action_chunks=args.chunk_size,
            enable_chunking=True,
            episode_indices=train_episode_indices,
            cache_vlm_states=args.cache_vlm_states,
            verbose=is_main_process(rank),
            skip_images=args.skip_images,
            shared_vlm_cache=shared_vlm_cache,
            # 离散化参数 (7个)
            discrete_actions=args.discrete_actions,
            discrete_columns=args.discrete_columns,
            discrete_deltas=args.discrete_deltas,
            discrete_method=args.discrete_method,
            undiscrete_actions=args.undiscrete_actions,
            undiscrete_columns=args.undiscrete_columns,
            undiscrete_deltas=args.undiscrete_deltas,
            # State 预处理参数
            state_process_order=args.state_process_order,
            hand_binary_columns=args.hand_binary_columns,
            hand_binary_threshold=args.hand_binary_threshold,
            state_euler_to_axisangle_columns=args.state_euler_to_axisangle_columns,
        )
        
        lerobot_collate_fn_with_dtype = partial(lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype)
        
        if is_distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
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
            persistent_workers=True if args.num_workers > 0 else False,
        )
        
        print_rank0(f"✓ 数据加载器创建完成，共 {len(train_dataset)} 个样本", rank)

    # ========== 创建模型 ==========
    # Pons Adapter
    pons = create_pons_adapter(
        q_seq_len=args.pons_q_seq_len,
        hidden_dim=args.vlm_output_dim,
        num_blocks=args.pons_num_blocks,
        num_heads=args.pons_num_heads,
        dropout=args.pons_dropout,
    ).to(device)
    
    # 可选: 加载预训练 Pons
    if args.pons_checkpoint:
        pons_state = torch.load(args.pons_checkpoint, map_location=device)
        pons.load_state_dict(pons_state)
        print_rank0(f"✓ Pons 预训练权重已加载: {args.pons_checkpoint}", rank)
    
    # ParaCAT Action Head
    paracat = create_paracat_action_head(
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        hidden_dim=args.vlm_output_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        num_mlp_layers=args.num_mlp_layers,
        mlp_expand_dim=args.mlp_expand_dim,
        num_heads=args.num_heads,
    ).to(device)
    
    # 可选: 加载预训练 ParaCAT
    if args.paracat_checkpoint:
        paracat_state = torch.load(args.paracat_checkpoint, map_location=device)
        paracat.load_state_dict(paracat_state)
        print_rank0(f"✓ ParaCAT 预训练权重已加载: {args.paracat_checkpoint}", rank)
    
    # 可选: 创建 ObservationStateMapper
    state_mapper = None
    if args.enable_state_mapper:
        # 获取 state 的 min/max 统计信息
        state_min = normalizers.get('state_min', None) if normalizers else None
        state_max = normalizers.get('state_max', None) if normalizers else None
        
        if state_min is None or state_max is None:
            print_rank0(f"[WARN] 未找到 observation.state 的统计信息，state_mapper 将使用默认的 [0,1] 范围", rank)
        
        state_mapper = ObservationStateMapper(
            state_dim=args.state_dim,
            hidden_dim=args.vlm_output_dim,
            minmax_columns=args.state_norm_columns_minmax,
            axisangle_columns=args.state_norm_columns_axisangle,
            state_min=state_min,
            state_max=state_max,
        ).to(device)
        
        print_rank0(f"✓ ObservationStateMapper 已创建:", rank)
        print_rank0(f"  State Dim: {args.state_dim}", rank)
        print_rank0(f"  MinMax Columns: {args.state_norm_columns_minmax}", rank)
        print_rank0(f"  AxisAngle Columns: {args.state_norm_columns_axisangle}", rank)
    
    # 联合模型
    model = PonsParaCATModel(pons=pons, paracat=paracat, state_mapper=state_mapper).to(device)
    
    # ========== 打印详细的参数量统计 ==========
    if is_main_process(rank):
        print("\n" + "=" * 70)
        print("📊 模型参数量详细统计")
        print("=" * 70)
        
        # Pons 参数量统计
        pons.print_model_summary(print_details=True)
        
        # ParaCAT 参数量统计
        paracat.print_model_summary(print_details=True)
        
        # State Mapper 参数量统计 (如果启用)
        if state_mapper is not None:
            state_mapper.print_model_summary(print_details=True)
        
        # 汇总
        pons_params = sum(p.numel() for p in pons.parameters())
        paracat_params = sum(p.numel() for p in paracat.parameters())
        state_mapper_params = sum(p.numel() for p in state_mapper.parameters()) if state_mapper else 0
        total_params = pons_params + paracat_params + state_mapper_params
        
        print("\n" + "=" * 70)
        print("📈 Pons + ParaCAT 联合模型 总参数量")
        print("=" * 70)
        print(f"   • Pons Adapter:       {pons_params:>15,} params")
        print(f"   • ParaCAT Head:       {paracat_params:>15,} params")
        if state_mapper is not None:
            print(f"   • State Mapper:       {state_mapper_params:>15,} params")
        print(f"   ─────────────────────────────────────────────────")
        print(f"   • 总计:               {total_params:>15,} params")
        print(f"   • 参数量 (MB):        {total_params * 4 / 1024 / 1024:>15.2f} MB (float32)")
        print(f"   • 参数量 (MB):        {total_params * 2 / 1024 / 1024:>15.2f} MB (float16/bf16)")
        print("=" * 70 + "\n")
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    model.train()
    
    # Loss
    if args.discrete_actions:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.L1Loss()
    
    # 优化器 - 支持不同的学习率
    if args.pons_lr_scale != 1.0:
        param_groups = [
            {'params': list(paracat.parameters()), 'lr': args.lr},
            {'params': list(pons.parameters()), 'lr': args.lr * args.pons_lr_scale},
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=args.weight_decay,
            betas=(0.95, 0.999),
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.95, 0.999),
        )
    
    # 学习率调度器
    total_steps = args.steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        else:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # 混合精度
    use_amp = args.use_amp and torch.cuda.is_available()
    amp_dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    scaler = GradScaler('cuda', enabled=use_amp)
    
    print_rank0(f"\n训练配置:", rank)
    print_rank0(f"  Steps: {args.steps}, LR: {args.lr}", rank)
    print_rank0(f"  Pons LR Scale: {args.pons_lr_scale}", rank)
    print_rank0(f"  Warmup: {warmup_steps} steps", rank)
    print_rank0(f"  AMP: {use_amp} ({args.amp_dtype})", rank)
    print_rank0(f"  四元数转轴角: {'启用' if args.convert_quat_to_axisangle else '禁用'}", rank)

    # ========== 训练循环 ==========
    step = 0
    gradient_accumulation_steps = args.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    
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
    
    # collate_fn 封装
    collate_with_normalizers = partial(
        pons_paracat_collate_fn,
        normalizers=normalizers,
        num_vlm_layers=args.num_vlm_layers,
        discrete_actions=args.discrete_actions,
        gripper_columns=args.gripper_columns,
        convert_quat_to_axisangle=args.convert_quat_to_axisangle,
    )
    
    model_config = {
        'type': 'pons_paracat',
        # Pons
        'pons_q_seq_len': args.pons_q_seq_len,
        'pons_num_blocks': args.pons_num_blocks,
        'pons_num_heads': args.pons_num_heads,
        'pons_dropout': args.pons_dropout,
        # ParaCAT
        'chunk_size': args.chunk_size,
        'action_dim': args.action_dim,
        'hidden_dim': args.vlm_output_dim,
        'num_transformer_blocks': args.num_transformer_blocks,
        'num_mlp_layers': args.num_mlp_layers,
        'mlp_expand_dim': args.mlp_expand_dim,
        # 离散化配置
        'discrete_actions': args.discrete_actions,
        'discrete_columns': args.discrete_columns if args.discrete_actions else None,
        'discrete_deltas': args.discrete_deltas if args.discrete_actions else None,
        'discrete_method': args.discrete_method if args.discrete_actions else None,
        # State 预处理配置
        'state_process_order': args.state_process_order,
        'hand_binary_columns': args.hand_binary_columns,
        'hand_binary_threshold': args.hand_binary_threshold,
        'state_euler_to_axisangle_columns': args.state_euler_to_axisangle_columns,
        'undiscrete_actions': args.undiscrete_actions,
        'undiscrete_columns': args.undiscrete_columns if args.undiscrete_actions else None,
        'undiscrete_deltas': args.undiscrete_deltas if args.undiscrete_actions else None,
        # Gripper 配置 (LIBERO 专用)
        'gripper_columns': args.gripper_columns if args.discrete_actions else None,
        'gripper_method': args.gripper_method if args.discrete_actions else None,
        # State Mapper 配置
        'enable_state_mapper': args.enable_state_mapper,
        'state_dim': args.state_dim if args.enable_state_mapper else None,
        'state_norm_columns_minmax': args.state_norm_columns_minmax if args.enable_state_mapper else None,
        'state_norm_columns_axisangle': args.state_norm_columns_axisangle if args.enable_state_mapper else None,
        'state_min': normalizers.get('state_min', None).tolist() if (args.enable_state_mapper and normalizers and 'state_min' in normalizers) else None,
        'state_max': normalizers.get('state_max', None).tolist() if (args.enable_state_mapper and normalizers and 'state_max' in normalizers) else None,
        # 其他
        'num_vlm_layers': args.num_vlm_layers,
    }
    
    # 记录训练开始时间，用于计算已用时间和剩余时间
    training_start_time = time.time()
    
    # 估算总 epoch 数
    try:
        steps_per_epoch = len(train_loader) // gradient_accumulation_steps
        if steps_per_epoch > 0:
            estimated_total_epochs = (args.steps + steps_per_epoch - 1) // steps_per_epoch
        else:
            estimated_total_epochs = None
    except:
        estimated_total_epochs = None
    
    epoch = 0
    while step < args.steps:
        model.train()
        
        # 分布式训练时，每个 epoch 设置 sampler 的 epoch
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # WebDataset 分批缓存模式：每个 epoch 重新打乱分片顺序
        if use_webdataset_cached_mode and shard_batch_cache is not None:
            shard_batch_cache.set_epoch(epoch)
        
        if is_main_process(rank):
            epoch_desc = f"Epoch {epoch}/{estimated_total_epochs-1}" if estimated_total_epochs else f"Epoch {epoch}"
            pbar = tqdm(train_loader, desc=epoch_desc, ncols=120)
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # 处理数据
            processed = collate_with_normalizers(batch)
            
            vlm_hidden_states = [v.to(device) for v in processed['vlm_hidden_states']]
            vlm_attention_mask = processed['vlm_attention_mask']
            if vlm_attention_mask is not None:
                vlm_attention_mask = vlm_attention_mask.to(device)
            
            gt_actions = processed['gt_actions'].to(device)
            
            # 获取 observation_state (如果启用 state_mapper)
            observation_state = None
            if args.enable_state_mapper:
                observation_state = processed.get('observation_state', None)
                if observation_state is not None:
                    observation_state = observation_state.to(device)
            
            is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0
            
            # Forward
            context_managers = []
            if is_distributed and is_accumulating:
                context_managers.append(model.no_sync())
            
            with ExitStack() as stack:
                for cm in context_managers:
                    stack.enter_context(cm)
                
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    predicted = model(vlm_hidden_states, attention_mask=vlm_attention_mask, observation_state=observation_state)
                    # predicted: (batch, chunk_size, action_dim, 3)
                    
                    if args.discrete_actions:
                        batch_size, chunk_size, action_dim, num_classes = predicted.shape
                        predicted_flat = predicted.view(-1, num_classes)
                        gt_flat = gt_actions.view(-1)
                        
                        # 验证标签范围
                        gt_min = gt_flat.min().item()
                        gt_max = gt_flat.max().item()
                        if gt_min < 0 or gt_max >= num_classes:
                            # 打印详细调试信息
                            print(f"\n❌ [Rank {rank}] 标签值超出范围!")
                            print(f"  gt_flat 范围: [{gt_min}, {gt_max}]，有效范围: [0, {num_classes - 1}]")
                            print(f"  gt_actions 形状: {gt_actions.shape}")
                            print(f"  gt_actions 唯一值: {torch.unique(gt_actions).tolist()}")
                            # 检查每列的值
                            for col_idx in range(gt_actions.shape[-1]):
                                col_vals = gt_actions[:, :, col_idx].unique().tolist()
                                print(f"  列 {col_idx} 唯一值: {col_vals}")
                            raise ValueError(
                                f"CrossEntropyLoss 标签超出范围: [{gt_min}, {gt_max}]，期望 [0, {num_classes - 1}]"
                            )
                        
                        loss = criterion(predicted_flat, gt_flat) / gradient_accumulation_steps
                    else:
                        # 连续模式: 取中间值 (index=1)
                        loss = criterion(predicted[..., 1], gt_actions) / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=1.0
                ).item()
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                step += 1
                
                batch_time = time.time() - batch_start_time
                
                # 记录每个 step 的 loss 到 custom 日志文件（只在主进程）
                if step_loss_log_file is not None and is_main_process(rank):
                    current_lr = optimizer.param_groups[0]['lr']
                    actual_loss = loss.item() * gradient_accumulation_steps
                    step_loss_log_file.write(f"{step},{epoch},{actual_loss:.6f},{current_lr:.8f},{grad_norm:.6f},{batch_time:.4f}\n")
                    # 每 100 步 flush 一次，确保数据写入磁盘
                    if step % 100 == 0:
                        step_loss_log_file.flush()
                
                # Wandb 日志
                if wandb_initialized and step % args.wandb_log_freq == 0:
                    try:
                        log_dict = {
                            'train/loss': loss.item() * gradient_accumulation_steps,
                            'train/step': step,
                            'train/batch_time': batch_time,
                            'train/grad_norm': grad_norm,
                            'train/learning_rate': optimizer.param_groups[0]['lr'],
                            'train/samples_per_sec': args.batch_size * gradient_accumulation_steps * world_size / batch_time if batch_time > 0 else 0,
                        }
                        # 如果有多个 param groups，记录每个的 learning rate
                        for i, pg in enumerate(optimizer.param_groups):
                            log_dict[f'train/lr_group{i}'] = pg['lr']
                        wandb.log(log_dict)
                    except Exception:
                        pass
                
                if is_main_process(rank):
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
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'step': f'{step}/{args.steps}',
                        'time': time_str,
                    })
                
                # 保存 checkpoint
                if is_main_process(rank) and step % args.save_every_steps == 0:
                    step_dir = out_dir / f"step_{step}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    
                    model_to_save = model.module if is_distributed else model
                    
                    # 分别保存 Pons 和 ParaCAT
                    torch.save(model_to_save.pons.state_dict(), step_dir / "pons.pt")
                    torch.save(model_to_save.paracat.state_dict(), step_dir / "paracat.pt")
                    
                    # 保存 State Mapper (如果启用)
                    if model_to_save.state_mapper is not None:
                        torch.save(model_to_save.state_mapper.state_dict(), step_dir / "state_mapper.pt")
                    
                    config_to_save = model_config.copy()
                    config_to_save['saved_at_step'] = step
                    with open(step_dir / "config.json", "w") as f:
                        json.dump(config_to_save, f, indent=2)
                    
                    print(f"\n✓ Checkpoint saved to {step_dir}/")
            
            if step >= args.steps:
                break
        
        if is_main_process(rank):
            pbar.close()
        
        epoch += 1
        
        if step >= args.steps:
            break
    
    # 最终保存
    if is_main_process(rank):
        final_dir = out_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        
        model_to_save = model.module if is_distributed else model
        
        torch.save(model_to_save.pons.state_dict(), final_dir / "pons.pt")
        torch.save(model_to_save.paracat.state_dict(), final_dir / "paracat.pt")
        
        # 保存 State Mapper (如果启用)
        if model_to_save.state_mapper is not None:
            torch.save(model_to_save.state_mapper.state_dict(), final_dir / "state_mapper.pt")
        
        config_to_save = model_config.copy()
        config_to_save['saved_at_step'] = step
        with open(final_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ 训练完成!")
        print(f"  Final checkpoint: {final_dir}/")
        print(f"  Total steps: {step}")
        print(f"{'='*60}")
    
    if wandb_initialized:
        wandb.finish()
    
    # 关闭 step loss 日志文件
    if step_loss_log_file is not None and is_main_process(rank):
        step_loss_log_file.flush()
        step_loss_log_file.close()
        print(f"✓ Step loss 日志已保存到 {custom_log_dir}/")
    
    cleanup_shared_memory_cache()
    cleanup_distributed()


if __name__ == "__main__":
    main()

