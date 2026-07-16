"""
ParaCAT Action Head 训练脚本 - 支持多GPU分布式训练 (仅 ParaCAT)

使用预训练的 Pons 输出或直接使用 VLM hidden states 训练 ParaCAT Action Head。

多GPU训练使用方法:
  torchrun --nproc_per_node=4 train_multigpu_only_paracat.py \
    --data_path /path/to/dataset \
    --pons_checkpoint /path/to/pons.pt \
    --steps 10000

单GPU训练使用方法:
  python train_multigpu_only_paracat.py \
    --data_path /path/to/dataset \
    --device cuda:0

参数说明:
  --pons_checkpoint: 预训练的 Pons checkpoint (可选，不提供则直接使用 VLM hidden states)
  --freeze_pons: 冻结 Pons 参数 (仅训练 ParaCAT)
  --discrete_actions: 启用 action 离散化 (ParaCAT 输出 3 类别)
"""

import argparse
import json
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
    """ParaCAT Action Head 训练配置"""
    
    # 数据相关参数
    data_path: str = ""
    batch_size: int = 32
    num_workers: int = 4
    
    # 训练超参数
    steps: int = 10000
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    
    # ParaCAT 参数
    chunk_size: int = 16
    action_dim: int = 7
    num_transformer_blocks: int = 2
    num_mlp_layers: int = 2
    mlp_expand_dim: int = 1024
    num_heads: int = 8
    
    # Pons 参数 (如果使用)
    pons_checkpoint: str = ""
    pons_q_seq_len: int = 64
    pons_num_blocks: int = 2
    freeze_pons: bool = False
    
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
    out_dir: str = "./experiments/paracat_training/checkpoints"
    log_dir: str = "./experiments/paracat_training/logs"
    save_every_steps: int = 1000
    
    # 系统配置
    device: str = "cuda:0"
    gradient_accumulation_steps: int = 1
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "paracat_training"


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


def load_normalization_stats(dataset_path: str) -> dict:
    """从数据集加载归一化统计信息"""
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"[WARN] stats.json 不存在于 {stats_path}")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    normalizers = {}
    
    if 'action' in stats:
        action_stats = stats['action']
        action_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        action_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        normalizers['action'] = MinMaxNormalizer(action_min, action_max)
    
    return normalizers if normalizers else None


# ============================================================================
# 数据处理函数
# ============================================================================

def paracat_collate_fn(batch, normalizers=None, num_vlm_layers=1, discrete_actions=False):
    """
    将 LeRobot batch 转换为 ParaCAT 训练需要的格式
    """
    vlm_tensor_raw = batch['vlm_hidden_states']  # (batch, num_layers, seq_len, hidden_dim)
    vlm_attention_mask_raw = batch.get('vlm_attention_mask', None)
    actions = batch['actions']  # (batch, num_chunks, action_dim)
    
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
    
    # 归一化 actions
    if normalizers is not None and 'action' in normalizers and not discrete_actions:
        actions_flat = actions.reshape(batch_size * num_chunks, action_dim)
        actions_normalized = normalizers['action'].normalize(actions_flat)
        actions = actions_normalized.reshape(batch_size, num_chunks, action_dim)
    
    # Ground truth actions 格式化
    # 如果是离散化模式，actions 已经是 (-delta, 0, delta)，需要转为类别 (0, 1, 2)
    # 只对 discrete_columns 指定的列进行类别转换
    # 如果是连续模式，保持原样
    if discrete_actions:
        # 离散化模式: 将离散值 {-1, 0, 1} 转换为类别索引 {0, 1, 2}
        # 映射关系: -1 -> 0, 0 -> 1, 1 -> 2
        # 公式: class_idx = discrete_val + 1
        discrete_columns = batch.get('discrete_columns', None)
        
        if discrete_columns is not None:
            # 按列处理：只对指定的离散化列进行类别转换
            gt_actions_discrete = torch.zeros_like(actions, dtype=torch.long)
            
            for col_idx in discrete_columns:
                if col_idx < action_dim:
                    # actions[:, :, col_idx] 已经是 {-1, 0, 1}
                    # 直接 +1 转为类别索引 {0, 1, 2}
                    gt_actions_discrete[:, :, col_idx] = (actions[:, :, col_idx] + 1).long()
            
            gt_actions = gt_actions_discrete
        else:
            # 兼容旧模式：所有列都是离散化的
            # actions 值为 {-1, 0, 1}，直接 +1 转为类别索引 {0, 1, 2}
            gt_actions = (actions + 1).long()
    else:
        gt_actions = actions
    
    return {
        'vlm_hidden_states': vlm_hidden_states,
        'vlm_attention_mask': vlm_attention_mask,
        'gt_actions': gt_actions,
        'discrete_columns': batch.get('discrete_columns', None),
        'discrete_deltas': batch.get('discrete_deltas', None),
    }


# ============================================================================
# 主训练函数
# ============================================================================

def main():
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    
    parser = argparse.ArgumentParser(description='Train ParaCAT Action Head (only)')
    config = TrainingConfig()
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num_workers", type=int, default=config.num_workers)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    
    # 训练参数
    parser.add_argument("--steps", type=int, default=config.steps)
    parser.add_argument("--lr", type=float, default=config.lr)
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
    parser.add_argument("--pons_checkpoint", type=str, default="",
                        help="预训练的 Pons checkpoint，不提供则直接使用 VLM hidden states")
    parser.add_argument("--pons_q_seq_len", type=int, default=config.pons_q_seq_len)
    parser.add_argument("--pons_num_blocks", type=int, default=config.pons_num_blocks)
    parser.add_argument("--freeze_pons", action="store_true", default=False)
    
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
    parser.add_argument("--undiscrete_actions", action="store_true", default=False,
                        help="启用反离散化配置 (用于推理)")
    parser.add_argument("--undiscrete_columns", type=int, nargs="+", default=None,
                        help="参与反离散化的列索引")
    parser.add_argument("--undiscrete_deltas", type=float, nargs="+", default=None,
                        help="对应列的反离散化 delta 值")
    
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
    print_rank0(f"ParaCAT Action Head Training (Only ParaCAT)", rank)
    print_rank0(f"{'='*60}", rank)
    if is_distributed:
        print_rank0(f"  World Size: {world_size}", rank)
    else:
        print_rank0(f"  Single GPU: {device}", rank)
    print_rank0(f"  Discrete Actions: {args.discrete_actions}", rank)
    if args.pons_checkpoint:
        print_rank0(f"  Pons Checkpoint: {args.pons_checkpoint}", rank)
        print_rank0(f"  Freeze Pons: {args.freeze_pons}", rank)
    else:
        print_rank0(f"  No Pons - Using VLM hidden states directly", rank)
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
            run_name = args.wandb_run_name or f'paracat_lr{args.lr}_{timestamp}'
            
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
    
    # 使用实际的 action_dim
    if args.action_dim != actual_action_dim:
        print_rank0(f"⚠️ 覆盖 action_dim: {args.action_dim} -> {actual_action_dim}", rank)
        args.action_dim = actual_action_dim
    
    print_rank0(f"\n数据集: {args.data_path}", rank)
    print_rank0(f"  Episodes: {total_episodes}, Action dim: {args.action_dim}", rank)
    
    # 归一化
    normalizers = load_normalization_stats(args.data_path) if not args.discrete_actions else None
    
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
            # 离散化参数 (6个)
            discrete_actions=args.discrete_actions,
            discrete_columns=args.discrete_columns,
            discrete_deltas=args.discrete_deltas,
            undiscrete_actions=args.undiscrete_actions,
            undiscrete_columns=args.undiscrete_columns,
            undiscrete_deltas=args.undiscrete_deltas,
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
    use_pons = bool(args.pons_checkpoint)
    
    pons = None
    if use_pons:
        # 加载 Pons
        pons = create_pons_adapter(
            q_seq_len=args.pons_q_seq_len,
            hidden_dim=args.vlm_output_dim,
            num_blocks=args.pons_num_blocks,
            num_heads=args.num_heads,
        ).to(device)
        
        # 加载权重
        pons_state = torch.load(args.pons_checkpoint, map_location=device)
        pons.load_state_dict(pons_state)
        print_rank0(f"✓ Pons 权重加载完成: {args.pons_checkpoint}", rank)
        
        if args.freeze_pons:
            for param in pons.parameters():
                param.requires_grad = False
            print_rank0("  Pons 参数已冻结", rank)
        
        pons_output_dim = args.vlm_output_dim
    else:
        # 不使用 Pons，直接使用 VLM hidden states
        pons_output_dim = args.vlm_output_dim
    
    # ParaCAT Action Head
    paracat = create_paracat_action_head(
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        hidden_dim=pons_output_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        num_mlp_layers=args.num_mlp_layers,
        mlp_expand_dim=args.mlp_expand_dim,
        num_heads=args.num_heads,
    ).to(device)
    
    # 收集需要训练的参数
    train_params = list(paracat.parameters())
    if use_pons and not args.freeze_pons:
        train_params += list(pons.parameters())
    
    total_params = sum(p.numel() for p in train_params if p.requires_grad)
    print_rank0(f"\n模型参数:", rank)
    print_rank0(f"  ParaCAT: {sum(p.numel() for p in paracat.parameters()):,}", rank)
    if use_pons:
        print_rank0(f"  Pons: {sum(p.numel() for p in pons.parameters()):,}", rank)
    print_rank0(f"  Trainable: {total_params:,}", rank)
    
    if is_distributed:
        paracat = DDP(paracat, device_ids=[local_rank], output_device=local_rank)
        if use_pons and not args.freeze_pons:
            pons = DDP(pons, device_ids=[local_rank], output_device=local_rank)
    
    paracat.train()
    if use_pons:
        pons.train() if not args.freeze_pons else pons.eval()
    
    # Loss
    if args.discrete_actions:
        # 离散化模式: 使用 CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()
    else:
        # 连续模式: 使用 L1Loss
        criterion = nn.L1Loss()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        [p for p in train_params if p.requires_grad],
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
    print_rank0(f"  Discrete: {args.discrete_actions}", rank)

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
        paracat_collate_fn,
        normalizers=normalizers,
        num_vlm_layers=args.num_vlm_layers,
        discrete_actions=args.discrete_actions,
    )
    
    model_config = {
        'type': 'paracat',
        'chunk_size': args.chunk_size,
        'action_dim': args.action_dim,
        'hidden_dim': pons_output_dim,
        'num_transformer_blocks': args.num_transformer_blocks,
        'num_mlp_layers': args.num_mlp_layers,
        'mlp_expand_dim': args.mlp_expand_dim,
        # 离散化配置
        'discrete_actions': args.discrete_actions,
        'discrete_columns': args.discrete_columns if args.discrete_actions else None,
        'discrete_deltas': args.discrete_deltas if args.discrete_actions else None,
        'undiscrete_actions': args.undiscrete_actions,
        'undiscrete_columns': args.undiscrete_columns if args.undiscrete_actions else None,
        'undiscrete_deltas': args.undiscrete_deltas if args.undiscrete_actions else None,
        # 其他
        'use_pons': use_pons,
        'pons_q_seq_len': args.pons_q_seq_len if use_pons else None,
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
        paracat.train()
        if use_pons and not args.freeze_pons:
            pons.train()
        
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
            
            is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0
            
            # Forward
            context_managers = []
            if is_distributed and is_accumulating:
                context_managers.append(paracat.no_sync())
                if use_pons and not args.freeze_pons:
                    context_managers.append(pons.no_sync())
            
            from contextlib import ExitStack
            with ExitStack() as stack:
                for cm in context_managers:
                    stack.enter_context(cm)
                
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    # Pons (可选)
                    if use_pons:
                        pons_output = pons(vlm_hidden_states, attention_mask=vlm_attention_mask)
                    else:
                        # 直接合并 VLM hidden states
                        pons_output = torch.cat(vlm_hidden_states, dim=1)
                    
                    # ParaCAT
                    predicted = paracat(pons_output)
                    # predicted: (batch, chunk_size, action_dim, 3)
                    
                    if args.discrete_actions:
                        # 离散化模式: CrossEntropyLoss
                        # predicted: (B, chunk, action_dim, 3)
                        # gt_actions: (B, chunk, action_dim) long
                        batch_size, chunk_size, action_dim, num_classes = predicted.shape
                        predicted_flat = predicted.view(-1, num_classes)  # (B*chunk*action, 3)
                        gt_flat = gt_actions.view(-1)  # (B*chunk*action,)
                        loss = criterion(predicted_flat, gt_flat) / gradient_accumulation_steps
                    else:
                        # 连续模式: L1Loss
                        # 需要把 ParaCAT 输出的 (B, chunk, action, 3) 转为连续值
                        # 这里假设第3维是 3 个分量，取 argmax 或其他方式
                        # 但根据设计，连续模式下 ParaCAT 输出最后一维应该是 1，不是 3
                        # 先按 L1 处理
                        loss = criterion(predicted[..., 1], gt_actions) / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in train_params if p.requires_grad], 
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
                        wandb.log({
                            'train/loss': loss.item() * gradient_accumulation_steps,
                            'train/step': step,
                            'train/batch_time': batch_time,
                            'train/grad_norm': grad_norm,
                            'train/learning_rate': optimizer.param_groups[0]['lr'],
                            'train/samples_per_sec': args.batch_size * gradient_accumulation_steps * world_size / batch_time if batch_time > 0 else 0,
                        })
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
                    
                    paracat_to_save = paracat.module if is_distributed else paracat
                    torch.save(paracat_to_save.state_dict(), step_dir / "paracat.pt")
                    
                    if use_pons and not args.freeze_pons:
                        pons_to_save = pons.module if is_distributed else pons
                        torch.save(pons_to_save.state_dict(), step_dir / "pons.pt")
                    
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
        
        paracat_to_save = paracat.module if is_distributed else paracat
        torch.save(paracat_to_save.state_dict(), final_dir / "paracat.pt")
        
        if use_pons:
            pons_to_save = pons.module if is_distributed else pons
            torch.save(pons_to_save.state_dict(), final_dir / "pons.pt")
        
        config_to_save = model_config.copy()
        config_to_save['saved_at_step'] = step
        with open(final_dir / "config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ 训练完成! Final checkpoint: {final_dir}/")
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

