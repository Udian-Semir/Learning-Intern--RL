"""
Pons Adapter 训练脚本 - 支持多GPU分布式训练

Pons Adapter 单独训练，使用自监督或重建目标进行预训练。
训练好的 Pons 可以用于下游任务（如 ParaCAT Action Head）。

多GPU训练使用方法:
  torchrun --nproc_per_node=4 train_multigpu.py \
    --data_path /path/to/dataset \
    --steps 10000

单GPU训练使用方法:
  python train_multigpu.py \
    --data_path /path/to/dataset \
    --device cuda:0

训练目标:
  - 自监督重建: Pons 输出经过投影后重建原始 VLM hidden states
  - 或者与下游任务（ParaCAT）联合训练
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
    print("⚠️ wandb 未安装，将禁用 wandb 日志")


# ============================================================================
# 训练配置类
# ============================================================================

@dataclass
class TrainingConfig:
    """Pons Adapter 训练配置"""
    
    # 数据相关参数
    data_path: str = ""
    batch_size: int = 32
    num_workers: int = 4
    
    # 训练超参数
    steps: int = 10000
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    adam_beta1: float = 0.95
    adam_beta2: float = 0.999
    
    # Pons 模型参数
    pons_q_seq_len: int = 64
    pons_num_blocks: int = 2
    pons_num_heads: int = 8
    pons_dropout: float = 0.1
    
    # VLM 参数
    num_vlm_layers: Optional[int] = None
    vlm_output_dim: Optional[int] = None
    
    # 保存参数
    out_dir: str = "./experiments/pons_training/checkpoints"
    log_dir: str = "./experiments/pons_training/logs"
    save_every_steps: int = 1000
    
    # 系统配置
    device: str = "cuda:0"
    gradient_accumulation_steps: int = 1
    
    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "pons_training"
    wandb_run_name: str = ""
    wandb_log_freq: int = 10


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
            if rank == 0:
                print(f"✓ 使用 NCCL 后端初始化分布式训练 (timeout={timeout_seconds}s)")
        except Exception as e:
            if rank == 0:
                print(f"⚠️ NCCL 初始化失败: {e}")
                print("  尝试使用 gloo 后端...")
            dist.init_process_group(backend='gloo', init_method='env://', timeout=timeout)
            if rank == 0:
                print(f"✓ 使用 gloo 后端初始化分布式训练")
        
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


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


# ============================================================================
# 共享内存缓存清理
# ============================================================================

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
        except Exception:
            pass


# ============================================================================
# 重建头 (用于自监督训练)
# ============================================================================

class ReconstructionHead(nn.Module):
    """
    重建头：将 Pons 压缩后的特征重建为原始 VLM hidden states
    
    用于自监督预训练 Pons Adapter
    """
    
    def __init__(
        self,
        pons_q_len: int,
        target_seq_len: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pons_q_len = pons_q_len
        self.target_seq_len = target_seq_len
        self.hidden_dim = hidden_dim
        
        # 使用 Cross-Attention 将 pons_q_len 扩展回 target_seq_len
        # Query: learnable tokens (target_seq_len)
        # Key/Value: Pons output (pons_q_len)
        self.query_tokens = nn.Parameter(
            torch.randn(1, target_seq_len, hidden_dim) * 0.02
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 最终投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, pons_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pons_output: (batch_size, pons_q_len, hidden_dim)
        
        Returns:
            reconstructed: (batch_size, target_seq_len, hidden_dim)
        """
        batch_size = pons_output.size(0)
        
        # 扩展 query tokens 到 batch
        query = self.query_tokens.expand(batch_size, -1, -1)
        
        # Cross-attention: query attends to pons output
        attn_out, _ = self.cross_attention(
            query=query,
            key=pons_output,
            value=pons_output
        )
        
        out = self.norm(query + attn_out)
        out = self.output_proj(out)
        
        return out


# ============================================================================
# 数据处理函数
# ============================================================================

def pons_collate_fn(batch, num_vlm_layers=1):
    """
    将 LeRobot batch 转换为 Pons 训练需要的格式
    
    输入:
        - vlm_hidden_states: (batch, num_layers, seq_len, hidden_dim)
    
    输出:
        - vlm_hidden_states: List[Tensor], 每个 (batch, seq_len, hidden_dim)
        - vlm_attention_mask: (batch, seq_len * num_layers), 1=valid, 0=padding
    """
    vlm_tensor_raw = batch['vlm_hidden_states']  # (batch, num_layers, seq_len, hidden_dim)
    vlm_attention_mask_raw = batch.get('vlm_attention_mask', None)
    
    batch_size = vlm_tensor_raw.size(0)
    num_layers = vlm_tensor_raw.size(1)
    seq_len = vlm_tensor_raw.size(2)
    
    # VLM hidden states: 拆分为列表
    vlm_hidden_states = [vlm_tensor_raw[:, i, :, :] for i in range(num_layers)]
    
    # VLM attention mask
    if vlm_attention_mask_raw is not None:
        vlm_attention_mask = vlm_attention_mask_raw.view(batch_size, num_layers * seq_len)
    else:
        vlm_attention_mask = None
    
    return {
        'vlm_hidden_states': vlm_hidden_states,
        'vlm_attention_mask': vlm_attention_mask,
        'vlm_tensor_raw': vlm_tensor_raw,  # 用于重建目标
    }


# ============================================================================
# 主训练函数
# ============================================================================

def main():
    global _shared_vlm_cache_global, _shared_vlm_cache_rank
    
    parser = argparse.ArgumentParser(description='Train Pons Adapter with Multi-GPU Support')
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
    
    # Pons 参数
    parser.add_argument("--pons_q_seq_len", type=int, default=config.pons_q_seq_len)
    parser.add_argument("--pons_num_blocks", type=int, default=config.pons_num_blocks)
    parser.add_argument("--pons_num_heads", type=int, default=config.pons_num_heads)
    parser.add_argument("--pons_dropout", type=float, default=config.pons_dropout)
    
    # VLM 参数
    parser.add_argument("--num_vlm_layers", type=int, default=None)
    parser.add_argument("--vlm_output_dim", type=int, default=None)
    
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
    
    # Action chunks (用于数据加载)
    parser.add_argument("--num_action_chunks", type=int, default=16)
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", default=config.use_wandb)
    parser.add_argument("--no_wandb", action="store_false", dest="use_wandb")
    parser.add_argument("--wandb_project", type=str, default=config.wandb_project)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_log_freq", type=int, default=config.wandb_log_freq)
    
    args = parser.parse_args()

    # ========== 初始化分布式环境 ==========
    rank, local_rank, world_size, is_distributed = setup_distributed()
    
    if is_distributed:
        device = torch.device(f'cuda:{local_rank}')
        args.device = str(device)
    else:
        device = torch.device(args.device)
    
    print_rank0(f"\n{'='*60}", rank)
    print_rank0(f"Pons Adapter Training", rank)
    print_rank0(f"{'='*60}", rank)
    if is_distributed:
        print_rank0(f"  World Size: {world_size}", rank)
        print_rank0(f"  Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps * world_size}", rank)
    else:
        print_rank0(f"  Single GPU Mode: {device}", rank)
    print_rank0(f"{'='*60}\n", rank)

    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    
    if is_main_process(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    if is_distributed:
        dist.barrier()
    
    # 初始化 wandb
    wandb_initialized = False
    if args.use_wandb and WANDB_AVAILABLE and is_main_process(rank):
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = args.wandb_run_name or f'pons_lr{args.lr}_q{args.pons_q_seq_len}_{timestamp}'
            
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=str(log_dir),
            )
            wandb_initialized = True
            print("✓ Wandb 初始化成功")
        except Exception as e:
            print(f"⚠️ Wandb 初始化失败: {e}")

    # ========== 自动检测参数 ==========
    match = re.search(r'hidden_dim_(\d+)_(\d+)_(\d+)', args.data_path)
    
    if args.num_vlm_layers is None:
        if match:
            args.num_vlm_layers = int(match.group(1))
        else:
            args.num_vlm_layers = 1
    
    if args.vlm_output_dim is None:
        if match:
            args.vlm_output_dim = int(match.group(3))
        else:
            args.vlm_output_dim = 1536  # Qwen2B default
    
    print_rank0(f"VLM 配置:", rank)
    print_rank0(f"  num_vlm_layers: {args.num_vlm_layers}", rank)
    print_rank0(f"  vlm_output_dim: {args.vlm_output_dim}", rank)

    # ========== 加载数据集 ==========
    info_path = Path(args.data_path) / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    total_episodes = info['total_episodes']
    print_rank0(f"\n数据集: {args.data_path}", rank)
    print_rank0(f"  Total episodes: {total_episodes}", rank)
    
    train_episode_indices = list(range(total_episodes))
    
    # 共享内存缓存
    shared_vlm_cache = None
    if args.use_shared_cache:
        cache_size = get_total_vlm_states(args.data_path)
        print_rank0(f"\n启用共享内存缓存，加载 {cache_size} 个样本...", rank)
        
        shared_vlm_cache = preload_vlm_cache_distributed(
            dataset_path=args.data_path,
            num_samples=cache_size,
            sample_shape=None,
            rank=rank,
            world_size=world_size,
            dtype=np.float32,
            verbose=is_main_process(rank),
            auto_detect_shape=True,
        )
        
        _shared_vlm_cache_global = shared_vlm_cache
        _shared_vlm_cache_rank = rank
        atexit.register(cleanup_shared_memory_cache)
        
        def signal_handler(signum, frame):
            cleanup_shared_memory_cache()
            sys.exit(0)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    # 创建数据集
    train_dataset = LeRobotDataset(
        dataset_path=args.data_path,
        num_action_chunks=args.num_action_chunks,
        enable_chunking=True,
        episode_indices=train_episode_indices,
        cache_vlm_states=args.cache_vlm_states,
        verbose=is_main_process(rank),
        skip_images=args.skip_images,
        shared_vlm_cache=shared_vlm_cache,
    )
    
    lerobot_collate_fn_with_dtype = partial(lerobot_default_collate_fn, vlm_dtype=args.vlm_dtype)
    
    train_sampler = None
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
    
    # 获取一个样本来确定 target_seq_len
    sample_batch = next(iter(train_loader))
    target_seq_len = sample_batch['vlm_hidden_states'].size(2) * args.num_vlm_layers
    
    # 重建头
    reconstruction_head = ReconstructionHead(
        pons_q_len=args.pons_q_seq_len,
        target_seq_len=target_seq_len,
        hidden_dim=args.vlm_output_dim,
        num_heads=args.pons_num_heads,
        dropout=args.pons_dropout,
    ).to(device)
    
    # 合并参数
    all_params = list(pons.parameters()) + list(reconstruction_head.parameters())
    
    total_params = sum(p.numel() for p in all_params)
    print_rank0(f"\n模型参数:", rank)
    print_rank0(f"  Pons: {sum(p.numel() for p in pons.parameters()):,}", rank)
    print_rank0(f"  Reconstruction Head: {sum(p.numel() for p in reconstruction_head.parameters()):,}", rank)
    print_rank0(f"  Total: {total_params:,}", rank)
    
    if is_distributed:
        pons = DDP(pons, device_ids=[local_rank], output_device=local_rank)
        reconstruction_head = DDP(reconstruction_head, device_ids=[local_rank], output_device=local_rank)
    
    pons.train()
    reconstruction_head.train()
    
    # Loss
    criterion = nn.MSELoss()
    
    # 优化器
    optimizer = torch.optim.AdamW(
        all_params,
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
    print_rank0(f"  Steps: {args.steps}", rank)
    print_rank0(f"  Learning rate: {args.lr}", rank)
    print_rank0(f"  Warmup steps: {warmup_steps}", rank)
    print_rank0(f"  AMP: {use_amp} ({args.amp_dtype})", rank)

    # ========== 训练循环 ==========
    step = 0
    gradient_accumulation_steps = args.gradient_accumulation_steps
    optimizer.zero_grad(set_to_none=True)
    
    model_config = {
        'type': 'pons',
        'q_seq_len': args.pons_q_seq_len,
        'hidden_dim': args.vlm_output_dim,
        'num_blocks': args.pons_num_blocks,
        'num_heads': args.pons_num_heads,
        'dropout': args.pons_dropout,
        'num_vlm_layers': args.num_vlm_layers,
    }
    
    epoch = 0
    while step < args.steps:
        pons.train()
        reconstruction_head.train()
        
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        if is_main_process(rank):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=120)
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            batch_start_time = time.time()
            
            # 处理数据
            processed = pons_collate_fn(batch, num_vlm_layers=args.num_vlm_layers)
            
            vlm_hidden_states = [v.to(device) for v in processed['vlm_hidden_states']]
            vlm_attention_mask = processed['vlm_attention_mask']
            if vlm_attention_mask is not None:
                vlm_attention_mask = vlm_attention_mask.to(device)
            
            # 目标: 原始 VLM hidden states (合并后)
            # (batch, num_layers, seq_len, hidden_dim) -> (batch, num_layers * seq_len, hidden_dim)
            vlm_tensor_raw = processed['vlm_tensor_raw'].to(device)
            batch_size, num_layers, seq_len, hidden_dim = vlm_tensor_raw.shape
            target = vlm_tensor_raw.view(batch_size, num_layers * seq_len, hidden_dim)
            
            is_accumulating = (batch_idx + 1) % gradient_accumulation_steps != 0
            
            # Forward
            if is_distributed and is_accumulating:
                with pons.no_sync(), reconstruction_head.no_sync():
                    with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                        pons_output = pons(vlm_hidden_states, attention_mask=vlm_attention_mask)
                        reconstructed = reconstruction_head(pons_output)
                        loss = criterion(reconstructed, target) / gradient_accumulation_steps
                    scaler.scale(loss).backward()
            else:
                with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                    pons_output = pons(vlm_hidden_states, attention_mask=vlm_attention_mask)
                    reconstructed = reconstruction_head(pons_output)
                    loss = criterion(reconstructed, target) / gradient_accumulation_steps
                scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0).item()
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
                step += 1
                
                batch_time = time.time() - batch_start_time
                
                # Wandb 日志
                if wandb_initialized and step % args.wandb_log_freq == 0:
                    try:
                        wandb.log({
                            'train/loss': loss.item() * gradient_accumulation_steps,
                            'train/step': step,
                            'train/learning_rate': optimizer.param_groups[0]['lr'],
                            'train/grad_norm': grad_norm,
                        })
                    except Exception:
                        pass
                
                if is_main_process(rank):
                    pbar.set_postfix({
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'step': step,
                    })
                
                # 保存 checkpoint
                if is_main_process(rank) and step % args.save_every_steps == 0:
                    step_dir = out_dir / f"step_{step}"
                    step_dir.mkdir(parents=True, exist_ok=True)
                    
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
        
        pons_to_save = pons.module if is_distributed else pons
        torch.save(pons_to_save.state_dict(), final_dir / "pons.pt")
        
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
    
    cleanup_shared_memory_cache()
    cleanup_distributed()


if __name__ == "__main__":
    main()

