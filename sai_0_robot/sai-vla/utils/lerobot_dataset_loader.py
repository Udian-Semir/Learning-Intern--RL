"""
Universal LeRobot Dataset Loader
通用 LeRobot 数据集加载器 - 加载完整数据，供各个模块使用

这个加载器会加载 LeRobot 格式数据集的所有数据：
- VLM hidden states
- Observation states (proprioception)
- Actions
- 其他元数据

各个模块（VLM, Action Heads等）可以从完整数据中提取所需的部分。

重要提示：
- 图像数据在加载时保持原始 uint8 格式（0-255），**不进行归一化处理**
- VLM 模块会根据各自需求进行图像预处理（如 Eagle 会归一化到 [-1, 1]，Qwen 有自己的归一化方式）
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import warnings
import contextlib
from collections import OrderedDict
from multiprocessing import shared_memory
from functools import partial
import av
from PIL import Image
import io
import torch.distributed as dist

# 导入离散化函数
try:
    from .discrete import discrete_constrain_delta, discrete_chunk_calculus, discrete_with_method, DISCRETE_METHODS
except ImportError:
    # 当直接运行此文件时的后备导入
    from discrete import discrete_constrain_delta, discrete_chunk_calculus, discrete_with_method, DISCRETE_METHODS


# ============================================================================
# 欧拉角转轴角函数 (NumPy 版本，用于 Action 预处理)
# ============================================================================

def euler_to_quat_numpy(euler: np.ndarray) -> np.ndarray:
    """
    欧拉角转四元数 (NumPy 批量版本)
    
    Args:
        euler: (..., 3) 欧拉角 (roll, pitch, yaw)
    
    Returns:
        quat: (..., 4) 四元数 (qx, qy, qz, qw)
    """
    roll = euler[..., 0]
    pitch = euler[..., 1]
    yaw = euler[..., 2]
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return np.stack([qx, qy, qz, qw], axis=-1)


def quat_to_axisangle_numpy(quat: np.ndarray) -> np.ndarray:
    """
    四元数转轴角 (NumPy 批量版本)
    
    Args:
        quat: (..., 4) 四元数 (qx, qy, qz, qw)
    
    Returns:
        axis_angle: (..., 3) 轴角 (ax, ay, az)，范围 [-pi, pi]
    """
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qw = np.clip(qw, -1.0, 1.0)
    
    den = np.sqrt(1.0 - qw * qw)
    angle = 2.0 * np.arccos(qw)
    
    # 处理小角度情况
    small_angle_mask = den < 1e-8
    
    axis_angle = np.zeros(quat.shape[:-1] + (3,), dtype=quat.dtype)
    
    # 正常情况
    if np.any(~small_angle_mask):
        scale = np.where(small_angle_mask, 0.0, angle / (den + 1e-8))
        axis_angle[..., 0] = qx * scale
        axis_angle[..., 1] = qy * scale
        axis_angle[..., 2] = qz * scale
    
    return axis_angle


def get_numpy_dtype(dtype_str: str):
    """
    将字符串数据类型转换为 numpy dtype
    
    支持的类型:
    - "float32", "fp32": np.float32
    - "float16", "fp16": np.float16  (减半内存，轻微精度损失)
    - "bfloat16", "bf16": 存储为 uint16，需要特殊处理
    
    注意: numpy 不原生支持 bfloat16，我们使用 float16 作为存储格式
    """
    dtype_map = {
        "float32": np.float32,
        "fp32": np.float32,
        "float16": np.float16,
        "fp16": np.float16,
        "bfloat16": np.float16,  # bfloat16 存储为 float16 以减半内存
        "bf16": np.float16,
    }
    if isinstance(dtype_str, str):
        return dtype_map.get(dtype_str.lower(), np.float32)
    return dtype_str  # 已经是 numpy dtype


class SharedVLMCache:
    """
    跨进程共享的 VLM 缓存
    
    使用 Python multiprocessing.shared_memory 实现多进程（多 GPU）共享缓存。
    
    工作流程：
    1. rank 0 进程创建共享内存并加载所有 VLM hidden states
    2. 其他进程通过共享内存名称附加到同一块内存
    3. 所有进程共享同一份数据，避免重复加载
    
    特性：
    - 支持可变序列长度（使用 max_seq_len 预分配，记录每个样本的实际 seq_len）
    - 返回时只返回实际数据（不包含 padding），保持与原始逻辑一致
    - collate_fn 仍然按 batch 内的 max_seq_len 进行 padding
    - 支持 float16 存储以减半内存占用（推荐用于大规模数据集）
    
    内存优化建议:
    - float32: 完整精度，内存占用最大
    - float16: 内存减半，推荐用于大规模训练
    - 使用 collate_fn 时会转换为训练所需的 dtype (如 bfloat16)
    """
    
    def __init__(self, shm_data: shared_memory.SharedMemory,
                 shm_seq_lens: shared_memory.SharedMemory,
                 num_samples: int, sample_shape: tuple, dtype=np.float32,
                 is_creator: bool = False):
        """内部构造函数，请使用 create() 或 attach() 类方法"""
        self.shm_data = shm_data
        self.shm_seq_lens = shm_seq_lens
        self.name = shm_data.name
        self.seq_lens_name = shm_seq_lens.name
        self.num_samples = num_samples
        self.sample_shape = sample_shape  # (num_layers, max_seq_len, hidden_dim)
        self.dtype = dtype
        self.is_creator = is_creator
        
        # 创建 numpy 数组视图（零拷贝）
        self.data = np.ndarray(
            (num_samples, *sample_shape), 
            dtype=dtype, 
            buffer=self.shm_data.buf
        )
        
        # 存储每个样本的实际 seq_len
        self.seq_lens = np.ndarray(
            (num_samples,),
            dtype=np.int32,
            buffer=self.shm_seq_lens.buf
        )
    
    @classmethod
    def create(cls, num_samples: int, sample_shape: tuple, 
               dtype=np.float32, name_prefix: str = "vlm_cache") -> "SharedVLMCache":
        """
        创建新的共享内存缓存（仅 rank 0 调用）
        
        Args:
            num_samples: 总样本数
            sample_shape: 单个样本形状 (num_layers, max_seq_len, hidden_dim)
            dtype: 数据类型，默认 float32
            name_prefix: 共享内存名称前缀
            
        Returns:
            SharedVLMCache 实例
            
        Raises:
            OSError: 如果共享内存大小超过系统限制
        """
        # 计算数据总大小
        total_elements = num_samples * np.prod(sample_shape)
        data_size = int(total_elements * np.dtype(dtype).itemsize)
        
        # seq_lens 数组大小
        seq_lens_size = num_samples * np.dtype(np.int32).itemsize
        
        total_size = data_size + seq_lens_size
        
        # 检查系统限制
        import shutil
        import resource
        
        print(f"  📊 System resource check:")
        
        # 检查 /dev/shm 可用空间
        try:
            shm_stats = shutil.disk_usage('/dev/shm')
            shm_available = shm_stats.free
            shm_total = shm_stats.total
            
            print(f"     /dev/shm total: {shm_total / 1024**3:.2f} GB")
            print(f"     /dev/shm available: {shm_available / 1024**3:.2f} GB")
            print(f"     Required: {total_size / 1024**3:.2f} GB")
            
            if total_size > shm_available:
                raise OSError(
                    f"共享内存需求 ({total_size / 1024**3:.2f} GB) 超过可用空间 "
                    f"({shm_available / 1024**3:.2f} GB).\n"
                    f"解决方案:\n"
                    f"  1. 增加 /dev/shm 大小: sudo mount -o remount,size=2T /dev/shm\n"
                    f"  2. 使用 mmap 模式 (CACHE_VLM_STATES=true, USE_SHARED_CACHE=false)\n"
                    f"  3. 减少缓存样本数"
                )
            elif total_size > shm_total * 0.8:
                print(f"     ⚠️  Warning: 共享内存需求接近系统限制 ({total_size/shm_total*100:.1f}%)")
        except Exception as e:
            print(f"     ⚠️  Could not check /dev/shm: {e}")
        
        # 检查 ulimit -l (max locked memory)
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_MEMLOCK)
            if soft_limit != resource.RLIM_INFINITY:
                soft_limit_gb = soft_limit / 1024**3
                print(f"     ulimit -l (soft): {soft_limit_gb:.2f} GB")
                if total_size > soft_limit:
                    print(f"     ⚠️  Warning: 需要 {total_size/1024**3:.2f} GB，但 ulimit -l 只有 {soft_limit_gb:.2f} GB")
                    print(f"     解决方案: ulimit -l unlimited 或增加限制")
        except Exception as e:
            print(f"     ⚠️  Could not check ulimit: {e}")
        
        # 创建共享内存
        try:
            shm_data = shared_memory.SharedMemory(create=True, size=data_size)
            shm_seq_lens = shared_memory.SharedMemory(create=True, size=seq_lens_size)
        except OSError as e:
            raise OSError(
                f"创建共享内存失败 (需要 {total_size / 1024**3:.2f} GB): {e}\n"
                f"可能原因:\n"
                f"  1. /dev/shm 空间不足\n"
                f"  2. 系统内存不足\n"
                f"  3. ulimit -l 限制过小\n"
                f"解决方案:\n"
                f"  1. 增加 /dev/shm: sudo mount -o remount,size=2T /dev/shm\n"
                f"  2. 使用 mmap 模式: CACHE_VLM_STATES=true, USE_SHARED_CACHE=false"
            ) from e
        
        print(f"  📦 Created shared memory:")
        print(f"     Data: {shm_data.name} ({data_size / 1024**3:.2f} GB)")
        print(f"     SeqLens: {shm_seq_lens.name} ({seq_lens_size / 1024**2:.2f} MB)")
        print(f"     Shape: ({num_samples}, {sample_shape})")
        
        return cls(shm_data, shm_seq_lens, num_samples, sample_shape, dtype, is_creator=True)
    
    @classmethod
    def attach(cls, name: str, seq_lens_name: str, num_samples: int, sample_shape: tuple, 
               dtype=np.float32) -> "SharedVLMCache":
        """
        附加到现有共享内存（非 rank 0 进程调用）
        
        Args:
            name: 数据共享内存名称
            seq_lens_name: seq_lens 共享内存名称
            num_samples: 总样本数
            sample_shape: 单个样本形状
            dtype: 数据类型
            
        Returns:
            SharedVLMCache 实例
        """
        shm_data = shared_memory.SharedMemory(name=name)
        shm_seq_lens = shared_memory.SharedMemory(name=seq_lens_name)
        return cls(shm_data, shm_seq_lens, num_samples, sample_shape, dtype, is_creator=False)
    
    def load_sample(self, index: int, data: np.ndarray):
        """
        加载单个样本到共享内存（仅 rank 0 调用）
        
        Args:
            index: 样本索引
            data: 原始数据 (num_layers, actual_seq_len, hidden_dim)
        """
        actual_seq_len = data.shape[1]  # 实际序列长度
        self.seq_lens[index] = actual_seq_len
        
        if data.shape != self.sample_shape:
            # 需要 padding 到 max_seq_len
            self.data[index] = 0  # 先清零
            self.data[index, :, :actual_seq_len, :] = data
        else:
            self.data[index] = data
    
    def get_sample(self, index: int) -> np.ndarray:
        """
        获取单个样本（所有进程可调用）
        
        返回实际数据（不包含 padding），保持与原始逻辑一致
        collate_fn 仍然按 batch 内的 max_seq_len 进行 padding
        
        注意：返回的是数据的拷贝而不是视图，以确保在多进程 DataLoader 中的安全性。
        torch.from_numpy 会与原 numpy 数组共享内存，如果返回的是视图，
        在跨进程传输时可能导致数据错误。
        """
        actual_seq_len = self.seq_lens[index]
        # 返回拷贝而不是视图，确保多进程安全
        return self.data[index, :, :actual_seq_len, :].copy()
    
    def close(self):
        """关闭共享内存连接"""
        self.shm_data.close()
        self.shm_seq_lens.close()
    
    def unlink(self):
        """删除共享内存（仅创建者在程序结束时调用）"""
        if self.is_creator:
            try:
                self.shm_data.unlink()
            except FileNotFoundError:
                pass
            try:
                self.shm_seq_lens.unlink()
            except FileNotFoundError:
                pass
    
    def __del__(self):
        """析构时自动关闭"""
        try:
            self.close()
        except:
            pass


def preload_vlm_cache_distributed(
    dataset_path: str,
    num_samples: int,
    sample_shape: tuple = None,
    rank: int = 0,
    world_size: int = 1,
    dtype=np.float32,
    verbose: bool = True,
    auto_detect_shape: bool = True,
    cache_dtype: str = None,
) -> SharedVLMCache:
    """
    分布式环境下预加载 VLM 缓存到共享内存
    
    Args:
        dataset_path: 数据集路径
        num_samples: 总样本数
        sample_shape: 样本形状 (num_layers, seq_len, hidden_dim)
                     如果 auto_detect_shape=True，可以为 None 或只提供 (num_layers, hidden_dim)
        rank: 当前进程 rank
        world_size: 总进程数
        dtype: 数据类型 (可以是 np.dtype 或字符串)
        verbose: 是否打印进度
        auto_detect_shape: 是否自动检测最大 seq_len（推荐 True）
        cache_dtype: 缓存存储类型 ("float32", "float16", "bfloat16")
                    - float16/bfloat16: 内存减半，推荐用于大规模数据
                    - 如果不指定，使用 dtype 参数
        
    Returns:
        SharedVLMCache 实例（所有进程共享同一块内存）
        
    Note:
        当 seq_len 变化时，使用最大 seq_len 预分配，较短的样本自动 padding
        
    内存优化:
        使用 cache_dtype="float16" 可以将共享内存占用减半:
        - 100k 样本, shape (1, 512, 2048): float32=400GB, float16=200GB
        
    示例:
        # 使用 float16 缓存以减半内存
        cache = preload_vlm_cache_distributed(
            dataset_path="/path/to/data",
            num_samples=100000,
            cache_dtype="float16",  # 减半内存
        )
    """
    # 处理 cache_dtype 参数
    if cache_dtype is not None:
        dtype = get_numpy_dtype(cache_dtype)
    elif isinstance(dtype, str):
        dtype = get_numpy_dtype(dtype)
    dataset_path = Path(dataset_path)
    
    # 所有 rank 先同步，确保都进入这个函数
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        if verbose and rank == 0:
            print(f"[Rank {rank}] Starting VLM cache preload...")
    
    # rank 0 加载数据，其他 rank 等待
    cache = None
    cache_name = None
    seq_lens_name = None
    final_shape = None
    
    if rank == 0:
        from tqdm import tqdm
        
        try:
            # ========== 自动检测形状 ==========
            if auto_detect_shape:
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"📏 Scanning VLM files to detect max sequence length...")
                    print(f"{'='*60}")
                
                max_seq_len = 0
                num_layers = None
                hidden_dim = None
                
                vlm_dir = dataset_path / "vlm_hidden_states"
                _has_chunk_npz = any(f.name.startswith("chunk-") and f.suffix == ".npz" for f in vlm_dir.iterdir()) if vlm_dir.exists() else False
                _has_per_ep_npy = (vlm_dir / "episode_000000.npy").exists() if vlm_dir.exists() else False
                if not _has_per_ep_npy and vlm_dir.exists():
                    _has_per_ep_npy = any(f.name.startswith("episode_") and f.suffix == ".npy" for f in vlm_dir.iterdir())
                _per_episode = _has_chunk_npz or _has_per_ep_npy
                
                def _scan_episode_shape(ep_data):
                    nonlocal num_layers, hidden_dim, max_seq_len
                    if ep_data.ndim == 4:
                        _, nl, sl, hd = ep_data.shape
                        num_layers, hidden_dim = nl, hd
                    elif ep_data.ndim == 3:
                        _, sl, hd = ep_data.shape
                        num_layers, hidden_dim = 1, hd
                    else:
                        return
                    max_seq_len = max(max_seq_len, sl)

                if _has_chunk_npz:
                    if verbose:
                        print(f"  检测到 chunk-npz VLM 格式")
                    npz_files = sorted(vlm_dir.glob("chunk-*.npz"))
                    for npz_file in (tqdm(npz_files, desc="Scanning chunk-npz shapes") if verbose else npz_files):
                        with np.load(npz_file, allow_pickle=False) as data:
                            for key in data.files:
                                _scan_episode_shape(data[key])
                elif _has_per_ep_npy:
                    if verbose:
                        print(f"  检测到 per-episode VLM 格式")
                    ep_files = sorted(vlm_dir.glob("episode_*.npy"))
                    for ep_file in (tqdm(ep_files, desc="Scanning episode shapes") if verbose else ep_files):
                        ep_data = np.load(ep_file, mmap_mode='r')
                        _scan_episode_shape(ep_data)
                else:
                    if verbose:
                        print(f"  检测到 per-frame VLM 格式")
                    scan_iterator = tqdm(range(num_samples), desc="Scanning shapes") if verbose else range(num_samples)
                    for i in scan_iterator:
                        vlm_path = vlm_dir / f"hidden_state_{i:06d}.npy"
                        vlm_state = np.load(vlm_path, mmap_mode='r')
                        if vlm_state.ndim == 2:
                            seq_len, hidden_dim = vlm_state.shape
                            num_layers = 1
                        else:
                            num_layers, seq_len, hidden_dim = vlm_state.shape
                        max_seq_len = max(max_seq_len, seq_len)
                
                sample_shape = (num_layers, max_seq_len, hidden_dim)
                
                if verbose:
                    print(f"✓ Detected shape: num_layers={num_layers}, max_seq_len={max_seq_len}, hidden_dim={hidden_dim}")
                    print(f"  Final shape: {sample_shape}")
            else:
                # 如果没有自动检测，sample_shape 应该已经提供
                if sample_shape is None:
                    raise ValueError("sample_shape must be provided when auto_detect_shape=False")
            
            # ========== 创建共享内存并加载数据 ==========
            if verbose:
                print(f"\n{'='*60}")
                print(f"🚀 Preloading VLM cache to shared memory...")
                print(f"{'='*60}")
            
            cache = SharedVLMCache.create(num_samples, sample_shape, dtype)
            
            # 加载所有样本
            def _load_ep_frames_to_cache(ep_data, start_idx):
                """将一个 episode 的帧加载到共享内存缓存，返回下一个 global_idx。"""
                idx = start_idx
                for frame_i in range(ep_data.shape[0]):
                    if idx >= num_samples:
                        break
                    vlm_state = ep_data[frame_i]
                    if vlm_state.ndim == 2:
                        vlm_state = vlm_state.reshape(1, vlm_state.shape[0], vlm_state.shape[1])
                    cache.load_sample(idx, vlm_state.astype(dtype))
                    idx += 1
                return idx

            if _has_chunk_npz:
                global_idx = 0
                npz_files = sorted(vlm_dir.glob("chunk-*.npz"))
                npz_iter = tqdm(npz_files, desc="Loading VLM states (chunk-npz)") if verbose else npz_files
                for npz_file in npz_iter:
                    with np.load(npz_file, allow_pickle=False) as data:
                        for key in sorted(data.files):
                            if global_idx >= num_samples:
                                break
                            global_idx = _load_ep_frames_to_cache(data[key], global_idx)
                    if global_idx >= num_samples:
                        break
            elif _has_per_ep_npy:
                global_idx = 0
                ep_files = sorted(vlm_dir.glob("episode_*.npy"))
                ep_iter = tqdm(ep_files, desc="Loading VLM states (per-episode)") if verbose else ep_files
                for ep_file in ep_iter:
                    ep_data = np.load(ep_file)
                    global_idx = _load_ep_frames_to_cache(ep_data, global_idx)
                    if global_idx >= num_samples:
                        break
            else:
                iterator = tqdm(range(num_samples), desc="Loading VLM states") if verbose else range(num_samples)
                for i in iterator:
                    vlm_path = vlm_dir / f"hidden_state_{i:06d}.npy"
                    vlm_state = np.load(vlm_path)
                    if vlm_state.ndim == 2:
                        vlm_state = vlm_state.reshape(1, vlm_state.shape[0], vlm_state.shape[1])
                    cache.load_sample(i, vlm_state.astype(dtype))
            
            if verbose:
                print(f"✅ VLM cache preloaded: {num_samples} samples")
                print(f"   Data shared memory: {cache.name}")
                print(f"   SeqLens shared memory: {cache.seq_lens_name}")
                print(f"   Max shape per sample: {sample_shape}")
                print(f"   Total size: {num_samples * np.prod(sample_shape) * np.dtype(dtype).itemsize / 1024**3:.2f} GB")
            
            # 广播共享内存名称和形状给其他进程
            cache_name = cache.name
            seq_lens_name = cache.seq_lens_name
            final_shape = sample_shape
            
            if verbose:
                print(f"[Rank {rank}] Cache loading completed, ready to broadcast...")
        except Exception as e:
            # 如果加载失败，清理已创建的共享内存
            if cache is not None:
                try:
                    cache.close()
                    cache.unlink()
                except:
                    pass
            raise  # 重新抛出异常
    else:
        # 非 rank 0 进程：等待 rank 0 完成加载
        if verbose:
            print(f"[Rank {rank}] Waiting for rank 0 to complete cache loading...")
    
    # 关键同步点：等待 rank 0 完成加载
    # 所有 rank 必须在这里同步，然后才能进行广播
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        if verbose and rank == 0:
            print(f"[Rank {rank}] All ranks synchronized, starting broadcast...")
        
        # 获取当前设备（NCCL 后端需要 CUDA tensor）
        # 使用 local_rank 对应的设备（如果可用）
        if torch.cuda.is_available():
            try:
                local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
                device = torch.device(f"cuda:{local_rank}")
            except:
                device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cpu")
        
        # 广播共享内存名称（数据和 seq_lens 两个）
        if rank == 0:
            name_tensor = torch.tensor([ord(c) for c in cache_name] + [0] * (256 - len(cache_name)), 
                                       dtype=torch.int32, device=device)
            seq_lens_name_tensor = torch.tensor([ord(c) for c in seq_lens_name] + [0] * (256 - len(seq_lens_name)), 
                                                dtype=torch.int32, device=device)
            shape_tensor = torch.tensor(list(final_shape), dtype=torch.int64, device=device)
        else:
            name_tensor = torch.zeros(256, dtype=torch.int32, device=device)
            seq_lens_name_tensor = torch.zeros(256, dtype=torch.int32, device=device)
            shape_tensor = torch.zeros(3, dtype=torch.int64, device=device)
        
        # 执行广播（所有 rank 必须同时执行）
        try:
            if verbose:
                print(f"[Rank {rank}] Starting broadcast operations...")
            dist.broadcast(name_tensor, src=0)
            dist.broadcast(seq_lens_name_tensor, src=0)
            dist.broadcast(shape_tensor, src=0)
            if verbose:
                print(f"[Rank {rank}] Broadcast operations completed")
        except Exception as e:
            print(f"[Rank {rank}] ERROR in broadcast: {e}")
            raise
        
        # 再次同步，确保广播完成
        dist.barrier()
        
        if rank != 0:
            # 解码名称和形状（移回 CPU）
            cache_name = ''.join(chr(c) for c in name_tensor.cpu().tolist() if c != 0)
            seq_lens_name = ''.join(chr(c) for c in seq_lens_name_tensor.cpu().tolist() if c != 0)
            final_shape = tuple(shape_tensor.cpu().tolist())
            # 附加到共享内存
            try:
                cache = SharedVLMCache.attach(cache_name, seq_lens_name, num_samples, final_shape, dtype)
                if verbose:
                    print(f"  [Rank {rank}] Attached to shared memory: {cache_name}, shape: {final_shape}")
            except Exception as e:
                print(f"[Rank {rank}] ERROR attaching to shared memory: {e}")
                raise
    
    return cache


def get_total_vlm_states(dataset_path: str) -> int:
    """
    获取数据集中所有 VLM hidden states 的总数（即所有轨迹的帧数总和）
    
    Args:
        dataset_path: 数据集路径
        
    Returns:
        total_frames: 总帧数（即 VLM hidden states 数量）
    """
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    return info['total_frames']


class LeRobotDataset(Dataset):
    """
    通用 LeRobot 数据集加载器
    
    加载完整的数据，包括：
    - Images (多相机图像)
    - VLM hidden states (所有层)
    - Observation states (proprioception)
    - Actions (支持 action chunking)
    - Episode 和 frame 信息
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        num_action_chunks: int = 25,
        enable_chunking: bool = True,
        episode_indices: Optional[List[int]] = None,
        cache_vlm_states: bool = False,
        cache_max_samples: int = -1,
        verbose: bool = True,
        max_cached_video_readers: int = 32,
        skip_images: bool = False,
        vlm_cache_dtype: str = "float32",
        shared_vlm_cache: Optional["SharedVLMCache"] = None,
        # 离散化参数 (10个参数)
        discrete_actions: bool = False,
        discrete_columns: Optional[List[int]] = None,
        discrete_deltas: Optional[List[float]] = None,
        discrete_method: str = "constrain_delta",
        discrete_beta: float = 0.6,
        discrete_alpha: float = 0.4,
        undiscrete_actions: bool = False,
        undiscrete_columns: Optional[List[int]] = None,
        undiscrete_deltas: Optional[List[float]] = None,
        # State 预处理参数
        state_process_order: Optional[List[str]] = None,
        hand_binary_columns: Optional[List[int]] = None,
        hand_binary_threshold: float = 442.0,
    ):
        """
        初始化数据集加载器
        
        Args:
            dataset_path: 数据集根目录路径
            split: 数据集划分 ('train', 'test', 'val')
            num_action_chunks: action chunk 数量（用于预测未来多步动作）
            enable_chunking: 是否启用 action chunking
            episode_indices: 特定 episode 索引列表，None 则加载所有
            cache_vlm_states: 是否缓存 VLM states 到内存 (使用 mmap 模式)
            cache_max_samples: 最大缓存样本数 (-1 表示缓存所有样本) [已弃用]
            verbose: 是否打印详细信息
            max_cached_video_readers: 缓存的视频 reader 数量上限，避免打开过多文件
            skip_images: 是否跳过加载图像 (仅使用预保存的 VLM hidden states 训练时设为 True)
            vlm_cache_dtype: VLM 数据类型，用于 collate_fn 中的转换
            shared_vlm_cache: 共享内存 VLM 缓存实例（多 GPU 共享）
                            如果提供，将使用共享内存缓存而非 mmap
                            使用 preload_vlm_cache_distributed() 创建
            
            离散化参数 (用于 ParaCAT 训练):
            discrete_actions: 是否启用离散化
            discrete_columns: 参与离散化的 action 列索引列表，从0开始
                              例如: [0, 1, 2] 表示对第0、1、2列进行离散化
            discrete_deltas: 对应列的离散化步长列表
                             例如: [0.01, 0.02, 0.01] 表示第0列delta=0.01，第1列delta=0.02...
                             长度必须与 discrete_columns 相同
            discrete_method: 离散化方法，可选:
                             - "constrain_delta": 简单累积误差方法 (默认)
                             - "chunk_calculus": 基于微积分的方法，带趋势预测
            discrete_beta: 趋势项权重 (仅 chunk_calculus 方法使用，默认 0.6)
            discrete_alpha: 趋势平滑系数 (仅 chunk_calculus 方法使用，默认 0.4)
            undiscrete_actions: 是否启用反离散化 (用于推理时，此处仅存储配置)
            undiscrete_columns: 参与反离散化的 action 列索引列表
            
            State 预处理参数 (按 state_process_order 顺序执行，索引基于原始 state):
            state_process_order: 预处理执行顺序列表，如 ["hand_binary"]
            hand_binary_columns: 原始 state 中手部数据列范围 [start, end)
            hand_binary_threshold: 手部二值化阈值，默认 442.0
            undiscrete_deltas: 对应列的反离散化步长列表
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.num_action_chunks = num_action_chunks
        self.enable_chunking = enable_chunking
        self.cache_vlm_states = cache_vlm_states
        self.cache_max_samples = cache_max_samples
        self.verbose = verbose
        self.max_video_readers = max(1, max_cached_video_readers)
        self.chunk_horizon = self.num_action_chunks if self.enable_chunking else 1
        self.skip_images = skip_images
        self.vlm_cache_dtype = vlm_cache_dtype
        self._vlm_cache_torch_dtype = torch.float32
        
        # 共享内存缓存（多 GPU 共享，优先级最高）
        self.shared_vlm_cache = shared_vlm_cache
        
        # Action 离散化参数 (用于 ParaCAT 训练)
        self.discrete_actions = discrete_actions
        self.discrete_columns = discrete_columns if discrete_columns is not None else []
        self.discrete_deltas = discrete_deltas if discrete_deltas is not None else []
        self.discrete_method = discrete_method
        self.discrete_beta = discrete_beta
        self.discrete_alpha = discrete_alpha
        
        # 反离散化参数 (用于推理时，此处仅存储配置，不在 __getitem__ 中应用)
        self.undiscrete_actions = undiscrete_actions
        self.undiscrete_columns = undiscrete_columns if undiscrete_columns is not None else []
        self.undiscrete_deltas = undiscrete_deltas if undiscrete_deltas is not None else []
        
        # State 预处理参数 (按顺序执行，索引基于原始 state)
        self.state_process_order = state_process_order if state_process_order is not None else []
        self.hand_binary_columns = hand_binary_columns  # [start, end)
        self.hand_binary_threshold = hand_binary_threshold

        # parquet / episode mmap 缓存 LRU 上限 (0 / None = 不限)
        # disk-shuffle 模式下一个 worker 会接触整个数据集的所有 parquet 文件,
        # 不设上限会缓慢吞掉 RAM。常规 SHM 模式 episode_indices 已被 set_active_episodes
        # 限制到单段 (千级别 ep), 上限为 0 不会触发驱逐, 行为与旧版完全一致。
        self.parquet_cache_max = 0
        self.episode_mmap_cache_max = 0
        self.chunk_npz_cache_max = 0
        
        # 验证离散化参数
        if self.discrete_actions:
            if len(self.discrete_columns) != len(self.discrete_deltas):
                raise ValueError(
                    f"discrete_columns 和 discrete_deltas 长度必须相同，"
                    f"当前: columns={len(self.discrete_columns)}, deltas={len(self.discrete_deltas)}"
                )
            if len(self.discrete_columns) == 0:
                raise ValueError("启用 discrete_actions 时，必须指定 discrete_columns 和 discrete_deltas")
        
        # 验证 State 预处理参数
        valid_processors = ["hand_binary"]
        for processor in self.state_process_order:
            if processor not in valid_processors:
                raise ValueError(
                    f"未知的 state 预处理器: {processor}. "
                    f"可用: {valid_processors}"
                )
        
        if self.hand_binary_columns is not None and len(self.hand_binary_columns) > 0:
            if len(self.hand_binary_columns) % 2 != 0:
                raise ValueError(
                    f"hand_binary_columns 长度必须是 2 的倍数 (每组 [start, end))，"
                    f"当前长度: {len(self.hand_binary_columns)}"
                )
        
        if self.discrete_actions:
            if self.discrete_method not in DISCRETE_METHODS:
                raise ValueError(
                    f"未知的离散化方法: {self.discrete_method}. "
                    f"可用方法: {DISCRETE_METHODS}"
                )
        
        # 加载元信息
        self._load_metadata()
        
        # 设置缓存
        # 缓存优先级：shared_vlm_cache > mmap (cache_vlm_states=True) > 无缓存
        # 用 OrderedDict 方便后续按 set_cache_limits() 启用 LRU 驱逐
        self.parquet_cache: "OrderedDict[str, Any]" = OrderedDict()
        self.video_readers: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()  # LRU 缓存视频 reader
        
        # 加载 episodes
        self._load_episodes(episode_indices)
        
        # 获取图像特征名称
        self.image_keys = [key for key in self.info['features'].keys() 
                  if key.startswith('observation.images.')]
        
        # 构建索引
        self._build_index()
        
        # 自动检测 VLM hidden state 存储格式: chunk_npz / per_episode / per_frame
        self._vlm_format = self._detect_vlm_storage_format()
        self._vlm_per_episode = self._vlm_format in ("per_episode", "chunk_npz")
        self._episode_mmap_cache: "OrderedDict[int, Any]" = OrderedDict()
        self._chunk_npz_cache: "OrderedDict[int, Any]" = OrderedDict()
        
        if self.verbose:
            self._print_dataset_info()
    
    def _load_metadata(self):
        """加载数据集元信息"""
        info_path = self.dataset_path / "meta" / "info.json"
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        # 提取关键信息
        self.action_dim = self.info['features']['action']['shape'][0]
        self.state_dim = self.info['features']['observation.state']['shape'][0]
        self.fps = self.info['fps']
        self.total_episodes = self.info['total_episodes']
        self.total_frames = self.info['total_frames']
        self.chunks_size = self.info['chunks_size']
        self.total_tasks = self.info.get('total_tasks', 0)
        
        # 加载 task descriptions (如果存在)
        self.task_descriptions = self._load_task_descriptions()
    
    def _load_task_descriptions(self) -> Dict[int, str]:
        """加载 task descriptions 映射"""
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            return {}
        
        task_map = {}
        with open(tasks_path, 'r') as f:
            for line in f:
                task = json.loads(line.strip())
                task_map[task['task_index']] = task['task']
        
        return task_map
    
    def _load_episodes(self, episode_indices: Optional[List[int]] = None):
        """加载 episode 信息"""
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        self.episodes = []
        
        with open(episodes_path, 'r') as f:
            for line in f:
                episode = json.loads(line.strip())
                if episode_indices is None or episode['episode_index'] in episode_indices:
                    self.episodes.append(episode)
        
        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found for split '{self.split}'")

    def _get_video_reader(self, video_path: Path):
        """按视频文件复用 reader，并限制并发打开的视频数量"""
        video_path_str = str(video_path)
        reader = self.video_readers.get(video_path_str)

        if reader is not None and reader.get("container") is not None:
            # 命中缓存，将其移动到队尾，表示最近使用
            self.video_readers.move_to_end(video_path_str)
            return reader

        # 缓存未命中或容器无效，重新打开
        container = av.open(video_path_str)
        stream = container.streams.video[0]
        reader = {"container": container, "stream": stream}
        self.video_readers[video_path_str] = reader
        self.video_readers.move_to_end(video_path_str)
        self._enforce_video_reader_limit()

        return reader

    def _enforce_video_reader_limit(self) -> None:
        """确保缓存的 video reader 数量不超过上限"""
        while len(self.video_readers) > self.max_video_readers:
            video_path_str, reader = self.video_readers.popitem(last=False)
            container = reader.get("container")
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()

    def close_video_readers(self):
        """关闭并清理所有缓存的 video reader"""
        for reader in self.video_readers.values():
            container = reader.get("container")
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
        self.video_readers.clear()

    def __del__(self):
        # 避免对象被回收时仍有打开的文件句柄
        self.close_video_readers()

    def _episode_parquet_path(self, episode_idx: int) -> Path:
        """根据 episode index 计算对应的 parquet 路径"""
        chunk_idx = episode_idx // self.chunks_size
        return self.dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    
    def _build_index(self):
        """构建数据索引
        
        当 chunk_horizon 大于 episode 剩余帧数时，__getitem__ 会自动零填充，
        因此每一帧都可以作为起始帧。
        """
        self.index_map = []
        
        horizon = self.chunk_horizon
        if horizon < 1:
            raise ValueError("num_action_chunks must be >= 1 when chunking is enabled")

        for episode in self.episodes:
            episode_idx = episode['episode_index']

            parquet_path = self._episode_parquet_path(episode_idx)

            df = pd.read_parquet(parquet_path)
            episode_length = len(df)
            if episode_length == 0:
                continue

            max_start = episode_length - 1
            frame_idx = 0
            done_column = df['next.done'] if 'next.done' in df.columns else None

            while frame_idx <= max_start:
                self.index_map.append((episode_idx, frame_idx, str(parquet_path)))

                if self.enable_chunking and done_column is not None:
                    actual_last = frame_idx + horizon - 1
                    if actual_last < episode_length:
                        # chunk 完全在 episode 内，检查终止标志
                        if bool(done_column.iloc[actual_last]):
                            break
                    # chunk 超出 episode（需要零填充），不做 done 检查，继续索引后续帧

                frame_idx += 1
    
    def _load_parquet(self, parquet_path: str) -> pd.DataFrame:
        """加载并缓存 parquet 文件 (可选 LRU 驱逐, 见 set_cache_limits)"""
        df = self.parquet_cache.get(parquet_path)
        if df is not None:
            if self.parquet_cache_max > 0:
                self.parquet_cache.move_to_end(parquet_path)
            return df
        df = pd.read_parquet(parquet_path)
        self.parquet_cache[parquet_path] = df
        if self.parquet_cache_max > 0:
            while len(self.parquet_cache) > self.parquet_cache_max:
                self.parquet_cache.popitem(last=False)
        return df

    def set_cache_limits(
        self,
        parquet_cache_max: int = 0,
        episode_mmap_cache_max: int = 0,
        chunk_npz_cache_max: int = 0,
    ) -> None:
        """
        启用各内置 RAM 缓存的 LRU 驱逐 (0 = 不限, 与旧行为一致)。

        disk-shuffle 模式下推荐:
            parquet_cache_max = 数千
            episode_mmap_cache_max = 数百 ~ 数千
            chunk_npz_cache_max = 0 (chunk_npz_cache 仅存 NpzFile 句柄, 占用极小)
        """
        self.parquet_cache_max = max(0, int(parquet_cache_max))
        self.episode_mmap_cache_max = max(0, int(episode_mmap_cache_max))
        self.chunk_npz_cache_max = max(0, int(chunk_npz_cache_max))
        if self.parquet_cache_max > 0:
            while len(self.parquet_cache) > self.parquet_cache_max:
                self.parquet_cache.popitem(last=False)
        if self.episode_mmap_cache_max > 0:
            while len(self._episode_mmap_cache) > self.episode_mmap_cache_max:
                self._episode_mmap_cache.popitem(last=False)
        if self.chunk_npz_cache_max > 0:
            while len(self._chunk_npz_cache) > self.chunk_npz_cache_max:
                _k, npz_obj = self._chunk_npz_cache.popitem(last=False)
                try:
                    npz_obj.close()
                except Exception:
                    pass
    
    def _apply_state_preprocessing(self, observation_state: np.ndarray) -> np.ndarray:
        """
        按配置顺序应用 state 预处理，自动处理索引偏移
        
        所有索引配置都基于原始 state，处理时自动计算偏移
        
        Args:
            observation_state: 原始 state 数组，shape (state_dim,)
            
        Returns:
            处理后的 state 数组
        """
        if not self.state_process_order:
            return observation_state
        
        # 记录累计索引偏移 (用于调整后续处理的索引)
        index_offset = 0
        
        for processor_name in self.state_process_order:
            if processor_name == "hand_binary" and self.hand_binary_columns is not None:
                # 支持多组手部数据，每组 [start, end)
                # 例如: [6, 12, 18, 24] 表示左手 6-12，右手 18-24
                # 按顺序处理，每组处理后会影响后续组的索引
                hand_offset = 0  # 手部处理内部的偏移累计
                
                for group_idx in range(0, len(self.hand_binary_columns), 2):
                    if group_idx + 1 < len(self.hand_binary_columns):
                        start = self.hand_binary_columns[group_idx]
                        end = self.hand_binary_columns[group_idx + 1]
                        
                        # 应用总偏移 (index_offset + 之前手部组的偏移)
                        adj_start = start + index_offset + hand_offset
                        adj_end = end + index_offset + hand_offset
                        
                        if adj_start < len(observation_state) and adj_end <= len(observation_state):
                            # 处理: 6维 -> 1维
                            hand_data = observation_state[adj_start:adj_end]
                            hand_binary = np.array([1.0 if np.mean(hand_data) > self.hand_binary_threshold else -1.0], 
                                                  dtype=np.float32)
                            # 重构 state
                            observation_state = np.concatenate([
                                observation_state[:adj_start],
                                hand_binary,
                                observation_state[adj_end:]
                            ])
                            # 更新手部内部偏移
                            hand_offset -= (end - start - 1)
                
                # 更新总偏移
                index_offset += hand_offset
        
        return observation_state
    
    def _detect_vlm_storage_format(self) -> str:
        """
        自动检测 VLM hidden state 存储格式。
        
        Returns:
            "chunk_npz" = chunk 打包格式 (chunk-XXX.npz)
            "per_episode" = 单文件 per-episode 格式 (episode_XXXXXX.npy)
            "per_frame" = 单文件 per-frame 格式 (hidden_state_XXXXXX.npy)
        """
        vlm_dir = self.dataset_path / "vlm_hidden_states"
        if not vlm_dir.exists():
            return "per_frame"
        
        chunk_npz = vlm_dir / "chunk-000.npz"
        if chunk_npz.exists():
            if self.verbose:
                print(f"  检测到 chunk-npz VLM 格式: {vlm_dir}")
            return "chunk_npz"
        
        for f in vlm_dir.iterdir():
            if f.name.startswith("chunk-") and f.suffix == ".npz":
                if self.verbose:
                    print(f"  检测到 chunk-npz VLM 格式: {vlm_dir}")
                return "chunk_npz"
        
        episode_file = vlm_dir / "episode_000000.npy"
        if episode_file.exists():
            if self.verbose:
                print(f"  检测到 per-episode VLM 格式: {vlm_dir}")
            return "per_episode"
        
        frame_file = vlm_dir / "hidden_state_000000.npy"
        if frame_file.exists():
            if self.verbose:
                print(f"  检测到 per-frame VLM 格式: {vlm_dir}")
            return "per_frame"
        
        for f in vlm_dir.iterdir():
            if f.name.startswith("episode_") and f.suffix == ".npy":
                if self.verbose:
                    print(f"  检测到 per-episode VLM 格式: {vlm_dir}")
                return "per_episode"
        
        return "per_frame"

    def _load_vlm_hidden_state(self, vlm_index: int, episode_index: int = None, frame_index: int = None):
        """
        加载 VLM hidden states
        自动检测格式：支持 per-episode 和 per-frame 两种存储方式
        
        Args:
            vlm_index: VLM 全局索引 (per-frame 模式使用)
            episode_index: episode 索引 (per-episode 模式使用)
            frame_index: episode 内帧索引 (per-episode 模式使用)
            
        Returns:
            numpy array of shape (num_layers, seq_len, hidden_dim)
            
        缓存模式优先级：
            1. shared_vlm_cache: 多 GPU 共享内存缓存（最快，推荐）
            2. mmap: 操作系统页缓存（较快，自动共享）
            3. 无缓存: 每次从磁盘读取（最慢）
        """
        # 优先级 1：使用共享内存缓存（多 GPU 共享，rank 0 预加载）
        if self.shared_vlm_cache is not None:
            if vlm_index < self.shared_vlm_cache.num_samples:
                vlm_state = self.shared_vlm_cache.get_sample(vlm_index)
                if self.verbose and vlm_index == 0 and not hasattr(self, '_shared_cache_logged'):
                    print(f"  🚀 Using shared memory cache (multi-GPU shared, zero-copy)")
                    self._shared_cache_logged = True
                return vlm_state
            else:
                if self.verbose and not hasattr(self, '_cache_out_of_range_logged'):
                    print(f"  ⚠️ VLM index {vlm_index} out of cache range ({self.shared_vlm_cache.num_samples}), falling back to mmap/disk")
                    self._cache_out_of_range_logged = True
        
        # per-episode 格式
        if self._vlm_per_episode and episode_index is not None and frame_index is not None:
            return self._load_vlm_from_episode(episode_index, frame_index)
        
        # per-frame 格式 (旧格式兼容)
        vlm_path = self.dataset_path / "vlm_hidden_states" / f"hidden_state_{vlm_index:06d}.npy"
        
        if self.cache_vlm_states:
            vlm_state = np.load(vlm_path, mmap_mode='r')
            if self.verbose and vlm_index == 0 and not hasattr(self, '_mmap_logged'):
                print(f"  📦 Using mmap mode for VLM cache (OS page cache, per-frame)")
                self._mmap_logged = True
        else:
            with open(vlm_path, "rb") as f:
                vlm_state = np.load(f, allow_pickle=True)
        
        # 自动适配单层和多层格式
        if vlm_state.ndim == 2:
            vlm_state = vlm_state.reshape(1, vlm_state.shape[0], vlm_state.shape[1])
        
        return vlm_state

    def _load_vlm_from_episode(self, episode_index: int, frame_index: int):
        """
        从 per-episode 格式读取单帧 hidden state。
        支持三种后端:
          1. chunk-npz: chunk-XXX.npz 内含多个 episode
          2. per-episode .npy: episode_XXXXXX.npy (旧格式, mmap)
          3. 回退: 旧 npy 不存在时尝试 chunk npz
        
        Args:
            episode_index: episode 索引
            frame_index: episode 内帧索引 (0-based)
            
        Returns:
            numpy array of shape (num_layers, seq_len, hidden_dim)
        """
        if episode_index not in self._episode_mmap_cache:
            vlm_dir = self.dataset_path / "vlm_hidden_states"
            loaded = False

            if self._vlm_format == "chunk_npz":
                loaded = self._load_episode_from_chunk_npz(episode_index, vlm_dir)

            if not loaded:
                ep_path = vlm_dir / f"episode_{episode_index:06d}.npy"
                if ep_path.exists():
                    self._episode_mmap_cache[episode_index] = np.load(ep_path, mmap_mode='r')
                    if self.episode_mmap_cache_max > 0:
                        while len(self._episode_mmap_cache) > self.episode_mmap_cache_max:
                            self._episode_mmap_cache.popitem(last=False)
                    loaded = True
                    if self.verbose and not hasattr(self, '_ep_mmap_logged'):
                        arr = self._episode_mmap_cache[episode_index]
                        print(f"  📦 Using mmap mode for VLM cache (per-episode npy, shape={arr.shape})")
                        self._ep_mmap_logged = True

            if not loaded:
                loaded = self._load_episode_from_chunk_npz(episode_index, vlm_dir)

            if not loaded:
                raise FileNotFoundError(
                    f"Episode {episode_index} VLM 文件不存在: "
                    f"未找到 chunk-npz 或 episode_{episode_index:06d}.npy"
                )
        elif self.episode_mmap_cache_max > 0:
            self._episode_mmap_cache.move_to_end(episode_index)
        
        ep_data = self._episode_mmap_cache[episode_index]
        
        if frame_index >= ep_data.shape[0]:
            raise IndexError(
                f"frame_index={frame_index} 超出 episode {episode_index} "
                f"的帧数={ep_data.shape[0]}"
            )
        
        vlm_state = ep_data[frame_index]
        
        if vlm_state.ndim == 2:
            vlm_state = vlm_state.reshape(1, vlm_state.shape[0], vlm_state.shape[1])
        
        return vlm_state

    def _load_episode_from_chunk_npz(self, episode_index: int, vlm_dir) -> bool:
        """从 chunk npz 文件中加载一个 episode 到缓存 (支持 LRU)。返回是否成功。"""
        chunks_size = self.info.get("chunks_size", 1000)
        chunk_idx = episode_index // chunks_size
        npz_path = vlm_dir / f"chunk-{chunk_idx:03d}.npz"
        if not npz_path.exists():
            return False

        npz_data = self._chunk_npz_cache.get(chunk_idx)
        if npz_data is None:
            npz_data = np.load(npz_path, allow_pickle=False)
            self._chunk_npz_cache[chunk_idx] = npz_data
            if self.verbose and not hasattr(self, '_chunk_npz_logged'):
                print(f"  📦 Using chunk-npz VLM cache: {npz_path.name}")
                self._chunk_npz_logged = True
            if self.chunk_npz_cache_max > 0:
                while len(self._chunk_npz_cache) > self.chunk_npz_cache_max:
                    _k, old = self._chunk_npz_cache.popitem(last=False)
                    try:
                        old.close()
                    except Exception:
                        pass
        elif self.chunk_npz_cache_max > 0:
            self._chunk_npz_cache.move_to_end(chunk_idx)

        ep_key = f"episode_{episode_index:06d}"
        if ep_key not in npz_data:
            return False

        self._episode_mmap_cache[episode_index] = npz_data[ep_key]
        if self.episode_mmap_cache_max > 0:
            while len(self._episode_mmap_cache) > self.episode_mmap_cache_max:
                self._episode_mmap_cache.popitem(last=False)
        return True
    
    def _load_image_from_video(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """
        从视频文件中加载指定帧的图像
        支持多种编码格式（FFV1无损编码、H264、mp4v等）
        PyAV会自动检测并使用相应的解码器
        
        Args:
            video_path: 视频文件路径
            frame_idx: 帧索引（从 0 开始的整数帧号）
            
        Returns:
            numpy array of shape (height, width, 3), dtype uint8, RGB格式
        """
        reader = self._get_video_reader(video_path)
        container = reader["container"]
        video_stream = reader["stream"]
        
        # 计算目标帧的 PTS (Presentation Timestamp)
        # time_base 是时间戳的单位（例如 1/30 秒）
        # PTS = frame_idx / fps，然后转换为 time_base 单位
        time_base = video_stream.time_base
        fps = video_stream.average_rate
        if fps is None or float(fps) <= 0:
            # 如果无法获取 fps，尝试从 time_base 推算
            fps = 1.0 / float(time_base) if time_base else 30.0
        
        # 将帧号转换为 PTS（time_base 单位）
        target_pts = int(frame_idx * (1.0 / float(fps)) / float(time_base))
        
        # Seek 到目标 PTS
        video_key = str(video_path)
        try:
            container.seek(target_pts, stream=video_stream)
        except av.AVError:
            # 如果 seek 失败，尝试重新打开容器
            self.video_readers.pop(video_key, None)
            reader = self._get_video_reader(video_path)
            container = reader["container"]
            video_stream = reader["stream"]
            container.seek(target_pts, stream=video_stream)
        
        # 读取帧并找到最接近的帧
        img = None
        for frame in container.decode(video=0):
            # 检查是否是目标帧（允许一定的误差）
            if frame.pts >= target_pts:
                # 将帧转换为RGB格式的numpy数组
                # PyAV会自动处理视频编码格式（FFV1/H264/mp4v等）的解码
                # 输出为RGB24格式，保持原始画质
                img = frame.to_ndarray(format='rgb24') # !
                break
        
        if img is None:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
        
        return img
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取一个完整的数据样本
        
        Returns:
            dict with keys:
                - 'images': dict of numpy arrays, 每个相机 (height, width, 3)
                - 'vlm_hidden_states': numpy array (num_layers, seq_len, hidden_dim)
                - 'observation_state': numpy array (state_dim,)
                - 'actions': numpy array (num_chunks, action_dim) 或 (action_dim,)
                - 'episode_index': int
                - 'frame_index': int
                - 'vlm_index': int
        """
        episode_idx, frame_idx, parquet_path = self.index_map[idx]
        
        # 加载 parquet 数据
        df = self._load_parquet(parquet_path)
        current_frame = df.iloc[frame_idx]
        
        # 1. 加载图像（所有相机）- 可选跳过以节省 I/O
        images = {}
        if not self.skip_images:
            chunk_idx = episode_idx // self.chunks_size
            for image_key in self.image_keys:
                # 构建视频路径
                video_path = self.dataset_path / "videos" / f"chunk-{chunk_idx:03d}" / image_key / f"episode_{episode_idx:06d}.mp4"
                # 加载图像帧
                img = self._load_image_from_video(video_path, frame_idx)
                images[image_key] = img  # (height, width, 3), uint8
        
        # 2. 加载当前帧对应的 VLM hidden states（未来帧无需重复加载）
        vlm_index = int(current_frame['vlm_hidden_state_index'])
        vlm_states = self._load_vlm_hidden_state(vlm_index, episode_index=episode_idx, frame_index=frame_idx)
        
        # 3. 加载 observation state (proprioception)
        observation_state = np.array(current_frame['observation.state'], dtype=np.float32) # !
        
        # 4. 加载 task description (如果存在)
        task_description = None
        if 'annotation.human.action.task_description' in current_frame.index:
            task_idx = int(current_frame['annotation.human.action.task_description'])
            task_description = self.task_descriptions.get(task_idx, f"Task {task_idx}")
        
        # 5. 加载 actions 和 state chunk (支持 chunking，超出 episode 末尾时零填充)
        observation_states_chunk = None  # 初始化
        action_chunk_mask = None  # 标记有效 vs 填充位置
        if self.enable_chunking:
            episode_length = len(df)
            valid_chunks = min(self.chunk_horizon, episode_length - frame_idx)

            actions_list = []
            states_list = []
            for chunk_offset in range(valid_chunks):
                action_frame = df.iloc[frame_idx + chunk_offset]
                action = np.array(action_frame['action'], dtype=np.float32)
                actions_list.append(action)
                state = np.array(action_frame['observation.state'], dtype=np.float32)
                states_list.append(state)
            actions_valid = np.stack(actions_list, axis=0)  # (valid_chunks, action_dim)
            states_valid = np.stack(states_list, axis=0)    # (valid_chunks, state_dim)

            pad_length = self.chunk_horizon - valid_chunks
            if pad_length > 0:
                action_dim = actions_valid.shape[1]
                state_dim = states_valid.shape[1]
                actions = np.concatenate([
                    actions_valid,
                    np.zeros((pad_length, action_dim), dtype=np.float32)
                ], axis=0)
                observation_states_chunk = np.concatenate([
                    states_valid,
                    np.zeros((pad_length, state_dim), dtype=np.float32)
                ], axis=0)
            else:
                actions = actions_valid
                observation_states_chunk = states_valid

            action_chunk_mask = np.zeros(self.chunk_horizon, dtype=np.float32)
            action_chunk_mask[:valid_chunks] = 1.0
        else:
            # 只使用当前帧动作
            actions = np.array(current_frame['action'], dtype=np.float32)  # (action_dim,)
        
        # 6. 可选: 对 observation_state 进行预处理
        # 按 state_process_order 顺序执行，索引基于原始 state，自动计算偏移
        observation_state = self._apply_state_preprocessing(observation_state)
        
        # 6.1 对 state chunk 也进行预处理 (用于 state 差分替代 action 功能)
        if observation_states_chunk is not None:
            processed_states = []
            for i in range(observation_states_chunk.shape[0]):
                processed_state = self._apply_state_preprocessing(observation_states_chunk[i])
                processed_states.append(processed_state)
            observation_states_chunk = np.stack(processed_states, axis=0)  # (num_chunks, processed_state_dim)
        
        # 7. 可选: 对 actions 进行离散化处理 (用于 ParaCAT 训练)
        # 离散化将连续动作转换为离散步进 (-delta, 0, delta)
        # 只对指定的列进行离散化，每列使用对应的 delta
        # 注意: 反离散化 (undiscrete) 应在推理时处理，不在 DataLoader 中
        if self.discrete_actions and self.enable_chunking and len(self.discrete_columns) > 0:
            # actions 形状: (num_chunks, action_dim)
            # 只对 discrete_columns 指定的列进行离散化
            for i, col_idx in enumerate(self.discrete_columns):
                if col_idx < actions.shape[1]:
                    delta = self.discrete_deltas[i]
                    actions[:, col_idx] = discrete_with_method(
                        actions[:, col_idx], 
                        delta,
                        method=self.discrete_method,
                        beta=self.discrete_beta,
                        alpha=self.discrete_alpha,
                    )
        
        return {
            'images': images,  # dict of (height, width, 3)
            'vlm_hidden_states': vlm_states,  # (num_layers, seq_len, hidden_dim)
            'observation_state': observation_state,  # (state_dim,)
            'observation_states_chunk': observation_states_chunk,  # (num_chunks, state_dim) or None
            'actions': actions,  # (num_chunks, action_dim) or (action_dim,)
            'action_chunk_mask': action_chunk_mask,  # (num_chunks,) 1=valid 0=pad, or None
            'task_description': task_description,  # str or None
            'episode_index': episode_idx,
            'frame_index': frame_idx,
            'vlm_index': vlm_index,
            # 离散化配置
            'discrete_actions': self.discrete_actions,
            'discrete_columns': self.discrete_columns if self.discrete_actions else None,
            'discrete_deltas': self.discrete_deltas if self.discrete_actions else None,
            'discrete_method': self.discrete_method if self.discrete_actions else None,
            'discrete_beta': self.discrete_beta if self.discrete_actions else None,
            'discrete_alpha': self.discrete_alpha if self.discrete_actions else None,
            # State 预处理配置
            'state_process_order': self.state_process_order if len(self.state_process_order) > 0 else None,
            'hand_binary_columns': self.hand_binary_columns,
            'hand_binary_threshold': self.hand_binary_threshold,
            # 反离散化配置 (用于推理时)
            'undiscrete_actions': self.undiscrete_actions,
            'undiscrete_columns': self.undiscrete_columns if self.undiscrete_actions else None,
            'undiscrete_deltas': self.undiscrete_deltas if self.undiscrete_actions else None,
        }
    
    def _print_dataset_info(self):
        """打印数据集信息"""
        print("=" * 70)
        print("LeRobot Dataset Information")
        print("=" * 70)
        print(f"Dataset path: {self.dataset_path}")
        print(f"Split: {self.split}")
        print(f"Total episodes: {len(self.episodes)}")
        print(f"Total samples: {len(self.index_map)}")
        print(f"FPS: {self.fps}")
        print(f"\nData Dimensions:")
        print(f"  Action dim: {self.action_dim}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Action chunks: {self.num_action_chunks}")
        print(f"  Chunking enabled: {self.enable_chunking}")
        print(f"\nSettings:")
        if self.shared_vlm_cache is not None:
            print(f"  VLM cache mode: 🚀 shared memory (multi-GPU shared, zero-copy)")
            print(f"  Shared memory name: {self.shared_vlm_cache.name}")
            print(f"  Shared memory size: {self.shared_vlm_cache.num_samples} samples")
        elif self.cache_vlm_states:
            print(f"  VLM cache mode: 📦 mmap (OS page cache, multi-GPU shared)")
            print(f"  VLM file dtype: float32 (mmap preserves original dtype)")
        else:
            print(f"  VLM cache mode: ❌ disabled (disk I/O every access)")
        print(f"  Skip images: {self.skip_images}")
        # 离散化设置
        if self.discrete_actions:
            print(f"  Action discretization: ✓ enabled")
            print(f"    Method:  {self.discrete_method}")
            print(f"    Columns: {self.discrete_columns}")
            print(f"    Deltas:  {self.discrete_deltas}")
            if self.discrete_method == "chunk_calculus":
                print(f"    Beta:    {self.discrete_beta}")
                print(f"    Alpha:   {self.discrete_alpha}")
        else:
            print(f"  Action discretization: ❌ disabled")
        # State 预处理设置
        if len(self.state_process_order) > 0:
            print(f"  State preprocessing: ✓ enabled")
            print(f"    Order: {self.state_process_order}")
            if self.hand_binary_columns:
                print(f"    hand_binary: columns={self.hand_binary_columns}, threshold={self.hand_binary_threshold}")
        # 反离散化设置 (仅配置，实际在推理时应用)
        if self.undiscrete_actions:
            print(f"  Action undiscretization config: ✓ enabled (for inference)")
            print(f"    Columns: {self.undiscrete_columns}")
            print(f"    Deltas:  {self.undiscrete_deltas}")
        print("=" * 70)


def collate_fn(
    batch: List[Dict[str, Any]], 
    vlm_dtype: str = "float32",
) -> Dict[str, Any]:
    """
    自定义 collate 函数，将 numpy 数组转换为 PyTorch tensors
    
    Args:
        batch: List of samples from __getitem__
        vlm_dtype: VLM hidden states 的数据类型 ('float32', 'float16', 'bfloat16')
                   使用半精度可以减少约 50% 显存占用
        
    Returns:
        Batched dict with:
            - 'images': dict of tensors, 每个相机 (batch_size, height, width, 3)，跳过时为空 dict
            - 'vlm_hidden_states': (batch_size, num_layers, seq_len, hidden_dim)
            - 'observation_state': (batch_size, state_dim)
            - 'actions': (batch_size, num_chunks, action_dim) or (batch_size, action_dim)
            - 'episode_index': (batch_size,)
            - 'frame_index': (batch_size,)
    """
    batch_size = len(batch)
    
    # 确定 VLM hidden states 的目标数据类型
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_vlm_dtype = dtype_map.get(vlm_dtype, torch.float32)
    
    # Stack images for each camera (如果有的话)
    # 重要: 图像保持原始 uint8 格式，不进行归一化，VLM 模块会自行处理
    image_keys = list(batch[0]['images'].keys())
    images = {}
    if image_keys:  # 只有当有图像时才处理
        for key in image_keys:
            images[key] = torch.stack([
                torch.from_numpy(sample['images'][key]).float()  # 只转换为 float，不除以 255
                for sample in batch
            ])  # (batch_size, height, width, 3)
    
    # Stack VLM hidden states with padding: (batch_size, num_layers, seq_len, hidden_dim)
    # Handle variable sequence lengths by padding to max length
    # 使用指定的数据类型，可通过 vlm_dtype 参数控制
    vlm_tensors = [torch.from_numpy(sample['vlm_hidden_states']).to(target_vlm_dtype) for sample in batch]
    
    # Get max sequence length in this batch
    max_seq_len = max(tensor.size(1) for tensor in vlm_tensors)  # shape is (num_layers, seq_len, hidden_dim)
    
    # Pad all tensors to max_seq_len and create attention masks
    # 使用 left padding (在左边填充)，保持序列末尾对齐
    padded_vlm_tensors = []
    vlm_attention_masks = []  # 1 for valid tokens, 0 for padding
    
    for tensor in vlm_tensors:
        num_layers, seq_len, hidden_dim = tensor.shape
        pad_len = max_seq_len - seq_len
        
        # Create attention mask: 1 for valid tokens, 0 for padding (left side)
        mask = torch.ones(num_layers, max_seq_len, dtype=torch.long)
        
        if pad_len > 0:
            # Left padding: pad on the left side (beginning) of sequence
            padding = torch.zeros(num_layers, pad_len, hidden_dim, dtype=tensor.dtype)
            padded_tensor = torch.cat([padding, tensor], dim=1)  # padding 在左边
            # Mark left padding positions as 0 in mask
            mask[:, :pad_len] = 0  # 左边的位置标记为 0
        else:
            padded_tensor = tensor
        
        padded_vlm_tensors.append(padded_tensor)
        vlm_attention_masks.append(mask)
    
    vlm_hidden_states = torch.stack(padded_vlm_tensors)
    vlm_attention_mask = torch.stack(vlm_attention_masks)
    
    # Stack observation states
    observation_state = torch.stack([
        torch.from_numpy(sample['observation_state']).float()
        for sample in batch
    ])
    
    # Stack actions
    actions = torch.stack([
        torch.from_numpy(sample['actions']).float()
        for sample in batch
    ])
    
    # Stack observation states chunk (用于 state 差分替代 action 功能)
    observation_states_chunk = None
    if batch[0]['observation_states_chunk'] is not None:
        observation_states_chunk = torch.stack([
            torch.from_numpy(sample['observation_states_chunk']).float()
            for sample in batch
        ])  # (batch_size, num_chunks, state_dim)
    
    # Stack action chunk mask (1=valid, 0=padding)
    action_chunk_mask = None
    if batch[0].get('action_chunk_mask') is not None:
        action_chunk_mask = torch.stack([
            torch.from_numpy(sample['action_chunk_mask']).float()
            for sample in batch
        ])  # (batch_size, num_chunks)
    
    # Stack metadata
    episode_index = torch.tensor([sample['episode_index'] for sample in batch], dtype=torch.long)
    frame_index = torch.tensor([sample['frame_index'] for sample in batch], dtype=torch.long)
    vlm_index = torch.tensor([sample['vlm_index'] for sample in batch], dtype=torch.long)
    
    # Collect task descriptions (list of strings)
    task_descriptions = [sample['task_description'] for sample in batch]
    
    # 获取离散化配置 (batch 中所有样本应该一致)
    discrete_actions = batch[0].get('discrete_actions', False)
    discrete_columns = batch[0].get('discrete_columns', None)
    discrete_deltas = batch[0].get('discrete_deltas', None)
    discrete_method = batch[0].get('discrete_method', None)
    discrete_beta = batch[0].get('discrete_beta', None)
    discrete_alpha = batch[0].get('discrete_alpha', None)
    
    # 获取 State 预处理配置
    state_process_order = batch[0].get('state_process_order', None)
    hand_binary_columns = batch[0].get('hand_binary_columns', None)
    hand_binary_threshold = batch[0].get('hand_binary_threshold', 442.0)
    
    # 获取反离散化配置 (用于推理时)
    undiscrete_actions = batch[0].get('undiscrete_actions', False)
    undiscrete_columns = batch[0].get('undiscrete_columns', None)
    undiscrete_deltas = batch[0].get('undiscrete_deltas', None)
    
    return {
        'images': images,  # dict of (batch_size, height, width, 3)，跳过时为空 dict
        'vlm_hidden_states': vlm_hidden_states,  # (batch_size, num_layers, seq_len, hidden_dim)
        'vlm_attention_mask': vlm_attention_mask,  # (batch_size, num_layers, seq_len), 1=valid, 0=padding
        'observation_state': observation_state,  # (batch_size, state_dim)
        'observation_states_chunk': observation_states_chunk,  # (batch_size, num_chunks, state_dim) or None
        'actions': actions,  # (batch_size, num_chunks, action_dim) or (batch_size, action_dim)
        'action_chunk_mask': action_chunk_mask,  # (batch_size, num_chunks) 1=valid 0=pad, or None
        'task_description': task_descriptions,  # List[str]
        'episode_index': episode_index,
        'frame_index': frame_index,
        'vlm_index': vlm_index,
        # 离散化配置
        'discrete_actions': discrete_actions,
        'discrete_columns': discrete_columns,
        'discrete_deltas': discrete_deltas,
        'discrete_method': discrete_method,
        'discrete_beta': discrete_beta,
        'discrete_alpha': discrete_alpha,
        # State 预处理配置
        'state_process_order': state_process_order,
        'hand_binary_columns': hand_binary_columns,
        'hand_binary_threshold': hand_binary_threshold,
        # 反离散化配置 (用于推理时)
        'undiscrete_actions': undiscrete_actions,
        'undiscrete_columns': undiscrete_columns,
        'undiscrete_deltas': undiscrete_deltas,
    }


def create_lerobot_dataloader(
    dataset_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    split: str = "train",
    num_action_chunks: int = 25,
    enable_chunking: bool = True,
    episode_indices: Optional[List[int]] = None,
    cache_vlm_states: bool = False,
    cache_max_samples: int = -1,
    verbose: bool = True,
    max_cached_video_readers: int = 32,
    skip_images: bool = False,
    vlm_dtype: str = "float32",
    # 离散化参数 (10个参数)
    discrete_actions: bool = False,
    discrete_columns: Optional[List[int]] = None,
    discrete_deltas: Optional[List[float]] = None,
    discrete_method: str = "constrain_delta",
    discrete_beta: float = 0.6,
    discrete_alpha: float = 0.4,
    undiscrete_actions: bool = False,
    undiscrete_columns: Optional[List[int]] = None,
    undiscrete_deltas: Optional[List[float]] = None,
    # State 预处理参数
    state_process_order: Optional[List[str]] = None,
    hand_binary_columns: Optional[List[int]] = None,
    hand_binary_threshold: float = 442.0,
    **kwargs
) -> DataLoader:
    """
    创建 LeRobot DataLoader
    
    Args:
        dataset_path: 数据集根目录路径
        batch_size: batch 大小
        num_workers: 数据加载 worker 数量
        shuffle: 是否 shuffle
        split: 数据集划分
        num_action_chunks: action chunk 数量
        enable_chunking: 是否启用 action chunking
        episode_indices: 特定 episode 索引列表
        cache_vlm_states: 是否缓存 VLM states
        cache_max_samples: 最大缓存样本数 (-1 表示缓存所有样本)
        verbose: 是否打印信息
        max_cached_video_readers: 视频 reader LRU 缓存上限
        skip_images: 是否跳过加载图像 (仅使用预保存的 VLM hidden states 训练时设为 True)
        vlm_dtype: VLM hidden states 的数据类型 ('float32', 'float16', 'bfloat16')
                   - 同时控制 RAM 缓存精度和 GPU tensor 精度
                   - float16: RAM 占用减半 (~14MB/样本)
                   - bfloat16: RAM 使用 float32 缓存 (无损)，GPU 使用 bfloat16
        
        离散化参数 (10个参数):
        discrete_actions: 是否启用离散化
        discrete_columns: 参与离散化的 action 列索引列表
        discrete_deltas: 对应列的离散化步长列表
        discrete_method: 离散化方法 ("constrain_delta" 或 "chunk_calculus")
        discrete_beta: 趋势项权重 (仅 chunk_calculus 使用)
        discrete_alpha: 趋势平滑系数 (仅 chunk_calculus 使用)
        undiscrete_actions: 是否启用反离散化配置
        undiscrete_columns: 参与反离散化的 action 列索引列表
        
        State 预处理参数:
        state_process_order: 预处理执行顺序列表
        hand_binary_columns: 原始 state 中手部数据列范围 [start, end)
        hand_binary_threshold: 手部二值化阈值
        undiscrete_deltas: 对应列的反离散化步长列表
        
        **kwargs: 传递给 DataLoader 的其他参数
    
    Returns:
        DataLoader 对象
    """
    from functools import partial
    
    dataset = LeRobotDataset(
        dataset_path=dataset_path,
        split=split,
        num_action_chunks=num_action_chunks,
        enable_chunking=enable_chunking,
        episode_indices=episode_indices,
        cache_vlm_states=cache_vlm_states,
        cache_max_samples=cache_max_samples,
        verbose=verbose,
        max_cached_video_readers=max_cached_video_readers,
        skip_images=skip_images,
        vlm_cache_dtype=vlm_dtype,  # 使用 vlm_dtype 控制缓存精度
        discrete_actions=discrete_actions,
        discrete_columns=discrete_columns,
        discrete_deltas=discrete_deltas,
        discrete_method=discrete_method,
        discrete_beta=discrete_beta,
        discrete_alpha=discrete_alpha,
        undiscrete_actions=undiscrete_actions,
        undiscrete_columns=undiscrete_columns,
        undiscrete_deltas=undiscrete_deltas,
        # State 预处理参数
        state_process_order=state_process_order,
        hand_binary_columns=hand_binary_columns,
        hand_binary_threshold=hand_binary_threshold,
    )
    
    # 创建带参数的 collate_fn
    collate_fn_with_dtype = partial(collate_fn, vlm_dtype=vlm_dtype)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_with_dtype,
        pin_memory=True,
        **kwargs
    )
    
    return dataloader


# ============================================================================
# WebDataset 集成 (大规模数据训练优化)
# ============================================================================

def create_webdataset_dataloader(
    shard_pattern: str,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    shuffle_buffer_size: int = 10000,
    vlm_dtype: str = "bfloat16",
    epoch_length: Optional[int] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    verbose: bool = True,
) -> DataLoader:
    """
    创建 WebDataset 数据加载器 (用于大规模数据训练)
    
    使用 WebDataset tar 分片格式，显著减少大量小文件的 I/O 开销。
    
    转换数据集:
        python -m utils.webdataset_utils convert \\
            --input_path /path/to/lerobot_dataset \\
            --output_path /path/to/webdataset_shards \\
            --samples_per_shard 5000
    
    Args:
        shard_pattern: tar 分片路径模式
                      示例: "/path/shards/shard-{000000..000099}.tar"
        batch_size: 批次大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        shuffle_buffer_size: 打乱缓冲区大小
        vlm_dtype: VLM hidden states 数据类型 ('float32', 'float16', 'bfloat16')
        epoch_length: 每个 epoch 的样本数 (None 则使用所有数据)
        distributed: 是否为分布式训练
        rank: 当前进程 rank
        world_size: 总进程数
        verbose: 是否打印信息
        
    Returns:
        DataLoader 实例
        
    Raises:
        ImportError: 如果 WebDataset 未安装
    """
    try:
        from .webdataset_utils import create_webdataset_loader
    except ImportError:
        from webdataset_utils import create_webdataset_loader
    
    return create_webdataset_loader(
        shard_pattern=shard_pattern,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
        vlm_dtype=vlm_dtype,
        epoch_length=epoch_length,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        verbose=verbose,
    )


def create_hybrid_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    vlm_dtype: str = "bfloat16",
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    use_webdataset: bool = False,
    webdataset_shard_pattern: Optional[str] = None,
    use_shared_cache: bool = True,
    shared_vlm_cache: Optional["SharedVLMCache"] = None,
    num_action_chunks: int = 25,
    skip_images: bool = True,
    prefetch_factor: int = 4,
    verbose: bool = True,
    **kwargs,
) -> Tuple[DataLoader, Optional["SharedVLMCache"]]:
    """
    创建混合数据加载器 - 自动选择最优加载方式
    
    优先级:
    1. WebDataset (如果提供了 shard_pattern)
    2. 共享内存缓存 (如果 use_shared_cache=True)
    3. mmap 模式 (如果 cache_vlm_states=True)
    4. 普通加载
    
    Args:
        dataset_path: 原始数据集路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        vlm_dtype: VLM 数据类型
        distributed: 是否分布式
        rank: 当前 rank
        world_size: 总进程数
        use_webdataset: 是否使用 WebDataset
        webdataset_shard_pattern: WebDataset 分片路径模式
        use_shared_cache: 是否使用共享内存缓存
        shared_vlm_cache: 预创建的共享缓存实例
        num_action_chunks: action chunk 数量
        skip_images: 是否跳过图像
        prefetch_factor: 预取因子
        verbose: 是否打印信息
        **kwargs: 其他参数
        
    Returns:
        (DataLoader, SharedVLMCache 或 None)
    """
    # 方式 1: WebDataset
    if use_webdataset and webdataset_shard_pattern:
        if verbose and rank == 0:
            print("🚀 Using WebDataset mode for large-scale training")
        
        loader = create_webdataset_dataloader(
            shard_pattern=webdataset_shard_pattern,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            vlm_dtype=vlm_dtype,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            verbose=verbose,
        )
        return loader, None
    
    # 方式 2/3/4: 标准 LeRobot 数据集
    from functools import partial
    from torch.utils.data.distributed import DistributedSampler
    
    # 创建或使用共享缓存
    cache = shared_vlm_cache
    if use_shared_cache and cache is None:
        cache_size = get_total_vlm_states(dataset_path)
        if verbose and rank == 0:
            print(f"📦 Creating shared memory cache for {cache_size} samples...")
        
        cache = preload_vlm_cache_distributed(
            dataset_path=dataset_path,
            num_samples=cache_size,
            sample_shape=None,
            rank=rank,
            world_size=world_size,
            dtype=np.float32,
            verbose=verbose and (rank == 0),
            auto_detect_shape=True,
        )
    
    # 创建数据集
    episode_indices = list(range(
        len(list(Path(dataset_path).glob("meta/episodes.jsonl")))
    )) if not kwargs.get('episode_indices') else kwargs.get('episode_indices')
    
    # 读取 episode 数量
    episodes_path = Path(dataset_path) / "meta" / "episodes.jsonl"
    episode_count = sum(1 for _ in open(episodes_path))
    episode_indices = list(range(episode_count))
    
    dataset = LeRobotDataset(
        dataset_path=dataset_path,
        num_action_chunks=num_action_chunks,
        enable_chunking=True,
        episode_indices=episode_indices,
        cache_vlm_states=kwargs.get('cache_vlm_states', False),
        verbose=verbose and (rank == 0),
        skip_images=skip_images,
        shared_vlm_cache=cache,
    )
    
    # 创建 sampler
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False
    
    # 创建 collate_fn
    collate_fn_with_dtype = partial(collate_fn, vlm_dtype=vlm_dtype)
    
    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn_with_dtype,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    if verbose and rank == 0:
        cache_mode = "shared memory" if cache else ("mmap" if kwargs.get('cache_vlm_states') else "disk")
        print(f"✓ DataLoader created: {len(dataset)} samples, cache mode: {cache_mode}")
    
    return loader, cache


if __name__ == '__main__':
    """测试数据加载器"""
    print("Testing Universal LeRobot Dataset Loader...")
    
    # 示例路径
    dataset_path = "/data/HuangWenlong/datasets/libero_github_convert_for_Sai0/libero_lerobot_object"
    
    # 创建数据集
    dataset = LeRobotDataset(
        dataset_path=dataset_path,
        split="train",
        enable_chunking=True,
        num_action_chunks=10,
        cache_vlm_states=True,
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # 测试单个样本
    print("\nTesting __getitem__...")
    sample = dataset[0]
    
    print(f"Sample keys: {sample.keys()}")
    print(f"VLM hidden states shape: {sample['vlm_hidden_states'].shape}")
    print(f"Observation state shape: {sample['observation_state'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Episode index: {sample['episode_index']}")
    print(f"Frame index: {sample['frame_index']}")
    
    # 测试 DataLoader
    print("\nTesting DataLoader...")
    dataloader = create_lerobot_dataloader(
        dataset_path=dataset_path,
        batch_size=4,
        num_workers=0,
        shuffle=False,
    )
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print("Batch keys:", batch.keys())
        print("\nImages:")
        for img_key, img_tensor in batch['images'].items():
            print(f"  {img_key}: {img_tensor.shape}")
        print(f"\nVLM hidden states: {batch['vlm_hidden_states'].shape}")
        print(f"Observation state: {batch['observation_state'].shape}")
        print(f"Actions: {batch['actions'].shape}")
        print(f"Task descriptions: {batch['task_description']}")
        print(f"Episode index: {batch['episode_index'].shape}")
        print(f"Frame index: {batch['frame_index'].shape}")
        print(f"VLM index: {batch['vlm_index'].shape}")
        
        if batch_idx >= 1:
            break
    
    print("\n✓ Universal dataset loader test completed successfully!")



#     Batch 0:
# Batch keys: dict_keys(['images', 'vlm_hidden_states', 'observation_state', 'actions', 'episode_index', 'frame_index', 'vlm_index'])

# Images:
#   observation.images.top: torch.Size([4, 800, 1280, 3])
#   observation.images.left_wrist: torch.Size([4, 800, 1280, 3])

# VLM hidden states: torch.Size([4, 3, 512, 2048])
# Observation state: torch.Size([4, 16])
# Actions: torch.Size([4, 25, 16])
# Episode index: torch.Size([4])
# Frame index: torch.Size([4])
# VLM index: torch.Size([4])
