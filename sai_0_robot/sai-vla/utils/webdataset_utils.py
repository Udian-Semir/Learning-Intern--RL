"""
WebDataset 工具集 - 大规模数据集训练优化
WebDataset Utilities for Large-Scale Dataset Training

功能：
1. 将 LeRobot 格式数据集转换为 WebDataset tar 分片
2. 提供 WebDataset 兼容的数据加载器
3. 支持分布式训练的数据分片

使用方法：

1. 转换数据集 (在项目根目录执行):
   
   # 保留原始精度 (默认)
   python -m utils.webdataset_utils convert \\
       --input_path /data/.../libero_lerobot_spatial_sys0_cosmos_-1 \\
       --samples_per_shard 5000 \\
       --num_action_chunks 25
   
   # 转换为 float16 (减半存储空间)
   python -m utils.webdataset_utils convert \\
       --input_path /data/.../libero_lerobot_spatial_sys0_cosmos_-1 \\
       --samples_per_shard 5000 \\
       --num_action_chunks 25 \\
       --convert_vlm_dtype \\
       --vlm_dtype float16
   
   # 输出目录自动创建: input_path/webdataset_shard5000_ac25/

2. 在训练脚本中配置 (train_qwen.sh):
   
   USE_WEBDATASET="true"
   # 使用 brace expansion 匹配多个分片: {000000..000012} 展开为 000000, 000001, ..., 000012
   WEBDATASET_SHARD_PATTERN="/data/.../webdataset_shard5000_ac25/shard-{000000..000012}.tar"

3. 在 Python 中使用:
   
   from utils.webdataset_utils import create_webdataset_loader
   train_loader = create_webdataset_loader(
       shard_pattern="/path/shards/shard-{000000..000012}.tar",
       batch_size=32,
       num_workers=8,
       distributed=True,
       rank=rank,
       world_size=world_size,
   )

转换后的 tar 文件结构:
   shard-000000.tar
   ├── 00000000.vlm.npy      # VLM hidden states (num_layers, seq_len, hidden_dim)
   ├── 00000000.state.npy    # observation state (state_dim,)
   ├── 00000000.action.npy   # actions (num_action_chunks, action_dim)
   ├── 00000000.meta.json    # {episode_index, frame_index, vlm_index, task_description}
   └── ...
"""

import os
import io
import json
import argparse
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Iterator
from functools import partial
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# WebDataset 导入（延迟导入以避免未安装时报错）
try:
    import webdataset as wds
    WEBDATASET_AVAILABLE = True
except ImportError:
    WEBDATASET_AVAILABLE = False
    print("⚠️ WebDataset not installed. Install with: pip install webdataset")


# ============================================================================
# 数据转换工具
# ============================================================================

class LeRobotToWebDatasetConverter:
    """
    将 LeRobot 格式数据集转换为 WebDataset tar 分片
    
    LeRobot 格式:
    - data/chunk-XXX/episode_XXXXXX.parquet (actions, states)
    - vlm_hidden_states/hidden_state_XXXXXX.npy
    - videos/chunk-XXX/observation.images.XXX/episode_XXXXXX.mp4
    
    WebDataset 格式:
    - shard-XXXXXX.tar 包含:
        - XXXXXX.vlm.npy (VLM hidden states)
        - XXXXXX.state.npy (observation state)
        - XXXXXX.action.npy (actions)
        - XXXXXX.meta.json (episode_idx, frame_idx, task_description)
    """
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        samples_per_shard: int = 5000,
        num_action_chunks: int = 25,
        skip_images: bool = True,
        compression: str = "none",  # "none", "gzip"
        verbose: bool = True,
        convert_vlm_dtype: bool = False,
        vlm_dtype: str = "float16",
    ):
        """
        初始化转换器
        
        Args:
            input_path: LeRobot 数据集路径
            output_path: 输出 WebDataset 分片目录
            samples_per_shard: 每个 tar 分片的样本数
            num_action_chunks: action chunk 数量
            skip_images: 是否跳过图像（VLM 训练通常只需要预计算的 hidden states）
            compression: tar 压缩方式 ("none" 或 "gzip")
            verbose: 是否打印详细信息
            convert_vlm_dtype: 是否对 VLM hidden states 做精度转换
                              - False: 保留原始 .npy 文件的精度（通常是 float32）
                              - True: 转换为 vlm_dtype 指定的精度
            vlm_dtype: VLM hidden states 的目标精度 (仅在 convert_vlm_dtype=True 时生效)
                      - "float16": 减半存储空间
                      - "float32": 保持完整精度
                      - "bfloat16": 存储为 float16（numpy 不支持 bfloat16）
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.samples_per_shard = samples_per_shard
        self.num_action_chunks = num_action_chunks
        self.skip_images = skip_images
        self.compression = compression
        self.verbose = verbose
        self.convert_vlm_dtype = convert_vlm_dtype
        self.vlm_dtype = vlm_dtype
        
        # 解析目标数据类型
        dtype_map = {
            "float16": np.float16,
            "float32": np.float32,
            "bfloat16": np.float16,  # numpy 不支持 bfloat16，使用 float16 存储
        }
        self.target_np_dtype = dtype_map.get(vlm_dtype, np.float16)
        
        # 加载元信息
        self._load_metadata()
        
    def _load_metadata(self):
        """加载数据集元信息"""
        info_path = self.input_path / "meta" / "info.json"
        with open(info_path, 'r') as f:
            self.info = json.load(f)
        
        self.action_dim = self.info['features']['action']['shape'][0]
        self.state_dim = self.info['features']['observation.state']['shape'][0]
        self.total_episodes = self.info['total_episodes']
        self.total_frames = self.info['total_frames']
        self.chunks_size = self.info['chunks_size']
        
        # 加载 task descriptions
        tasks_path = self.input_path / "meta" / "tasks.jsonl"
        self.task_descriptions = {}
        if tasks_path.exists():
            with open(tasks_path, 'r') as f:
                for line in f:
                    task = json.loads(line.strip())
                    self.task_descriptions[task['task_index']] = task['task']
        
        # 加载 episodes
        episodes_path = self.input_path / "meta" / "episodes.jsonl"
        self.episodes = []
        with open(episodes_path, 'r') as f:
            for line in f:
                self.episodes.append(json.loads(line.strip()))
        
        if self.verbose:
            print(f"📊 Dataset info:")
            print(f"  Total episodes: {self.total_episodes}")
            print(f"  Total frames: {self.total_frames}")
            print(f"  Action dim: {self.action_dim}")
            print(f"  State dim: {self.state_dim}")
            print(f"  Convert VLM dtype: {self.convert_vlm_dtype}")
            if self.convert_vlm_dtype:
                print(f"  Target VLM dtype: {self.vlm_dtype}")
            else:
                print(f"  VLM dtype: 保留原始精度")
    
    def _build_index(self) -> List[Tuple[int, int, str]]:
        """构建数据索引"""
        index_map = []
        horizon = self.num_action_chunks
        
        for episode in self.episodes:
            episode_idx = episode['episode_index']
            chunk_idx = episode_idx // self.chunks_size
            parquet_path = self.input_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
            
            df = pd.read_parquet(parquet_path)
            episode_length = len(df)
            
            if episode_length < horizon:
                continue
            
            max_start = episode_length - horizon
            frame_idx = 0
            done_column = df['next.done'] if 'next.done' in df.columns else None
            
            while frame_idx <= max_start:
                index_map.append((episode_idx, frame_idx, str(parquet_path)))
                
                if done_column is not None:
                    last_frame_done = bool(done_column.iloc[frame_idx + horizon - 1])
                    if last_frame_done:
                        break
                
                frame_idx += 1
        
        return index_map
    
    def _load_vlm_hidden_state(self, vlm_index: int) -> np.ndarray:
        """加载 VLM hidden states"""
        vlm_path = self.input_path / "vlm_hidden_states" / f"hidden_state_{vlm_index:06d}.npy"
        vlm_state = np.load(vlm_path)
        
        # 自动适配单层和多层格式
        if vlm_state.ndim == 2:
            vlm_state = vlm_state.reshape(1, vlm_state.shape[0], vlm_state.shape[1])
        
        return vlm_state
    
    def convert(self):
        """执行转换"""
        # 创建输出目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 构建索引
        if self.verbose:
            print("🔍 Building index...")
        index_map = self._build_index()
        total_samples = len(index_map)
        
        if self.verbose:
            print(f"  Total samples: {total_samples}")
            print(f"  Samples per shard: {self.samples_per_shard}")
            print(f"  Expected shards: {(total_samples + self.samples_per_shard - 1) // self.samples_per_shard}")
        
        # 缓存 parquet 数据
        parquet_cache = {}
        
        def load_parquet(path: str) -> pd.DataFrame:
            if path not in parquet_cache:
                parquet_cache[path] = pd.read_parquet(path)
            return parquet_cache[path]
        
        # 分片写入
        shard_idx = 0
        sample_in_shard = 0
        tar_file = None
        
        ext = ".tar.gz" if self.compression == "gzip" else ".tar"
        mode = "w:gz" if self.compression == "gzip" else "w"
        
        pbar = tqdm(enumerate(index_map), total=total_samples, desc="Converting") if self.verbose else enumerate(index_map)
        
        for global_idx, (episode_idx, frame_idx, parquet_path) in pbar:
            # 开启新分片
            if sample_in_shard == 0:
                shard_path = self.output_path / f"shard-{shard_idx:06d}{ext}"
                tar_file = tarfile.open(shard_path, mode)
            
            # 加载数据
            df = load_parquet(parquet_path)
            current_frame = df.iloc[frame_idx]
            
            # 1. VLM hidden states
            vlm_index = int(current_frame['vlm_hidden_state_index'])
            vlm_states = self._load_vlm_hidden_state(vlm_index)
            
            # 2. Observation state
            observation_state = np.array(current_frame['observation.state'], dtype=np.float32)
            
            # 3. Actions (chunk)
            actions_list = []
            for chunk_offset in range(self.num_action_chunks):
                action_frame = df.iloc[frame_idx + chunk_offset]
                action = np.array(action_frame['action'], dtype=np.float32)
                actions_list.append(action)
            actions = np.stack(actions_list, axis=0)
            
            # 4. Task description
            task_description = None
            if 'annotation.human.action.task_description' in current_frame.index:
                task_idx = int(current_frame['annotation.human.action.task_description'])
                task_description = self.task_descriptions.get(task_idx, f"Task {task_idx}")
            
            # 5. 元数据
            meta = {
                'episode_index': episode_idx,
                'frame_index': frame_idx,
                'vlm_index': vlm_index,
                'task_description': task_description,
                'global_index': global_idx,
            }
            
            # 写入 tar
            sample_key = f"{global_idx:08d}"
            
            # VLM hidden states (.npy)
            vlm_buffer = io.BytesIO()
            if self.convert_vlm_dtype:
                # 转换为指定精度
                np.save(vlm_buffer, vlm_states.astype(self.target_np_dtype))
            else:
                # 保留原始精度
                np.save(vlm_buffer, vlm_states)
            vlm_buffer.seek(0)
            vlm_info = tarfile.TarInfo(name=f"{sample_key}.vlm.npy")
            vlm_info.size = vlm_buffer.getbuffer().nbytes
            tar_file.addfile(vlm_info, vlm_buffer)
            
            # Observation state (.npy)
            state_buffer = io.BytesIO()
            np.save(state_buffer, observation_state)
            state_buffer.seek(0)
            state_info = tarfile.TarInfo(name=f"{sample_key}.state.npy")
            state_info.size = state_buffer.getbuffer().nbytes
            tar_file.addfile(state_info, state_buffer)
            
            # Actions (.npy)
            action_buffer = io.BytesIO()
            np.save(action_buffer, actions)
            action_buffer.seek(0)
            action_info = tarfile.TarInfo(name=f"{sample_key}.action.npy")
            action_info.size = action_buffer.getbuffer().nbytes
            tar_file.addfile(action_info, action_buffer)
            
            # Metadata (.json)
            meta_bytes = json.dumps(meta).encode('utf-8')
            meta_buffer = io.BytesIO(meta_bytes)
            meta_info = tarfile.TarInfo(name=f"{sample_key}.meta.json")
            meta_info.size = len(meta_bytes)
            tar_file.addfile(meta_info, meta_buffer)
            
            sample_in_shard += 1
            
            # 关闭当前分片，开启新分片
            if sample_in_shard >= self.samples_per_shard:
                tar_file.close()
                shard_idx += 1
                sample_in_shard = 0
                parquet_cache.clear()  # 清理缓存
        
        # 关闭最后一个分片
        if tar_file is not None and sample_in_shard > 0:
            tar_file.close()
            shard_idx += 1
        
        # 保存元信息
        meta_info = {
            'total_samples': total_samples,
            'num_shards': shard_idx,
            'samples_per_shard': self.samples_per_shard,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'num_action_chunks': self.num_action_chunks,
            'compression': self.compression,
            'source_dataset': str(self.input_path),
        }
        
        with open(self.output_path / "meta.json", 'w') as f:
            json.dump(meta_info, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Conversion complete!")
            print(f"  Output directory: {self.output_path}")
            print(f"  Total shards: {shard_idx}")
            print(f"  Total samples: {total_samples}")


# ============================================================================
# WebDataset 数据加载器
# ============================================================================

def decode_vlm_npy(data: bytes) -> np.ndarray:
    """解码 VLM hidden states"""
    return np.load(io.BytesIO(data))

def decode_state_npy(data: bytes) -> np.ndarray:
    """解码 observation state"""
    return np.load(io.BytesIO(data))

def decode_action_npy(data: bytes) -> np.ndarray:
    """解码 actions"""
    return np.load(io.BytesIO(data))

def decode_meta_json(data: bytes) -> dict:
    """解码元数据"""
    return json.loads(data.decode('utf-8'))


def webdataset_collate_fn(
    batch: List[Dict[str, Any]],
    vlm_dtype: str = "bfloat16",
) -> Dict[str, torch.Tensor]:
    """
    WebDataset collate 函数
    
    Args:
        batch: WebDataset 样本列表
        vlm_dtype: VLM hidden states 的数据类型
        
    Returns:
        批量张量字典
    """
    batch_size = len(batch)
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_vlm_dtype = dtype_map.get(vlm_dtype, torch.bfloat16)
    
    # VLM hidden states with padding
    vlm_tensors = [torch.from_numpy(sample['vlm']).to(target_vlm_dtype) for sample in batch]
    max_seq_len = max(tensor.size(1) for tensor in vlm_tensors)
    
    padded_vlm_tensors = []
    vlm_attention_masks = []
    
    for tensor in vlm_tensors:
        num_layers, seq_len, hidden_dim = tensor.shape
        pad_len = max_seq_len - seq_len
        
        mask = torch.ones(num_layers, max_seq_len, dtype=torch.long)
        
        if pad_len > 0:
            padding = torch.zeros(num_layers, pad_len, hidden_dim, dtype=tensor.dtype)
            mask[:, :pad_len] = 0  # Left padding
            tensor = torch.cat([padding, tensor], dim=1)
        
        padded_vlm_tensors.append(tensor)
        vlm_attention_masks.append(mask)
    
    vlm_hidden_states = torch.stack(padded_vlm_tensors)
    vlm_attention_mask = torch.stack(vlm_attention_masks)
    
    # Observation states
    observation_states = torch.stack([
        torch.from_numpy(sample['state']).float() for sample in batch
    ])
    
    # Actions
    actions = torch.stack([
        torch.from_numpy(sample['action']).float() for sample in batch
    ])
    
    # Indices
    episode_indices = torch.tensor([sample['meta']['episode_index'] for sample in batch])
    frame_indices = torch.tensor([sample['meta']['frame_index'] for sample in batch])
    vlm_indices = torch.tensor([sample['meta']['vlm_index'] for sample in batch])
    
    return {
        'vlm_hidden_states': vlm_hidden_states,
        'vlm_attention_mask': vlm_attention_mask,
        'observation_state': observation_states,
        'actions': actions,
        'episode_index': episode_indices,
        'frame_index': frame_indices,
        'vlm_index': vlm_indices,
    }


def create_webdataset_loader(
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
    创建 WebDataset 数据加载器
    
    Args:
        shard_pattern: tar 分片路径模式 (e.g., "/path/shards/shard-{000000..000099}.tar")
        batch_size: 批次大小
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        shuffle_buffer_size: 打乱缓冲区大小
        vlm_dtype: VLM hidden states 数据类型
        epoch_length: 每个 epoch 的样本数（None 则使用所有数据）
        distributed: 是否为分布式训练
        rank: 当前进程 rank
        world_size: 总进程数
        verbose: 是否打印信息
        
    Returns:
        DataLoader 实例
    """
    if not WEBDATASET_AVAILABLE:
        raise ImportError("WebDataset not installed. Install with: pip install webdataset")
    
    # 创建 WebDataset pipeline
    # 注意：确保 分片数 >= num_gpus * num_workers，否则会报错
    if distributed:
        # 分布式模式：按 rank 分片
        dataset = wds.WebDataset(
            shard_pattern,
            nodesplitter=wds.split_by_node,
            shardshuffle=shuffle,
        )
    else:
        dataset = wds.WebDataset(
            shard_pattern,
            shardshuffle=shuffle,
        )
    
    # 自定义解码器：手动处理 .npy 和 .json 文件
    # WebDataset 的 decode() 默认不处理 .npy 文件
    def custom_decoder(sample):
        """解码 tar 中的文件"""
        result = {'__key__': sample.get('__key__')}
        
        # 解码 .npy 文件 (VLM, state, action)
        for npy_key in ['vlm.npy', 'state.npy', 'action.npy']:
            if npy_key in sample:
                data = sample[npy_key]
                if isinstance(data, bytes):
                    result[npy_key.replace('.npy', '')] = np.load(io.BytesIO(data))
                else:
                    result[npy_key.replace('.npy', '')] = data
        
        # 解码 .json 文件 (meta)
        if 'meta.json' in sample:
            data = sample['meta.json']
            if isinstance(data, bytes):
                result['meta'] = json.loads(data.decode('utf-8'))
            elif isinstance(data, str):
                result['meta'] = json.loads(data)
            else:
                result['meta'] = data
        
        return result
    
    dataset = dataset.map(custom_decoder)
    
    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    
    # Epoch 长度 (在 batching 之前设置)
    if epoch_length is not None:
        dataset = dataset.with_epoch(epoch_length)
    
    # 创建 DataLoader
    # 注意：不使用 WebDataset 的 .batched()，而是让 WebLoader 的 batch_size 来处理
    # 这样 collate_fn 会正确接收到 List[Dict] 格式的数据
    collate_fn = partial(webdataset_collate_fn, vlm_dtype=vlm_dtype)
    
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,  # 由 WebLoader 处理批处理
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,  # 丢弃不完整的最后一个批次
    )
    
    if verbose and rank == 0:
        print(f"📦 WebDataset loader created:")
        print(f"  Shard pattern: {shard_pattern}")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {num_workers}")
        print(f"  Shuffle buffer: {shuffle_buffer_size}")
        print(f"  VLM dtype: {vlm_dtype}")
        print(f"  Distributed: {distributed} (rank {rank}/{world_size})")
    
    return loader


# ============================================================================
# 单文件合并格式 (Arrow/Memory-mapped)
# ============================================================================

class MergedVLMStatesLoader:
    """
    合并的 VLM Hidden States 加载器
    
    将所有 VLM hidden states 合并到单个内存映射文件，
    减少大量小文件的 I/O 开销。
    
    文件格式:
    - vlm_merged.npy: 合并的 VLM hidden states (N, num_layers, max_seq_len, hidden_dim)
    - vlm_seq_lens.npy: 每个样本的实际 seq_len (N,)
    - vlm_index.json: 索引映射信息
    """
    
    def __init__(
        self,
        merged_file_path: str,
        dtype: str = "float16",
    ):
        """
        初始化加载器
        
        Args:
            merged_file_path: 合并文件目录路径
            dtype: 数据类型
        """
        self.merged_path = Path(merged_file_path)
        self.dtype = dtype
        
        # 加载元信息
        with open(self.merged_path / "vlm_index.json", 'r') as f:
            self.index_info = json.load(f)
        
        self.num_samples = self.index_info['num_samples']
        self.max_seq_len = self.index_info['max_seq_len']
        self.num_layers = self.index_info['num_layers']
        self.hidden_dim = self.index_info['hidden_dim']
        
        # 内存映射文件
        self.vlm_data = np.load(
            self.merged_path / "vlm_merged.npy",
            mmap_mode='r'
        )
        self.seq_lens = np.load(self.merged_path / "vlm_seq_lens.npy")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """获取单个样本，返回实际长度（不含 padding）"""
        actual_len = self.seq_lens[idx]
        return self.vlm_data[idx, :, :actual_len, :].copy()
    
    @staticmethod
    def create_merged_file(
        input_path: str,
        output_path: str,
        dtype: str = "float16",
        verbose: bool = True,
    ):
        """
        创建合并的 VLM hidden states 文件
        
        Args:
            input_path: 原始数据集路径（包含 vlm_hidden_states/ 目录）
            output_path: 输出目录
            dtype: 存储数据类型 ("float16", "float32")
            verbose: 是否打印信息
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 扫描所有 VLM hidden states 文件
        vlm_dir = input_path / "vlm_hidden_states"
        vlm_files = sorted(vlm_dir.glob("hidden_state_*.npy"))
        num_samples = len(vlm_files)
        
        if verbose:
            print(f"📊 Found {num_samples} VLM hidden state files")
        
        # 检测形状
        sample = np.load(vlm_files[0])
        if sample.ndim == 2:
            sample = sample.reshape(1, sample.shape[0], sample.shape[1])
        
        num_layers, _, hidden_dim = sample.shape
        
        # 扫描最大 seq_len
        if verbose:
            print("🔍 Scanning max sequence length...")
        
        max_seq_len = 0
        seq_lens = []
        
        for vlm_file in tqdm(vlm_files, desc="Scanning", disable=not verbose):
            data = np.load(vlm_file)
            if data.ndim == 2:
                seq_len = data.shape[0]
            else:
                seq_len = data.shape[1]
            seq_lens.append(seq_len)
            max_seq_len = max(max_seq_len, seq_len)
        
        seq_lens = np.array(seq_lens, dtype=np.int32)
        
        if verbose:
            print(f"  Max seq_len: {max_seq_len}")
            print(f"  Num layers: {num_layers}")
            print(f"  Hidden dim: {hidden_dim}")
        
        # 创建合并文件
        np_dtype = np.float16 if dtype == "float16" else np.float32
        
        if verbose:
            print(f"📦 Creating merged file ({dtype})...")
        
        # 使用内存映射创建大文件
        merged_shape = (num_samples, num_layers, max_seq_len, hidden_dim)
        merged_path = output_path / "vlm_merged.npy"
        
        # 创建空的 npy 文件
        fp = np.lib.format.open_memmap(
            merged_path,
            mode='w+',
            dtype=np_dtype,
            shape=merged_shape,
        )
        
        # 写入数据
        for idx, vlm_file in enumerate(tqdm(vlm_files, desc="Writing", disable=not verbose)):
            data = np.load(vlm_file)
            if data.ndim == 2:
                data = data.reshape(1, data.shape[0], data.shape[1])
            
            actual_len = data.shape[1]
            fp[idx, :, :actual_len, :] = data.astype(np_dtype)
        
        # 刷新到磁盘
        del fp
        
        # 保存 seq_lens
        np.save(output_path / "vlm_seq_lens.npy", seq_lens)
        
        # 保存索引信息
        index_info = {
            'num_samples': num_samples,
            'max_seq_len': max_seq_len,
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'dtype': dtype,
            'source_path': str(input_path),
        }
        
        with open(output_path / "vlm_index.json", 'w') as f:
            json.dump(index_info, f, indent=2)
        
        if verbose:
            total_size_gb = (num_samples * num_layers * max_seq_len * hidden_dim * 
                           (2 if dtype == "float16" else 4)) / (1024**3)
            print(f"\n✅ Merged file created!")
            print(f"  Output: {output_path}")
            print(f"  Size: {total_size_gb:.2f} GB")
            print(f"  Samples: {num_samples}")


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="WebDataset 工具集 - 大规模数据集训练优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换数据集为 WebDataset 格式 (保留原始精度，输出目录自动生成)
  python -m utils.webdataset_utils convert \\
      --input_path /data/.../libero_lerobot_spatial_sys0_cosmos_-1 \\
      --samples_per_shard 5000 \\
      --num_action_chunks 25
  
  # 输出目录: /data/.../libero_lerobot_spatial_sys0_cosmos_-1/webdataset_shard5000_ac25/
  
  # 转换并将 VLM hidden states 转为 float16 (减半存储空间)
  python -m utils.webdataset_utils convert \\
      --input_path /data/.../libero_dataset \\
      --samples_per_shard 5000 \\
      --convert_vlm_dtype \\
      --vlm_dtype float16

  # 合并 VLM hidden states 到单个文件
  python -m utils.webdataset_utils merge \\
      --input_path /data/.../libero_dataset \\
      --dtype float16
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 转换命令
    convert_parser = subparsers.add_parser('convert', help='转换为 WebDataset 格式')
    convert_parser.add_argument('--input_path', type=str, required=True,
                               help='LeRobot 数据集路径 (包含 data/, vlm_hidden_states/, meta/ 目录)')
    convert_parser.add_argument('--output_path', type=str, default=None,
                               help='输出目录 (可选，默认在 input_path 下创建 webdataset_shard{N}_ac{M}/ 子目录)')
    convert_parser.add_argument('--samples_per_shard', type=int, default=5000,
                               help='每个分片的样本数 (默认: 5000)')
    convert_parser.add_argument('--num_action_chunks', type=int, default=25,
                               help='Action chunk 数量 (默认: 25)')
    convert_parser.add_argument('--compression', type=str, default='none',
                               choices=['none', 'gzip'],
                               help='压缩方式 (默认: none，gzip 会减小文件但增加 CPU 开销)')
    convert_parser.add_argument('--convert_vlm_dtype', action='store_true', default=False,
                               help='是否对 VLM hidden states 做精度转换 (默认: False，保留原始精度)')
    convert_parser.add_argument('--vlm_dtype', type=str, default='float16',
                               choices=['float16', 'float32'],
                               help='VLM hidden states 的目标精度 (仅在 --convert_vlm_dtype 时生效，默认: float16)')
    
    # 合并命令
    merge_parser = subparsers.add_parser('merge', help='合并 VLM states 到单个文件')
    merge_parser.add_argument('--input_path', type=str, required=True,
                             help='原始数据集路径')
    merge_parser.add_argument('--output_path', type=str, default=None,
                             help='输出目录 (可选，默认在 input_path 下创建 vlm_merged_fp16/ 或 vlm_merged_fp32/ 子目录)')
    merge_parser.add_argument('--dtype', type=str, default='float16',
                             choices=['float16', 'float32'],
                             help='存储数据类型 (默认: float16，减半存储空间)')
    
    args = parser.parse_args()
    
    if args.command == 'convert':
        if not WEBDATASET_AVAILABLE:
            print("❌ WebDataset not installed. Install with: pip install webdataset")
            return
        
        # 自动生成输出路径
        input_path = Path(args.input_path)
        if args.output_path is None:
            output_dir_name = f"webdataset_shard{args.samples_per_shard}_ac{args.num_action_chunks}"
            output_path = input_path / output_dir_name
            print(f"📁 输出目录自动生成: {output_path}")
        else:
            output_path = Path(args.output_path)
        
        converter = LeRobotToWebDatasetConverter(
            input_path=args.input_path,
            output_path=str(output_path),
            samples_per_shard=args.samples_per_shard,
            num_action_chunks=args.num_action_chunks,
            compression=args.compression,
            convert_vlm_dtype=args.convert_vlm_dtype,
            vlm_dtype=args.vlm_dtype,
        )
        converter.convert()
        
        # 打印使用提示
        num_shards = converter.info['total_frames'] // args.samples_per_shard
        if converter.info['total_frames'] % args.samples_per_shard > 0:
            num_shards += 1
        
        print(f"\n" + "="*60)
        print(f"📋 训练时配置 (train_qwen.sh):")
        print(f"="*60)
        print(f'USE_WEBDATASET="true"')
        print(f'WEBDATASET_SHARD_PATTERN="{output_path}/shard-{{000000..{num_shards-1:06d}}}.tar"')
        print(f"="*60)
    
    elif args.command == 'merge':
        # 自动生成输出路径
        input_path = Path(args.input_path)
        if args.output_path is None:
            output_dir_name = f"vlm_merged_{'fp16' if args.dtype == 'float16' else 'fp32'}"
            output_path = input_path / output_dir_name
            print(f"📁 输出目录自动生成: {output_path}")
        else:
            output_path = Path(args.output_path)
        
        MergedVLMStatesLoader.create_merged_file(
            input_path=args.input_path,
            output_path=str(output_path),
            dtype=args.dtype,
        )
    
    else:
        parser.print_help()


# ============================================================================
# WebDataset 分批共享内存缓存 (Shard Batch Caching)
# ============================================================================

def expand_shard_pattern(shard_pattern: str) -> List[str]:
    """
    展开分片路径模式为文件列表
    
    Args:
        shard_pattern: 包含 brace expansion 的路径模式
                      例如: "/path/shard-{000000..000010}.tar"
    
    Returns:
        展开后的文件路径列表
    """
    import re
    import glob
    
    # 尝试 brace expansion: {000000..000010}
    brace_pattern = re.search(r'\{(\d+)\.\.(\d+)\}', shard_pattern)
    
    if brace_pattern:
        start = int(brace_pattern.group(1))
        end = int(brace_pattern.group(2))
        width = len(brace_pattern.group(1))
        
        prefix = shard_pattern[:brace_pattern.start()]
        suffix = shard_pattern[brace_pattern.end():]
        
        files = [f"{prefix}{i:0{width}d}{suffix}" for i in range(start, end + 1)]
        # 过滤存在的文件
        files = [f for f in files if os.path.exists(f)]
        return files
    else:
        # 尝试 glob 模式
        files = sorted(glob.glob(shard_pattern))
        return files


def _parse_npy_fast(data: bytes) -> np.ndarray:
    """
    快速解析 npy 格式数据（跳过 np.load 的开销）
    
    npy 格式: magic(6) + version(2) + header_len(2/4) + header + data
    """
    # npy magic: \x93NUMPY
    if data[:6] != b'\x93NUMPY':
        # 回退到标准方法
        return np.load(io.BytesIO(data))
    
    version = (data[6], data[7])
    
    if version[0] == 1:
        header_len = int.from_bytes(data[8:10], 'little')
        header_start = 10
    elif version[0] in (2, 3):
        header_len = int.from_bytes(data[8:12], 'little')
        header_start = 12
    else:
        return np.load(io.BytesIO(data))
    
    header_end = header_start + header_len
    header = data[header_start:header_end].decode('latin1')
    
    # 解析 header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (1, 757, 1536)}
    import ast
    try:
        d = ast.literal_eval(header.strip())
    except:
        return np.load(io.BytesIO(data))
    
    dtype = np.dtype(d['descr'])
    shape = d['shape']
    fortran_order = d.get('fortran_order', False)
    
    # 直接从 bytes 创建 ndarray（零拷贝转换）
    arr = np.frombuffer(data, dtype=dtype, offset=header_end)
    
    if fortran_order:
        arr = arr.reshape(shape, order='F')
    else:
        arr = arr.reshape(shape)
    
    # 必须 copy，因为 data (bytes) 是临时对象
    return arr.copy()


def load_shard_to_memory(shard_path: str) -> List[Dict[str, Any]]:
    """
    将单个 tar 分片加载到内存（优化版本）
    
    Args:
        shard_path: tar 文件路径
        
    Returns:
        样本列表，每个样本是一个字典
    """
    samples = []
    
    with tarfile.open(shard_path, 'r') as tar:
        members = tar.getmembers()
        
        # 按 key 分组（使用 defaultdict 减少判断）
        from collections import defaultdict
        sample_dict = defaultdict(dict)
        
        for member in members:
            if member.isfile():
                # 解析文件名: 00000000.vlm.npy -> key=00000000, ext=vlm.npy
                name = member.name
                dot_idx = name.find('.')
                if dot_idx > 0:
                    key = name[:dot_idx]
                    ext = name[dot_idx + 1:]
                    
                    # 读取文件内容
                    f = tar.extractfile(member)
                    if f:
                        sample_dict[key][ext] = f.read()
        
        # 解码并转换为样本列表
        for key in sorted(sample_dict.keys()):
            raw = sample_dict[key]
            sample = {'__key__': key}
            
            # 快速解码 .npy 文件
            if 'vlm.npy' in raw:
                sample['vlm'] = _parse_npy_fast(raw['vlm.npy'])
            if 'state.npy' in raw:
                sample['state'] = _parse_npy_fast(raw['state.npy'])
            if 'action.npy' in raw:
                sample['action'] = _parse_npy_fast(raw['action.npy'])
            
            # 解码 .json 文件
            if 'meta.json' in raw:
                sample['meta'] = json.loads(raw['meta.json'].decode('utf-8'))
            
            samples.append(sample)
    
    return samples


def load_shards_parallel(
    shard_paths: List[str], 
    num_workers: int = 4,
    use_multiprocessing: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int, int, int]:
    """
    并行加载多个分片，同时统计 shape 信息
    
    Args:
        shard_paths: 分片路径列表
        num_workers: 并行工作线程/进程数
        use_multiprocessing: 是否使用多进程（绕过 GIL，但有序列化开销）
        
    Returns:
        (所有样本的列表, max_seq_len, num_layers, hidden_dim, state_dim)
    """
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    
    all_samples = []
    max_seq_len = 0
    num_layers = None
    hidden_dim = None
    state_dim = None
    
    # 选择执行器：
    # - 多线程：适合 IO 密集型，但受 GIL 限制
    # - 多进程：绕过 GIL，但有进程间数据序列化开销
    # 对于 numpy 密集操作，多线程通常够用（numpy 会释放 GIL）
    Executor = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
    
    with Executor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(load_shard_to_memory, path): path for path in shard_paths}
        
        # as_completed 让我们在分片加载完成后立即处理
        for future in as_completed(futures):
            samples = future.result()
            
            # 统计 shape（只需要第一个样本）
            if samples and num_layers is None:
                vlm = samples[0]['vlm']
                if vlm.ndim == 2:
                    num_layers = 1
                    hidden_dim = vlm.shape[1]
                else:
                    num_layers = vlm.shape[0]
                    hidden_dim = vlm.shape[2]
                state_dim = samples[0]['state'].shape[0]
            
            # 统计 max_seq_len
            for sample in samples:
                vlm = sample['vlm']
                seq_len = vlm.shape[1] if vlm.ndim == 3 else vlm.shape[0]
                if seq_len > max_seq_len:
                    max_seq_len = seq_len
            
            all_samples.extend(samples)
    
    return all_samples, max_seq_len, num_layers, hidden_dim, state_dim


def _fill_vlm_data_fast(
    vlm_data: np.ndarray,
    state_data: np.ndarray, 
    action_data: np.ndarray,
    seq_lens: np.ndarray,
    samples: List[Dict],
    dtype: np.dtype,
    max_seq_len: int,
):
    """
    快速填充数据到共享内存（优化版本）
    
    优化策略：
    1. 整体清零一次（比循环中逐个清零快）
    2. 批量处理 state/action（先收集再一次性赋值）
    3. VLM 数据因为需要 padding，仍需循环但优化内存访问
    """
    num_samples = len(samples)
    
    # 1. 整体清零（使用 fill 比 [:] = 0 更快）
    vlm_data.fill(0)
    
    # 2. 批量收集 state 和 action（减少 Python 循环开销）
    states = np.array([s['state'] for s in samples], dtype=np.float32)
    actions = np.array([s['action'] for s in samples], dtype=np.float32)
    
    # 一次性赋值（利用 numpy 的内存连续性优化）
    np.copyto(state_data, states)
    np.copyto(action_data, actions)
    
    # 释放临时数组
    del states, actions
    
    # 3. VLM 数据需要按样本 padding，但可以优化内存访问模式
    # 按 seq_len 分组处理，减少 if 判断
    for i in range(num_samples):
        vlm = samples[i]['vlm']
        
        if vlm.ndim == 2:
            vlm = vlm[np.newaxis, :, :]  # 比 reshape 更快
        
        actual_seq_len = vlm.shape[1]
        seq_lens[i] = actual_seq_len
        
        # Left padding: 数据放在右边
        start_idx = max_seq_len - actual_seq_len
        
        # 直接使用 numpy 的内存视图，避免类型转换
        if vlm.dtype == dtype:
            vlm_data[i, :, start_idx:, :] = vlm
        else:
            # 类型转换时使用 out 参数避免额外分配
            vlm_data[i, :, start_idx:, :] = vlm.astype(dtype, copy=False)


class ShardBatchCache:
    """
    分批加载 WebDataset 分片到共享内存的缓存管理器
    
    工作流程:
    1. 将所有分片分成若干批次
    2. 每批次加载到共享内存
    3. 训练完当前批次后，清除并加载下一批
    
    Usage:
        cache_manager = ShardBatchCache(
            shard_pattern="/path/shard-{000000..000010}.tar",
            shards_per_batch=4,
            rank=rank,
            world_size=world_size,
        )
        
        while step < total_steps:
            # 加载下一批分片
            samples, seq_lens = cache_manager.load_next_batch()
            
            # 创建 DataLoader 并训练
            dataset = CachedSamplesDataset(samples, seq_lens)
            for batch in DataLoader(dataset, ...):
                train_step(batch)
    """
    
    def __init__(
        self,
        shard_pattern: str,
        shards_per_batch: int = 4,
        cache_dtype: str = "float32",
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
        shuffle_shards: bool = True,
        seed: int = 42,
    ):
        """
        初始化分批缓存管理器
        
        Args:
            shard_pattern: tar 分片路径模式
            shards_per_batch: 每批加载的分片数量
            cache_dtype: 缓存数据类型 ("float32", "float16")
            rank: 当前进程 rank
            world_size: 总进程数
            verbose: 是否打印信息
            shuffle_shards: 是否在每个 epoch 开始时打乱分片顺序
            seed: 随机种子（用于分布式同步）
        """
        self.shard_pattern = shard_pattern
        self.shards_per_batch = shards_per_batch
        self.cache_dtype = cache_dtype
        self.rank = rank
        self.world_size = world_size
        self.verbose = verbose
        self.shuffle_shards = shuffle_shards
        self.base_seed = seed
        self.current_epoch = 0
        
        # 展开分片路径
        self.all_shards = expand_shard_pattern(shard_pattern)
        self.num_shards = len(self.all_shards)
        
        if self.num_shards == 0:
            raise ValueError(f"No shard files found for pattern: {shard_pattern}")
        
        # 打乱后的分片顺序（每个 epoch 可能不同）
        self.shuffled_shards = self.all_shards.copy()
        if shuffle_shards:
            self._shuffle_shards_for_epoch(0)
        
        # 计算批次数量
        self.num_batches = (self.num_shards + shards_per_batch - 1) // shards_per_batch
        self.current_batch_idx = 0
        
        # 当前批次的数据
        self.current_samples = None
        self.current_vlm_data = None
        self.current_state_data = None
        self.current_action_data = None
        self.current_meta_data = None
        self.current_seq_lens = None
        
        # 共享内存句柄
        self.shm_vlm = None
        self.shm_state = None
        self.shm_action = None
        self.shm_seqlens = None
        
        if verbose and rank == 0:
            print(f"\n{'='*60}")
            print(f"📦 ShardBatchCache initialized:")
            print(f"   Total shards: {self.num_shards}")
            print(f"   Shards per batch: {shards_per_batch}")
            print(f"   Number of batches: {self.num_batches}")
            print(f"   Cache dtype: {cache_dtype}")
            print(f"   Shuffle shards: {shuffle_shards}")
            if shuffle_shards:
                print(f"   Initial shard order: {[os.path.basename(s) for s in self.shuffled_shards[:min(5, len(self.shuffled_shards))]]}" + 
                      ("..." if len(self.shuffled_shards) > 5 else ""))
            print(f"{'='*60}")
    
    def _shuffle_shards_for_epoch(self, epoch: int):
        """
        为指定 epoch 打乱分片顺序
        
        使用固定的 seed + epoch 保证所有进程的打乱顺序一致
        """
        import random
        
        # 使用 epoch 相关的种子，保证：
        # 1. 不同 epoch 顺序不同
        # 2. 所有进程（rank）顺序一致
        rng = random.Random(self.base_seed + epoch)
        self.shuffled_shards = self.all_shards.copy()
        rng.shuffle(self.shuffled_shards)
        self.current_epoch = epoch
        
        if self.verbose and self.rank == 0:
            print(f"🔀 Shuffled shards for epoch {epoch}: {[os.path.basename(s) for s in self.shuffled_shards[:min(5, len(self.shuffled_shards))]]}" +
                  ("..." if len(self.shuffled_shards) > 5 else ""))
    
    def set_epoch(self, epoch: int):
        """
        设置当前 epoch 并重新打乱分片顺序
        
        在每个 epoch 开始时调用此方法以获得不同的分片顺序
        """
        if self.shuffle_shards and epoch != self.current_epoch:
            self._shuffle_shards_for_epoch(epoch)
        self.current_batch_idx = 0
    
    def get_batch_shards(self, batch_idx: int) -> List[str]:
        """获取指定批次的分片文件列表（使用打乱后的顺序）"""
        start = batch_idx * self.shards_per_batch
        end = min(start + self.shards_per_batch, self.num_shards)
        return self.shuffled_shards[start:end]
    
    def load_next_batch(self) -> Tuple[int, 'CachedSamplesDataset']:
        """
        加载下一批分片到内存
        
        Returns:
            (num_samples, CachedSamplesDataset): 样本数量和数据集对象
        """
        import torch.distributed as dist
        
        # 同步所有进程
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        # 清除当前批次数据
        self._clear_current_batch()
        
        # 获取当前批次的分片
        batch_shards = self.get_batch_shards(self.current_batch_idx)
        
        if self.verbose and self.rank == 0:
            print(f"\n📂 Loading batch {self.current_batch_idx + 1}/{self.num_batches}:")
            print(f"   Shards: {[os.path.basename(s) for s in batch_shards]}")
        
        # ========== 使用共享内存：rank 0 加载，其他 rank 共享访问 ==========
        import torch
        from multiprocessing import shared_memory
        
        # 共享内存名称（基于批次索引，避免冲突）
        shm_prefix = f"wds_cache_batch{self.current_batch_idx}"
        
        if self.rank == 0:
            # rank 0 加载数据（使用并行加载加速，同时统计 shape）
            if self.verbose:
                print(f"   Loading {len(batch_shards)} shards in parallel...")
            
            import time
            load_start = time.time()
            all_samples, max_seq_len, num_layers, hidden_dim, state_dim = load_shards_parallel(
                batch_shards, num_workers=min(len(batch_shards), 8)
            )
            load_time = time.time() - load_start
            
            num_samples = len(all_samples)
            
            if self.verbose:
                print(f"   ✓ Loaded in {load_time:.2f}s ({num_samples} samples)")
            
            if num_samples == 0:
                raise ValueError(f"No samples found in batch {self.current_batch_idx}")
            
            # action shape 需要单独获取
            action_shape = all_samples[0]['action'].shape
            
            if self.verbose:
                print(f"   Max seq_len: {max_seq_len}")
                print(f"   VLM shape: ({num_layers}, {max_seq_len}, {hidden_dim})")
            
            # 分配内存并填充数据
            dtype = np.float16 if self.cache_dtype == "float16" else np.float32
            
            vlm_shape = (num_samples, num_layers, max_seq_len, hidden_dim)
            state_shape = (num_samples, state_dim)
            action_shape_full = (num_samples, *action_shape)
            seq_lens_shape = (num_samples,)
            
            vlm_size = int(np.prod(vlm_shape)) * (2 if dtype == np.float16 else 4)
            state_size = int(np.prod(state_shape)) * 4
            action_size = int(np.prod(action_shape_full)) * 4
            seq_lens_size = num_samples * 4
            
            # 创建共享内存
            try:
                # 先尝试删除可能存在的旧共享内存
                for suffix in ['_vlm', '_state', '_action', '_seqlens']:
                    try:
                        old_shm = shared_memory.SharedMemory(name=f"{shm_prefix}{suffix}")
                        old_shm.close()
                        old_shm.unlink()
                    except:
                        pass
                
                self.shm_vlm = shared_memory.SharedMemory(create=True, size=vlm_size, name=f"{shm_prefix}_vlm")
                self.shm_state = shared_memory.SharedMemory(create=True, size=state_size, name=f"{shm_prefix}_state")
                self.shm_action = shared_memory.SharedMemory(create=True, size=action_size, name=f"{shm_prefix}_action")
                self.shm_seqlens = shared_memory.SharedMemory(create=True, size=seq_lens_size, name=f"{shm_prefix}_seqlens")
            except Exception as e:
                print(f"⚠️ 创建共享内存失败: {e}")
                raise
            
            # 创建 numpy 数组视图
            self.current_vlm_data = np.ndarray(vlm_shape, dtype=dtype, buffer=self.shm_vlm.buf)
            self.current_state_data = np.ndarray(state_shape, dtype=np.float32, buffer=self.shm_state.buf)
            self.current_action_data = np.ndarray(action_shape_full, dtype=np.float32, buffer=self.shm_action.buf)
            self.current_seq_lens = np.ndarray(seq_lens_shape, dtype=np.int32, buffer=self.shm_seqlens.buf)
            self.current_meta_data = []
            
            # 快速填充数据到共享内存
            fill_start = time.time()
            _fill_vlm_data_fast(
                self.current_vlm_data,
                self.current_state_data,
                self.current_action_data,
                self.current_seq_lens,
                all_samples,
                dtype,
                max_seq_len,
            )
            
            # meta 数据单独处理
            self.current_meta_data = [sample.get('meta', {}) for sample in all_samples]
            
            if self.verbose:
                fill_time = time.time() - fill_start
                print(f"   ✓ Filled shared memory in {fill_time:.2f}s")
            
            # 释放原始样本内存
            del all_samples
            
            if self.verbose:
                vlm_size_gb = self.current_vlm_data.nbytes / 1024**3
                print(f"   VLM shared memory: {vlm_size_gb:.2f} GB")
            
            # 广播 shape 信息给其他 rank
            shape_info = torch.tensor([num_samples, num_layers, max_seq_len, hidden_dim, 
                                       state_dim, action_shape[0], action_shape[1] if len(action_shape) > 1 else 0], 
                                      dtype=torch.long, device='cuda')
        else:
            # 其他 rank 等待接收 shape 信息
            shape_info = torch.zeros(7, dtype=torch.long, device='cuda')
        
        # 同步并广播 shape 信息
        if self.world_size > 1 and dist.is_initialized():
            dist.broadcast(shape_info, src=0)
            dist.barrier()
        
        if self.rank != 0:
            # 其他 rank 从 shape_info 解析信息
            num_samples = int(shape_info[0].item())
            num_layers = int(shape_info[1].item())
            max_seq_len = int(shape_info[2].item())
            hidden_dim = int(shape_info[3].item())
            state_dim = int(shape_info[4].item())
            action_dim1 = int(shape_info[5].item())
            action_dim2 = int(shape_info[6].item())
            action_shape = (action_dim1, action_dim2) if action_dim2 > 0 else (action_dim1,)
            
            dtype = np.float16 if self.cache_dtype == "float16" else np.float32
            
            vlm_shape = (num_samples, num_layers, max_seq_len, hidden_dim)
            state_shape = (num_samples, state_dim)
            action_shape_full = (num_samples, *action_shape)
            seq_lens_shape = (num_samples,)
            
            # 连接到共享内存
            try:
                self.shm_vlm = shared_memory.SharedMemory(name=f"{shm_prefix}_vlm")
                self.shm_state = shared_memory.SharedMemory(name=f"{shm_prefix}_state")
                self.shm_action = shared_memory.SharedMemory(name=f"{shm_prefix}_action")
                self.shm_seqlens = shared_memory.SharedMemory(name=f"{shm_prefix}_seqlens")
            except Exception as e:
                print(f"⚠️ [Rank {self.rank}] 连接共享内存失败: {e}")
                raise
            
            # 创建 numpy 数组视图
            self.current_vlm_data = np.ndarray(vlm_shape, dtype=dtype, buffer=self.shm_vlm.buf)
            self.current_state_data = np.ndarray(state_shape, dtype=np.float32, buffer=self.shm_state.buf)
            self.current_action_data = np.ndarray(action_shape_full, dtype=np.float32, buffer=self.shm_action.buf)
            self.current_seq_lens = np.ndarray(seq_lens_shape, dtype=np.int32, buffer=self.shm_seqlens.buf)
            self.current_meta_data = [{} for _ in range(num_samples)]  # 其他 rank 不需要 meta_data
        
        # 同步
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        # 更新批次索引（循环）
        self.current_batch_idx = (self.current_batch_idx + 1) % self.num_batches
        
        # 创建数据集
        dataset = CachedSamplesDataset(
            vlm_data=self.current_vlm_data,
            state_data=self.current_state_data,
            action_data=self.current_action_data,
            meta_data=self.current_meta_data,
            seq_lens=self.current_seq_lens,
        )
        
        return len(self.current_vlm_data), dataset
    
    def _clear_current_batch(self):
        """清除当前批次的数据和共享内存"""
        import torch.distributed as dist
        
        # 同步所有进程
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        # 清除 numpy 数组引用
        self.current_vlm_data = None
        self.current_state_data = None
        self.current_action_data = None
        self.current_meta_data = None
        self.current_seq_lens = None
        
        # 关闭共享内存连接
        for attr in ['shm_vlm', 'shm_state', 'shm_action', 'shm_seqlens']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                try:
                    shm = getattr(self, attr)
                    shm.close()
                    # 只有 rank 0 负责删除共享内存
                    if self.rank == 0:
                        try:
                            shm.unlink()
                        except:
                            pass
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ [Rank {self.rank}] 清理共享内存 {attr} 时出错: {e}")
                setattr(self, attr, None)
        
        # 同步确保所有进程都完成清理
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        # 强制垃圾回收
        import gc
        gc.collect()
    
    def get_total_samples_estimate(self) -> int:
        """估算总样本数（基于分片数量）"""
        # 假设每个分片有相同数量的样本
        # 实际使用时可以从 meta.json 获取精确值
        return self.num_shards * 5000  # 默认估计
    
    def cleanup(self):
        """清理所有资源"""
        self._clear_current_batch()


class CachedSamplesDataset(torch.utils.data.Dataset):
    """
    基于内存缓存的数据集
    
    从 ShardBatchCache 加载的数据创建 PyTorch Dataset
    """
    
    def __init__(
        self,
        vlm_data: np.ndarray,
        state_data: np.ndarray,
        action_data: np.ndarray,
        meta_data: List[Dict],
        seq_lens: np.ndarray,
    ):
        """
        初始化数据集
        
        Args:
            vlm_data: VLM hidden states, shape (N, num_layers, max_seq_len, hidden_dim)
            state_data: Observation states, shape (N, state_dim)
            action_data: Actions, shape (N, num_chunks, action_dim)
            meta_data: 元数据列表
            seq_lens: 每个样本的实际 seq_len, shape (N,)
        """
        self.vlm_data = vlm_data
        self.state_data = state_data
        self.action_data = action_data
        self.meta_data = meta_data
        self.seq_lens = seq_lens
        self.num_samples = len(vlm_data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        actual_seq_len = int(self.seq_lens[idx])
        max_seq_len = self.vlm_data.shape[2]
        
        # 返回实际数据的副本（重要：共享内存必须 copy，否则会有数据安全问题）
        vlm = self.vlm_data[idx, :, max_seq_len - actual_seq_len:, :].copy()
        state = self.state_data[idx].copy()
        action = self.action_data[idx].copy()
        
        return {
            'vlm_hidden_states': vlm,  # (num_layers, actual_seq_len, hidden_dim)
            'observation_state': state,
            'actions': action,
            'meta': self.meta_data[idx] if idx < len(self.meta_data) else {},
            'seq_len': actual_seq_len,
        }


def cached_samples_collate_fn(
    batch: List[Dict[str, Any]],
    vlm_dtype: str = "bfloat16",
) -> Dict[str, torch.Tensor]:
    """
    CachedSamplesDataset 的 collate 函数
    
    处理可变序列长度，进行 left padding
    """
    batch_size = len(batch)
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    target_vlm_dtype = dtype_map.get(vlm_dtype, torch.bfloat16)
    
    # VLM hidden states with padding
    vlm_tensors = [torch.from_numpy(sample['vlm_hidden_states']).to(target_vlm_dtype) for sample in batch]
    max_seq_len = max(tensor.size(1) for tensor in vlm_tensors)
    
    padded_vlm_tensors = []
    vlm_attention_masks = []
    
    for tensor in vlm_tensors:
        num_layers, seq_len, hidden_dim = tensor.shape
        pad_len = max_seq_len - seq_len
        
        mask = torch.ones(num_layers, max_seq_len, dtype=torch.long)
        
        if pad_len > 0:
            padding = torch.zeros(num_layers, pad_len, hidden_dim, dtype=tensor.dtype)
            mask[:, :pad_len] = 0  # Left padding
            tensor = torch.cat([padding, tensor], dim=1)
        
        padded_vlm_tensors.append(tensor)
        vlm_attention_masks.append(mask)
    
    vlm_hidden_states = torch.stack(padded_vlm_tensors)
    vlm_attention_mask = torch.stack(vlm_attention_masks)
    
    # Observation states
    observation_states = torch.stack([
        torch.from_numpy(sample['observation_state']).float() for sample in batch
    ])
    
    # Actions
    actions = torch.stack([
        torch.from_numpy(sample['actions']).float() for sample in batch
    ])
    
    return {
        'vlm_hidden_states': vlm_hidden_states,
        'vlm_attention_mask': vlm_attention_mask,
        'observation_state': observation_states,
        'actions': actions,
    }


if __name__ == "__main__":
    main()

