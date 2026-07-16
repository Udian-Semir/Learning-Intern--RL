"""
Sai0_1 数据工具模块

整合 LeRobot Dataset Loader，提供统一的数据加载接口。

使用示例:
    from VLAs.Sai0_1 import create_dataloader, Sai0Config
    
    config = Sai0Config.for_qwen3_libero(dataset_path="/path/to/dataset")
    dataloader = create_dataloader(config)
    
    for batch in dataloader:
        # 使用数据
        pass
"""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers.feature_extraction_utils import BatchFeature

# 添加 utils 路径
_UTILS_PATH = Path(__file__).parent.parent.parent / 'utils'
if str(_UTILS_PATH) not in sys.path:
    sys.path.insert(0, str(_UTILS_PATH))

from lerobot_dataset_loader import (
    LeRobotDataset,
    collate_fn as lerobot_collate_fn,
    create_lerobot_dataloader,
)


# ============================================================================
# 常量
# ============================================================================

DEFAULT_EMBODIMENT_ID = 31


# ============================================================================
# 四元数转轴角
# ============================================================================

def quat2axisangle_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    四元数转轴角 (PyTorch 批量版本)
    
    参考: Isaac-GR00T/examples/Libero/eval/utils.py quat2axisangle()
    
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
    将 9 维 state 转换为 8 维 state (四元数→轴角)
    
    用户 state 格式 (9维):
        [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
    
    转换后 (8维):
        [gripper1, gripper2, x, y, z, ax, ay, az]
    """
    gripper = state[:, 0:2]
    position = state[:, 2:5]
    quat = state[:, 5:9]
    axis_angle = quat2axisangle_torch(quat)
    return torch.cat([gripper, position, axis_angle], dim=1)


# ============================================================================
# 归一化器
# ============================================================================

class MinMaxNormalizer:
    """
    Min-Max 归一化器，与原始 GR00T N1.5 完全对齐
    
    归一化公式: normalized = 2 * (x - min) / (max - min) - 1
    输出范围: [-1, 1]
    """
    
    def __init__(self, min_vals: torch.Tensor, max_vals: torch.Tensor):
        self.min_vals = min_vals
        self.max_vals = max_vals
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化到 [-1, 1]"""
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        
        mask = min_vals != max_vals
        normalized = torch.zeros_like(x)
        
        if mask.any():
            normalized[..., mask] = (x[..., mask] - min_vals[mask]) / (max_vals[mask] - min_vals[mask])
            normalized[..., mask] = 2.0 * normalized[..., mask] - 1.0
        
        return normalized
    
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化"""
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        return (x + 1.0) / 2.0 * (max_vals - min_vals) + min_vals


def load_normalization_stats(
    dataset_path: str,
    convert_quat_to_axisangle: bool = True
) -> Optional[Dict[str, MinMaxNormalizer]]:
    """
    从数据集加载归一化统计信息
    
    Args:
        dataset_path: 数据集路径
        convert_quat_to_axisangle: 是否进行 9维→8维 转换
    
    Returns:
        包含 'state' 和 'action' 归一化器的字典
    """
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"[WARN] stats.json 不存在于 {stats_path}，将不进行归一化")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    normalizers = {}
    
    # State 归一化
    if 'observation.state' in stats:
        state_stats = stats['observation.state']
        original_min = state_stats['min']
        original_max = state_stats['max']
        original_dim = len(original_min)
        
        if convert_quat_to_axisangle and original_dim == 9:
            # 9维 → 8维
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
        else:
            state_min = torch.tensor(original_min, dtype=torch.float32)
            state_max = torch.tensor(original_max, dtype=torch.float32)
        
        normalizers['state'] = MinMaxNormalizer(state_min, state_max)
    
    # Action 归一化
    if 'action' in stats:
        action_stats = stats['action']
        action_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        action_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        normalizers['action'] = MinMaxNormalizer(action_min, action_max)
    
    return normalizers if normalizers else None


# ============================================================================
# Sai0 专用 Collate 函数
# ============================================================================

def sai0_collate_fn(
    batch: List[Dict[str, Any]],
    max_state_dim: int = 64,
    max_action_dim: int = 32,
    normalizers: Optional[Dict[str, MinMaxNormalizer]] = None,
    convert_quat_to_axisangle: bool = True,
    embodiment_id: int = DEFAULT_EMBODIMENT_ID,
) -> Tuple[BatchFeature, BatchFeature]:
    """
    Sai0 专用 collate 函数
    
    将 LeRobot batch 转换为 FlowMatching Action Head 需要的格式
    
    Args:
        batch: LeRobot dataloader 返回的 batch
        max_state_dim: 最大状态维度
        max_action_dim: 最大动作维度
        normalizers: 归一化器字典
        convert_quat_to_axisangle: 是否将四元数转为轴角
        embodiment_id: Embodiment ID
    
    Returns:
        (backbone_output, action_head_inputs) 元组
    """
    # 使用 LeRobot 的基础 collate
    collated = lerobot_collate_fn(batch)
    
    vlm_tensor_raw = collated['vlm_hidden_states']  # (batch, num_layers, seq_len, hidden_dim)
    observation_state = collated['observation_state']  # (batch, state_dim)
    actions = collated['actions']  # (batch, num_chunks, action_dim)
    
    # 处理 VLM hidden states: 取第一层
    if vlm_tensor_raw.dim() == 4:
        vlm_tensor = vlm_tensor_raw[:, 0, :, :]
    else:
        vlm_tensor = vlm_tensor_raw
    
    batch_size = vlm_tensor.size(0)
    seq_len = vlm_tensor.size(1)
    num_chunks = actions.size(1)
    
    # State 四元数转轴角
    if convert_quat_to_axisangle and observation_state.size(1) == 9:
        observation_state = convert_state_quat_to_axisangle(observation_state)
    
    actual_state_dim = observation_state.size(1)
    actual_action_dim = actions.size(2)
    
    # 归一化
    if normalizers is not None:
        if 'state' in normalizers:
            observation_state = normalizers['state'].normalize(observation_state)
        if 'action' in normalizers:
            actions_flat = actions.reshape(batch_size * num_chunks, actual_action_dim)
            actions_normalized = normalizers['action'].normalize(actions_flat)
            actions = actions_normalized.reshape(batch_size, num_chunks, actual_action_dim)
    
    # VLM 特征
    backbone_features = vlm_tensor
    backbone_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=vlm_tensor.device)
    
    # State padding
    n_state_dims = actual_state_dim
    if n_state_dims > max_state_dim:
        observation_state = observation_state[:, :max_state_dim]
        n_state_dims = max_state_dim
    else:
        padding = torch.zeros(batch_size, max_state_dim - n_state_dims,
                             dtype=observation_state.dtype, device=observation_state.device)
        observation_state = torch.cat([observation_state, padding], dim=1)
    
    state = observation_state.unsqueeze(1)  # (batch, 1, max_state_dim)
    state_mask = torch.zeros(batch_size, 1, max_state_dim, dtype=torch.bool, device=state.device)
    state_mask[:, :, :n_state_dims] = True
    
    # Action padding
    n_action_dims = actual_action_dim
    if n_action_dims > max_action_dim:
        actions = actions[:, :, :max_action_dim]
        n_action_dims = max_action_dim
    else:
        padding = torch.zeros(batch_size, num_chunks, max_action_dim - n_action_dims,
                             dtype=actions.dtype, device=actions.device)
        actions = torch.cat([actions, padding], dim=2)
    
    action = actions
    action_mask = torch.zeros_like(action)
    action_mask[:, :, :n_action_dims] = 1.0
    
    # Embodiment ID
    embodiment_ids = torch.full(
        (batch_size,), embodiment_id, dtype=torch.long, device=vlm_tensor.device
    )
    
    # 构建输出
    backbone_output = BatchFeature(data={
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attention_mask,
    })
    
    action_head_inputs = BatchFeature(data={
        "state": state,
        "action": action,
        "action_mask": action_mask,
        "embodiment_id": embodiment_ids,
    })
    
    # 附加元信息
    action_head_inputs["_meta"] = {
        "actual_state_dim": n_state_dims,
        "actual_action_dim": n_action_dims,
        "task_descriptions": collated.get('task_description', None),
        "episode_index": collated.get('episode_index', None),
        "frame_index": collated.get('frame_index', None),
    }
    
    return backbone_output, action_head_inputs


# ============================================================================
# DataLoader 创建函数
# ============================================================================

def create_sai0_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    num_action_chunks: int = 16,
    enable_chunking: bool = True,
    max_state_dim: int = 64,
    max_action_dim: int = 32,
    convert_quat_to_axisangle: bool = True,
    use_normalization: bool = True,
    embodiment_id: int = DEFAULT_EMBODIMENT_ID,
    cache_vlm_states: bool = False,
    max_cached_video_readers: int = 32,
    episode_indices: Optional[List[int]] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    verbose: bool = True,
) -> Tuple[DataLoader, Optional[Dict[str, MinMaxNormalizer]]]:
    """
    创建 Sai0 专用 DataLoader
    
    Args:
        dataset_path: LeRobot 数据集路径
        batch_size: 批次大小
        num_workers: 数据加载 worker 数
        shuffle: 是否 shuffle
        num_action_chunks: Action chunk 数量
        enable_chunking: 是否启用 chunking
        max_state_dim: 最大状态维度
        max_action_dim: 最大动作维度
        convert_quat_to_axisangle: 是否将四元数转为轴角
        use_normalization: 是否使用归一化
        embodiment_id: Embodiment ID
        cache_vlm_states: 是否缓存 VLM states
        max_cached_video_readers: 视频 reader 缓存上限
        episode_indices: 特定 episode 索引列表
        distributed: 是否分布式训练
        rank: 进程排名
        world_size: 总进程数
        verbose: 详细输出
    
    Returns:
        (DataLoader, normalizers) 元组
    """
    # 加载归一化统计
    normalizers = None
    if use_normalization:
        normalizers = load_normalization_stats(dataset_path, convert_quat_to_axisangle)
    
    # 创建 Dataset
    dataset = LeRobotDataset(
        dataset_path=dataset_path,
        split="train",
        num_action_chunks=num_action_chunks,
        enable_chunking=enable_chunking,
        episode_indices=episode_indices,
        cache_vlm_states=cache_vlm_states,
        verbose=verbose and (rank == 0),
        max_cached_video_readers=max_cached_video_readers,
    )
    
    # 创建 Sampler
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # 使用 sampler 时禁用 shuffle
    
    # 创建 collate 函数
    from functools import partial
    collate = partial(
        sai0_collate_fn,
        max_state_dim=max_state_dim,
        max_action_dim=max_action_dim,
        normalizers=normalizers,
        convert_quat_to_axisangle=convert_quat_to_axisangle,
        embodiment_id=embodiment_id,
    )
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        sampler=sampler,
        drop_last=True,  # 分布式训练时丢弃不完整 batch
    )
    
    return dataloader, normalizers


def create_dataloader_from_config(config) -> Tuple[DataLoader, Optional[Dict[str, MinMaxNormalizer]]]:
    """
    从 Sai0Config 创建 DataLoader
    
    Args:
        config: Sai0Config 实例
    
    Returns:
        (DataLoader, normalizers) 元组
    """
    return create_sai0_dataloader(
        dataset_path=config.data.dataset_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        num_action_chunks=config.data.num_action_chunks,
        enable_chunking=config.data.enable_chunking,
        max_state_dim=config.action_head.max_state_dim,
        max_action_dim=config.action_head.max_action_dim,
        convert_quat_to_axisangle=config.action_head.convert_quat_to_axisangle,
        use_normalization=config.action_head.use_normalization,
        embodiment_id=config.action_head.embodiment_id,
        cache_vlm_states=config.data.cache_vlm_states,
        max_cached_video_readers=config.data.max_cached_video_readers,
    )


# ============================================================================
# 数据集信息工具
# ============================================================================

def get_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """
    获取数据集信息
    
    Args:
        dataset_path: 数据集路径
    
    Returns:
        数据集信息字典
    """
    info_path = Path(dataset_path) / "meta" / "info.json"
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # 加载任务数
    tasks_path = Path(dataset_path) / "meta" / "tasks.jsonl"
    task_count = 0
    tasks = []
    if tasks_path.exists():
        with open(tasks_path, 'r') as f:
            for line in f:
                task = json.loads(line.strip())
                tasks.append(task['task'])
                task_count += 1
    
    return {
        "dataset_path": dataset_path,
        "total_episodes": info.get('total_episodes', 0),
        "total_frames": info.get('total_frames', 0),
        "fps": info.get('fps', 0),
        "action_dim": info['features']['action']['shape'][0] if 'features' in info else 0,
        "state_dim": info['features']['observation.state']['shape'][0] if 'features' in info else 0,
        "total_tasks": task_count,
        "tasks": tasks,
    }


def print_dataset_info(dataset_path: str):
    """打印数据集信息"""
    info = get_dataset_info(dataset_path)
    
    print("=" * 70)
    print("Dataset Information")
    print("=" * 70)
    print(f"Path: {info['dataset_path']}")
    print(f"Total Episodes: {info['total_episodes']}")
    print(f"Total Frames: {info['total_frames']}")
    print(f"FPS: {info['fps']}")
    print(f"Action Dim: {info['action_dim']}")
    print(f"State Dim: {info['state_dim']}")
    print(f"Total Tasks: {info['total_tasks']}")
    if info['tasks']:
        print("\nTasks:")
        for i, task in enumerate(info['tasks'][:5]):  # 只显示前5个
            print(f"  {i}: {task}")
        if len(info['tasks']) > 5:
            print(f"  ... and {len(info['tasks']) - 5} more")
    print("=" * 70)

