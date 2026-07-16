"""
OFT Action Head 轨迹评估脚本

功能:
- 加载 OFT checkpoint 和数据集
- 进行 state 预处理 (hand_binary + minmax 归一化)
- 进行 action 反归一化 (从 [-1,1] 到原始范围)
- 支持单帧测试和轨迹评估模式
- 每个 chunk 从 GT 位置开始预测，评估单个 chunk 的预测质量

使用方法:
    # 单帧测试
    python tools/evaluate_oft_action_trajectory.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet \
        --frame_idx 0

    # 轨迹评估 (每隔 chunk_size 帧从 GT 位置开始预测)
    python tools/evaluate_oft_action_trajectory.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet \
        --trajectory_mode \
        --chunk_size 25 \
        --dims 0 1 2 \
        --plot

    # 指定 stats.json 路径
    python tools/evaluate_oft_action_trajectory.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet \
        --stats_path /path/to/meta/stats.json \
        --trajectory_mode \
        --plot
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch

# 设置环境变量避免 tokenizer 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Action_Heads.OFT1_0.vlm2oft_pipeline import create_vlm2oft_pipeline


# ============================================================================
# 归一化工具类
# ============================================================================

class MinMaxNormalizer:
    """Min-Max 归一化器 (与训练时一致)"""
    
    def __init__(self, min_vals: np.ndarray, max_vals: np.ndarray):
        self.min_vals = min_vals.astype(np.float32)
        self.max_vals = max_vals.astype(np.float32)
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """归一化到 [-1, 1]"""
        mask = self.min_vals != self.max_vals
        normalized = np.zeros_like(x, dtype=np.float32)
        if mask.any():
            normalized[..., mask] = (x[..., mask] - self.min_vals[mask]) / (self.max_vals[mask] - self.min_vals[mask])
            normalized[..., mask] = 2.0 * normalized[..., mask] - 1.0
        return normalized
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """从 [-1, 1] 反归一化"""
        return (x + 1.0) / 2.0 * (self.max_vals - self.min_vals) + self.min_vals


def load_stats(stats_path: str) -> Dict[str, Any]:
    """加载 stats.json"""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats


def create_action_denormalizer(stats: Dict[str, Any]) -> MinMaxNormalizer:
    """从 stats 创建 action 反归一化器"""
    action_stats = stats.get('action', {})
    action_min = np.array(action_stats.get('min', []), dtype=np.float32)
    action_max = np.array(action_stats.get('max', []), dtype=np.float32)
    return MinMaxNormalizer(action_min, action_max)


def preprocess_state(
    state: np.ndarray,
    config: Dict[str, Any],
    stats: Dict[str, Any] = None
) -> np.ndarray:
    """
    State 预处理 (与训练时一致)
    
    处理顺序由 config['state_process_order'] 决定:
    1. hand_binary: 手部数据二值化 (多维 -> 1维)
    2. minmax: MinMax 归一化到 [-1, 1]
    
    Args:
        state: 原始 state
        config: checkpoint 的 config.json
        stats: 数据集的 stats.json (用于 minmax 归一化)
    """
    state = state.copy().astype(np.float32)
    
    state_process_order = config.get('state_process_order', [])
    
    # hand_binary 配置
    hand_binary_columns = config.get('hand_binary_columns', [])
    hand_binary_threshold = config.get('hand_binary_threshold', 442.0)
    
    # minmax 配置
    state_norm_columns_minmax = config.get('state_norm_columns_minmax', [])
    state_min = None
    state_max = None
    if stats and 'observation.state' in stats:
        state_min = np.array(stats['observation.state'].get('min', []), dtype=np.float32)
        state_max = np.array(stats['observation.state'].get('max', []), dtype=np.float32)
    
    for processor in state_process_order:
        if processor == 'hand_binary' and hand_binary_columns:
            # hand_binary: 每组 [start, end) 的数据取平均，> threshold -> 1, else -> 0
            # hand_binary_columns: [start1, end1, start2, end2, ...]
            offset = 0
            new_state_parts = []
            last_end = 0
            
            for i in range(0, len(hand_binary_columns), 2):
                if i + 1 < len(hand_binary_columns):
                    start = hand_binary_columns[i] + offset
                    end = hand_binary_columns[i + 1] + offset
                    
                    if start < len(state) and end <= len(state):
                        # 保留 start 之前的部分
                        if last_end < start:
                            new_state_parts.append(state[last_end:start])
                        
                        # 计算二值化结果
                        hand_data = state[start:end]
                        binary_val = 1.0 if np.mean(hand_data) > hand_binary_threshold else 0.0
                        new_state_parts.append(np.array([binary_val]))
                        
                        last_end = end
                        offset -= (end - start - 1)
            
            # 添加剩余部分
            if last_end < len(state):
                new_state_parts.append(state[last_end:])
            
            state = np.concatenate(new_state_parts) if new_state_parts else state
    
    # MinMax 归一化 (在 hand_binary 处理后进行)
    if state_norm_columns_minmax and state_min is not None and state_max is not None:
        for col in state_norm_columns_minmax:
            if col < len(state) and col < len(state_min):
                col_range = state_max[col] - state_min[col]
                if col_range > 1e-8:
                    # 归一化到 [-1, 1]
                    normalized = (state[col] - state_min[col]) / col_range
                    state[col] = 2.0 * normalized - 1.0
                else:
                    state[col] = 0.0
    
    return state


def load_config(checkpoint_path: str) -> Dict[str, Any]:
    """从 checkpoint 目录加载 config.json"""
    ckpt_dir = Path(checkpoint_path).parent
    config_path = ckpt_dir / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.json 不存在: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def load_parquet_frame(parquet_path: str, frame_idx: int = 0) -> Dict[str, Any]:
    """
    从 parquet 文件加载指定帧的数据
    
    Returns:
        dict containing:
        - observation_state: np.ndarray
        - action: np.ndarray (ground truth)
        - frame_index: int (global frame index)
        - timestamp: float
    """
    df = pd.read_parquet(parquet_path)
    
    if frame_idx >= len(df):
        raise ValueError(f"frame_idx {frame_idx} 超出范围，总帧数: {len(df)}")
    
    row = df.iloc[frame_idx]
    
    result = {
        'frame_idx_in_episode': frame_idx,
    }
    
    # 加载 state
    if 'observation.state' in df.columns:
        result['observation_state'] = np.array(row['observation.state'], dtype=np.float32)
    elif 'state' in df.columns:
        result['observation_state'] = np.array(row['state'], dtype=np.float32)
    else:
        print(f"警告: 未找到 state 列")
        result['observation_state'] = None
    
    # 加载 action (ground truth)
    if 'action' in df.columns:
        result['action'] = np.array(row['action'], dtype=np.float32)
    else:
        result['action'] = None
    
    # 加载 frame_index (全局帧索引，用于找 VLM hidden states)
    if 'frame_index' in df.columns:
        result['frame_index'] = int(row['frame_index'])
    elif 'index' in df.columns:
        result['frame_index'] = int(row['index'])
    else:
        result['frame_index'] = frame_idx
    
    # 加载 timestamp
    if 'timestamp' in df.columns:
        result['timestamp'] = float(row['timestamp'])
    else:
        result['timestamp'] = None
    
    print(f"Parquet 列: {df.columns.tolist()}")
    print(f"总帧数: {len(df)}")
    
    return result


def load_parquet_all(parquet_path: str) -> Dict[str, Any]:
    """
    加载整个 parquet 文件的所有数据
    
    Returns:
        dict containing:
        - observation_states: np.ndarray (N, state_dim)
        - actions: np.ndarray (N, action_dim) - ground truth delta actions
        - frame_indices: List[int] - 全局帧索引列表
        - num_frames: int
    """
    df = pd.read_parquet(parquet_path)
    
    result = {
        'num_frames': len(df),
    }
    
    # 加载所有 state
    if 'observation.state' in df.columns:
        result['observation_states'] = np.array(df['observation.state'].tolist(), dtype=np.float32)
    elif 'state' in df.columns:
        result['observation_states'] = np.array(df['state'].tolist(), dtype=np.float32)
    else:
        result['observation_states'] = None
    
    # 加载所有 action (ground truth)
    if 'action' in df.columns:
        result['actions'] = np.array(df['action'].tolist(), dtype=np.float32)
    else:
        result['actions'] = None
    
    # 加载 frame_index 列表
    if 'frame_index' in df.columns:
        result['frame_indices'] = df['frame_index'].tolist()
    elif 'index' in df.columns:
        result['frame_indices'] = df['index'].tolist()
    else:
        result['frame_indices'] = list(range(len(df)))
    
    print(f"Parquet 列: {df.columns.tolist()}")
    print(f"总帧数: {len(df)}")
    if result['observation_states'] is not None:
        print(f"State Shape: {result['observation_states'].shape}")
    if result['actions'] is not None:
        print(f"Action Shape: {result['actions'].shape}")
    
    return result


def load_vlm_hidden_states(
    vlm_hidden_states_dir: str, 
    frame_index: int,
    num_vlm_layers: int = 1
) -> List[torch.Tensor]:
    """
    加载 VLM hidden states
    
    Args:
        vlm_hidden_states_dir: VLM hidden states 目录
        frame_index: 全局帧索引
        num_vlm_layers: VLM 层数
    
    Returns:
        List of tensors, each shape (1, seq_len, hidden_dim)
    """
    npy_path = Path(vlm_hidden_states_dir) / f"hidden_state_{frame_index:06d}.npy"
    
    if not npy_path.exists():
        raise FileNotFoundError(f"VLM hidden states 文件不存在: {npy_path}")
    
    # 加载 npy 文件
    hidden_states = np.load(npy_path)
    
    print(f"加载 VLM hidden states: {npy_path}")
    print(f"  Shape: {hidden_states.shape}")
    
    # hidden_states 可能是 (seq_len, hidden_dim) 或 (num_layers, seq_len, hidden_dim)
    if hidden_states.ndim == 2:
        # (seq_len, hidden_dim) -> (1, seq_len, hidden_dim)
        hidden_states = hidden_states[np.newaxis, :]
        # 转换为 tensor
        tensor = torch.from_numpy(hidden_states).float()
        return [tensor]
    elif hidden_states.ndim == 3:
        # (num_layers, seq_len, hidden_dim)
        tensors = []
        for i in range(min(num_vlm_layers, hidden_states.shape[0])):
            # (seq_len, hidden_dim) -> (1, seq_len, hidden_dim)
            layer_hs = hidden_states[i:i+1]
            tensor = torch.from_numpy(layer_hs).float()
            tensors.append(tensor)
        return tensors
    else:
        raise ValueError(f"Unexpected hidden states shape: {hidden_states.shape}")


def setup_matplotlib(show: bool = False):
    """根据是否需要显示窗口来设置 matplotlib 后端"""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return plt


def run_single_frame_test(
    pipeline,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    action_denormalizer: MinMaxNormalizer,
    vlm_dir: Path,
    parquet_path: Path,
    frame_idx: int,
    device: str,
    chunk_size: int,
    show_all_actions: bool
):
    """单帧测试模式"""
    num_vlm_layers = config.get('num_vlm_hidden_layers', 1)
    action_dim = config.get('action_dim', 14)
    proprio_dim = config.get('proprio_dim', 14)
    
    # 加载数据
    print("\n[加载数据]")
    frame_data = load_parquet_frame(str(parquet_path), frame_idx)
    
    print(f"  Frame Index (episode 内): {frame_data['frame_idx_in_episode']}")
    print(f"  Frame Index (全局): {frame_data['frame_index']}")
    
    if frame_data['observation_state'] is not None:
        print(f"  State Shape: {frame_data['observation_state'].shape}")
        print(f"  State (原始): {frame_data['observation_state'][:8].tolist()}...")
    
    if frame_data['action'] is not None:
        print(f"  GT Action Shape: {frame_data['action'].shape}")
        print(f"  GT Action: {frame_data['action'][:8].tolist()}...")
    
    # 加载 VLM hidden states
    print("\n[加载 VLM Hidden States]")
    vlm_hidden_states = load_vlm_hidden_states(
        str(vlm_dir), 
        frame_data['frame_index'],
        num_vlm_layers
    )
    vlm_hidden_states = [v.to(device) for v in vlm_hidden_states]
    
    # State 预处理 (hand_binary + minmax 归一化)
    print("\n[推理]")
    if frame_data['observation_state'] is not None:
        processed_state = preprocess_state(
            frame_data['observation_state'],
            config,
            stats
        )
        print(f"  处理后 State Shape: {processed_state.shape}")
        print(f"  处理后 State (前8维): {processed_state[:8].tolist()}...")
        
        proprio_state = processed_state[:proprio_dim]
        proprio_tensor = torch.tensor(
            proprio_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
    else:
        proprio_tensor = torch.zeros(1, proprio_dim, dtype=torch.float32, device=device)
    
    # 推理
    with torch.no_grad():
        action_predictions = pipeline(vlm_hidden_states, proprio_tensor)
    
    action_predictions_np = action_predictions[0, 0].cpu().numpy()
    predicted_actions = action_predictions_np.reshape(chunk_size, action_dim)
    
    print("\n" + "=" * 70)
    print("推理结果 (归一化)")
    print("=" * 70)
    print(f"Predicted Actions Shape: {predicted_actions.shape}")
    print(f"Predicted Actions Range (归一化): [{predicted_actions.min():.4f}, {predicted_actions.max():.4f}]")
    
    # Action 反归一化
    if action_denormalizer is not None:
        predicted_actions = action_denormalizer.denormalize(predicted_actions)
        print(f"Predicted Actions Range (反归一化): [{predicted_actions.min():.6f}, {predicted_actions.max():.6f}]")
    
    print(f"\n预测动作 (前 5 步):")
    for i in range(min(5, chunk_size)):
        print(f"  Step {i}: {predicted_actions[i, :6].tolist()}...")
    
    if show_all_actions:
        print(f"\n预测动作 (全部 {chunk_size} 步):")
        for i in range(chunk_size):
            print(f"  Step {i:2d}: {predicted_actions[i].tolist()}")
    
    # 与 GT 对比
    if frame_data['action'] is not None:
        gt_action = frame_data['action']
        pred_first = predicted_actions[0]
        
        print(f"\n与 Ground Truth 对比 (第一步):")
        print(f"  GT Action:   {gt_action[:6].tolist()}...")
        print(f"  Pred Action: {pred_first[:6].tolist()}...")
        
        error = np.abs(gt_action - pred_first)
        print(f"  Abs Error (前6维): {error[:6].tolist()}")
        print(f"  Mean Abs Error: {error.mean():.6f}")
        print(f"  Max Abs Error:  {error.max():.6f}")


def run_trajectory_test(
    pipeline,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    action_denormalizer: MinMaxNormalizer,
    vlm_dir: Path,
    parquet_path: Path,
    device: str,
    chunk_size: int,
    dims: List[int],
    plot: bool,
    output_dir: str,
    show: bool
):
    """
    轨迹测试模式:
    每隔 chunk_size 帧取一次图片预测，使用 delta 累计计算轨迹误差
    
    流程:
    1. 用第 0 帧图像预测 chunk_size 个 delta action
    2. 对预测的 action 进行反归一化
    3. 累加这些 delta 得到第一个 chunk 的预测轨迹
    4. 用第 chunk_size 帧图像预测下一个 chunk_size 个 delta action
    5. 将新预测的 delta 累加到上一个 chunk 的终点位置
    6. 以此类推...
    """
    num_vlm_layers = config.get('num_vlm_hidden_layers', 1)
    action_dim = config.get('action_dim', 14)
    proprio_dim = config.get('proprio_dim', 14)
    
    # 加载所有数据
    print("\n[加载所有数据]")
    all_data = load_parquet_all(str(parquet_path))
    
    num_frames = all_data['num_frames']
    gt_actions = all_data['actions']  # (N, action_dim) - delta actions
    observation_states = all_data['observation_states']
    frame_indices = all_data['frame_indices']
    
    if gt_actions is None:
        print("错误: 无法获取 GT actions")
        return
    
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    print(f"  总帧数: {num_frames}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  预测次数 (Chunk 数): {num_chunks}")
    print(f"  评估维度: {dims}")
    
    # 计算 GT 轨迹 (delta 累积)
    gt_trajectory = np.cumsum(gt_actions[:, dims], axis=0)  # (N, len(dims))
    print(f"  GT 轨迹 Shape: {gt_trajectory.shape}")
    
    # 预测轨迹 - 使用显式累加方式
    pred_trajectory = np.zeros((num_frames, len(dims)), dtype=np.float32)
    
    # 当前累计位置 (从 0 开始)
    current_position = np.zeros(len(dims), dtype=np.float32)
    
    # 记录每个 chunk 的预测起点帧索引 (用于绘图标记)
    chunk_start_frames = []
    chunk_start_positions = []  # 预测轨迹在每个 chunk 起点的位置
    
    # 每隔 chunk_size 帧进行一次预测
    predict_frame_indices = list(range(0, num_frames, chunk_size))
    print(f"\n[开始预测] 预测帧索引: {predict_frame_indices[:10]}{'...' if len(predict_frame_indices) > 10 else ''}")
    
    for pred_idx, start_frame in enumerate(predict_frame_indices):
        end_frame = min(start_frame + chunk_size, num_frames)
        actual_chunk_len = end_frame - start_frame
        global_frame_idx = frame_indices[start_frame]
        
        # 记录当前 chunk 的起点 (使用 GT 位置)
        chunk_start_frames.append(start_frame)
        if start_frame == 0:
            chunk_start_positions.append(np.zeros(len(dims), dtype=np.float32))
        else:
            chunk_start_positions.append(gt_trajectory[start_frame - 1].copy())
        
        # 加载 VLM hidden states
        try:
            vlm_hidden_states = load_vlm_hidden_states(
                str(vlm_dir), 
                global_frame_idx,
                num_vlm_layers
            )
            vlm_hidden_states = [v.to(device) for v in vlm_hidden_states]
            use_gt = False
        except FileNotFoundError as e:
            print(f"  [!] 跳过帧 {start_frame}: {e}")
            use_gt = True
        
        if use_gt:
            # 使用 GT delta 填充 (用于没有 hidden states 的情况)
            chunk_deltas = gt_actions[start_frame:end_frame, dims]
        else:
            # State 预处理 (hand_binary + minmax 归一化)
            if observation_states is not None:
                processed_state = preprocess_state(
                    observation_states[start_frame],
                    config,
                    stats
                )
                proprio_state = processed_state[:proprio_dim]
                proprio_tensor = torch.tensor(
                    proprio_state, dtype=torch.float32, device=device
                ).unsqueeze(0)
            else:
                proprio_tensor = torch.zeros(1, proprio_dim, dtype=torch.float32, device=device)
            
            # 推理
            with torch.no_grad():
                action_predictions = pipeline(vlm_hidden_states, proprio_tensor)
            
            action_predictions_np = action_predictions[0, 0].cpu().numpy()
            predicted_actions = action_predictions_np.reshape(chunk_size, action_dim)
            
            # Action 反归一化
            if action_denormalizer is not None:
                predicted_actions = action_denormalizer.denormalize(predicted_actions)
            
            # 提取需要的维度
            chunk_deltas = predicted_actions[:actual_chunk_len, dims]
        
        # 每个 chunk 从 GT 的当前位置开始累加预测的 delta
        # 这样每个 chunk 的起点都是原始数据集的真实位置
        if start_frame == 0:
            # 第一个 chunk 从 0 开始
            chunk_start_pos = np.zeros(len(dims), dtype=np.float32)
        else:
            # 后续 chunk 从 GT 在该帧的位置开始
            chunk_start_pos = gt_trajectory[start_frame - 1].copy()
        
        current_position = chunk_start_pos.copy()
        
        # 累加每个时间步的 delta 到当前位置
        for i in range(actual_chunk_len):
            frame_idx = start_frame + i
            current_position = current_position + chunk_deltas[i]
            pred_trajectory[frame_idx] = current_position.copy()
        
        # 打印进度
        if pred_idx < 3 or pred_idx == len(predict_frame_indices) - 1:
            chunk_end_pos = current_position
            gt_end_pos = gt_trajectory[end_frame - 1]
            chunk_error = np.linalg.norm(chunk_end_pos - gt_end_pos)
            print(f"  Chunk {pred_idx}: 帧 {start_frame}-{end_frame-1}, "
                  f"全局帧 {global_frame_idx}, "
                  f"Chunk内误差: {chunk_error:.4f}"
                  f"{' [GT填充]' if use_gt else ''}")
    
    # 计算误差
    print("\n" + "=" * 70)
    print("轨迹评估结果")
    print("=" * 70)
    
    # 每步误差
    step_errors = np.abs(gt_trajectory - pred_trajectory)
    
    # 终点误差
    end_error = np.linalg.norm(gt_trajectory[-1] - pred_trajectory[-1])
    
    # 平均轨迹误差
    mean_traj_error = np.mean(step_errors)
    max_traj_error = np.max(step_errors)
    
    # 各维度终点误差
    dim_end_errors = np.abs(gt_trajectory[-1] - pred_trajectory[-1])
    
    print(f"总帧数: {num_frames}")
    print(f"Chunk Size: {chunk_size}")
    print(f"评估维度: {dims}")
    print(f"\n终点误差 (欧氏距离): {end_error:.6f}")
    print(f"各维度终点误差: {dim_end_errors.tolist()}")
    print(f"\n平均轨迹误差: {mean_traj_error:.6f}")
    print(f"最大轨迹误差: {max_traj_error:.6f}")
    
    # 每个维度的统计
    print(f"\n各维度统计:")
    for i, dim in enumerate(dims):
        dim_mean = np.mean(step_errors[:, i])
        dim_max = np.max(step_errors[:, i])
        dim_end = dim_end_errors[i]
        print(f"  Dim {dim}: Mean={dim_mean:.6f}, Max={dim_max:.6f}, End={dim_end:.6f}")
    
    # 绘图
    if plot:
        plt = setup_matplotlib(show)
        os.makedirs(output_dir, exist_ok=True)
        
        parquet_name = parquet_path.stem
        
        # 获取每个 chunk 起点的位置 (用于标记)
        chunk_pred_starts = np.array(chunk_start_positions)  # (num_chunks, len(dims))
        chunk_gt_starts = gt_trajectory[chunk_start_frames]  # (num_chunks, len(dims))
        
        if len(dims) >= 3:
            # 3D 轨迹图
            fig = plt.figure(figsize=(16, 6))
            
            # 3D 轨迹
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2],
                    'b-', label='GT', linewidth=1.5, alpha=0.8)
            ax1.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2],
                    'r--', label='Pred', linewidth=1.5, alpha=0.8)
            
            # 标记起点和终点
            ax1.scatter(*gt_trajectory[0], color='green', s=150, marker='o', label='Start', zorder=10)
            ax1.scatter(*gt_trajectory[-1], color='blue', s=150, marker='x', label='GT End', zorder=10)
            ax1.scatter(*pred_trajectory[-1], color='red', s=150, marker='^', label='Pred End', zorder=10)
            
            # 标记每个 chunk 的预测起点 (用小圆点)
            if len(chunk_start_frames) > 1:
                ax1.scatter(chunk_pred_starts[1:, 0], chunk_pred_starts[1:, 1], chunk_pred_starts[1:, 2],
                           color='orange', s=50, marker='o', alpha=0.7, label=f'Chunk starts (n={len(chunk_start_frames)})')
            
            ax1.set_xlabel(f'Dim {dims[0]}')
            ax1.set_ylabel(f'Dim {dims[1]}')
            ax1.set_zlabel(f'Dim {dims[2]}')
            ax1.set_title(f'3D Trajectory\nChunk={chunk_size}, End Error={end_error:.4f}')
            ax1.legend(loc='upper left', fontsize=8)
            
            # 误差曲线
            ax2 = fig.add_subplot(122)
            for i, dim in enumerate(dims[:3]):
                ax2.plot(step_errors[:, i], label=f'Dim {dim}', alpha=0.8)
            ax2.axhline(y=mean_traj_error, color='k', linestyle='--', label=f'Mean: {mean_traj_error:.4f}')
            
            # 标记每个 chunk 的边界
            for frame in chunk_start_frames[1:]:
                ax2.axvline(x=frame, color='gray', linestyle=':', alpha=0.5)
            
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Abs Error')
            ax2.set_title(f'Per-Step Error (Chunk boundaries in gray)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{parquet_name}_trajectory_chunk{chunk_size}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n轨迹图已保存: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()
        else:
            # 2D 轨迹图
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            for i, dim in enumerate(dims):
                axes[0].plot(gt_trajectory[:, i], label=f'GT Dim {dim}', alpha=0.8)
                axes[0].plot(pred_trajectory[:, i], '--', label=f'Pred Dim {dim}', alpha=0.8)
            
            # 标记每个 chunk 的边界
            for frame in chunk_start_frames[1:]:
                axes[0].axvline(x=frame, color='gray', linestyle=':', alpha=0.5)
            
            axes[0].set_xlabel('Frame')
            axes[0].set_ylabel('Cumsum Position')
            axes[0].set_title('Trajectory Comparison (Chunk boundaries in gray)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            for i, dim in enumerate(dims):
                axes[1].plot(step_errors[:, i], label=f'Dim {dim}', alpha=0.8)
            axes[1].axhline(y=mean_traj_error, color='k', linestyle='--', label=f'Mean: {mean_traj_error:.4f}')
            
            # 标记每个 chunk 的边界
            for frame in chunk_start_frames[1:]:
                axes[1].axvline(x=frame, color='gray', linestyle=':', alpha=0.5)
            
            axes[1].set_xlabel('Frame')
            axes[1].set_ylabel('Abs Error')
            axes[1].set_title('Per-Step Error')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f'{parquet_name}_trajectory_chunk{chunk_size}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n轨迹图已保存: {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()


def main():
    parser = argparse.ArgumentParser(description='OFT Checkpoint 测试')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint 路径 (action_head.pt)')
    parser.add_argument('--parquet', type=str, required=True,
                        help='Parquet 文件路径')
    parser.add_argument('--stats_path', type=str, default=None,
                        help='stats.json 路径 (用于 action 反归一化和 state 归一化，默认: parquet 同级的 meta/stats.json)')
    parser.add_argument('--vlm_hidden_states_dir', type=str, default=None,
                        help='VLM hidden states 目录 (默认: parquet 同级的 vlm_hidden_states)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk 大小 (默认从 config 读取)')
    
    # 单帧模式参数
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='单帧模式: 要测试的帧索引 (episode 内，默认 0)')
    parser.add_argument('--show_all_actions', action='store_true',
                        help='单帧模式: 显示所有预测动作')
    
    # 轨迹模式参数
    parser.add_argument('--trajectory_mode', action='store_true',
                        help='启用轨迹模式: 每隔 chunk_size 帧取图片预测，计算累计误差')
    parser.add_argument('--dims', type=int, nargs='+', default=[0, 1, 2],
                        help='轨迹模式: 要评估的 action 维度 (默认: 0 1 2)')
    parser.add_argument('--plot', action='store_true',
                        help='轨迹模式: 绘制轨迹图')
    parser.add_argument('--output_dir', type=str, default='./trajectory_plots',
                        help='轨迹模式: 输出目录')
    parser.add_argument('--show', action='store_true',
                        help='轨迹模式: 弹出交互式窗口')
    
    args = parser.parse_args()
    
    # 检查文件
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: Checkpoint 不存在: {checkpoint_path}")
        return
    
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        print(f"错误: Parquet 文件不存在: {parquet_path}")
        return
    
    # VLM hidden states 目录
    if args.vlm_hidden_states_dir:
        vlm_dir = Path(args.vlm_hidden_states_dir)
    else:
        vlm_dir = parquet_path.parent.parent.parent / "vlm_hidden_states"
    
    if not vlm_dir.exists():
        print(f"错误: VLM hidden states 目录不存在: {vlm_dir}")
        return
    
    # Stats 路径
    if args.stats_path:
        stats_path = Path(args.stats_path)
    else:
        stats_path = parquet_path.parent.parent.parent / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"警告: stats.json 不存在: {stats_path}")
        print("  将不进行 action 反归一化和 state 归一化")
        stats = None
    else:
        stats = load_stats(str(stats_path))
    
    mode = "轨迹模式" if args.trajectory_mode else "单帧模式"
    print("=" * 70)
    print(f"OFT Checkpoint 测试 - {mode}")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Stats: {stats_path}")
    print(f"VLM Hidden States: {vlm_dir}")
    print(f"Device: {args.device}")
    print("=" * 70)
    
    # 加载配置
    print("\n[加载配置]")
    config = load_config(str(checkpoint_path))
    
    chunk_size = args.chunk_size or config.get('num_actions_chunk', 50)
    action_dim = config.get('action_dim', 14)
    
    print(f"  Action Head Type: {config.get('type', 'unknown')}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Action Dim: {action_dim}")
    print(f"  Proprio Dim: {config.get('proprio_dim', 'unknown')}")
    print(f"  VLM Output Dim: {config.get('llm_output_dim', 'unknown')}")
    print(f"  Num VLM Layers: {config.get('num_vlm_hidden_layers', 1)}")
    print(f"  State Process Order: {config.get('state_process_order', [])}")
    print(f"  Hand Binary Columns: {config.get('hand_binary_columns', [])}")
    print(f"  State Norm Columns MinMax: {config.get('state_norm_columns_minmax', [])}")
    
    # 创建 action 反归一化器
    action_denormalizer = None
    if stats:
        action_denormalizer = create_action_denormalizer(stats)
        print(f"\n[Action 反归一化]")
        print(f"  Action Min (前3维): {action_denormalizer.min_vals[:3].tolist()}")
        print(f"  Action Max (前3维): {action_denormalizer.max_vals[:3].tolist()}")
    
    # 加载模型
    print("\n[加载模型]")
    num_vlm_layers = config.get('num_vlm_hidden_layers', 1)
    vlm_output_dim = config.get('llm_output_dim', 2048)
    num_transformer_blocks = config.get('num_blocks', 2)
    num_attention_heads = config.get('num_attention_heads', 8)
    action_head_hidden_dim = config.get('action_head_hidden_dim', 4096)
    
    pipeline = create_vlm2oft_pipeline(
        num_transformer_blocks=num_transformer_blocks,
        num_attention_heads=num_attention_heads,
        num_vlm_layers=num_vlm_layers,
        vlm_output_dim=vlm_output_dim,
        action_head_hidden_dim=action_head_hidden_dim,
    ).to(args.device)
    
    state_dict = torch.load(checkpoint_path, map_location=args.device)
    pipeline.load_state_dict(state_dict)
    pipeline.eval()
    
    total_params = sum(p.numel() for p in pipeline.parameters())
    print(f"  模型参数量: {total_params:,}")
    
    # 运行测试
    if args.trajectory_mode:
        run_trajectory_test(
            pipeline=pipeline,
            config=config,
            stats=stats,
            action_denormalizer=action_denormalizer,
            vlm_dir=vlm_dir,
            parquet_path=parquet_path,
            device=args.device,
            chunk_size=chunk_size,
            dims=args.dims,
            plot=args.plot,
            output_dir=args.output_dir,
            show=args.show
        )
    else:
        run_single_frame_test(
            pipeline=pipeline,
            config=config,
            stats=stats,
            action_denormalizer=action_denormalizer,
            vlm_dir=vlm_dir,
            parquet_path=parquet_path,
            frame_idx=args.frame_idx,
            device=args.device,
            chunk_size=chunk_size,
            show_all_actions=args.show_all_actions
        )
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
