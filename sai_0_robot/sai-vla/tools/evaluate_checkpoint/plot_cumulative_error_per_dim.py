"""
python /home/dev/文档/huangwenlong/sai0-vla/tools/evaluate_checkpoint/plot_cumulative_error_per_dim.py \
    --checkpoint /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset73_256_filter_weight_eagle_parad/p600020k201e4constant_bsz128*1*8_tb4_20000steps_layer-1_20260203_wpflow_matching_0_eagle_libero_test_newest_dataset73_256_filter_USE_SHARED_CACHE_true_CACHE_VLM_STATES_false_USE_AMP_true_webdataset_false/checkpoints/step_20000/action_head.pt \
    --parquet /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset73_256_filter/data/chunk-000/episode_000000.parquet \
    --end_to_end \
    --vlm_model_path /home/dev/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
    --vlm_type eagle2_5_vl \
    --vlm_layers -1 \
    --vlm_content_order images_first \
    --vlm_lowercase_instruction \
    --vlm_add_generation_prompt \
    --instruction "pick up the bottle" \
    --image_resize 256 256 \
    --state_norm_columns 0 1 2 3 4 5 6 7 8 9 10 11 \
    --state_hand_binary_columns 12 18 18 24 \
    --state_hand_binary_threshold 442 \
    --state_process_order hand_binary minmax \
    --chunk_size 20

# ! --image_bgr_to_rgb \

每个 Action 维度的累积误差可视化脚本

功能:
- 加载 OFT checkpoint 和数据集
- 计算 GT action 和预测 action 的累积轨迹
- 为每个 action 维度绘制累积轨迹对比图和累积误差图
- 支持按 chunk 预测模式
- 每个 chunk 从 GT 位置开始预测

使用方法:
    # 基本用法 (使用 checkpoint config.json 中的 state 处理配置)
    python tools/evaluate_checkpoint/plot_cumulative_error_per_dim.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet

    # 指定维度范围
    python tools/evaluate_checkpoint/plot_cumulative_error_per_dim.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet \
        --dims 0 1 2 3 4 5 6

    # 指定 state 处理配置 (与训练脚本 train_eagle.sh 类似)
    python tools/evaluate_checkpoint/plot_cumulative_error_per_dim.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet \
        --state_norm_columns 0 1 2 3 4 5 6 7 8 9 10 11 \
        --state_norm_stats_key observation.state \
        --state_hand_binary_columns 12 18 18 24 \
        --state_hand_binary_threshold 442 \
        --state_process_order hand_binary minmax

    # 指定输出目录和 chunk_size
    python tools/evaluate_checkpoint/plot_cumulative_error_per_dim.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet \
        --chunk_size 20 \
        --output_dir ./cumulative_error_plots

State 处理参数说明 (参考训练脚本 train_eagle.sh):
    --state_norm_columns: 需要归一化的 state 列索引
    --state_norm_stats_key: stats.json 中 state 统计信息的 key (默认: observation.state)
    --state_hand_binary_columns: 手部二值化列范围 (起始 结束 起始 结束...)
        - 每组 [start, end) 的数据取平均后二值化
        - 多维 -> 1维，支持多组 (例如: 12 18 18 24 表示两组)
    --state_hand_binary_threshold: 手部二值化阈值 (平均值 > threshold -> 1, 否则 -> 0) (默认: 442.0)
    --state_process_order: State 处理顺序 (默认: hand_binary minmax)

图像预处理参数说明 (端到端推理模式):
    --image_resize: 图像 resize 尺寸 [width, height]
    --image_flip_horizontal: 水平翻转图像
    --image_flip_vertical: 垂直翻转图像
    --image_rotate_180: 旋转图像 180 度
    --image_bgr_to_rgb: BGR 转 RGB

端到端推理模式:
    使用 --end_to_end 启用，自动从 parquet 对应的视频中提取每个 chunk 起始帧图像，
    通过 VLM 获取 hidden states 后再进行预测。

    # 端到端推理示例
    python tools/evaluate_checkpoint/plot_cumulative_error_per_dim.py \
        --checkpoint /path/to/action_head.pt \
        --parquet /path/to/episode_000000.parquet \
        --end_to_end \
        --vlm_model_path /path/to/vlm_model \
        --vlm_type eagle2_5_vl \
        --vlm_layers -1 \
        --instruction "pick up the bottle" \
        --image_resize 256 256 \
        --image_bgr_to_rgb \
        --state_norm_columns 0 1 2 3 4 5 6 7 8 9 10 11 \
        --state_binarize_columns 12 13 \
        --state_binarize_threshold 0

    视频文件查找规则:
        数据集结构: dataset/data/train/episode_000000.parquet
        视频路径:   dataset/videos/observation.images.main/episode_000000.mp4

VLM 配置参数说明 (端到端推理模式):
    --vlm_type: VLM 模型类型 (qwen3_vl, eagle2_5_vl, cosmos_reason_2b_vl)
    --vlm_layers: VLM 提取层 (Eagle: -1, Qwen2B: 14, Qwen4B: 16)
    --vlm_flip_images: 是否翻转图像
    --vlm_content_order: 内容顺序 (images_first, text_first)
    --vlm_lowercase_instruction: 指令小写
    --vlm_add_generation_prompt: 添加 generation prompt
    --instruction: 任务指令
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

# 设置环境变量避免 tokenizer 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Action_Heads.OFT1_0.vlm2oft_pipeline import create_vlm2oft_pipeline


# ============================================================================
# 图像处理工具
# ============================================================================

def preprocess_image(
    image: Image.Image,
    resize: Optional[List[int]] = None,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotate_180: bool = False,
    bgr_to_rgb: bool = False,
) -> Image.Image:
    """
    预处理图像
    
    Args:
        image: PIL 图像
        resize: [width, height] 或 None
        flip_horizontal: 水平翻转
        flip_vertical: 垂直翻转
        rotate_180: 旋转 180 度
        bgr_to_rgb: BGR 转 RGB (已在加载时处理)
        
    Returns:
        预处理后的图像
    """
    # Resize
    if resize is not None and len(resize) == 2:
        image = image.resize((resize[0], resize[1]), Image.Resampling.LANCZOS)
    
    # 旋转 180 度 (先于翻转)
    if rotate_180:
        image = image.rotate(180)
    
    # 水平翻转
    if flip_horizontal:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    
    # 垂直翻转
    if flip_vertical:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    
    return image


def extract_frame_from_video(video_path: str, frame_index: int, bgr_to_rgb: bool = True) -> Image.Image:
    """
    从视频中提取指定帧
    
    Args:
        video_path: 视频文件路径
        frame_index: 帧索引
        bgr_to_rgb: 是否将 BGR 转换为 RGB
        
    Returns:
        PIL 图像
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("需要安装 opencv-python: pip install opencv-python")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 设置帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取视频帧: {video_path}, frame_index={frame_index}")
    
    # BGR -> RGB
    if bgr_to_rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(frame)


def find_video_path(parquet_path: Path, video_key: str = "observation.images.main") -> Optional[Path]:
    """
    根据 parquet 路径找到对应的视频文件
    
    支持多种数据集结构:
    
    结构1 (LeRobot chunk 结构):
    dataset/
    ├── data/
    │   └── chunk-000/
    │       └── episode_000000.parquet
    └── videos/
        └── chunk-000/
            └── observation.images.main/
                └── episode_000000.mp4
    
    结构2 (标准 LeRobot 结构):
    dataset/
    ├── data/
    │   └── train/
    │       └── episode_000000.parquet
    └── videos/
        └── observation.images.main/
            └── episode_000000.mp4
    
    Args:
        parquet_path: parquet 文件路径
        video_key: 视频键名 (默认 observation.images.main)
        
    Returns:
        视频文件路径，如果未找到返回 None
    """
    # 获取 episode 名称 (去掉 .parquet 后缀)
    episode_name = parquet_path.stem
    
    # 获取 chunk 名称 (parquet 所在目录名，如 chunk-000 或 train)
    chunk_name = parquet_path.parent.name
    
    # 尝试找到数据集根目录
    # parquet 路径通常是: dataset/data/chunk-000/episode_000000.parquet
    dataset_root = parquet_path.parent.parent.parent
    
    # 尝试多种可能的路径
    possible_paths = [
        # LeRobot chunk 结构: videos/chunk-000/observation.images.main/episode_000000.mp4
        dataset_root / "videos" / chunk_name / video_key / f"{episode_name}.mp4",
        # 标准 LeRobot 结构: videos/observation.images.main/episode_000000.mp4
        dataset_root / "videos" / video_key / f"{episode_name}.mp4",
        # 简化结构: videos/chunk-000/episode_000000.mp4
        dataset_root / "videos" / chunk_name / f"{episode_name}.mp4",
        # 简化结构: videos/episode_000000.mp4
        dataset_root / "videos" / f"{episode_name}.mp4",
        # 直接在 parquet 同级目录
        parquet_path.parent / "videos" / video_key / f"{episode_name}.mp4",
        parquet_path.parent / "videos" / f"{episode_name}.mp4",
    ]
    
    print(f"  [调试] 尝试查找视频文件...")
    print(f"  [调试] Episode 名称: {episode_name}")
    print(f"  [调试] Chunk 名称: {chunk_name}")
    print(f"  [调试] 数据集根目录: {dataset_root}")
    
    for video_path in possible_paths:
        print(f"  [调试] 尝试路径: {video_path} -> {'存在' if video_path.exists() else '不存在'}")
        if video_path.exists():
            return video_path
    
    return None


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
    stats: Dict[str, Any] = None,
    state_config: Dict[str, Any] = None
) -> np.ndarray:
    """
    State 预处理
    
    支持两种配置方式:
    1. 从 checkpoint 的 config.json 读取 (旧方式)
    2. 从命令行参数指定的 state_config (新方式，优先级更高)
    
    state_config 格式 (参考训练脚本 train_eagle.sh):
    {
        'state_process_order': ['hand_binary', 'minmax'],  # 处理顺序
        'state_normalize': {
            'columns': [0, 1, 2, ...],  # 需要归一化的列
            'stats_key': 'observation.state'  # stats.json 中的 key
        },
        'state_hand_binary': {
            'columns': [12, 18, 18, 24],  # 手部数据范围 (起始 结束 起始 结束...)
            'threshold': 442.0  # 阈值 (平均值 > threshold -> 1, 否则 -> 0)
        }
    }
    """
    state = state.copy().astype(np.float32)
    
    # 如果提供了 state_config，使用新的处理方式
    if state_config is not None:
        # 获取处理顺序
        process_order = state_config.get('state_process_order', ['hand_binary', 'minmax'])
        
        # 获取配置
        norm_config = state_config.get('state_normalize', {})
        norm_columns = norm_config.get('columns', [])
        stats_key = norm_config.get('stats_key', 'observation.state')
        
        hand_binary_config = state_config.get('state_hand_binary', {})
        hand_binary_columns = hand_binary_config.get('columns', [])
        hand_binary_threshold = hand_binary_config.get('threshold', 442.0)
        
        # 加载 state 统计信息
        state_min = None
        state_max = None
        if stats is not None and stats_key in stats:
            state_min = np.array(stats[stats_key].get('min', []), dtype=np.float32)
            state_max = np.array(stats[stats_key].get('max', []), dtype=np.float32)
        
        # 按顺序处理
        for processor in process_order:
            if processor == 'hand_binary' and hand_binary_columns:
                # 手部二值化: 每组 [start, end) 的数据取平均，> threshold -> 1, else -> 0
                # 多维 -> 1维，需要调整索引
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
            
            elif processor == 'minmax' and norm_columns and state_min is not None:
                # MinMax 归一化
                for col in norm_columns:
                    if col < len(state) and col < len(state_min):
                        col_range = state_max[col] - state_min[col]
                        if col_range > 1e-8:
                            normalized = (state[col] - state_min[col]) / col_range
                            state[col] = 2.0 * normalized - 1.0
                        else:
                            state[col] = 0.0
        
        return state
    
    # 旧的处理方式 (从 checkpoint config.json 读取)
    state_process_order = config.get('state_process_order', [])
    hand_binary_columns = config.get('hand_binary_columns', [])
    hand_binary_threshold = config.get('hand_binary_threshold', 442.0)
    state_norm_columns_minmax = config.get('state_norm_columns_minmax', [])
    state_min = None
    state_max = None
    if stats and 'observation.state' in stats:
        state_min = np.array(stats['observation.state'].get('min', []), dtype=np.float32)
        state_max = np.array(stats['observation.state'].get('max', []), dtype=np.float32)
    
    for processor in state_process_order:
        if processor == 'hand_binary' and hand_binary_columns:
            offset = 0
            new_state_parts = []
            last_end = 0
            
            for i in range(0, len(hand_binary_columns), 2):
                if i + 1 < len(hand_binary_columns):
                    start = hand_binary_columns[i] + offset
                    end = hand_binary_columns[i + 1] + offset
                    
                    if start < len(state) and end <= len(state):
                        if last_end < start:
                            new_state_parts.append(state[last_end:start])
                        
                        hand_data = state[start:end]
                        binary_val = 1.0 if np.mean(hand_data) > hand_binary_threshold else 0.0
                        new_state_parts.append(np.array([binary_val]))
                        
                        last_end = end
                        offset -= (end - start - 1)
            
            if last_end < len(state):
                new_state_parts.append(state[last_end:])
            
            state = np.concatenate(new_state_parts) if new_state_parts else state
    
    if state_norm_columns_minmax and state_min is not None and state_max is not None:
        for col in state_norm_columns_minmax:
            if col < len(state) and col < len(state_min):
                col_range = state_max[col] - state_min[col]
                if col_range > 1e-8:
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


def load_parquet_all(parquet_path: str) -> Dict[str, Any]:
    """加载整个 parquet 文件的所有数据"""
    df = pd.read_parquet(parquet_path)
    
    result = {
        'num_frames': len(df),
    }
    
    if 'observation.state' in df.columns:
        result['observation_states'] = np.array(df['observation.state'].tolist(), dtype=np.float32)
    elif 'state' in df.columns:
        result['observation_states'] = np.array(df['state'].tolist(), dtype=np.float32)
    else:
        result['observation_states'] = None
    
    if 'action' in df.columns:
        result['actions'] = np.array(df['action'].tolist(), dtype=np.float32)
    else:
        result['actions'] = None
    
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
    """加载 VLM hidden states"""
    npy_path = Path(vlm_hidden_states_dir) / f"hidden_state_{frame_index:06d}.npy"
    
    if not npy_path.exists():
        raise FileNotFoundError(f"VLM hidden states 文件不存在: {npy_path}")
    
    hidden_states = np.load(npy_path)
    
    if hidden_states.ndim == 2:
        hidden_states = hidden_states[np.newaxis, :]
        tensor = torch.from_numpy(hidden_states).float()
        return [tensor]
    elif hidden_states.ndim == 3:
        tensors = []
        for i in range(min(num_vlm_layers, hidden_states.shape[0])):
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
    return plt


def plot_cumulative_error_per_dim(
    gt_actions: np.ndarray,
    pred_actions: np.ndarray,
    dims: List[int],
    output_dir: str,
    parquet_name: str,
    chunk_size: int,
    chunk_start_frames: List[int],
    show: bool = False,
    dim_names: Dict[int, str] = None
):
    """
    为每个维度绘制累积误差图
    
    Args:
        gt_actions: GT delta actions (N, action_dim)
        pred_actions: 预测 delta actions (N, action_dim)
        dims: 要绘制的维度列表
        output_dir: 输出目录
        parquet_name: parquet 文件名
        chunk_size: chunk 大小
        chunk_start_frames: 每个 chunk 的起始帧
        show: 是否弹出显示窗口
        dim_names: 维度名称映射 {dim_idx: name}
    """
    plt = setup_matplotlib(show)
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认维度名称
    if dim_names is None:
        dim_names = {
            0: 'X Position',
            1: 'Y Position',
            2: 'Z Position',
            3: 'Roll',
            4: 'Pitch',
            5: 'Yaw',
            6: 'Gripper',
            7: 'Dim 7',
            8: 'Dim 8',
            9: 'Dim 9',
            10: 'Dim 10',
            11: 'Dim 11',
            12: 'Dim 12',
            13: 'Dim 13',
        }
    
    num_frames = gt_actions.shape[0]
    action_dim = gt_actions.shape[1]
    
    # 计算 GT 累积轨迹
    gt_cumsum = np.cumsum(gt_actions, axis=0)  # (N, action_dim)
    
    # 计算预测累积轨迹 - 每个 chunk 从 GT 位置开始
    # 这样评估的是每个 chunk 内的预测质量
    pred_cumsum = np.zeros((num_frames, action_dim), dtype=np.float32)
    
    for chunk_idx, start_frame in enumerate(chunk_start_frames):
        # 确定当前 chunk 的结束帧
        if chunk_idx + 1 < len(chunk_start_frames):
            end_frame = chunk_start_frames[chunk_idx + 1]
        else:
            end_frame = num_frames
        
        # 每个 chunk 从 GT 位置开始
        if start_frame == 0:
            chunk_start_pos = np.zeros(action_dim, dtype=np.float32)
        else:
            chunk_start_pos = gt_cumsum[start_frame - 1].copy()
        
        # 在 chunk 内累加预测的 delta
        current_pos = chunk_start_pos.copy()
        for frame_idx in range(start_frame, end_frame):
            current_pos = current_pos + pred_actions[frame_idx]
            pred_cumsum[frame_idx] = current_pos.copy()
    
    # 计算累积误差 (绝对值)
    cumsum_error = np.abs(gt_cumsum - pred_cumsum)  # (N, action_dim)
    
    # ========================================
    # 1. 所有维度的汇总图
    # ========================================
    num_dims = len(dims)
    cols = min(4, num_dims)
    rows = (num_dims + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols * 2, figsize=(6 * cols, 4 * rows))
    if rows == 1 and cols * 2 == 2:
        axes = np.array([axes])
    axes = axes.reshape(rows, -1)
    
    fig.suptitle(f'Cumulative Trajectory vs Error - {parquet_name}\nChunk Size: {chunk_size}', fontsize=14)
    
    for idx, dim in enumerate(dims):
        row = idx // cols
        col_base = (idx % cols) * 2
        
        dim_name = dim_names.get(dim, f'Dim {dim}')
        
        # 轨迹对比图
        ax_traj = axes[row, col_base]
        ax_traj.plot(gt_cumsum[:, dim], 'b-', label='GT Cumsum', linewidth=1.2, alpha=0.8)
        ax_traj.plot(pred_cumsum[:, dim], 'r--', label='Pred Cumsum', linewidth=1.2, alpha=0.8)
        
        # 标记 chunk 边界
        for frame in chunk_start_frames[1:]:
            ax_traj.axvline(x=frame, color='gray', linestyle=':', alpha=0.4)
        
        ax_traj.set_xlabel('Frame')
        ax_traj.set_ylabel('Cumulative Value')
        ax_traj.set_title(f'Dim {dim}: {dim_name}')
        ax_traj.legend(fontsize=8)
        ax_traj.grid(True, alpha=0.3)
        
        # 累积误差图
        ax_err = axes[row, col_base + 1]
        ax_err.fill_between(range(num_frames), cumsum_error[:, dim], alpha=0.3, color='red')
        ax_err.plot(cumsum_error[:, dim], 'r-', linewidth=1.0)
        
        # 标记 chunk 边界
        for frame in chunk_start_frames[1:]:
            ax_err.axvline(x=frame, color='gray', linestyle=':', alpha=0.4)
        
        mean_err = np.mean(cumsum_error[:, dim])
        max_err = np.max(cumsum_error[:, dim])
        end_err = cumsum_error[-1, dim]
        
        ax_err.axhline(y=mean_err, color='orange', linestyle='--', 
                       label=f'Mean: {mean_err:.4f}', alpha=0.8)
        ax_err.set_xlabel('Frame')
        ax_err.set_ylabel('Cumulative Error (|GT-Pred|)')
        ax_err.set_title(f'End: {end_err:.4f}, Max: {max_err:.4f}')
        ax_err.legend(fontsize=8)
        ax_err.grid(True, alpha=0.3)
    
    # 隐藏空白子图
    for idx in range(num_dims, rows * cols):
        row = idx // cols
        col_base = (idx % cols) * 2
        axes[row, col_base].set_visible(False)
        axes[row, col_base + 1].set_visible(False)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{parquet_name}_cumsum_all_dims_chunk{chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n汇总图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # ========================================
    # 2. 每个维度单独的详细图
    # ========================================
    for dim in dims:
        dim_name = dim_names.get(dim, f'Dim {dim}')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Dim {dim}: {dim_name} - Cumulative Trajectory & Error Analysis\n{parquet_name}, Chunk Size: {chunk_size}', 
                     fontsize=14)
        
        # (0,0) 累积轨迹对比
        ax = axes[0, 0]
        ax.plot(gt_cumsum[:, dim], 'b-', label='GT Cumsum', linewidth=1.5, alpha=0.8)
        ax.plot(pred_cumsum[:, dim], 'r--', label='Pred Cumsum', linewidth=1.5, alpha=0.8)
        for frame in chunk_start_frames[1:]:
            ax.axvline(x=frame, color='gray', linestyle=':', alpha=0.4)
        ax.scatter([0], [0], color='green', s=100, marker='o', label='Start', zorder=10)
        ax.scatter([num_frames-1], [gt_cumsum[-1, dim]], color='blue', s=100, marker='x', label='GT End', zorder=10)
        ax.scatter([num_frames-1], [pred_cumsum[-1, dim]], color='red', s=100, marker='^', label='Pred End', zorder=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Cumulative Value')
        ax.set_title('Cumulative Trajectory Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # (0,1) 原始 delta 对比
        ax = axes[0, 1]
        ax.plot(gt_actions[:, dim], 'b-', label='GT delta', linewidth=0.8, alpha=0.7)
        ax.plot(pred_actions[:, dim], 'r-', label='Pred delta', linewidth=0.8, alpha=0.7)
        for frame in chunk_start_frames[1:]:
            ax.axvline(x=frame, color='gray', linestyle=':', alpha=0.4)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Delta Value')
        ax.set_title('Raw Delta Action Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # (1,0) 累积误差
        ax = axes[1, 0]
        ax.fill_between(range(num_frames), cumsum_error[:, dim], alpha=0.3, color='red')
        ax.plot(cumsum_error[:, dim], 'r-', linewidth=1.2)
        for frame in chunk_start_frames[1:]:
            ax.axvline(x=frame, color='gray', linestyle=':', alpha=0.4)
        
        mean_err = np.mean(cumsum_error[:, dim])
        max_err = np.max(cumsum_error[:, dim])
        end_err = cumsum_error[-1, dim]
        max_idx = np.argmax(cumsum_error[:, dim])
        
        ax.axhline(y=mean_err, color='orange', linestyle='--', 
                   label=f'Mean: {mean_err:.4f}', alpha=0.8)
        ax.scatter([max_idx], [max_err], color='purple', s=100, marker='*', 
                   label=f'Max: {max_err:.4f} @ frame {max_idx}', zorder=10)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Cumulative Error |GT_cumsum - Pred_cumsum|')
        ax.set_title(f'Cumulative Error (End Error: {end_err:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # (1,1) Delta 误差
        ax = axes[1, 1]
        delta_error = np.abs(gt_actions[:, dim] - pred_actions[:, dim])
        ax.fill_between(range(num_frames), delta_error, alpha=0.3, color='blue')
        ax.plot(delta_error, 'b-', linewidth=0.8)
        for frame in chunk_start_frames[1:]:
            ax.axvline(x=frame, color='gray', linestyle=':', alpha=0.4)
        
        mean_delta_err = np.mean(delta_error)
        ax.axhline(y=mean_delta_err, color='orange', linestyle='--', 
                   label=f'Mean: {mean_delta_err:.4f}', alpha=0.8)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Delta Error |GT_delta - Pred_delta|')
        ax.set_title(f'Per-Frame Delta Error (Mean: {mean_delta_err:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{parquet_name}_dim{dim}_chunk{chunk_size}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Dim {dim} 详细图已保存: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    # ========================================
    # 3. 所有维度的误差统计汇总
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Error Statistics Summary (All Dimensions) - {parquet_name}\nChunk Size: {chunk_size}', fontsize=14)
    
    # 累积误差的终点值对比
    ax = axes[0]
    end_errors = cumsum_error[-1, dims]
    mean_errors = np.mean(cumsum_error[:, dims], axis=0)
    max_errors = np.max(cumsum_error[:, dims], axis=0)
    
    x = np.arange(len(dims))
    width = 0.25
    
    ax.bar(x - width, end_errors, width, label='End Error', alpha=0.8)
    ax.bar(x, mean_errors, width, label='Mean Error', alpha=0.8)
    ax.bar(x + width, max_errors, width, label='Max Error', alpha=0.8)
    
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Cumulative Error')
    ax.set_title('Cumulative Error Statistics per Dimension')
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{d}' for d in dims])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 累积误差随时间的演化 (所有维度)
    ax = axes[1]
    for dim in dims:
        dim_name = dim_names.get(dim, f'D{dim}')
        ax.plot(cumsum_error[:, dim], label=f'{dim_name}', alpha=0.7)
    
    for frame in chunk_start_frames[1:]:
        ax.axvline(x=frame, color='gray', linestyle=':', alpha=0.3)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel('Cumulative Error')
    ax.set_title('Cumulative Error Over Time (All Dimensions)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{parquet_name}_error_summary_chunk{chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n误差统计汇总图已保存: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # 打印统计信息
    print("\n" + "=" * 70)
    print("各维度累积误差统计")
    print("=" * 70)
    print(f"{'Dim':<8} {'Name':<15} {'End Error':<12} {'Mean Error':<12} {'Max Error':<12}")
    print("-" * 70)
    for i, dim in enumerate(dims):
        dim_name = dim_names.get(dim, f'Dim {dim}')[:15]
        print(f"{dim:<8} {dim_name:<15} {end_errors[i]:<12.6f} {mean_errors[i]:<12.6f} {max_errors[i]:<12.6f}")
    print("=" * 70)


def run_evaluation(
    pipeline,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    action_denormalizer: MinMaxNormalizer,
    vlm_dir: Path,
    parquet_path: Path,
    device: str,
    chunk_size: int,
    dims: List[int],
    output_dir: str,
    show: bool,
    state_config: Dict[str, Any] = None,
    image_config: Dict[str, Any] = None,
    vlm_backbone = None,
    video_path: Path = None,
    instruction: str = "pick up the bottle",
):
    """
    运行评估并绘图
    
    Args:
        state_config: State 处理配置，格式:
            {
                'state_normalize': {'columns': [...], 'stats_key': '...'},
                'state_gripper_binarize': {'columns': [...], 'threshold': ...}
            }
        image_config: 图像预处理配置，格式:
            {
                'resize': [width, height],
                'flip_horizontal': bool,
                'flip_vertical': bool,
                'rotate_180': bool,
                'bgr_to_rgb': bool
            }
        vlm_backbone: VLM backbone 模型 (端到端模式)
        video_path: 视频文件路径 (端到端模式)
        instruction: 任务指令 (端到端模式)
    """
    num_vlm_layers = config.get('num_vlm_hidden_layers', 1)
    action_dim = config.get('action_dim', 14)
    proprio_dim = config.get('proprio_dim', 14)
    
    # 判断是否使用端到端推理
    use_end_to_end = (vlm_backbone is not None and video_path is not None)
    if use_end_to_end:
        print(f"\n[端到端推理模式]")
        print(f"  视频路径: {video_path}")
    
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
    
    # 预测 actions (存储每帧的预测 delta)
    pred_actions = np.zeros_like(gt_actions, dtype=np.float32)  # (N, action_dim)
    
    # 记录每个 chunk 的起始帧
    chunk_start_frames = []
    
    # 每隔 chunk_size 帧进行一次预测
    predict_frame_indices = list(range(0, num_frames, chunk_size))
    print(f"\n[开始预测] 预测帧索引: {predict_frame_indices[:10]}{'...' if len(predict_frame_indices) > 10 else ''}")
    
    for pred_idx, start_frame in enumerate(predict_frame_indices):
        end_frame = min(start_frame + chunk_size, num_frames)
        actual_chunk_len = end_frame - start_frame
        global_frame_idx = frame_indices[start_frame]
        
        chunk_start_frames.append(start_frame)
        
        # 获取 VLM hidden states
        use_gt = False
        vlm_hidden_states = None
        
        if use_end_to_end:
            # 端到端模式: 从视频提取图像，通过 VLM 获取 hidden states
            try:
                # 提取帧图像
                bgr_to_rgb = image_config.get('bgr_to_rgb', True) if image_config else True
                image = extract_frame_from_video(str(video_path), global_frame_idx, bgr_to_rgb=bgr_to_rgb)
                
                # 图像预处理
                if image_config:
                    image = preprocess_image(
                        image,
                        resize=image_config.get('resize'),
                        flip_horizontal=image_config.get('flip_horizontal', False),
                        flip_vertical=image_config.get('flip_vertical', False),
                        rotate_180=image_config.get('rotate_180', False),
                    )
                
                # 通过 VLM 获取 hidden states
                with torch.no_grad():
                    output = vlm_backbone.get_hidden_states(
                        images=[image],
                        instruction=instruction,
                    )
                    # HiddenStateOutput 对象，通过 .hidden_states 获取列表
                    vlm_hidden_states = [v.to(device) for v in output.hidden_states]
                    
            except Exception as e:
                print(f"  [!] 端到端推理失败，帧 {start_frame}: {e}")
                use_gt = True
        else:
            # 使用预提取的 VLM hidden states
            try:
                vlm_hidden_states = load_vlm_hidden_states(
                    str(vlm_dir), 
                    global_frame_idx,
                    num_vlm_layers
                )
                vlm_hidden_states = [v.to(device) for v in vlm_hidden_states]
            except FileNotFoundError as e:
                print(f"  [!] 跳过帧 {start_frame}: {e}")
                use_gt = True
        
        if use_gt:
            # 使用 GT delta 填充 (用于没有 hidden states 的情况)
            pred_actions[start_frame:end_frame] = gt_actions[start_frame:end_frame]
        else:
            # State 预处理
            if observation_states is not None:
                processed_state = preprocess_state(
                    observation_states[start_frame],
                    config,
                    stats,
                    state_config
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
            predicted_chunk = action_predictions_np.reshape(chunk_size, action_dim)
            
            # Action 反归一化
            if action_denormalizer is not None:
                predicted_chunk = action_denormalizer.denormalize(predicted_chunk)
            
            # 填充预测结果
            pred_actions[start_frame:end_frame] = predicted_chunk[:actual_chunk_len]
        
        # 打印进度
        if pred_idx < 3 or pred_idx == len(predict_frame_indices) - 1:
            print(f"  Chunk {pred_idx}: 帧 {start_frame}-{end_frame-1}, "
                  f"全局帧 {global_frame_idx}"
                  f"{' [GT填充]' if use_gt else ''}")
    
    # 绘制每个维度的累积误差图
    print("\n[绘制累积误差图]")
    plot_cumulative_error_per_dim(
        gt_actions=gt_actions,
        pred_actions=pred_actions,
        dims=dims,
        output_dir=output_dir,
        parquet_name=parquet_path.stem,
        chunk_size=chunk_size,
        chunk_start_frames=chunk_start_frames,
        show=show
    )


def main():
    parser = argparse.ArgumentParser(description='每个 Action 维度的累积误差可视化')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Checkpoint 路径 (action_head.pt)')
    parser.add_argument('--parquet', type=str, required=True,
                        help='Parquet 文件路径')
    parser.add_argument('--stats_path', type=str, default=None,
                        help='stats.json 路径 (默认: parquet 同级的 meta/stats.json)')
    parser.add_argument('--vlm_hidden_states_dir', type=str, default=None,
                        help='VLM hidden states 目录 (默认: parquet 同级的 vlm_hidden_states)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='Chunk 大小 (默认从 config 读取)')
    parser.add_argument('--dims', type=int, nargs='+', default=None,
                        help='要评估的 action 维度 (默认: 所有维度)')
    parser.add_argument('--output_dir', type=str, default='./cumulative_error_plots',
                        help='输出目录')
    parser.add_argument('--show', action='store_true',
                        help='弹出交互式窗口')
    
    # State 处理配置参数
    parser.add_argument('--state_norm_columns', type=int, nargs='+', default=None,
                        help='State 归一化列索引 (例如: 0 1 2 3 4 5 6 7 8 9 10 11)')
    parser.add_argument('--state_norm_stats_key', type=str, default='observation.state',
                        help='State 归一化使用的 stats.json key (默认: observation.state)')
    parser.add_argument('--state_hand_binary_columns', type=int, nargs='+', default=None,
                        help='手部二值化列范围 (起始 结束 起始 结束...)，每组范围取平均后二值化 (例如: 12 18 18 24)')
    parser.add_argument('--state_hand_binary_threshold', type=float, default=442.0,
                        help='手部二值化阈值 (平均值 > threshold -> 1, 否则 -> 0) (默认: 442.0)')
    parser.add_argument('--state_process_order', type=str, nargs='+', default=None,
                        help='State 处理顺序 (例如: hand_binary minmax)')
    
    # 图像预处理配置参数 (用于端到端推理模式)
    parser.add_argument('--image_resize', type=int, nargs=2, default=None,
                        help='图像 resize 尺寸 [width, height] (例如: 256 256)')
    parser.add_argument('--image_flip_horizontal', action='store_true',
                        help='水平翻转图像')
    parser.add_argument('--image_flip_vertical', action='store_true',
                        help='垂直翻转图像')
    parser.add_argument('--image_rotate_180', action='store_true',
                        help='旋转图像 180 度')
    parser.add_argument('--image_bgr_to_rgb', action='store_true',
                        help='BGR 转 RGB (如果图像是 BGR 格式)')
    
    # 端到端推理模式 (使用原始图像而非预提取的 VLM hidden states)
    parser.add_argument('--end_to_end', action='store_true',
                        help='启用端到端推理模式 (从视频提取图像推理，需要完整的 VLM 模型)')
    parser.add_argument('--vlm_model_path', type=str, default=None,
                        help='端到端模式: VLM 模型路径')
    parser.add_argument('--video_path', type=str, default=None,
                        help='端到端模式: 直接指定视频文件路径 (如果自动查找失败)')
    parser.add_argument('--video_key', type=str, default='observation.images.main',
                        help='端到端模式: 视频文件夹名称 (默认: observation.images.main)')
    
    # VLM 配置参数
    parser.add_argument('--vlm_type', type=str, default='eagle2_5_vl',
                        choices=['qwen3_vl', 'eagle2_5_vl', 'cosmos_reason_2b_vl'],
                        help='VLM 模型类型 (默认: eagle2_5_vl)')
    parser.add_argument('--vlm_layers', type=int, nargs='+', default=[-1],
                        help='VLM 提取层 (Eagle: -1, Qwen2B: 14, Qwen4B: 16)')
    parser.add_argument('--vlm_flip_images', action='store_true',
                        help='VLM 图像翻转')
    parser.add_argument('--vlm_content_order', type=str, default='images_first',
                        choices=['images_first', 'text_first'],
                        help='VLM 内容顺序 (默认: images_first)')
    parser.add_argument('--vlm_lowercase_instruction', action='store_true', default=True,
                        help='VLM 指令小写 (默认: True)')
    parser.add_argument('--vlm_add_generation_prompt', action='store_true', default=True,
                        help='VLM 添加 generation prompt (默认: True)')
    
    # 指令参数
    parser.add_argument('--instruction', type=str, default='pick up the bottle',
                        help='任务指令 (默认: pick up the bottle)')
    
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
    
    # VLM hidden states 目录 (非端到端模式使用)
    if args.vlm_hidden_states_dir:
        vlm_dir = Path(args.vlm_hidden_states_dir)
    else:
        vlm_dir = parquet_path.parent.parent.parent / "vlm_hidden_states"
    
    # Stats 路径
    if args.stats_path:
        stats_path = Path(args.stats_path)
    else:
        stats_path = parquet_path.parent.parent.parent / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"警告: stats.json 不存在: {stats_path}")
        stats = None
    else:
        stats = load_stats(str(stats_path))
    
    print("=" * 70)
    print("每个 Action 维度的累积误差可视化")
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
    
    # 默认评估所有维度
    dims = args.dims if args.dims is not None else list(range(action_dim))
    
    print(f"  Action Head Type: {config.get('type', 'unknown')}")
    print(f"  Chunk Size: {chunk_size}")
    print(f"  Action Dim: {action_dim}")
    print(f"  评估维度: {dims}")
    
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
    
    # 构建 state 处理配置
    state_config = None
    if args.state_norm_columns is not None or args.state_hand_binary_columns is not None:
        state_config = {}
        
        # 处理顺序
        if args.state_process_order is not None:
            state_config['state_process_order'] = args.state_process_order
        else:
            # 默认顺序: 先 hand_binary，再 minmax
            state_config['state_process_order'] = ['hand_binary', 'minmax']
        
        if args.state_norm_columns is not None:
            state_config['state_normalize'] = {
                'columns': args.state_norm_columns,
                'stats_key': args.state_norm_stats_key
            }
        
        if args.state_hand_binary_columns is not None:
            state_config['state_hand_binary'] = {
                'columns': args.state_hand_binary_columns,
                'threshold': args.state_hand_binary_threshold
            }
        
        print(f"\n[State 处理配置]")
        print(f"  处理顺序: {state_config['state_process_order']}")
        if 'state_normalize' in state_config:
            print(f"  归一化列: {state_config['state_normalize']['columns']}")
            print(f"  Stats Key: {state_config['state_normalize']['stats_key']}")
        if 'state_hand_binary' in state_config:
            print(f"  手部二值化列范围: {state_config['state_hand_binary']['columns']}")
            print(f"  手部二值化阈值: {state_config['state_hand_binary']['threshold']}")
    else:
        print(f"\n[State 处理配置]")
        print(f"  使用 checkpoint config.json 中的配置")
    
    # 构建图像预处理配置
    image_config = None
    if args.image_resize is not None or args.image_flip_horizontal or args.image_flip_vertical or args.image_rotate_180 or args.image_bgr_to_rgb:
        image_config = {
            'resize': args.image_resize,
            'flip_horizontal': args.image_flip_horizontal,
            'flip_vertical': args.image_flip_vertical,
            'rotate_180': args.image_rotate_180,
            'bgr_to_rgb': args.image_bgr_to_rgb
        }
        
        print(f"\n[图像预处理配置]")
        print(f"  Resize: {image_config['resize']}")
        print(f"  水平翻转: {image_config['flip_horizontal']}")
        print(f"  垂直翻转: {image_config['flip_vertical']}")
        print(f"  旋转180度: {image_config['rotate_180']}")
        print(f"  BGR转RGB: {image_config['bgr_to_rgb']}")
    
    # 端到端模式相关变量
    vlm_backbone = None
    video_path = None
    
    # 检查端到端模式
    if args.end_to_end:
        print(f"\n[端到端推理模式]")
        
        # 检查 VLM 模型路径
        if not args.vlm_model_path:
            print(f"  错误: 端到端模式需要指定 --vlm_model_path")
            return
        
        vlm_model_path = Path(args.vlm_model_path)
        if not vlm_model_path.exists():
            print(f"  错误: VLM 模型路径不存在: {vlm_model_path}")
            return
        
        # 查找视频文件
        if args.video_path:
            # 用户直接指定了视频路径
            video_path = Path(args.video_path)
            if not video_path.exists():
                print(f"  错误: 指定的视频文件不存在: {video_path}")
                return
            print(f"  使用指定的视频路径: {video_path}")
        else:
            # 自动查找视频文件
            video_path = find_video_path(parquet_path, args.video_key)
            if video_path is None:
                print(f"  错误: 未找到视频文件")
                print(f"  提示: 可以使用 --video_path 直接指定视频文件路径")
                return
        
        print(f"  VLM 模型: {vlm_model_path}")
        print(f"  VLM 类型: {args.vlm_type}")
        print(f"  VLM 提取层: {args.vlm_layers}")
        print(f"  视频文件: {video_path}")
        print(f"  指令: {args.instruction}")
        
        # 加载 VLM backbone
        print(f"\n[加载 VLM Backbone]")
        try:
            from VLMs.S0_1.backbone import create_vlm_backbone
            vlm_backbone = create_vlm_backbone(
                model_path=str(vlm_model_path),
                model_type=args.vlm_type,
                device=args.device,
                layers=args.vlm_layers,
                flip_images=args.vlm_flip_images,
                content_order=args.vlm_content_order,
                lowercase_instruction=args.vlm_lowercase_instruction,
                add_generation_prompt=args.vlm_add_generation_prompt,
            )
            print(f"  VLM Backbone 加载成功")
        except Exception as e:
            print(f"  VLM Backbone 加载失败: {e}")
            return
    else:
        # 非端到端模式，检查 VLM hidden states 目录
        if not vlm_dir.exists():
            print(f"错误: VLM hidden states 目录不存在: {vlm_dir}")
            print(f"  提示: 可以使用 --end_to_end 模式从视频直接推理")
            return
    
    # 运行评估
    run_evaluation(
        pipeline=pipeline,
        config=config,
        stats=stats,
        action_denormalizer=action_denormalizer,
        vlm_dir=vlm_dir,
        parquet_path=parquet_path,
        device=args.device,
        chunk_size=chunk_size,
        dims=dims,
        output_dir=args.output_dir,
        show=args.show,
        state_config=state_config,
        image_config=image_config,
        vlm_backbone=vlm_backbone,
        video_path=video_path,
        instruction=args.instruction,
    )
    
    print("\n" + "=" * 70)
    print("评估完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
