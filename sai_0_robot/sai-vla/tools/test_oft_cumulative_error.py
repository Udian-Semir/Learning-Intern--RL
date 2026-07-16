#!/usr/bin/env python3
"""
OFT Action Head 累计误差测试脚本

用于评估 VLM + OFT Action Head 模型的预测准确性，
通过计算和可视化累计误差来对比预测动作与真实动作。

使用方法:
    python /home/dev/文档/huangwenlong/sai0-vla/tools/test_oft_cumulative_error.py \
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
        --vlm_type qwen3_vl \
        --vlm_layers 14 \
        --hidden_dim 2048 \
        --oft_checkpoint /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset111_single_512_filter_weight/qwen2b/parad/p6000Task1lr1changestepchange_bsz128*1*8_tb4_40000steps_layer14_20260206_wptele_train_dataset111_single_512_filter_USE_SHARED_CACHE_true_CACHE_VLM_STATES_false_USE_AMP_true_webdataset_false/checkpoints/step_5000/action_head.pt \
        --num_transformer_blocks 4 \
        --num_attention_heads 8 \
        --action_head_hidden_dim 4096 \
        --dataset_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset111_single_512_filter \
        --episode_idx 0 \
        --chunk_size 8 \
        --action_dim 7 \
        --proprio_dim 7 \
        --image_keys main \
        --instruction "pick up the bottle" \
        --prompt_template detailed \
        --output_dir ./eval_output \
        --state_process_order "hand_binary minmax_normalize" \
        --hand_binary_columns "6 12" \
        --hand_binary_threshold 500 \
        --state_norm_columns_minmax "0 1 2 3 4 5" \
        --stats_file "/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset111_single_512_filter/meta/stats.json" \
        --stats_key "observation.state"

功能:
    1. 加载指定的 VLM backbone 和 OFT Action Head
    2. 从 Parquet 数据集读取 GT actions 和 states
    3. 从 videos/ 目录提取图像
    4. 按 chunk 进行预测，每个 chunk 从 GT 位置开始
    5. 计算累计误差并绘制对比图

State 预处理参数说明:
    --state_process_order: 处理顺序，空格分隔 (如 "hand_binary minmax_normalize")
        - hand_binary: 手部数据二值化 (6维->1维)
        - minmax_normalize: MinMax 归一化到 [-1, 1]
        - gripper_binarize: Gripper 二值化
    --hand_binary_columns: 手部二值化列范围 (如 "6 12" 表示列6-11)
    --hand_binary_threshold: 二值化阈值 (默认 500)
    --state_norm_columns_minmax: MinMax 归一化列 (如 "0 1 2 3 4 5")
    --stats_file: stats.json 路径 (用于获取 min/max)
    --stats_key: stats.json 中的 key (默认 "observation.state")
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ============================================================================
# 路径设置
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # tools -> sai0-vla
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入 VLM Backbone
from VLMs.S0_1.backbone import create_vlm_backbone

# 导入 OFT Pipeline
from Action_Heads.OFT1_0.vlm2oft_pipeline import create_vlm2oft_pipeline

# 动态修改 constants.py 中的值
import Action_Heads.OFT1_0.constants as oft_constants


# ============================================================================
# State 预处理函数
# ============================================================================

def apply_state_preprocessing(
    observation_state: np.ndarray,
    state_process_order: List[str],
    hand_binary_columns: Optional[List[int]] = None,
    hand_binary_threshold: float = 442.0,
    state_euler_to_axisangle_columns: Optional[List[int]] = None,
    # MinMax 归一化参数
    minmax_columns: Optional[List[int]] = None,
    minmax_min: Optional[np.ndarray] = None,
    minmax_max: Optional[np.ndarray] = None,
    # Gripper 二值化参数
    gripper_binarize_columns: Optional[List[int]] = None,
    gripper_binarize_threshold: float = 0.5,
) -> np.ndarray:
    """
    按配置顺序应用 state 预处理
    
    Args:
        observation_state: 原始状态数组
        state_process_order: 处理顺序列表，如 ["hand_binary", "minmax_normalize"]
        hand_binary_columns: 手部二值化列范围 [start1, end1, start2, end2, ...]
        hand_binary_threshold: 手部二值化阈值
        state_euler_to_axisangle_columns: 欧拉角转轴角列
        minmax_columns: MinMax 归一化列
        minmax_min: 每列最小值数组
        minmax_max: 每列最大值数组
        gripper_binarize_columns: Gripper 二值化列
        gripper_binarize_threshold: Gripper 二值化阈值
    
    Returns:
        处理后的状态数组
    """
    if not state_process_order:
        return observation_state
    
    index_offset = 0
    
    for processor_name in state_process_order:
        if processor_name == "hand_binary" and hand_binary_columns is not None:
            hand_offset = 0
            
            for group_idx in range(0, len(hand_binary_columns), 2):
                if group_idx + 1 < len(hand_binary_columns):
                    start = hand_binary_columns[group_idx]
                    end = hand_binary_columns[group_idx + 1]
                    
                    adj_start = start + index_offset + hand_offset
                    adj_end = end + index_offset + hand_offset
                    
                    if adj_start < len(observation_state) and adj_end <= len(observation_state):
                        hand_data = observation_state[adj_start:adj_end]
                        hand_binary = np.array(
                            [1.0 if np.mean(hand_data) > hand_binary_threshold else 0.0], 
                            dtype=np.float32
                        )
                        observation_state = np.concatenate([
                            observation_state[:adj_start],
                            hand_binary,
                            observation_state[adj_end:]
                        ])
                        hand_offset -= (end - start - 1)
            
            index_offset += hand_offset
                    
        elif processor_name == "euler_to_axisangle" and state_euler_to_axisangle_columns:
            # 欧拉角转轴角，简化实现
            pass
        
        elif processor_name == "minmax_normalize" and minmax_columns is not None:
            # 最大最小值归一化: 2 * (x - min) / (max - min) - 1 -> [-1, 1]
            if minmax_min is not None and minmax_max is not None:
                for col in minmax_columns:
                    if col < len(observation_state):
                        col_min = minmax_min[col]
                        col_max = minmax_max[col]
                        col_range = col_max - col_min
                        if col_range > 1e-8:
                            observation_state[col] = 2.0 * (observation_state[col] - col_min) / col_range - 1.0
                        else:
                            observation_state[col] = 0.0
        
        elif processor_name == "gripper_binarize" and gripper_binarize_columns is not None:
            # Gripper 二值化: > threshold 为 1, <= threshold 为 0
            for col in gripper_binarize_columns:
                if col < len(observation_state):
                    if observation_state[col] > gripper_binarize_threshold:
                        observation_state[col] = 1.0
                    else:
                        observation_state[col] = 0.0
    
    return observation_state


def load_stats_minmax(stats_file: str, stats_key: str = "observation.state") -> tuple:
    """
    从 stats.json 加载最小最大值
    
    Args:
        stats_file: stats.json 文件路径
        stats_key: 字段名 (如 "observation.state")
    
    Returns:
        (min_array, max_array)
    """
    import json
    
    if not os.path.exists(stats_file):
        print(f"警告: stats 文件不存在: {stats_file}")
        return None, None
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    if stats_key not in stats:
        print(f"警告: stats 中不存在 key: {stats_key}")
        return None, None
    
    state_stats = stats[stats_key]
    min_vals = np.array(state_stats.get("min", []), dtype=np.float32)
    max_vals = np.array(state_stats.get("max", []), dtype=np.float32)
    
    return min_vals, max_vals


# ============================================================================
# 数据加载函数
# ============================================================================

def load_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """加载数据集元信息"""
    import json
    
    info_path = Path(dataset_path) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"数据集 info.json 不存在: {info_path}")
    
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    # 加载任务描述 (可选)
    tasks_path = Path(dataset_path) / "meta" / "tasks.jsonl"
    tasks = {}
    if tasks_path.exists():
        with open(tasks_path, 'r') as f:
            for line in f:
                if line.strip():
                    task = json.loads(line)
                    tasks[task.get("task_index", 0)] = task.get("task", "")
    
    return {"info": info, "tasks": tasks}


def load_episode_data(
    dataset_path: str,
    episode_idx: int,
    info: Dict[str, Any]
) -> pd.DataFrame:
    """加载指定 episode 的 Parquet 数据"""
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_idx // chunks_size
    
    parquet_path = (
        Path(dataset_path) / "data" / f"chunk-{chunk_idx:03d}" /
        f"episode_{episode_idx:06d}.parquet"
    )
    
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet 文件不存在: {parquet_path}")
    
    return pd.read_parquet(parquet_path)


def get_video_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """从视频中提取指定帧"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取视频帧: {video_path}, frame {frame_idx}")
    
    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_images_from_video(
    dataset_path: str,
    episode_idx: int,
    frame_idx: int,
    chunk_idx: int,
    image_keys: List[str] = None,
    flip: bool = False
) -> List[Image.Image]:
    """从视频中获取指定视角的图像"""
    if image_keys is None:
        image_keys = ["main"]
    
    videos_path = Path(dataset_path) / "videos"
    images = []
    
    for key in image_keys:
        video_path = (
            videos_path / f"chunk-{chunk_idx:03d}" /
            f"observation.images.{key}" / f"episode_{episode_idx:06d}.mp4"
        )
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        img = get_video_frame(str(video_path), frame_idx)
        
        if flip:
            img = img[::-1, ::-1, :].copy()
        
        images.append(Image.fromarray(img))
    
    return images


# ============================================================================
# OFT 评估类
# ============================================================================

class OFTEvaluator:
    """OFT Action Head 评估器"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        
        # 解析 VLM 层
        if isinstance(args.vlm_layers, str):
            self.vlm_layers = [int(x.strip()) for x in args.vlm_layers.split(',')]
        else:
            self.vlm_layers = args.vlm_layers if isinstance(args.vlm_layers, list) else [args.vlm_layers]
        
        # 动态修改 OFT constants
        oft_constants.LLM_OUTPUT_DIM_MLP_INPUT_DIM = args.hidden_dim
        oft_constants.NUM_VLM_HIDDEN_LAYERS = len(self.vlm_layers)
        oft_constants.ACTION_DIM = args.action_dim
        oft_constants.NUM_ACTIONS_CHUNK = args.chunk_size
        oft_constants.PROPRIO_DIM = args.proprio_dim
        
        # 解析 State 预处理参数
        self._parse_state_preprocess_args()
        
        self._load_models()
    
    def _parse_state_preprocess_args(self):
        """解析 State 预处理参数"""
        args = self.args
        
        # 处理顺序
        self.state_process_order = []
        if args.state_process_order:
            self.state_process_order = args.state_process_order.split()
        
        # 手部二值化参数
        self.hand_binary_columns = None
        if args.hand_binary_columns:
            self.hand_binary_columns = [int(x) for x in args.hand_binary_columns.split()]
        self.hand_binary_threshold = args.hand_binary_threshold
        
        # 欧拉角转轴角参数
        self.state_euler_to_axisangle_columns = None
        if args.state_euler_to_axisangle_columns:
            self.state_euler_to_axisangle_columns = [int(x) for x in args.state_euler_to_axisangle_columns.split()]
        
        # MinMax 归一化参数
        self.minmax_columns = None
        self.minmax_min = None
        self.minmax_max = None
        if args.state_norm_columns_minmax:
            self.minmax_columns = [int(x) for x in args.state_norm_columns_minmax.split()]
            if args.stats_file:
                self.minmax_min, self.minmax_max = load_stats_minmax(
                    args.stats_file, args.stats_key
                )
                if self.minmax_min is not None:
                    print(f"  - 加载 stats min/max: {args.stats_file}")
                    print(f"    MinMax 列: {self.minmax_columns}")
        
        # Gripper 二值化参数
        self.gripper_binarize_columns = None
        if args.gripper_binarize_columns:
            self.gripper_binarize_columns = [int(x) for x in args.gripper_binarize_columns.split()]
        self.gripper_binarize_threshold = args.gripper_binarize_threshold
        
        # 打印配置
        if self.state_process_order:
            print(f"\n📋 State 预处理配置:")
            print(f"  - 处理顺序: {self.state_process_order}")
            if self.hand_binary_columns:
                print(f"  - 手部二值化列: {self.hand_binary_columns}, 阈值: {self.hand_binary_threshold}")
            if self.minmax_columns:
                print(f"  - MinMax 归一化列: {self.minmax_columns}")
    
    def _load_models(self):
        """加载 VLM 和 OFT Pipeline"""
        args = self.args
        
        # ========== 1. 加载 VLM Backbone ==========
        print(f"\n📥 加载 VLM Backbone...")
        print(f"  - Model Path: {args.vlm_model_path}")
        print(f"  - Model Type: {args.vlm_type}")
        print(f"  - Layers: {self.vlm_layers}")
        
        self.vlm_backbone = create_vlm_backbone(
            model_type=args.vlm_type,
            model_path=args.vlm_model_path,
            device=args.device,
            layers=self.vlm_layers,
            prompt_template=args.prompt_template,
            content_order=args.content_order,
            lowercase_instruction=args.lowercase_instruction,
            add_generation_prompt=args.add_generation_prompt,
            flip_images=args.flip_images,
        )
        print("  ✓ VLM Backbone 加载成功")
        
        # ========== 2. 加载 OFT Pipeline ==========
        print(f"\n📥 加载 OFT Pipeline...")
        print(f"  - Checkpoint: {args.oft_checkpoint}")
        print(f"  - Transformer Blocks: {args.num_transformer_blocks}")
        print(f"  - Attention Heads: {args.num_attention_heads}")
        print(f"  - Action Head Hidden Dim: {args.action_head_hidden_dim}")
        
        self.oft_pipeline = create_vlm2oft_pipeline(
            num_transformer_blocks=args.num_transformer_blocks,
            num_attention_heads=args.num_attention_heads,
            num_vlm_layers=len(self.vlm_layers),
            vlm_output_dim=args.hidden_dim,
            action_head_hidden_dim=args.action_head_hidden_dim,
        ).to(self.device)
        
        # 加载权重
        if not os.path.exists(args.oft_checkpoint):
            raise FileNotFoundError(f"OFT checkpoint 不存在: {args.oft_checkpoint}")
        
        oft_state = torch.load(args.oft_checkpoint, map_location=self.device)
        self.oft_pipeline.load_state_dict(oft_state)
        self.oft_pipeline.eval()
        
        print("  ✓ OFT Pipeline 加载成功")
        print(f"  - Chunk Size: {args.chunk_size}")
        print(f"  - Action Dim: {args.action_dim}")
    
    def predict_chunk(
        self,
        images: List[Image.Image],
        instruction: str,
        observation_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        预测单个 chunk 的动作
        
        Args:
            images: 图像列表
            instruction: 任务指令
            observation_state: 观测状态，shape (proprio_dim,)
        
        Returns:
            预测的动作 delta，shape (chunk_size, action_dim)
        """
        args = self.args
        
        # 获取 VLM 隐藏状态
        vlm_output = self.vlm_backbone.get_hidden_states(
            images=images,
            instruction=instruction,
        )
        vlm_hidden_states = vlm_output.hidden_states  # List[Tensor]
        
        with torch.no_grad():
            vlm_hidden_states_device = [v.to(self.device) for v in vlm_hidden_states]
            
            # 准备 proprioception
            if observation_state is not None:
                proprio_state = observation_state[:args.proprio_dim]
                proprio_tensor = torch.tensor(
                    proprio_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)  # (1, proprio_dim)
            else:
                proprio_tensor = torch.zeros(
                    1, args.proprio_dim, dtype=torch.float32, device=self.device
                )
            
            # OFT Pipeline 推理
            action_predictions = self.oft_pipeline(
                vlm_hidden_states_device,
                proprio_tensor
            )
            # action_predictions: (batch_size, 1, chunk_size * action_dim)
        
        # 转换为 numpy 并重塑
        action_predictions_np = action_predictions[0, 0].cpu().numpy()  # (chunk_size * action_dim,)
        pred_deltas = action_predictions_np.reshape(args.chunk_size, args.action_dim)
        
        return pred_deltas
    
    def evaluate_episode(self) -> Dict[str, Any]:
        """
        评估单个 episode
        
        Returns:
            评估结果字典
        """
        args = self.args
        
        # 加载数据集信息
        print(f"\n📂 加载数据集: {args.dataset_path}")
        dataset_info = load_dataset_info(args.dataset_path)
        info = dataset_info["info"]
        tasks = dataset_info["tasks"]
        
        print(f"  - 总 episodes: {info.get('total_episodes', 'N/A')}")
        
        # 获取任务指令
        instruction = args.instruction
        if not instruction and tasks:
            task_description = tasks.get(0, "Complete the task")
            instruction = task_description.lower() if args.lowercase_instruction else task_description
        print(f"  - 任务指令: {instruction}")
        
        # 加载 episode 数据
        print(f"\n📋 加载 Episode {args.episode_idx}...")
        df = load_episode_data(args.dataset_path, args.episode_idx, info)
        episode_length = len(df)
        print(f"  - Episode 长度: {episode_length} 帧")
        
        # 提取 GT actions
        gt_actions = np.array(df['action'].tolist(), dtype=np.float32)
        print(f"  - GT actions shape: {gt_actions.shape}")
        
        # 提取 observation.state (如果存在)
        observation_states = None
        if 'observation.state' in df.columns:
            raw_states = np.array(df['observation.state'].tolist(), dtype=np.float32)
            print(f"  - Raw observation states shape: {raw_states.shape}")
            
            # 应用 state 预处理
            if self.state_process_order:
                print(f"  - 应用 State 预处理: {self.state_process_order}")
                observation_states = []
                for i in range(len(raw_states)):
                    processed_state = apply_state_preprocessing(
                        raw_states[i].copy(),
                        state_process_order=self.state_process_order,
                        hand_binary_columns=self.hand_binary_columns,
                        hand_binary_threshold=self.hand_binary_threshold,
                        state_euler_to_axisangle_columns=self.state_euler_to_axisangle_columns,
                        minmax_columns=self.minmax_columns,
                        minmax_min=self.minmax_min,
                        minmax_max=self.minmax_max,
                        gripper_binarize_columns=self.gripper_binarize_columns,
                        gripper_binarize_threshold=self.gripper_binarize_threshold,
                    )
                    observation_states.append(processed_state)
                observation_states = np.array(observation_states, dtype=np.float32)
                print(f"  - Processed observation states shape: {observation_states.shape}")
            else:
                observation_states = raw_states
        
        # 计算 GT 累计位置
        gt_position = np.cumsum(gt_actions, axis=0)
        
        # 初始化预测数组
        pred_deltas = np.zeros_like(gt_actions)
        pred_position = np.zeros_like(gt_actions)
        
        # 计算 chunk 信息
        chunks_size = info.get("chunks_size", 1000)
        chunk_idx = args.episode_idx // chunks_size
        
        # 按 chunk 进行预测
        print(f"\n🔮 开始预测 (chunk_size={args.chunk_size})...")
        num_chunks = (episode_length + args.chunk_size - 1) // args.chunk_size
        
        for i, chunk_start in enumerate(tqdm(range(0, episode_length, args.chunk_size), desc="预测进度")):
            chunk_end = min(chunk_start + args.chunk_size, episode_length)
            actual_chunk_len = chunk_end - chunk_start
            
            # 获取当前帧的图像
            images = get_images_from_video(
                args.dataset_path,
                args.episode_idx,
                chunk_start,  # 使用 chunk 起始帧的图像
                chunk_idx,
                image_keys=args.image_keys,
                flip=args.flip_images
            )
            
            # 获取当前帧的 observation.state
            current_state = None
            if observation_states is not None:
                current_state = observation_states[chunk_start]
            
            # 预测
            chunk_pred = self.predict_chunk(images, instruction, current_state)
            
            # 截取实际长度
            pred_deltas[chunk_start:chunk_end] = chunk_pred[:actual_chunk_len]
            
            # 计算累计位置 (每个 chunk 从 GT 位置开始，评估单 chunk 预测质量)
            if chunk_start == 0:
                chunk_start_pos = np.zeros(args.action_dim)
            else:
                chunk_start_pos = gt_position[chunk_start - 1]  # 从 GT 位置开始
            
            pred_position[chunk_start:chunk_end] = (
                chunk_start_pos + np.cumsum(pred_deltas[chunk_start:chunk_end], axis=0)
            )
        
        # 计算误差
        position_error = np.abs(gt_position - pred_position)
        max_error_per_dim = np.max(position_error, axis=0)
        mean_error_per_dim = np.mean(position_error, axis=0)
        
        print(f"\n📊 误差统计:")
        print(f"  - 最大误差 (各维度): {max_error_per_dim}")
        print(f"  - 平均误差 (各维度): {mean_error_per_dim}")
        
        return {
            "episode_idx": args.episode_idx,
            "episode_length": episode_length,
            "gt_actions": gt_actions,
            "gt_position": gt_position,
            "pred_deltas": pred_deltas,
            "pred_position": pred_position,
            "position_error": position_error,
            "max_error_per_dim": max_error_per_dim,
            "mean_error_per_dim": mean_error_per_dim,
            "instruction": instruction,
        }
    
    def plot_results(self, results: Dict[str, Any]):
        """
        绘制评估结果
        
        Args:
            results: 评估结果字典
        """
        args = self.args
        os.makedirs(args.output_dir, exist_ok=True)
        
        episode_idx = results["episode_idx"]
        episode_length = results["episode_length"]
        gt_actions = results["gt_actions"]
        gt_position = results["gt_position"]
        pred_deltas = results["pred_deltas"]
        pred_position = results["pred_position"]
        action_dim = gt_actions.shape[1]
        
        x = np.arange(episode_length)
        
        # 绘制每个维度的单独图
        print(f"\n🎨 绘制图表...")
        for dim in range(action_dim):
            fig, axes = plt.subplots(2, 1, figsize=(20, 10))
            
            # 子图1: Delta Action 对比
            ax1 = axes[0]
            ax1.plot(x, gt_actions[:, dim], 'b-', label='GT Delta Action', alpha=0.7, linewidth=1)
            
            # 绘制预测的 delta
            for chunk_start in range(0, episode_length, args.chunk_size):
                chunk_end = min(chunk_start + args.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pred = pred_deltas[chunk_start:chunk_end, dim]
                label = f'Pred Delta (chunk={args.chunk_size})' if chunk_start == 0 else None
                ax1.plot(chunk_x, chunk_pred, 'r-', alpha=0.8, linewidth=1.5, label=label)
                ax1.axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.5)
            
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Delta Value')
            ax1.set_title(f'Delta Action Comparison - Dimension {dim}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 子图2: 累计位置对比
            ax2 = axes[1]
            ax2.plot(x, gt_position[:, dim], 'b-', label='GT Position (cumsum)',
                     alpha=0.7, linewidth=1.5)
            
            # 绘制预测的累计位置
            for chunk_start in range(0, episode_length, args.chunk_size):
                chunk_end = min(chunk_start + args.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pos = pred_position[chunk_start:chunk_end, dim]
                label = 'Pred Position' if chunk_start == 0 else None
                ax2.plot(chunk_x, chunk_pos, 'r-', alpha=0.8, linewidth=1.5, label=label)
                ax2.axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.5)
            
            max_err = results["max_error_per_dim"][dim]
            mean_err = results["mean_error_per_dim"][dim]
            ax2.set_xlabel('Frame Index')
            ax2.set_ylabel('Position Value')
            ax2.set_title(f'Position Comparison - Dimension {dim} | '
                          f'Max Error: {max_err:.4f}, Mean Error: {mean_err:.4f}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存单维度图
            save_path = os.path.join(
                args.output_dir,
                f'episode_{episode_idx:06d}_dim_{dim}_chunk_{args.chunk_size}.png'
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [Dim {dim}] max_err={max_err:.6f}, mean_err={mean_err:.6f} -> {save_path}")
        
        # 绘制汇总图
        self._plot_summary(results)
    
    def _plot_summary(self, results: Dict[str, Any]):
        """绘制汇总图"""
        args = self.args
        
        episode_idx = results["episode_idx"]
        episode_length = results["episode_length"]
        gt_actions = results["gt_actions"]
        gt_position = results["gt_position"]
        pred_deltas = results["pred_deltas"]
        pred_position = results["pred_position"]
        action_dim = gt_actions.shape[1]
        
        x = np.arange(episode_length)
        
        # 绘制汇总图
        num_cols = 2
        num_rows = action_dim
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 5 * num_rows))
        
        # 确保 axes 是 2D 数组
        if action_dim == 1:
            axes = axes.reshape(1, -1)
        
        for dim in range(action_dim):
            # 第一列: Delta 对比
            axes[dim, 0].plot(x, gt_actions[:, dim], 'b-', label='GT', alpha=0.7, linewidth=1)
            for chunk_start in range(0, episode_length, args.chunk_size):
                chunk_end = min(chunk_start + args.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pred = pred_deltas[chunk_start:chunk_end, dim]
                label = 'Pred' if chunk_start == 0 else None
                axes[dim, 0].plot(chunk_x, chunk_pred, 'r-', alpha=0.8, linewidth=1.5, label=label)
                axes[dim, 0].axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.3)
            
            axes[dim, 0].set_ylabel(f'Dim {dim}')
            axes[dim, 0].legend(loc='upper right', fontsize=8)
            axes[dim, 0].grid(True, alpha=0.3)
            
            # 第二列: 位置对比
            axes[dim, 1].plot(x, gt_position[:, dim], 'b-', label='GT', alpha=0.7, linewidth=1.5)
            for chunk_start in range(0, episode_length, args.chunk_size):
                chunk_end = min(chunk_start + args.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pos = pred_position[chunk_start:chunk_end, dim]
                label = 'Pred' if chunk_start == 0 else None
                axes[dim, 1].plot(chunk_x, chunk_pos, 'r-', alpha=0.8, linewidth=1.5, label=label)
                axes[dim, 1].axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.3)
            
            max_err = results["max_error_per_dim"][dim]
            mean_err = results["mean_error_per_dim"][dim]
            axes[dim, 1].set_ylabel(f'Dim {dim}')
            axes[dim, 1].set_title(f'Max: {max_err:.4f}, Mean: {mean_err:.4f}', fontsize=9)
            axes[dim, 1].legend(loc='upper right', fontsize=8)
            axes[dim, 1].grid(True, alpha=0.3)
        
        axes[0, 0].set_title('Delta Action Comparison')
        axes[0, 1].set_title('Position Comparison (Cumsum)')
        axes[-1, 0].set_xlabel('Frame Index')
        axes[-1, 1].set_xlabel('Frame Index')
        
        instruction_display = results["instruction"][:80] + "..." if len(results["instruction"]) > 80 else results["instruction"]
        plt.suptitle(
            f'Episode {episode_idx} - OFT Evaluation (chunk={args.chunk_size})\n'
            f'Instruction: {instruction_display}',
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        
        summary_path = os.path.join(
            args.output_dir,
            f'episode_{episode_idx:06d}_summary_chunk_{args.chunk_size}.png'
        )
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n汇总图已保存: {summary_path}")


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='OFT Action Head 累计误差测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ========== VLM 相关 ==========
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="VLM 模型路径")
    parser.add_argument("--vlm_type", type=str, default="qwen3_vl",
                        choices=["qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"],
                        help="VLM 类型")
    parser.add_argument("--vlm_layers", type=str, default="14",
                        help="提取的 VLM 隐藏层 (如 '14' 或 '1,14,28')")
    parser.add_argument("--hidden_dim", type=int, default=2048,
                        help="VLM 隐藏层维度 (Eagle: 2048, Qwen2B: 2048, Qwen4B: 2560)")
    
    # ========== OFT Pipeline 相关 ==========
    parser.add_argument("--oft_checkpoint", type=str, required=True,
                        help="OFT Action Head checkpoint 路径 (action_head.pt)")
    parser.add_argument("--num_transformer_blocks", type=int, default=4,
                        help="Transformer 块数量")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="注意力头数量")
    parser.add_argument("--action_head_hidden_dim", type=int, default=4096,
                        help="Action Head 隐藏层维度")
    
    # ========== 动作配置 ==========
    parser.add_argument("--chunk_size", type=int, default=8,
                        help="动作块大小")
    parser.add_argument("--action_dim", type=int, default=7,
                        help="动作维度")
    parser.add_argument("--proprio_dim", type=int, default=7,
                        help="Proprioception 维度")
    
    # ========== 数据集相关 ==========
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集路径 (LeRobot 格式)")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="要评估的 episode 索引")
    parser.add_argument("--image_keys", type=str, nargs='+', default=["main"],
                        help="图像视角名称列表")
    
    # ========== State 预处理相关 ==========
    parser.add_argument("--state_process_order", type=str, default="",
                        help="State 处理顺序，空格分隔 (如 'hand_binary minmax_normalize')")
    parser.add_argument("--hand_binary_columns", type=str, default="",
                        help="手部二值化列范围，空格分隔 (如 '6 12' 表示列6-12)")
    parser.add_argument("--hand_binary_threshold", type=float, default=500.0,
                        help="手部二值化阈值")
    parser.add_argument("--state_euler_to_axisangle_columns", type=str, default="",
                        help="欧拉角转轴角列，空格分隔 (如 '9 10 11')")
    parser.add_argument("--state_norm_columns_minmax", type=str, default="",
                        help="MinMax 归一化列，空格分隔 (如 '0 1 2 3 4 5')")
    parser.add_argument("--stats_file", type=str, default="",
                        help="stats.json 文件路径 (用于 MinMax 归一化)")
    parser.add_argument("--stats_key", type=str, default="observation.state",
                        help="stats.json 中的 key 名称")
    parser.add_argument("--gripper_binarize_columns", type=str, default="",
                        help="Gripper 二值化列，空格分隔")
    parser.add_argument("--gripper_binarize_threshold", type=float, default=0.5,
                        help="Gripper 二值化阈值")
    
    # ========== Prompt 配置 ==========
    parser.add_argument("--instruction", type=str, default="",
                        help="任务指令 (默认从数据集 tasks.jsonl 读取)")
    parser.add_argument("--prompt_template", type=str, default="detailed",
                        help="Prompt 模板 (action, simple, detailed, step_by_step 或自定义)")
    parser.add_argument("--content_order", type=str, default="images_first",
                        help="内容顺序 (images_first, text_first, interleaved)")
    parser.add_argument("--lowercase_instruction", action="store_true", default=True,
                        help="将指令转为小写")
    parser.add_argument("--no_lowercase_instruction", dest="lowercase_instruction", action="store_false",
                        help="不将指令转为小写")
    parser.add_argument("--add_generation_prompt", action="store_true", default=True,
                        help="添加 generation prompt")
    parser.add_argument("--no_generation_prompt", dest="add_generation_prompt", action="store_false",
                        help="不添加 generation prompt")
    
    # ========== 图像配置 ==========
    parser.add_argument("--flip_images", action="store_true", default=False,
                        help="翻转图像 (180度旋转)")
    
    # ========== 输出配置 ==========
    parser.add_argument("--output_dir", type=str, default="./eval_output",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    
    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

def main():
    args = parse_args()
    
    print("=" * 60)
    print("OFT Action Head 累计误差测试")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  VLM Model: {args.vlm_model_path}")
    print(f"  VLM Type: {args.vlm_type}")
    print(f"  VLM Layers: {args.vlm_layers}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  OFT Checkpoint: {args.oft_checkpoint}")
    print(f"  Chunk Size: {args.chunk_size}")
    print(f"  Action Dim: {args.action_dim}")
    print(f"  Proprio Dim: {args.proprio_dim}")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Episode: {args.episode_idx}")
    print(f"  Image Keys: {args.image_keys}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Device: {args.device}")
    
    # State 预处理配置
    if args.state_process_order:
        print(f"\n  State 预处理:")
        print(f"    处理顺序: {args.state_process_order}")
        if args.hand_binary_columns:
            print(f"    手部二值化: 列={args.hand_binary_columns}, 阈值={args.hand_binary_threshold}")
        if args.state_norm_columns_minmax:
            print(f"    MinMax 归一化: 列={args.state_norm_columns_minmax}")
            if args.stats_file:
                print(f"    Stats 文件: {args.stats_file}")
    
    # 创建评估器
    evaluator = OFTEvaluator(args)
    
    # 评估
    start_time = time.time()
    results = evaluator.evaluate_episode()
    eval_time = time.time() - start_time
    print(f"\n⏱️ 评估耗时: {eval_time:.2f}s")
    
    # 绘图
    evaluator.plot_results(results)
    
    print("\n" + "=" * 60)
    print("评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
