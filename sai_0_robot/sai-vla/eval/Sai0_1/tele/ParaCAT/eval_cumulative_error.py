#!/usr/bin/env python3
"""
遥操作数据 ParaCAT 评估脚本 - 累计误差绘图

使用 ParaCAT Action Head 对遥操作数据进行评估，
绘制 GT action delta 和预测 delta 的累计误差对比图。

使用方法:
    CUDA_VISIBLE_DEVICES=0 python -m eval.Sai0_1.tele.ParaCAT.eval_cumulative_error \
        --paracat_checkpoint ./checkpoints/paracat.pt \
        --pons_checkpoint ./checkpoints/pons.pt \
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
        --dataset_path /path/to/lerobot_dataset \
        --episode_idx 0 \
        --chunk_size 16 \
        --output_dir ./eval_plots

功能:
    1. 加载指定 episode 的数据 (action, images, task_description)
    2. 实时通过 VLM backbone 提取 hidden states
    3. 按 chunk 进行 ParaCAT 预测
    4. 计算累计位置 (GT 和预测)
    5. 绘制每个 action 维度的对比图和汇总图
"""

import sys
import os
import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# ============================================================================
# 路径设置
# ============================================================================
EVAL_DIR = Path(__file__).resolve().parent
SAI0_ROOT = EVAL_DIR.parents[3]  # eval/Sai0_1/tele/ParaCAT -> sai0-vla
if str(SAI0_ROOT) not in sys.path:
    sys.path.insert(0, str(SAI0_ROOT))

# 导入 VLM Backbone
from VLMs.S0_1.backbone import create_vlm_backbone

# 导入 ParaCAT Action Head
from Action_Heads.ParaCAT.model.action_head.paracat_action_head import (
    ParaCATActionHead, create_paracat_action_head
)

# 导入 Pons Adapter
from Adapter.Pons.pons_adapter import PonsAdapter, create_pons_adapter

# 导入离散化/反离散化工具函数
from utils.discrete import undiscrete_constrain_delta


# ============================================================================
# 常量定义
# ============================================================================
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


# ============================================================================
# ObservationStateMapper 类 (从训练代码复制)
# ============================================================================

class ObservationStateMapper(nn.Module):
    """
    将 observation_state 按列归一化并映射到 VLM 隐藏空间
    
    支持两种归一化方式:
    1. minmax: 从 stats.json 读取 min/max 进行 [0, 1] 归一化
    2. axisangle: 对已转换的轴角值 (范围 [-pi, pi]) 进行 [0, 1] 归一化
    
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
        minmax_columns: List[int] = None,
        axisangle_columns: List[int] = None,
        state_min: torch.Tensor = None,
        state_max: torch.Tensor = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.minmax_columns = minmax_columns or []
        self.axisangle_columns = axisangle_columns or []
        
        if state_min is not None:
            self.register_buffer('state_min', state_min)
        else:
            self.register_buffer('state_min', torch.zeros(state_dim))
            
        if state_max is not None:
            self.register_buffer('state_max', state_max)
        else:
            self.register_buffer('state_max', torch.ones(state_dim))
        
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
        normalized = state.clone()
        
        for col in self.minmax_columns:
            col_min = self.state_min[col]
            col_max = self.state_max[col]
            col_range = col_max - col_min + 1e-8
            normalized[:, col] = (state[:, col] - col_min) / col_range
        
        for col in self.axisangle_columns:
            normalized[:, col] = (state[:, col] + math.pi) / (2 * math.pi)
        
        state_embedding = self.mlp(normalized)
        return state_embedding.unsqueeze(1)


# ============================================================================
# State 预处理函数
# ============================================================================

def euler_to_quat_numpy(euler: np.ndarray) -> np.ndarray:
    """欧拉角转四元数 (NumPy 批量版本)"""
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
    """四元数转轴角 (NumPy 批量版本)"""
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qw = np.clip(qw, -1.0, 1.0)
    
    den = np.sqrt(1.0 - qw * qw)
    angle = 2.0 * np.arccos(qw)
    
    small_angle_mask = den < 1e-8
    axis_angle = np.zeros(quat.shape[:-1] + (3,), dtype=quat.dtype)
    
    if np.any(~small_angle_mask):
        scale = np.where(small_angle_mask, 0.0, angle / (den + 1e-8))
        axis_angle[..., 0] = qx * scale
        axis_angle[..., 1] = qy * scale
        axis_angle[..., 2] = qz * scale
    
    return axis_angle


def euler_to_axisangle_numpy(euler: np.ndarray) -> np.ndarray:
    """欧拉角转轴角"""
    quat = euler_to_quat_numpy(euler)
    return quat_to_axisangle_numpy(quat)


def apply_state_preprocessing(
    observation_state: np.ndarray,
    state_process_order: List[str],
    hand_binary_columns: Optional[List[int]] = None,
    hand_binary_threshold: float = 442.0,
    state_euler_to_axisangle_columns: Optional[List[int]] = None,
) -> np.ndarray:
    """
    按配置顺序应用 state 预处理
    
    Args:
        observation_state: 原始 state 数组
        state_process_order: 预处理执行顺序
        hand_binary_columns: 手部列索引范围
        hand_binary_threshold: 手部二值化阈值
        state_euler_to_axisangle_columns: 欧拉角列索引
        
    Returns:
        处理后的 state 数组
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
            for i in range(0, len(state_euler_to_axisangle_columns), 3):
                if i + 2 < len(state_euler_to_axisangle_columns):
                    original_cols = state_euler_to_axisangle_columns[i:i+3]
                    cols = [c + index_offset for c in original_cols]
                    
                    if all(col < len(observation_state) for col in cols):
                        euler = observation_state[cols]
                        axisangle = euler_to_axisangle_numpy(euler.reshape(1, 3)).flatten()
                        observation_state[cols] = axisangle
    
    return observation_state


# ============================================================================
# 数据加载函数
# ============================================================================

def load_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """加载数据集元信息"""
    meta_path = Path(dataset_path) / "meta"
    
    with open(meta_path / "info.json", "r") as f:
        info = json.load(f)
    
    tasks = {}
    tasks_path = meta_path / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                task = json.loads(line)
                tasks[task["task_index"]] = task["task"]
    
    return {"info": info, "tasks": tasks}


def load_episode_data(
    dataset_path: str, 
    episode_idx: int,
    info: Dict[str, Any]
) -> pd.DataFrame:
    """加载指定 episode 的数据"""
    chunks_size = info["chunks_size"]
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
# 评估主类
# ============================================================================

@dataclass
class EvalConfig:
    """评估配置"""
    # 模型参数
    paracat_checkpoint: str = ""
    pons_checkpoint: str = ""
    state_mapper_checkpoint: str = ""  # State Mapper checkpoint 路径
    vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_type: str = "qwen3_vl"
    
    # VLM 相关
    vlm_layers: List[int] = field(default_factory=lambda: [14])
    vlm_output_dim: int = 2048
    
    # ParaCAT 参数
    chunk_size: int = 16
    action_dim: int = 14
    num_transformer_blocks: int = 2
    num_mlp_layers: int = 2
    mlp_expand_dim: int = 1024
    num_heads: int = 8
    
    # Pons 参数
    pons_q_seq_len: int = 64
    pons_num_blocks: int = 2
    
    # 数据集参数
    dataset_path: str = ""
    episode_idx: int = 0
    image_keys: List[str] = field(default_factory=lambda: ["main"])
    
    # 离散化参数
    undiscrete_columns: Optional[List[int]] = None
    undiscrete_deltas: Optional[List[float]] = None
    gripper_columns: Optional[List[int]] = None  # Gripper 列 (原始值已是 {-1, 0, 1})
    
    # State 预处理参数
    state_process_order: Optional[List[str]] = None
    hand_binary_columns: Optional[List[int]] = None
    hand_binary_threshold: float = 442.0
    state_euler_to_axisangle_columns: Optional[List[int]] = None
    state_dim: int = 14
    
    # State Mapper 参数
    enable_state_mapper: bool = False
    state_norm_columns_minmax: Optional[List[int]] = None
    state_norm_columns_axisangle: Optional[List[int]] = None
    
    # Prompt 配置
    content_order: str = "images_first"
    lowercase_instruction: bool = True
    add_generation_prompt: bool = True
    add_action_prompt: bool = True
    
    # 系统配置
    device: str = "cuda:0"
    flip_images: bool = False
    output_dir: str = "./eval_plots"
    verbose: bool = False


class TeleEvaluator:
    """
    遥操作数据 ParaCAT 评估器
    
    评估 ParaCAT 模型在遥操作数据上的预测精度，
    通过绘制累计误差图进行可视化分析。
    """
    
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        self.device = cfg.device
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载所有模型组件"""
        cfg = self.cfg
        
        # ====== 1. 加载 VLM Backbone ======
        print(f"\n📦 加载 VLM Backbone: {cfg.vlm_model_path}")
        print(f"  - 类型: {cfg.vlm_type}")
        print(f"  - 提取层: {cfg.vlm_layers}")
        
        self.vlm_backbone = create_vlm_backbone(
            model_type=cfg.vlm_type,
            model_path=cfg.vlm_model_path,
            device=cfg.device,
            layers=cfg.vlm_layers,
            flip_images=cfg.flip_images,
            content_order=cfg.content_order,
            prompt_template="simple",
            lowercase_instruction=cfg.lowercase_instruction,
            verbose=cfg.verbose,
        )
        print(f"✓ VLM Backbone 加载成功!")
        
        # ====== 2. 加载 Pons Adapter (可选) ======
        self.pons = None
        self.use_pons = cfg.pons_checkpoint and cfg.pons_checkpoint != ""
        
        if self.use_pons:
            print(f"\n📦 加载 Pons Adapter: {cfg.pons_checkpoint}")
            
            self.pons = create_pons_adapter(
                q_seq_len=cfg.pons_q_seq_len,
                hidden_dim=cfg.vlm_output_dim,
                num_blocks=cfg.pons_num_blocks,
                num_heads=cfg.num_heads,
            ).to(cfg.device)
            
            pons_state = torch.load(cfg.pons_checkpoint, map_location=cfg.device)
            self.pons.load_state_dict(pons_state)
            self.pons.eval()
            print(f"✓ Pons Adapter 加载成功!")
        else:
            print(f"\n📦 不使用 Pons Adapter，直接使用 VLM hidden states")
        
        # ====== 3. 加载 ParaCAT Action Head ======
        print(f"\n📦 加载 ParaCAT Action Head: {cfg.paracat_checkpoint}")
        
        # 加载 checkpoint 检测配置
        ckpt_dir = Path(cfg.paracat_checkpoint).parent
        config_path = ckpt_dir / "config.json"
        
        cfg_dict = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg_dict = json.load(f)
            print(f"  - 从 config.json 加载配置")
            
            # 使用配置文件中的参数
            if 'hidden_dim' in cfg_dict:
                cfg.vlm_output_dim = cfg_dict['hidden_dim']
            if 'chunk_size' in cfg_dict:
                cfg.chunk_size = cfg_dict['chunk_size']
            if 'action_dim' in cfg_dict:
                cfg.action_dim = cfg_dict['action_dim']
            if 'num_transformer_blocks' in cfg_dict:
                cfg.num_transformer_blocks = cfg_dict['num_transformer_blocks']
            if 'num_mlp_layers' in cfg_dict:
                cfg.num_mlp_layers = cfg_dict['num_mlp_layers']
            if 'mlp_expand_dim' in cfg_dict:
                cfg.mlp_expand_dim = cfg_dict['mlp_expand_dim']
            
            # 加载反离散化配置
            if cfg.undiscrete_columns is None and 'undiscrete_columns' in cfg_dict:
                cfg.undiscrete_columns = cfg_dict['undiscrete_columns']
            if cfg.undiscrete_deltas is None and 'undiscrete_deltas' in cfg_dict:
                cfg.undiscrete_deltas = cfg_dict['undiscrete_deltas']
            
            # 加载 Gripper 列配置
            if cfg.gripper_columns is None and 'gripper_columns' in cfg_dict:
                cfg.gripper_columns = cfg_dict['gripper_columns']
        
        # 创建 ParaCAT 模型
        self.paracat = create_paracat_action_head(
            chunk_size=cfg.chunk_size,
            action_dim=cfg.action_dim,
            hidden_dim=cfg.vlm_output_dim,
            num_transformer_blocks=cfg.num_transformer_blocks,
            num_mlp_layers=cfg.num_mlp_layers,
            mlp_expand_dim=cfg.mlp_expand_dim,
            num_heads=cfg.num_heads,
        ).to(cfg.device)
        
        # 加载权重
        paracat_state = torch.load(cfg.paracat_checkpoint, map_location=cfg.device)
        self.paracat.load_state_dict(paracat_state)
        self.paracat.eval()
        print(f"✓ ParaCAT Action Head 加载成功!")
        print(f"  - chunk_size: {cfg.chunk_size}")
        print(f"  - action_dim: {cfg.action_dim}")
        
        # 打印反离散化配置
        if cfg.undiscrete_columns and cfg.undiscrete_deltas:
            print(f"  - 反离散化列: {cfg.undiscrete_columns}")
            print(f"  - 反离散化 delta: {cfg.undiscrete_deltas}")
        
        # 打印 Gripper 列配置
        if cfg.gripper_columns:
            print(f"  - Gripper 列: {cfg.gripper_columns}")
        
        # ====== 4. 加载 State Mapper (可选) ======
        self.state_mapper = None
        self.use_state_mapper = False
        
        # 检查是否有 state_mapper.pt 文件
        state_mapper_path = ckpt_dir / "state_mapper.pt"
        if cfg.state_mapper_checkpoint:
            state_mapper_path = Path(cfg.state_mapper_checkpoint)
        
        # 从 config.json 检查是否启用了 state_mapper
        if cfg_dict.get('enable_state_mapper', False) and state_mapper_path.exists():
            print(f"\n📦 加载 State Mapper: {state_mapper_path}")
            
            # 从 config.json 获取 state_mapper 配置
            state_dim = cfg_dict.get('state_dim', cfg.state_dim)
            minmax_columns = cfg_dict.get('state_norm_columns_minmax', cfg.state_norm_columns_minmax)
            axisangle_columns = cfg_dict.get('state_norm_columns_axisangle', cfg.state_norm_columns_axisangle)
            state_min = cfg_dict.get('state_min', None)
            state_max = cfg_dict.get('state_max', None)
            
            # 创建 state_mapper
            self.state_mapper = ObservationStateMapper(
                state_dim=state_dim,
                hidden_dim=cfg.vlm_output_dim,
                minmax_columns=minmax_columns,
                axisangle_columns=axisangle_columns,
                state_min=torch.tensor(state_min, dtype=torch.float32) if state_min else None,
                state_max=torch.tensor(state_max, dtype=torch.float32) if state_max else None,
            ).to(cfg.device)
            
            # 加载权重
            state_mapper_state = torch.load(state_mapper_path, map_location=cfg.device)
            self.state_mapper.load_state_dict(state_mapper_state)
            self.state_mapper.eval()
            self.use_state_mapper = True
            
            # 更新配置
            cfg.enable_state_mapper = True
            cfg.state_dim = state_dim
            cfg.state_norm_columns_minmax = minmax_columns
            cfg.state_norm_columns_axisangle = axisangle_columns
            
            print(f"✓ State Mapper 加载成功!")
            print(f"  - state_dim: {state_dim}")
            print(f"  - minmax_columns: {minmax_columns}")
            print(f"  - axisangle_columns: {axisangle_columns}")
        elif cfg_dict.get('enable_state_mapper', False):
            print(f"\n⚠️ config.json 启用了 state_mapper 但未找到权重文件: {state_mapper_path}")
        elif cfg.enable_state_mapper and state_mapper_path.exists():
            # 手动启用 state_mapper
            print(f"\n📦 加载 State Mapper: {state_mapper_path}")
            
            self.state_mapper = ObservationStateMapper(
                state_dim=cfg.state_dim,
                hidden_dim=cfg.vlm_output_dim,
                minmax_columns=cfg.state_norm_columns_minmax,
                axisangle_columns=cfg.state_norm_columns_axisangle,
            ).to(cfg.device)
            
            state_mapper_state = torch.load(state_mapper_path, map_location=cfg.device)
            self.state_mapper.load_state_dict(state_mapper_state)
            self.state_mapper.eval()
            self.use_state_mapper = True
            
            print(f"✓ State Mapper 加载成功!")
            print(f"  - state_dim: {cfg.state_dim}")
    
    def undiscretize_action(self, discrete_action: np.ndarray) -> np.ndarray:
        """
        将离散动作转换为连续动作
        
        Args:
            discrete_action: 离散动作值 {-1, 0, 1}
        
        Returns:
            continuous_action: 连续动作值
            - 对于 undiscrete_columns: {-delta, 0, delta}
            - 对于 gripper_columns: 保持原值 {-1, 0, 1}
        """
        continuous_action = discrete_action.copy().astype(np.float32)
        
        # 处理需要 delta 反离散化的列
        if self.cfg.undiscrete_columns and self.cfg.undiscrete_deltas:
            for i, col in enumerate(self.cfg.undiscrete_columns):
                if col < len(continuous_action) and i < len(self.cfg.undiscrete_deltas):
                    delta = self.cfg.undiscrete_deltas[i]
                    continuous_action[col] = undiscrete_constrain_delta(
                        np.array([discrete_action[col]]), delta
                    )[0]
        
        # Gripper 列保持原值 {-1, 0, 1}，不需要乘以 delta
        # (discrete_action 已经是 {-1, 0, 1}，直接使用)
        if self.cfg.gripper_columns:
            for col in self.cfg.gripper_columns:
                if col < len(continuous_action):
                    continuous_action[col] = float(discrete_action[col])
        
        return continuous_action
    
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
            observation_state: 观测状态 (预处理后的)，shape (state_dim,)
        
        Returns:
            预测的动作 delta，shape (chunk_size, action_dim)
        """
        # 获取 VLM 隐藏状态
        vlm_output = self.vlm_backbone.get_hidden_states(
            images=images,
            instruction=instruction,
        )
        vlm_hidden_states = vlm_output.hidden_states  # List[Tensor]
        
        with torch.no_grad():
            vlm_hidden_states_device = [v.to(self.device) for v in vlm_hidden_states]
            
            if self.use_pons:
                # 使用 Pons Adapter 聚合特征
                pons_output = self.pons(vlm_hidden_states_device)
            else:
                # 直接拼接 VLM hidden states
                pons_output = torch.cat(vlm_hidden_states_device, dim=1)
            
            # 如果启用 state_mapper，拼接 state embedding
            if self.use_state_mapper and observation_state is not None:
                state_tensor = torch.tensor(
                    observation_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)  # (1, state_dim)
                state_embedding = self.state_mapper(state_tensor)  # (1, 1, hidden_dim)
                pons_output = torch.cat([pons_output, state_embedding], dim=1)
                # pons_output: (1, pons_q_seq_len + 1, hidden_dim)
            
            # ParaCAT 推理 - 获取离散动作
            discrete_actions = self.paracat.predict_discrete_action(pons_output)
            # discrete_actions: (1, chunk_size, action_dim)，值为 {-1, 0, 1}
        
        # 转换为 numpy
        discrete_actions = discrete_actions[0].cpu().numpy()  # (chunk_size, action_dim)
        
        # 反离散化
        pred_deltas = np.zeros_like(discrete_actions, dtype=np.float32)
        for t in range(discrete_actions.shape[0]):
            pred_deltas[t] = self.undiscretize_action(discrete_actions[t])
        
        return pred_deltas
    
    def evaluate_episode(self) -> Dict[str, Any]:
        """
        评估单个 episode
        
        Returns:
            评估结果字典
        """
        cfg = self.cfg
        
        # 加载数据集信息
        print(f"\n📂 加载数据集: {cfg.dataset_path}")
        dataset_info = load_dataset_info(cfg.dataset_path)
        info = dataset_info["info"]
        tasks = dataset_info["tasks"]
        
        print(f"  - 总 episodes: {info['total_episodes']}")
        print(f"  - action_dim: {info['features']['action']['shape'][0]}")
        
        # 获取任务描述
        task_description = tasks.get(0, "Complete the task")
        if cfg.add_action_prompt:
            instruction = f"What action should the robot take to {task_description.lower()}?"
        else:
            instruction = task_description.lower() if cfg.lowercase_instruction else task_description
        
        print(f"  - 任务指令: {instruction}")
        
        # 加载 episode 数据
        print(f"\n📋 加载 Episode {cfg.episode_idx}...")
        df = load_episode_data(cfg.dataset_path, cfg.episode_idx, info)
        episode_length = len(df)
        print(f"  - Episode 长度: {episode_length} 帧")
        
        # 提取 GT actions
        gt_actions = np.array(df['action'].tolist(), dtype=np.float32)
        print(f"  - GT actions shape: {gt_actions.shape}")
        
        # 提取 observation.state (如果使用 state_mapper)
        observation_states = None
        if self.use_state_mapper and 'observation.state' in df.columns:
            raw_states = np.array(df['observation.state'].tolist(), dtype=np.float32)
            print(f"  - Raw observation.state shape: {raw_states.shape}")
            
            # 应用 state 预处理
            observation_states = []
            for i in range(len(raw_states)):
                processed_state = apply_state_preprocessing(
                    raw_states[i].copy(),
                    state_process_order=cfg.state_process_order or [],
                    hand_binary_columns=cfg.hand_binary_columns,
                    hand_binary_threshold=cfg.hand_binary_threshold,
                    state_euler_to_axisangle_columns=cfg.state_euler_to_axisangle_columns,
                )
                observation_states.append(processed_state)
            observation_states = np.array(observation_states, dtype=np.float32)
            print(f"  - Processed observation.state shape: {observation_states.shape}")
        elif self.use_state_mapper:
            print(f"  ⚠️ 启用了 state_mapper 但数据集中没有 observation.state 列")
        
        # 计算 GT 累计位置
        gt_position = np.cumsum(gt_actions, axis=0)
        
        # 初始化预测数组
        pred_deltas = np.zeros_like(gt_actions)
        pred_position = np.zeros_like(gt_actions)
        
        # 计算 chunk 信息
        chunks_size = info["chunks_size"]
        chunk_idx = cfg.episode_idx // chunks_size
        
        # 按 chunk 进行预测
        print(f"\n🔮 开始预测 (chunk_size={cfg.chunk_size})...")
        if self.use_state_mapper:
            print(f"  - 使用 State Mapper")
        num_chunks = (episode_length + cfg.chunk_size - 1) // cfg.chunk_size
        
        for i, chunk_start in enumerate(tqdm(range(0, episode_length, cfg.chunk_size), desc="预测进度")):
            chunk_end = min(chunk_start + cfg.chunk_size, episode_length)
            actual_chunk_len = chunk_end - chunk_start
            
            # 获取当前帧的图像
            images = get_images_from_video(
                cfg.dataset_path,
                cfg.episode_idx,
                chunk_start,  # 使用 chunk 起始帧的图像
                chunk_idx,
                image_keys=cfg.image_keys,
                flip=cfg.flip_images
            )
            
            # 获取当前帧的 observation.state
            current_state = None
            if observation_states is not None:
                current_state = observation_states[chunk_start]
            
            # 预测
            chunk_pred = self.predict_chunk(images, instruction, current_state)
            
            # 截取实际长度
            pred_deltas[chunk_start:chunk_end] = chunk_pred[:actual_chunk_len]
            
            # 计算累计位置 (从预测位置继续，误差会累积)
            if chunk_start == 0:
                chunk_start_pos = np.zeros(cfg.action_dim)
            else:
                chunk_start_pos = pred_position[chunk_start - 1]  # 从预测位置继续
            
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
            "episode_idx": cfg.episode_idx,
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
        cfg = self.cfg
        os.makedirs(cfg.output_dir, exist_ok=True)
        
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
            
            # 绘制预测的离散水平线段
            for chunk_start in range(0, episode_length, cfg.chunk_size):
                chunk_end = min(chunk_start + cfg.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pred = pred_deltas[chunk_start:chunk_end, dim]
                label = f'Pred Delta (chunk={cfg.chunk_size})' if chunk_start == 0 else None
                ax1.hlines(chunk_pred, chunk_x - 0.3, chunk_x + 0.3, 
                          colors='r', alpha=0.8, linewidth=2, label=label)
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
            for chunk_start in range(0, episode_length, cfg.chunk_size):
                chunk_end = min(chunk_start + cfg.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pos = pred_position[chunk_start:chunk_end, dim]
                label = 'Pred Position' if chunk_start == 0 else None
                ax2.hlines(chunk_pos, chunk_x - 0.3, chunk_x + 0.3,
                          colors='r', alpha=0.8, linewidth=2, label=label)
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
                cfg.output_dir, 
                f'episode_{episode_idx:06d}_dim_{dim}_chunk_{cfg.chunk_size}.png'
            )
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  [Dim {dim}] max_err={max_err:.6f}, mean_err={mean_err:.6f} -> {save_path}")
        
        # 绘制汇总图
        num_cols = 2
        num_rows = action_dim
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 5 * num_rows))
        
        for dim in range(action_dim):
            # 第一列: Delta 对比
            axes[dim, 0].plot(x, gt_actions[:, dim], 'b-', label='GT', alpha=0.7, linewidth=1)
            for chunk_start in range(0, episode_length, cfg.chunk_size):
                chunk_end = min(chunk_start + cfg.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pred = pred_deltas[chunk_start:chunk_end, dim]
                label = 'Pred' if chunk_start == 0 else None
                axes[dim, 0].hlines(chunk_pred, chunk_x - 0.3, chunk_x + 0.3,
                                   colors='r', alpha=0.8, linewidth=1.5, label=label)
                axes[dim, 0].axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.3)
            
            axes[dim, 0].set_ylabel(f'Dim {dim}')
            axes[dim, 0].legend(loc='upper right', fontsize=8)
            axes[dim, 0].grid(True, alpha=0.3)
            
            # 第二列: 位置对比
            axes[dim, 1].plot(x, gt_position[:, dim], 'b-', label='GT', alpha=0.7, linewidth=1.5)
            for chunk_start in range(0, episode_length, cfg.chunk_size):
                chunk_end = min(chunk_start + cfg.chunk_size, episode_length)
                chunk_x = x[chunk_start:chunk_end]
                chunk_pos = pred_position[chunk_start:chunk_end, dim]
                label = 'Pred' if chunk_start == 0 else None
                axes[dim, 1].hlines(chunk_pos, chunk_x - 0.3, chunk_x + 0.3,
                                   colors='r', alpha=0.8, linewidth=1.5, label=label)
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
        
        plt.suptitle(
            f'Episode {episode_idx} - ParaCAT Evaluation (chunk={cfg.chunk_size})\n'
            f'Instruction: {results["instruction"][:80]}...', 
            fontsize=12, y=1.01
        )
        plt.tight_layout()
        
        summary_path = os.path.join(
            cfg.output_dir, 
            f'episode_{episode_idx:06d}_summary_chunk_{cfg.chunk_size}.png'
        )
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n汇总图已保存: {summary_path}")
        
        # 保存统计信息
        stats = {
            "episode_idx": episode_idx,
            "episode_length": episode_length,
            "chunk_size": cfg.chunk_size,
            "action_dim": action_dim,
            "max_error_per_dim": results["max_error_per_dim"].tolist(),
            "mean_error_per_dim": results["mean_error_per_dim"].tolist(),
            "instruction": results["instruction"],
        }
        
        stats_path = os.path.join(
            cfg.output_dir, 
            f'episode_{episode_idx:06d}_stats_chunk_{cfg.chunk_size}.json'
        )
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"统计信息已保存: {stats_path}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='遥操作数据 ParaCAT 评估 - 累计误差绘图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 模型相关
    parser.add_argument("--paracat_checkpoint", type=str, required=True,
                        help="训练好的 paracat.pt 文件路径")
    parser.add_argument("--pons_checkpoint", type=str, default="",
                        help="预训练的 pons.pt 文件路径 (可选)")
    parser.add_argument("--state_mapper_checkpoint", type=str, default="",
                        help="预训练的 state_mapper.pt 文件路径 (可选，默认从 checkpoint 目录自动查找)")
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="VLM 模型路径")
    parser.add_argument("--vlm_type", type=str, default="qwen3_vl",
                        choices=["qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"],
                        help="VLM 类型")
    
    # VLM 相关参数
    parser.add_argument("--vlm_layers", type=str, default="14",
                        help="提取的 VLM 隐藏层 (如 '14' 或 '1,14,28')")
    parser.add_argument("--vlm_output_dim", type=int, default=2048,
                        help="VLM 输出维度")
    
    # ParaCAT 参数
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="动作块大小")
    parser.add_argument("--action_dim", type=int, default=14,
                        help="动作维度")
    parser.add_argument("--num_transformer_blocks", type=int, default=2,
                        help="Transformer 块数量")
    parser.add_argument("--num_mlp_layers", type=int, default=2,
                        help="MLP 层数量")
    parser.add_argument("--mlp_expand_dim", type=int, default=1024,
                        help="MLP 扩展维度")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="注意力头数量")
    
    # Pons 参数
    parser.add_argument("--pons_q_seq_len", type=int, default=64,
                        help="Pons query 序列长度")
    parser.add_argument("--pons_num_blocks", type=int, default=2,
                        help="Pons 块数量")
    
    # 数据集参数
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="数据集路径")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="要评估的 episode 索引")
    parser.add_argument("--image_keys", type=str, default="main",
                        help="图像视角键名，逗号分隔")
    
    # 离散化参数
    parser.add_argument("--undiscrete_columns", type=int, nargs="+", default=None,
                        help="需要反离散化的列索引")
    parser.add_argument("--undiscrete_deltas", type=float, nargs="+", default=None,
                        help="对应列的 delta 值")
    parser.add_argument("--gripper_columns", type=int, nargs="+", default=None,
                        help="Gripper 列索引 (原始值已是 {-1, 0, 1})")
    
    # State 预处理参数
    parser.add_argument("--state_process_order", type=str, nargs="+", default=None,
                        help="State 预处理顺序 (如 'hand_binary euler_to_axisangle')")
    parser.add_argument("--hand_binary_columns", type=int, nargs="+", default=None,
                        help="手部二值化列索引范围")
    parser.add_argument("--hand_binary_threshold", type=float, default=442.0,
                        help="手部二值化阈值")
    parser.add_argument("--state_euler_to_axisangle_columns", type=int, nargs="+", default=None,
                        help="欧拉角转轴角列索引")
    parser.add_argument("--state_dim", type=int, default=14,
                        help="处理后的 State 维度")
    
    # State Mapper 参数
    parser.add_argument("--enable_state_mapper", action="store_true", default=False,
                        help="启用 State Mapper (会从 config.json 自动检测)")
    parser.add_argument("--state_norm_columns_minmax", type=int, nargs="+", default=None,
                        help="State Mapper 使用 minmax 归一化的列索引")
    parser.add_argument("--state_norm_columns_axisangle", type=int, nargs="+", default=None,
                        help="State Mapper 使用 axisangle 归一化的列索引")
    
    # Prompt 配置
    parser.add_argument("--content_order", type=str, default="images_first",
                        choices=["images_first", "text_first", "interleaved", "single_image"],
                        help="内容顺序")
    parser.add_argument("--lowercase_instruction", action="store_true", default=True,
                        help="将指令转为小写")
    parser.add_argument("--no_lowercase_instruction", action="store_false", 
                        dest="lowercase_instruction", help="不转换指令为小写")
    parser.add_argument("--add_action_prompt", action="store_true", default=True,
                        help="添加 action prompt 前缀")
    parser.add_argument("--no_action_prompt", action="store_false", 
                        dest="add_action_prompt", help="不添加 action prompt 前缀")
    
    # 系统配置
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--flip_images", action="store_true", default=False,
                        help="翻转图像")
    parser.add_argument("--output_dir", type=str, default="./eval_plots",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 解析参数
    vlm_layers = [int(x.strip()) for x in args.vlm_layers.split(",")]
    image_keys = [k.strip() for k in args.image_keys.split(",")]
    
    # 创建配置
    cfg = EvalConfig(
        paracat_checkpoint=args.paracat_checkpoint,
        pons_checkpoint=args.pons_checkpoint,
        state_mapper_checkpoint=args.state_mapper_checkpoint,
        vlm_model_path=args.vlm_model_path,
        vlm_type=args.vlm_type,
        vlm_layers=vlm_layers,
        vlm_output_dim=args.vlm_output_dim,
        # ParaCAT 参数
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        num_transformer_blocks=args.num_transformer_blocks,
        num_mlp_layers=args.num_mlp_layers,
        mlp_expand_dim=args.mlp_expand_dim,
        num_heads=args.num_heads,
        # Pons 参数
        pons_q_seq_len=args.pons_q_seq_len,
        pons_num_blocks=args.pons_num_blocks,
        # 数据集参数
        dataset_path=args.dataset_path,
        episode_idx=args.episode_idx,
        image_keys=image_keys,
        # 离散化参数
        undiscrete_columns=args.undiscrete_columns,
        undiscrete_deltas=args.undiscrete_deltas,
        gripper_columns=args.gripper_columns,
        # State 预处理参数
        state_process_order=args.state_process_order,
        hand_binary_columns=args.hand_binary_columns,
        hand_binary_threshold=args.hand_binary_threshold,
        state_euler_to_axisangle_columns=args.state_euler_to_axisangle_columns,
        state_dim=args.state_dim,
        # State Mapper 参数
        enable_state_mapper=args.enable_state_mapper,
        state_norm_columns_minmax=args.state_norm_columns_minmax,
        state_norm_columns_axisangle=args.state_norm_columns_axisangle,
        # Prompt 配置
        content_order=args.content_order,
        lowercase_instruction=args.lowercase_instruction,
        add_action_prompt=args.add_action_prompt,
        # 系统配置
        device=args.device,
        flip_images=args.flip_images,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )
    
    # 打印配置
    print("\n" + "=" * 70)
    print("遥操作数据 ParaCAT 评估 - 累计误差绘图")
    print("=" * 70)
    print(f"数据集: {cfg.dataset_path}")
    print(f"Episode: {cfg.episode_idx}")
    print(f"Chunk Size: {cfg.chunk_size}")
    print(f"ParaCAT: {cfg.paracat_checkpoint}")
    print(f"Pons: {cfg.pons_checkpoint or '未使用'}")
    print(f"VLM: {cfg.vlm_model_path}")
    print(f"输出目录: {cfg.output_dir}")
    if cfg.state_process_order:
        print(f"State 预处理: {cfg.state_process_order}")
    if cfg.gripper_columns:
        print(f"Gripper 列: {cfg.gripper_columns}")
    print("=" * 70)
    
    # 创建评估器
    evaluator = TeleEvaluator(cfg)
    
    # 评估
    results = evaluator.evaluate_episode()
    
    # 绘图
    evaluator.plot_results(results)
    
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
