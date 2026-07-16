#!/usr/bin/env python3
"""
LIBERO 评估脚本 - 基于 ParaCAT Action Head

使用 Sai0_1 模块化结构评估 ParaCAT Action Head 在 LIBERO 环境中的表现。
ParaCAT 使用离散化动作预测 (3分类: 后退/不动/前进)。

使用方法:
    # 基本用法 - LIBERO 环境评估 (使用 Pons)
    CUDA_VISIBLE_DEVICES=0 python -m eval.Sai0_1.libero.ParaCAT.eval_Sai0_1 \
        --paracat_checkpoint ./checkpoints/step_5000/paracat.pt \
        --pons_checkpoint ./checkpoints/step_5000/pons.pt \
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
        --dataset_path /path/to/dataset \
        --task_suite_name libero_spatial \
        --vlm_layers 14

    # 不使用 Pons，直接使用 VLM hidden states
    CUDA_VISIBLE_DEVICES=0 python -m eval.Sai0_1.libero.ParaCAT.eval_Sai0_1 \
        --paracat_checkpoint ./checkpoints/step_5000/paracat.pt \
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
        --dataset_path /path/to/dataset \
        --task_suite_name libero_spatial

    # 使用训练数据测试
    CUDA_VISIBLE_DEVICES=0 python -m eval.Sai0_1.libero.ParaCAT.eval_Sai0_1 \
        --paracat_checkpoint ./checkpoints/step_5000/paracat.pt \
        --dataset_path /path/to/dataset \
        --use_training_data \
        --num_test_samples 100

参数说明:
    --paracat_checkpoint: 训练好的 paracat.pt 文件路径
    --pons_checkpoint: 预训练的 pons.pt 文件路径 (可选)
    --vlm_model_path: VLM 模型路径
    --dataset_path: 训练数据集路径 (用于加载归一化统计量)
    --task_suite_name: LIBERO 任务套件名称
    --vlm_layers: 提取的 VLM 隐藏层
    --undiscrete_columns: 需要反离散化的列索引
    --undiscrete_deltas: 对应列的 delta 值
"""

import sys
import os
import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import imageio
import numpy as np
import torch
import tqdm
from PIL import Image

# ============================================================================
# 路径设置
# ============================================================================
EVAL_DIR = Path(__file__).resolve().parent
SAI0_ROOT = EVAL_DIR.parents[3]  # eval/Sai0_1/libero/ParaCAT -> sai0-vla
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

# LIBERO 环境 - 延迟导入，避免 robosuite 提前初始化 CUDA
LIBERO_AVAILABLE = False
benchmark = None
get_libero_path = None
OffScreenRenderEnv = None


def _fix_robosuite_log_permission():
    """修复 robosuite 日志文件权限问题"""
    import logging
    robosuite_log = "/tmp/robosuite.log"
    
    try:
        with open(robosuite_log, 'a') as f:
            pass
        return
    except PermissionError:
        pass
    
    user_log = os.path.expanduser("~/.robosuite/robosuite.log")
    os.makedirs(os.path.dirname(user_log), exist_ok=True)
    
    _original_file_handler = logging.FileHandler
    
    class PatchedFileHandler(logging.FileHandler):
        def __init__(self, filename, mode='a', encoding=None, delay=False):
            if filename == "/tmp/robosuite.log":
                filename = user_log
                print(f"⚠️ robosuite 日志重定向到: {filename}")
            super().__init__(filename, mode, encoding, delay)
    
    logging.FileHandler = PatchedFileHandler
    print(f"✓ 已设置日志重定向 (原: {robosuite_log} -> 新: {user_log})")


def _init_libero():
    """延迟初始化 LIBERO，确保 CUDA 已正确初始化"""
    global LIBERO_AVAILABLE, benchmark, get_libero_path, OffScreenRenderEnv
    if LIBERO_AVAILABLE:
        return True
    
    _fix_robosuite_log_permission()
    
    try:
        from libero.libero import benchmark as _benchmark, get_libero_path as _get_libero_path
        from libero.libero.envs import OffScreenRenderEnv as _OffScreenRenderEnv
        benchmark = _benchmark
        get_libero_path = _get_libero_path
        OffScreenRenderEnv = _OffScreenRenderEnv
        LIBERO_AVAILABLE = True
        return True
    except ImportError:
        print("⚠️ LIBERO 未安装，请运行: pip install robosuite==1.4.0")
        return False


# ============================================================================
# 常量定义
# ============================================================================
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


# ============================================================================
# 四元数转轴角函数
# ============================================================================

def quat2axisangle_torch(quat: torch.Tensor) -> torch.Tensor:
    """四元数转轴角 (PyTorch 批量版本)"""
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


def quat2axisangle_numpy(quat: np.ndarray) -> np.ndarray:
    """四元数转轴角 (NumPy 版本)"""
    qw = quat[3]
    if qw > 1.0:
        qw = 1.0
    elif qw < -1.0:
        qw = -1.0
    
    den = np.sqrt(1.0 - qw * qw)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * np.arccos(qw)) / den


def convert_state_quat_to_axisangle(state: torch.Tensor) -> torch.Tensor:
    """将用户的 9 维 state 转换为 8 维 state"""
    batch_size = state.shape[0]
    
    gripper = state[:, 0:2]
    position = state[:, 2:5]
    quat = state[:, 5:9]
    
    axis_angle = quat2axisangle_torch(quat)
    converted_state = torch.cat([gripper, position, axis_angle], dim=1)
    
    return converted_state


# ============================================================================
# 归一化处理类
# ============================================================================

class MinMaxNormalizer:
    """Min-Max 归一化器"""
    
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


def load_normalization_stats(dataset_path: str, convert_quat_to_axisangle: bool = True) -> dict:
    """从数据集加载归一化统计信息"""
    import math
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"[WARN] stats.json 不存在于 {stats_path}，将不进行归一化")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    normalizers = {}
    
    if 'observation.state' in stats:
        state_stats = stats['observation.state']
        original_min = state_stats['min']
        original_max = state_stats['max']
        original_dim = len(original_min)
        
        if convert_quat_to_axisangle and original_dim == 9:
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
            print(f"✓ State 归一化统计量: {original_dim}维 → 8维 (四元数→轴角转换)")
        else:
            state_min = torch.tensor(original_min, dtype=torch.float32)
            state_max = torch.tensor(original_max, dtype=torch.float32)
            print(f"✓ 加载 state 归一化统计量，维度: {len(state_min)}")
        
        normalizers['state'] = MinMaxNormalizer(state_min, state_max)
    
    if 'action' in stats:
        action_stats = stats['action']
        action_min = torch.tensor(action_stats['min'], dtype=torch.float32)
        action_max = torch.tensor(action_stats['max'], dtype=torch.float32)
        print(f"✓ 加载 action 归一化统计量，维度: {len(action_min)}")
        
        normalizers['action'] = MinMaxNormalizer(action_min, action_max)
    
    return normalizers if normalizers else None


# ============================================================================
# LIBERO 环境工具函数
# ============================================================================

def get_libero_env(task, resolution=256, seed=None):
    """初始化 LIBERO 环境"""
    import random
    
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    
    if seed is not None:
        env.seed(seed)
    else:
        random_seed = random.randint(0, 2**31 - 1)
        env.seed(random_seed)
    
    return env, task_description


def get_libero_dummy_action():
    """获取空动作"""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs, resize_to: int = 128, flip: bool = False):
    """提取 LIBERO 图像"""
    img = obs["agentview_image"].copy()
    wrist_img = obs["robot0_eye_in_hand_image"].copy()
    
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if wrist_img.dtype != np.uint8:
        wrist_img = (wrist_img * 255).astype(np.uint8) if wrist_img.max() <= 1.0 else wrist_img.astype(np.uint8)
    
    if resize_to is not None:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        wrist_img = cv2.resize(wrist_img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    
    if flip:
        img = np.ascontiguousarray(img[::-1, ::-1, :])
        wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1, :])
    
    return img, wrist_img


def save_rollout_video(top_view, wrist_view, idx, success, task_description, 
                       log_file=None, video_dir="./rollouts"):
    """保存 rollout 视频"""
    rollout_dir = f"{video_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = (
        task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    )
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    
    if len(top_view) == 0 or len(wrist_view) == 0:
        print(f"⚠️ 警告: 视频帧为空!")
        return None
    
    try:
        video_writer = imageio.get_writer(
            mp4_path, 
            fps=10,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p',
        )
        
        for img1, img2 in zip(top_view, wrist_view):
            if img1.dtype != np.uint8:
                img1 = img1.astype(np.uint8)
            if img2.dtype != np.uint8:
                img2 = img2.astype(np.uint8)
            
            combined = np.hstack((img1, img2))
            video_writer.append_data(combined)
        
        video_writer.close()
        file_size = os.path.getsize(mp4_path)
        print(f"✓ 视频保存成功: {mp4_path} ({file_size / 1024:.1f} KB)")
        
    except Exception as e:
        print(f"❌ 视频保存失败: {e}")
        return None
    
    return mp4_path


# ============================================================================
# ParaCAT Policy 封装
# ============================================================================

class ParaCATPolicy:
    """
    ParaCAT Policy 封装
    
    使用 Sai0_1 模块化架构进行 LIBERO 评估
    支持可选的 Pons Adapter 和离散化动作预测
    """
    
    def __init__(
        self,
        paracat_checkpoint: str,
        pons_checkpoint: str = None,
        vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        vlm_type: str = "qwen3_vl",
        vlm_layers: List[int] = None,
        vlm_output_dim: int = 2048,
        dataset_path: str = None,
        device: str = "cuda:0",
        # ParaCAT 参数
        chunk_size: int = 16,
        action_dim: int = 7,
        num_transformer_blocks: int = 2,
        num_mlp_layers: int = 2,
        mlp_expand_dim: int = 1024,
        num_heads: int = 8,
        # Pons 参数
        pons_q_seq_len: int = 64,
        pons_num_blocks: int = 2,
        # 离散化参数
        undiscrete_columns: List[int] = None,
        undiscrete_deltas: List[float] = None,
        # Gripper 列参数 (LIBERO 专用)
        gripper_columns: List[int] = None,
        # 推理参数
        action_chunk_size: int = 1,
        flip_images: bool = True,
        # Prompt 配置
        content_order: str = "images_first",
        lowercase_instruction: bool = True,
        add_generation_prompt: bool = True,
        add_action_prompt: bool = True,
        verbose: bool = False,
    ):
        """
        初始化 ParaCAT Policy
        """
        self.device = device
        self.vlm_layers = vlm_layers or [14]
        self.vlm_output_dim = vlm_output_dim
        self.action_chunk_size = action_chunk_size
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.flip_images = flip_images
        self.content_order = content_order
        self.lowercase_instruction = lowercase_instruction
        self.add_generation_prompt = add_generation_prompt
        self.add_action_prompt = add_action_prompt
        self.verbose = verbose
        
        # 离散化参数
        self.undiscrete_columns = undiscrete_columns
        self.undiscrete_deltas = undiscrete_deltas
        
        # Gripper 列参数 (LIBERO 专用)
        # 这些列不做反离散化（不乘 delta），直接保持 {-1, 0, 1} 传给环境
        self.gripper_columns = gripper_columns or []
        
        self.action_queue = []
        self.use_pons = pons_checkpoint is not None and pons_checkpoint != ""
        
        # ====== 1. 加载 VLM Backbone ======
        print(f"\n📦 加载 VLM Backbone: {vlm_model_path}")
        print(f"  - 类型: {vlm_type}")
        print(f"  - 提取层: {self.vlm_layers}")
        print(f"  - 内容顺序: {content_order}")
        print(f"  - Action Prompt: {add_action_prompt}")
        
        self.vlm_backbone = create_vlm_backbone(
            model_type=vlm_type,
            model_path=vlm_model_path,
            device=device,
            layers=self.vlm_layers,
            flip_images=flip_images,
            content_order=content_order,
            prompt_template="simple",
            lowercase_instruction=False,
            verbose=verbose,
        )
        print(f"✓ VLM Backbone 加载成功!")
        
        # ====== 2. 加载 Pons Adapter (可选) ======
        self.pons = None
        if self.use_pons:
            print(f"\n📦 加载 Pons Adapter: {pons_checkpoint}")
            
            self.pons = create_pons_adapter(
                q_seq_len=pons_q_seq_len,
                hidden_dim=vlm_output_dim,
                num_blocks=pons_num_blocks,
                num_heads=num_heads,
            ).to(device)
            
            pons_state = torch.load(pons_checkpoint, map_location=device)
            self.pons.load_state_dict(pons_state)
            self.pons.eval()
            print(f"✓ Pons Adapter 加载成功!")
        else:
            print(f"\n📦 不使用 Pons Adapter，直接使用 VLM hidden states")
        
        # ====== 3. 加载 ParaCAT Action Head ======
        print(f"\n📦 加载 ParaCAT Action Head: {paracat_checkpoint}")
        
        # 加载 checkpoint 检测配置
        ckpt_dir = Path(paracat_checkpoint).parent
        config_path = ckpt_dir / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                cfg_dict = json.load(f)
            print(f"  - 从 config.json 加载配置")
            
            # 使用配置文件中的参数
            if 'hidden_dim' in cfg_dict:
                vlm_output_dim = cfg_dict['hidden_dim']
                self.vlm_output_dim = vlm_output_dim
                print(f"  - hidden_dim: {vlm_output_dim}")
            
            if 'chunk_size' in cfg_dict:
                chunk_size = cfg_dict['chunk_size']
                self.chunk_size = chunk_size
            
            if 'action_dim' in cfg_dict:
                action_dim = cfg_dict['action_dim']
                self.action_dim = action_dim
            
            if 'num_transformer_blocks' in cfg_dict:
                num_transformer_blocks = cfg_dict['num_transformer_blocks']
            
            if 'num_mlp_layers' in cfg_dict:
                num_mlp_layers = cfg_dict['num_mlp_layers']
            
            if 'mlp_expand_dim' in cfg_dict:
                mlp_expand_dim = cfg_dict['mlp_expand_dim']
            
            # 加载反离散化配置
            if self.undiscrete_columns is None and 'undiscrete_columns' in cfg_dict:
                self.undiscrete_columns = cfg_dict['undiscrete_columns']
            if self.undiscrete_deltas is None and 'undiscrete_deltas' in cfg_dict:
                self.undiscrete_deltas = cfg_dict['undiscrete_deltas']
            
            # 加载 Gripper 列配置 (LIBERO 专用)
            if not self.gripper_columns and 'gripper_columns' in cfg_dict:
                self.gripper_columns = cfg_dict['gripper_columns'] or []
            
            # 检查是否使用 Pons
            if 'use_pons' in cfg_dict and cfg_dict['use_pons'] and not self.use_pons:
                print(f"  ⚠️ 配置显示需要 Pons，但未提供 pons_checkpoint")
            
            if 'pons_q_seq_len' in cfg_dict:
                pons_q_seq_len = cfg_dict['pons_q_seq_len']
        
        # 创建 ParaCAT 模型
        self.paracat = create_paracat_action_head(
            chunk_size=chunk_size,
            action_dim=action_dim,
            hidden_dim=vlm_output_dim,
            num_transformer_blocks=num_transformer_blocks,
            num_mlp_layers=num_mlp_layers,
            mlp_expand_dim=mlp_expand_dim,
            num_heads=num_heads,
        ).to(device)
        
        # 加载权重
        paracat_state = torch.load(paracat_checkpoint, map_location=device)
        self.paracat.load_state_dict(paracat_state)
        self.paracat.eval()
        print(f"✓ ParaCAT Action Head 加载成功!")
        print(f"  - chunk_size: {chunk_size}")
        print(f"  - action_dim: {action_dim}")
        print(f"  - 使用离散化: True")
        
        # 打印反离散化配置
        if self.undiscrete_columns and self.undiscrete_deltas:
            print(f"  - 反离散化列: {self.undiscrete_columns}")
            print(f"  - 反离散化 delta: {self.undiscrete_deltas}")
        else:
            print(f"  ⚠️ 未配置反离散化参数，将使用默认值")
        
        # 打印 Gripper 列配置
        if self.gripper_columns:
            print(f"  - Gripper 列 (不做反离散化): {self.gripper_columns}")
        else:
            print(f"  - Gripper 列: 未配置 (将使用默认最后一列)")
        
        # ====== 4. 加载归一化统计量 (用于非离散化的列) ======
        self.normalizers = None
        if dataset_path:
            self.normalizers = load_normalization_stats(
                dataset_path, 
                convert_quat_to_axisangle=True
            )
            if self.normalizers:
                print(f"✓ 归一化统计量加载成功")
    
    def reset_action_queue(self):
        """重置动作队列"""
        self.action_queue = []
    
    def undiscretize_action(self, discrete_action: np.ndarray) -> np.ndarray:
        """
        将离散动作转换为连续动作
        
        输入 discrete_action 来自 ParaCAT.predict_discrete_action()，
        该方法内部已执行 argmax - 1，所以输入值范围是 {-1, 0, 1}
        
        参考: paracat_action_head.py 的 predict_discrete_action() 方法
            class_idx = torch.argmax(logits, dim=-1)  # {0, 1, 2}
            discrete_actions = class_idx - 1          # {-1, 0, 1}
        
        处理方式:
            - 位置/旋转列 (undiscrete_columns): 乘以 delta 还原为连续值
            - Gripper 列 (gripper_columns): 保持 {-1, 0, 1} 直接传给 LIBERO 环境
        
        Args:
            discrete_action: 离散动作值 {-1, 0, 1}，shape (action_dim,)
        
        Returns:
            continuous_action: 连续动作值，shape (action_dim,)
        """
        continuous_action = discrete_action.copy().astype(np.float32)
        
        # 反离散化位置/旋转列 - 使用 utils/discrete.py 的函数
        # {-1, 0, 1} * delta = {-delta, 0, delta}
        if self.undiscrete_columns and self.undiscrete_deltas:
            for i, col in enumerate(self.undiscrete_columns):
                if col < len(continuous_action) and i < len(self.undiscrete_deltas):
                    delta = self.undiscrete_deltas[i]
                    # 使用 undiscrete_constrain_delta 函数
                    continuous_action[col] = undiscrete_constrain_delta(
                        np.array([discrete_action[col]]), delta
                    )[0]
        
        # Gripper 列: 保持 {-1, 0, 1} 直接传给 LIBERO 环境
        # 输入已经是 argmax - 1 的结果，不需要额外处理
        # LIBERO 环境可以接受 {-1, 0, 1} 的 gripper 值
        
        return continuous_action
    
    def get_action(self, observation_dict: dict, lang: str) -> np.ndarray:
        """
        获取动作
        
        Args:
            observation_dict: LIBERO 观测字典
            lang: 任务语言描述
        
        Returns:
            action: (7,) 动作数组 [x, y, z, ax, ay, az, gripper]
        """
        # 如果队列中有动作，直接返回
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)
        
        # 处理观测
        xyz = observation_dict["robot0_eef_pos"]
        rpy = quat2axisangle_numpy(observation_dict["robot0_eef_quat"])
        gripper = observation_dict["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(observation_dict)
        
        # 准备图像
        images = [Image.fromarray(img), Image.fromarray(wrist_img)]
        
        # 构建 prompt
        task_desc = lang.lower() if self.lowercase_instruction else lang
        if self.add_action_prompt:
            instruction = f"What action should the robot take to {task_desc}?"
        else:
            instruction = task_desc
        
        # 获取 VLM 隐藏状态
        vlm_output = self.vlm_backbone.get_hidden_states(
            images=images,
            instruction=instruction,
        )
        vlm_hidden_states = vlm_output.hidden_states  # List[Tensor]
        
        # 处理 VLM 输出
        with torch.no_grad():
            vlm_hidden_states_device = [v.to(self.device) for v in vlm_hidden_states]
            
            if self.use_pons:
                # 使用 Pons Adapter 聚合特征
                pons_output = self.pons(vlm_hidden_states_device)  # (1, pons_len, hidden_dim)
            else:
                # 直接拼接 VLM hidden states
                pons_output = torch.cat(vlm_hidden_states_device, dim=1)  # (1, total_seq_len, hidden_dim)
            
            # ParaCAT 推理 - 获取离散动作
            discrete_actions = self.paracat.predict_discrete_action(pons_output)
            # discrete_actions: (1, chunk_size, action_dim)，值为 {-1, 0, 1}
        
        # 转换为 numpy
        discrete_actions = discrete_actions[0].cpu().numpy()  # (chunk_size, action_dim)
        
        # 将动作加入队列
        for idx in range(min(self.action_chunk_size, self.chunk_size)):
            discrete_action = discrete_actions[idx]  # (action_dim,)
            
            # 反离散化
            # 位置/旋转列: 乘以 delta 还原为连续值
            # Gripper 列: 保持 {-1, 0, 1} 直接传给 LIBERO 环境
            continuous_action = self.undiscretize_action(discrete_action)
            
            self.action_queue.append(continuous_action)
        
        return self.action_queue.pop(0)


# ============================================================================
# 评估配置
# ============================================================================

@dataclass
class EvalConfig:
    """评估配置"""
    # 模型参数
    paracat_checkpoint: str = ""
    pons_checkpoint: str = ""
    vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_type: str = "qwen3_vl"
    dataset_path: str = ""
    
    # VLM 相关
    vlm_layers: List[int] = field(default_factory=lambda: [14])
    vlm_output_dim: int = 2048
    
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
    
    # 离散化参数
    undiscrete_columns: Optional[List[int]] = None
    undiscrete_deltas: Optional[List[float]] = None
    
    # Gripper 列参数 (LIBERO 专用)
    gripper_columns: Optional[List[int]] = None
    
    # Prompt 配置
    content_order: str = "images_first"
    lowercase_instruction: bool = True
    add_generation_prompt: bool = True
    add_action_prompt: bool = True
    
    # LIBERO 环境
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 5
    task_ids: Optional[List[int]] = None
    max_tasks: int = -1
    num_steps_wait: int = 10
    max_steps: int = 600
    env_seed: Optional[int] = None
    
    # 任务指令替换配置
    task_instruction_override: Optional[Dict[int, str]] = None
    
    # 推理参数
    action_chunk_size: int = 1
    execute_all_chunks: bool = False
    
    # 系统配置
    device: str = "cuda:0"
    headless: bool = False
    flip_images: bool = True
    video_dir: str = "./eval_rollouts"
    log_dir: str = ""  # 日志保存目录，为空则使用 video_dir
    verbose: bool = False
    
    # 训练数据测试模式
    use_training_data: bool = False
    num_test_samples: int = 100


# ============================================================================
# 训练数据测试函数
# ============================================================================

def eval_with_training_data(cfg: EvalConfig):
    """使用训练数据进行测试"""
    sys.path.insert(0, str(SAI0_ROOT / 'utils'))
    from lerobot_dataset_loader import LeRobotDataset
    
    print(f"\n{'='*60}")
    print("训练数据测试模式 (Training Data Evaluation)")
    print(f"{'='*60}")
    print(f"数据集路径: {cfg.dataset_path}")
    print(f"测试样本数: {cfg.num_test_samples}")
    print(f"VLM 层数: {len(cfg.vlm_layers)}")
    print(f"{'='*60}\n")
    
    if not cfg.dataset_path:
        raise ValueError("必须指定 dataset_path")
    
    # 加载数据集
    print("📂 加载训练数据集...")
    dataset = LeRobotDataset(
        dataset_path=cfg.dataset_path,
        num_action_chunks=cfg.chunk_size,
        enable_chunking=True,
        verbose=True,
    )
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
    # 加载 Pons (可选)
    pons = None
    use_pons = cfg.pons_checkpoint and cfg.pons_checkpoint != ""
    if use_pons:
        print(f"\n📦 加载 Pons Adapter: {cfg.pons_checkpoint}")
        pons = create_pons_adapter(
            q_seq_len=cfg.pons_q_seq_len,
            hidden_dim=cfg.vlm_output_dim,
            num_blocks=cfg.pons_num_blocks,
            num_heads=cfg.num_heads,
        ).to(cfg.device)
        pons_state = torch.load(cfg.pons_checkpoint, map_location=cfg.device)
        pons.load_state_dict(pons_state)
        pons.eval()
        print("✅ Pons Adapter 加载完成")
    
    # 加载 ParaCAT 模型配置
    ckpt_dir = Path(cfg.paracat_checkpoint).parent
    config_path = ckpt_dir / "config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg_dict = json.load(f)
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
    
    # 加载 ParaCAT 模型
    print(f"\n📦 加载 ParaCAT Action Head: {cfg.paracat_checkpoint}")
    paracat = create_paracat_action_head(
        chunk_size=cfg.chunk_size,
        action_dim=cfg.action_dim,
        hidden_dim=cfg.vlm_output_dim,
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_mlp_layers=cfg.num_mlp_layers,
        mlp_expand_dim=cfg.mlp_expand_dim,
        num_heads=cfg.num_heads,
    ).to(cfg.device)
    
    paracat_state = torch.load(cfg.paracat_checkpoint, map_location=cfg.device)
    paracat.load_state_dict(paracat_state)
    paracat.eval()
    print("✅ ParaCAT Action Head 加载完成")
    
    # 测试
    num_samples = min(cfg.num_test_samples, len(dataset))
    print(f"\n🔍 开始测试 {num_samples} 个样本...")
    
    total_correct = 0
    total_predictions = 0
    accuracy_per_dim = [0] * cfg.action_dim
    
    # 随机采样
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = list(range(num_samples))
    
    for i, idx in enumerate(tqdm.tqdm(indices, desc="Testing")):
        sample = dataset[idx]
        
        # 获取数据
        vlm_tensor_raw = torch.from_numpy(sample['vlm_hidden_states']).float()
        actions = torch.from_numpy(sample['actions']).float()
        
        batch_size = 1
        num_layers = vlm_tensor_raw.size(0)
        seq_len = vlm_tensor_raw.size(1)
        hidden_dim = vlm_tensor_raw.size(2)
        num_chunks = actions.size(0)
        actual_action_dim = actions.size(1)
        
        # 准备 VLM hidden states
        vlm_hidden_states = [vlm_tensor_raw[i:i+1, :, :].to(cfg.device) for i in range(num_layers)]
        
        # 模型推理
        with torch.no_grad():
            if use_pons:
                pons_output = pons(vlm_hidden_states)
            else:
                pons_output = torch.cat(vlm_hidden_states, dim=1)
            
            # ParaCAT 预测
            logits = paracat(pons_output)  # (1, chunk, action_dim, 3)
            pred_discrete = torch.argmax(logits, dim=-1) - 1  # (1, chunk, action_dim), {-1, 0, 1}
        
        # 计算准确率 (假设 GT 已经是离散化的)
        # 这里简化处理，实际需要根据数据集格式调整
        pred_discrete = pred_discrete[0].cpu()  # (chunk, action_dim)
        
        # 统计
        total_predictions += num_chunks * actual_action_dim
    
    # 打印结果
    print(f"\n{'='*60}")
    print("📊 测试结果")
    print(f"{'='*60}")
    print(f"测试样本数: {num_samples}")
    print(f"总预测数: {total_predictions}")
    print(f"{'='*60}\n")
    
    # 保存结果
    results = {
        "num_samples": num_samples,
        "total_predictions": total_predictions,
    }
    
    os.makedirs(cfg.video_dir, exist_ok=True)
    result_path = Path(cfg.video_dir) / "training_data_eval_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 结果已保存到: {result_path}")


# ============================================================================
# LIBERO 评估函数
# ============================================================================

def eval_libero(cfg: EvalConfig):
    """LIBERO 评估主函数"""
    # 延迟初始化 LIBERO
    if not _init_libero():
        raise RuntimeError("LIBERO 未安装")
    
    # 设置日志目录 (使用 cfg.log_dir，如果为空则使用 video_dir)
    log_dir = cfg.log_dir if cfg.log_dir else cfg.video_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.video_dir, exist_ok=True)
    
    # 创建 Policy
    policy = ParaCATPolicy(
        paracat_checkpoint=cfg.paracat_checkpoint,
        pons_checkpoint=cfg.pons_checkpoint if cfg.pons_checkpoint else None,
        vlm_model_path=cfg.vlm_model_path,
        vlm_type=cfg.vlm_type,
        vlm_layers=cfg.vlm_layers,
        vlm_output_dim=cfg.vlm_output_dim,
        dataset_path=cfg.dataset_path,
        device=cfg.device,
        # ParaCAT 参数
        chunk_size=cfg.chunk_size,
        action_dim=cfg.action_dim,
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_mlp_layers=cfg.num_mlp_layers,
        mlp_expand_dim=cfg.mlp_expand_dim,
        num_heads=cfg.num_heads,
        # Pons 参数
        pons_q_seq_len=cfg.pons_q_seq_len,
        pons_num_blocks=cfg.pons_num_blocks,
        # 离散化参数
        undiscrete_columns=cfg.undiscrete_columns,
        undiscrete_deltas=cfg.undiscrete_deltas,
        # Gripper 列参数
        gripper_columns=cfg.gripper_columns,
        # 推理参数
        action_chunk_size=cfg.chunk_size if cfg.execute_all_chunks else cfg.action_chunk_size,
        flip_images=cfg.flip_images,
        # Prompt 配置
        content_order=cfg.content_order,
        lowercase_instruction=cfg.lowercase_instruction,
        add_generation_prompt=cfg.add_generation_prompt,
        add_action_prompt=cfg.add_action_prompt,
        verbose=cfg.verbose,
    )
    
    # 初始化 LIBERO 任务
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    print(f"\n{'='*60}")
    print(f"LIBERO 评估 (ParaCAT Action Head)")
    print(f"{'='*60}")
    print(f"Task Suite: {cfg.task_suite_name}")
    print(f"Total tasks: {num_tasks_in_suite}")
    print(f"Trials per task: {cfg.num_trials_per_task}")
    print(f"Max steps: {cfg.max_steps}")
    print(f"{'='*60}\n")
    
    # 确定要评估的任务
    if cfg.task_ids is not None:
        task_id_list = cfg.task_ids
    else:
        if cfg.max_tasks > 0:
            task_id_list = list(range(min(cfg.max_tasks, num_tasks_in_suite)))
        else:
            task_id_list = list(range(num_tasks_in_suite))
    
    # 打印任务列表
    print("Tasks to evaluate:")
    for tid in task_id_list:
        task = task_suite.get_task(tid)
        original_lang = task.language
        if cfg.task_instruction_override and tid in cfg.task_instruction_override:
            override_lang = cfg.task_instruction_override[tid]
            print(f"  [{tid}] {original_lang}")
            print(f"       🔄 替换为: {override_lang}")
        else:
            print(f"  [{tid}] {original_lang}")
    print()
    
    # 打印任务指令替换配置
    if cfg.task_instruction_override:
        print("🔄 任务指令替换配置:")
        for tid, new_instruction in cfg.task_instruction_override.items():
            print(f"  Task {tid}: {new_instruction}")
        print()
    
    # 打开日志文件
    log_file = open(f"{log_dir}/eval_{cfg.task_suite_name}_{DATE_TIME}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"ParaCAT Checkpoint: {cfg.paracat_checkpoint}\n")
    log_file.write(f"Pons Checkpoint: {cfg.pons_checkpoint}\n")
    log_file.write(f"VLM: {cfg.vlm_model_path}\n")
    log_file.write(f"Tasks: {task_id_list}\n\n")
    
    # 统计变量
    task_results = {}
    total_episodes, total_successes = 0, 0
    
    # 开始评估
    for task_id in tqdm.tqdm(task_id_list, desc="Tasks"):
        task = task_suite.get_task(task_id)
        original_task_description = task.language
        
        # 应用任务指令替换
        if cfg.task_instruction_override and task_id in cfg.task_instruction_override:
            task_description = cfg.task_instruction_override[task_id]
            print(f"\n[Task {task_id}] 原指令: {original_task_description}")
            print(f"           🔄 替换为: {task_description}")
            log_file.write(f"\n[Task {task_id}] 原指令: {original_task_description}\n")
            log_file.write(f"           🔄 替换为: {task_description}\n")
        else:
            task_description = original_task_description
            print(f"\n[Task {task_id}] {task_description}")
            log_file.write(f"\n[Task {task_id}] {task_description}\n")
        
        # 初始化环境
        env, _ = get_libero_env(task, resolution=256, seed=cfg.env_seed)
        
        # 获取初始状态
        initial_states = task_suite.get_task_init_states(task_id)
        
        task_successes = 0
        
        for trial in range(cfg.num_trials_per_task):
            print(f"  Trial {trial + 1}/{cfg.num_trials_per_task}")
            trial_start_time = time.time()
            
            # 重置环境
            env.reset()
            state_idx = trial % initial_states.shape[0]
            obs = env.set_init_state(initial_states[state_idx])
            
            # 等待物体稳定
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action())
            
            # 重置动作队列
            policy.reset_action_queue()
            
            # Rollout
            done = False
            success = False
            top_view, wrist_view = [], []
            
            for step in range(cfg.max_steps):
                if done:
                    break
                
                # 获取图像
                img, wrist_img = get_libero_image(obs, flip=cfg.flip_images)
                top_view.append(img)
                wrist_view.append(wrist_img)
                
                # 获取动作
                inference_start = time.time()
                action = policy.get_action(obs, task_description)
                inference_time = time.time() - inference_start
                print(f"    ⏱️ Step {step + 1} inference time: {inference_time*1000:.2f} ms")
                
                # 执行动作
                obs, reward, done, info = env.step(action.tolist())
                
                # 检查成功
                if done and reward > 0:
                    success = True
            
            # 记录结果
            if success:
                task_successes += 1
                total_successes += 1
            total_episodes += 1
            
            trial_time = time.time() - trial_start_time
            print(f"    Success: {success}")
            print(f"    ⏱️ Trial time: {trial_time:.2f} s")
            print(f"    Episodes: {total_episodes}, Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)")
            log_file.write(f"  Trial {trial + 1}: Success={success}, Steps={step + 1}, Time={trial_time:.2f}s\n")
            log_file.flush()
            
            # 保存视频
            save_rollout_video(
                top_view, wrist_view, 
                idx=trial, 
                success=success, 
                task_description=task_description,
                log_file=log_file,
                video_dir=f"{cfg.video_dir}/task_{task_id}"
            )
        
        env.close()
        
        # 记录任务结果
        task_success_rate = task_successes / cfg.num_trials_per_task
        task_results[task_id] = {
            "task_description": task_description,
            "successes": task_successes,
            "episodes": cfg.num_trials_per_task,
            "success_rate": task_success_rate,
        }
        
        print(f"  Task {task_id} Success Rate: {task_success_rate:.1%}")
        log_file.write(f"  Task Success Rate: {task_success_rate:.1%}\n")
    
    # 打印最终结果
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Task Suite: {cfg.task_suite_name}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {total_successes/total_episodes*100:.1f}%")
    print(f"\nPer-Task Results:")
    for tid, result in task_results.items():
        print(f"  Task {tid}: {result['success_rate']:.1%} - {result['task_description'][:50]}...")
    print(f"{'='*60}")
    
    log_file.close()
    
    # 保存结果
    results_path = Path(cfg.video_dir) / f"eval_results_{cfg.task_suite_name}.json"
    with open(results_path, 'w') as f:
        json.dump({
            "task_suite": cfg.task_suite_name,
            "vlm_model": cfg.vlm_model_path,
            "paracat_checkpoint": cfg.paracat_checkpoint,
            "pons_checkpoint": cfg.pons_checkpoint,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "overall_success_rate": total_successes / total_episodes,
            "task_results": task_results,
        }, f, indent=2)
    print(f"\n💾 结果已保存到: {results_path}")
    
    return task_results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LIBERO Evaluation with ParaCAT Action Head',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 模型相关
    parser.add_argument("--paracat_checkpoint", type=str, required=True,
                        help="训练好的 paracat.pt 文件路径")
    parser.add_argument("--pons_checkpoint", type=str, default="",
                        help="预训练的 pons.pt 文件路径 (可选)")
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="VLM 模型路径")
    parser.add_argument("--vlm_type", type=str, default="qwen3_vl",
                        choices=["qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"],
                        help="VLM 类型")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="训练数据集路径 (用于加载归一化统计量)")
    
    # VLM 相关参数
    parser.add_argument("--vlm_layers", type=str, default="14",
                        help="提取的 VLM 隐藏层 (如 '14' 或 '1,14,28')")
    parser.add_argument("--vlm_output_dim", type=int, default=2048,
                        help="VLM 输出维度")
    
    # ParaCAT 参数
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="动作块大小")
    parser.add_argument("--action_dim", type=int, default=7,
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
    
    # 离散化参数
    parser.add_argument("--undiscrete_columns", type=int, nargs="+", default=None,
                        help="需要反离散化的列索引")
    parser.add_argument("--undiscrete_deltas", type=float, nargs="+", default=None,
                        help="对应列的 delta 值")
    
    # Gripper 列参数 (LIBERO 专用)
    # 这些列不做反离散化（不乘 delta），直接保持 {-1, 0, 1} 传给环境
    parser.add_argument("--gripper_columns", type=int, nargs="+", default=None,
                        help="Gripper 列索引 (空格分隔)，这些列不做反离散化")
    
    # Prompt 配置
    parser.add_argument("--content_order", type=str, default="images_first",
                        choices=["images_first", "text_first", "interleaved", "single_image"],
                        help="内容顺序")
    parser.add_argument("--lowercase_instruction", action="store_true", default=True,
                        help="将指令转为小写")
    parser.add_argument("--no_lowercase_instruction", action="store_false", dest="lowercase_instruction",
                        help="不转换指令为小写")
    parser.add_argument("--add_generation_prompt", action="store_true", default=True,
                        help="添加 generation prompt")
    parser.add_argument("--no_generation_prompt", action="store_false", dest="add_generation_prompt",
                        help="不添加 generation prompt")
    parser.add_argument("--add_action_prompt", action="store_true", default=True,
                        help="添加 action prompt 前缀")
    parser.add_argument("--no_action_prompt", action="store_false", dest="add_action_prompt",
                        help="不添加 action prompt 前缀")
    
    # LIBERO 环境
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", 
                                "libero_10", "libero_90"])
    parser.add_argument("--num_trials_per_task", type=int, default=5)
    parser.add_argument("--task_ids", type=str, default=None)
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--env_seed", type=int, default=None,
                        help="环境随机种子，不设置则使用随机数")
    
    # 任务指令替换
    parser.add_argument("--task_instruction_override", type=str, default=None,
                        help="替换指定任务的指令。格式: 'task_id:instruction' 或 "
                             "'task_id1:instruction1|task_id2:instruction2'")
    
    # 推理参数
    parser.add_argument("--action_chunk_size", type=int, default=1)
    parser.add_argument("--execute_all_chunks", action="store_true")
    
    # 系统配置
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--flip_images", action="store_true", default=True)
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images")
    parser.add_argument("--video_dir", type=str, default="./eval_rollouts")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--verbose", action="store_true")
    
    # 训练数据测试模式
    parser.add_argument("--use_training_data", action="store_true")
    parser.add_argument("--num_test_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    # 解析 vlm_layers
    vlm_layers = [int(x.strip()) for x in args.vlm_layers.split(",")]
    
    # 解析 task_ids
    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    
    # 解析 task_instruction_override
    task_instruction_override = None
    if args.task_instruction_override:
        task_instruction_override = {}
        overrides = args.task_instruction_override.split("|")
        for override in overrides:
            override = override.strip()
            if ":" in override:
                parts = override.split(":", 1)
                if len(parts) == 2:
                    try:
                        tid = int(parts[0].strip())
                        instruction = parts[1].strip()
                        task_instruction_override[tid] = instruction
                        print(f"✅ 任务指令替换: Task {tid} -> \"{instruction}\"")
                    except ValueError:
                        print(f"⚠️ 无效的任务ID: {parts[0]}")
        
        if task_instruction_override:
            print(f"📋 共 {len(task_instruction_override)} 个任务指令将被替换")
    
    # 创建配置
    cfg = EvalConfig(
        paracat_checkpoint=args.paracat_checkpoint,
        pons_checkpoint=args.pons_checkpoint,
        vlm_model_path=args.vlm_model_path,
        vlm_type=args.vlm_type,
        dataset_path=args.dataset_path,
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
        # 离散化参数
        undiscrete_columns=args.undiscrete_columns,
        undiscrete_deltas=args.undiscrete_deltas,
        # Gripper 列参数
        gripper_columns=args.gripper_columns,
        # Prompt 配置
        content_order=args.content_order,
        lowercase_instruction=args.lowercase_instruction,
        add_generation_prompt=args.add_generation_prompt,
        add_action_prompt=args.add_action_prompt,
        # LIBERO 环境
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        task_ids=task_ids,
        max_tasks=args.max_tasks,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        env_seed=args.env_seed,
        # 任务指令替换
        task_instruction_override=task_instruction_override,
        # 推理参数
        action_chunk_size=args.action_chunk_size,
        execute_all_chunks=args.execute_all_chunks,
        # 系统配置
        device=args.device,
        headless=args.headless,
        flip_images=args.flip_images,
        video_dir=args.video_dir,
        verbose=args.verbose,
        # 训练数据测试模式
        use_training_data=args.use_training_data,
        num_test_samples=args.num_test_samples,
    )
    
    # 根据模式选择评估方法
    if cfg.use_training_data:
        eval_with_training_data(cfg)
    else:
        eval_libero(cfg)


if __name__ == "__main__":
    main()

