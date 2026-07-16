#!/usr/bin/env python3
"""
LIBERO 评估脚本 - 基于自定义训练的 OFT Action Head (Qwen VLM 版本)

该脚本与 train_for_libero_qwen2b_multigpu.py 训练方式对齐，使用相同的:
- 数据处理流程 (四元数→轴角转换)
- 归一化方式 (min_max 到 [-1, 1])
- 模型配置 (VLM2OFTPipeline + L1RegressionActionHead)
- VLM backbone (Qwen/Qwen3-VL-2B-Instruct 或 Qwen/Qwen3-VL-4B-Instruct)

使用方法:
    # 基本使用 - LIBERO 环境评估 (注意: 需要在命令前设置 CUDA_VISIBLE_DEVICES)
    CUDA_VISIBLE_DEVICES=0 python eval_new.py \\
        --checkpoint_path ./experiments/checkpoints/step_5000/action_head.pt \\
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \\
        --data_path /path/to/dataset \\
        --task_suite_name libero_spatial \\
        --vlm_layers 14

    # 使用训练数据测试
    CUDA_VISIBLE_DEVICES=0 python eval_new.py \\
        --checkpoint_path ./experiments/checkpoints/step_5000/action_head.pt \\
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \\
        --data_path /path/to/dataset \\
        --use_training_data \\
        --num_test_samples 100

    # 多层 VLM (与训练配置一致)
    CUDA_VISIBLE_DEVICES=1 python eval_new.py \\
        --checkpoint_path ./experiments/checkpoints/step_5000/action_head.pt \\
        --vlm_model_path Qwen/Qwen3-VL-4B-Instruct \\
        --data_path /path/to/dataset \\
        --vlm_layers 16,17,18 \\
        --vlm_output_dim 2560 \\
        --task_suite_name libero_spatial

参数说明:
    --checkpoint_path: 训练好的 action_head.pt 文件路径
    --vlm_model_path: VLM 模型路径 (Qwen/Qwen3-VL-2B-Instruct 或 Qwen/Qwen3-VL-4B-Instruct)
    --data_path: 训练数据集路径 (用于加载归一化统计量)
    --task_suite_name: LIBERO 任务套件名称
    --vlm_layers: 提取的 VLM 隐藏层 (可以是单层如 "14" 或多层如 "16,17,18")
    --vlm_output_dim: VLM backbone 输出维度 (Qwen2B=1536, Qwen4B=2560)
    
注意: 使用 CUDA_VISIBLE_DEVICES=X 在命令前设置要使用的 GPU，--device 参数应该设为 cuda:0
"""

import sys
import os
import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
import tqdm
from PIL import Image

# ============================================================================
# 路径设置
# ============================================================================
EVAL_DIR = Path(__file__).resolve().parent
SAI0_ROOT = EVAL_DIR.parents[2]  # .../sai0-vla
if str(SAI0_ROOT) not in sys.path:
    sys.path.insert(0, str(SAI0_ROOT))

# 添加 OFT1_0 路径
OFT_DIR = SAI0_ROOT / "Action_Heads" / "OFT1_0"
if str(OFT_DIR) not in sys.path:
    sys.path.insert(0, str(OFT_DIR))

# 添加 utils 路径
UTILS_DIR = SAI0_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

# 导入 OFT 模型
from vlm2oft_pipeline import VLM2OFTPipeline, create_vlm2oft_pipeline
from constants import (
    ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM,
    LLM_OUTPUT_DIM_MLP_INPUT_DIM, NUM_VLM_HIDDEN_LAYERS
)

# LIBERO 环境 - 延迟导入，避免 robosuite 提前初始化 CUDA
LIBERO_AVAILABLE = False
benchmark = None
get_libero_path = None
OffScreenRenderEnv = None

def _init_libero():
    """延迟初始化 LIBERO，确保 CUDA 已正确初始化"""
    global LIBERO_AVAILABLE, benchmark, get_libero_path, OffScreenRenderEnv
    if LIBERO_AVAILABLE:
        return True
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

# Qwen VLM - 使用 transformers 加载
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ transformers 未正确安装，VLM 功能不可用: {e}")


# ============================================================================
# 常量定义
# ============================================================================
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


# ============================================================================
# 工具函数 - 与训练代码对齐
# ============================================================================

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    四元数转轴角 (NumPy 版本)
    
    与训练代码 quat2axisangle_torch 对应
    
    Args:
        quat: (4,) 四元数 [qx, qy, qz, qw]
    
    Returns:
        axis_angle: (3,) 轴角 [ax, ay, az]
    """
    qw = quat[3]
    if qw > 1.0:
        qw = 1.0
    elif qw < -1.0:
        qw = -1.0
    
    den = np.sqrt(1.0 - qw * qw)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * np.arccos(qw)) / den


def quat2axisangle_torch(quat: torch.Tensor) -> torch.Tensor:
    """
    四元数转轴角 (PyTorch 批量版本) - 与训练代码一致
    
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
    将用户的 9 维 state 转换为 8 维 state - 与训练代码一致
    
    用户 state 格式 (9维):
        [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
    
    转换后 state 格式 (8维):
        [gripper1, gripper2, x, y, z, ax, ay, az]
    """
    batch_size = state.shape[0]
    gripper = state[:, 0:2]
    position = state[:, 2:5]
    quat = state[:, 5:9]
    axis_angle = quat2axisangle_torch(quat)
    converted_state = torch.cat([gripper, position, axis_angle], dim=1)
    return converted_state


class MinMaxNormalizer:
    """
    Min-Max 归一化器 - 与训练代码完全一致
    
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
        """反归一化，从 [-1, 1] 恢复到原始范围"""
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        
        return (x + 1.0) / 2.0 * (max_vals - min_vals) + min_vals
    
    def denormalize_numpy(self, x: np.ndarray) -> np.ndarray:
        """反归一化 NumPy 版本"""
        min_vals = self.min_vals.numpy()
        max_vals = self.max_vals.numpy()
        
        return (x + 1.0) / 2.0 * (max_vals - min_vals) + min_vals


def load_normalization_stats(dataset_path: str, convert_quat_to_axisangle: bool = True) -> dict:
    """
    从数据集加载归一化统计信息 - 与训练代码完全一致
    """
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    
    if not stats_path.exists():
        print(f"[WARN] stats.json 不存在于 {stats_path}，将不进行归一化")
        return None
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    normalizers = {}
    
    # 加载 state 归一化统计量
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
        
        normalizers['state'] = MinMaxNormalizer(state_min, state_max)
    
    # 加载 action 归一化统计量
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

def get_libero_env(task, resolution=256):
    """初始化 LIBERO 环境"""
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
    env.seed(0)
    return env, task_description


def get_libero_dummy_action():
    """获取空动作"""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs, flip: bool = False, resize_to: int = 128):
    """提取 LIBERO 图像
    
    Args:
        obs: LIBERO 环境观测
        flip: 是否翻转图像 180 度 (默认 False)
        resize_to: resize 目标尺寸 (默认 128x128，与训练数据一致)
    """
    img = obs["agentview_image"].copy()
    wrist_img = obs["robot0_eye_in_hand_image"].copy()
    
    if flip:
        img = img[::-1, ::-1].copy()
        wrist_img = wrist_img[::-1, ::-1].copy()
    
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if wrist_img.dtype != np.uint8:
        wrist_img = (wrist_img * 255).astype(np.uint8) if wrist_img.max() <= 1.0 else wrist_img.astype(np.uint8)
    
    if resize_to is not None:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        wrist_img = cv2.resize(wrist_img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    
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
    
    print(f"保存视频: {len(top_view)} 帧")
    
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
# Qwen VLM Backbone 封装
# ============================================================================

class QwenVLMBackbone:
    """
    Qwen VLM Backbone 封装
    
    支持单层或多层隐藏状态提取，与训练代码对齐
    """
    
    def __init__(
        self, 
        model_path: str = "Qwen/Qwen3-VL-2B-Instruct", 
        device: str = "cuda:0", 
        layers: List[int] = [14],
        add_action_prompt: bool = True,
        verbose: bool = False
    ):
        """
        初始化 Qwen VLM
        
        Args:
            model_path: 模型路径
            device: 设备 (如 "cuda:0", "cuda:1")
            layers: 要提取的 transformer 层号列表 (如 [14] 或 [16, 17, 18])
            add_action_prompt: 是否添加 action prompt
            verbose: 是否打印详细信息
        """
        self.device = device
        self.layers = layers
        self.add_action_prompt = add_action_prompt
        self.verbose = verbose
        
        print(f"Loading Qwen VLM from: {model_path}")
        print(f"  - 提取层: {layers}")
        print(f"  - 设备: {device}")
        
        # 直接使用传入的 device (应该是 cuda:0，因为 CUDA_VISIBLE_DEVICES 在命令行设置)
        actual_device = device
        
        # 加载模型 - 使用指定设备而非 device_map="auto"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=actual_device,  # 指定单个设备，而非 "auto"
            trust_remote_code=True,
        )
        self.model.eval()
        
        # 更新 self.device 为实际使用的设备
        self.device = actual_device
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # 加载 tokenizer (用于 verbose 模式下解码 tokens)
        self.tokenizer = None
        if self.verbose:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print(f"✓ Tokenizer loaded for verbose mode")
            except Exception as e:
                print(f"⚠️ Could not load tokenizer: {e}")
                print("  Token decoding will be disabled.")
        
        print(f"✓ Qwen VLM 加载成功!")
        if self.verbose:
            print(f"  Verbose mode: ENABLED (will print input_ids and decoded tokens)")
    
    @torch.no_grad()
    def get_hidden_states(
        self,
        images: List[np.ndarray],
        text: str,
    ) -> List[torch.Tensor]:
        """
        提取 VLM 隐藏状态
        
        Args:
            images: 图像列表 [agent_view, wrist_view]
            text: 任务描述文本
        
        Returns:
            hidden_states_list: 每层的隐藏状态列表，每个形状 (1, seq_len, hidden_dim)
        """
        # 处理图像
        pil_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        
        # 构建文本内容
        if self.add_action_prompt:
            text_content = f"What action should the robot take to {text.lower()}?"
        else:
            text_content = text
        
        # 构建 messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[0]},
                    {"type": "image", "image": pil_images[1]},
                    {"type": "text", "text": text_content}
                ]
            }
        ]
        
        # 处理输入
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Verbose 模式下打印完整 token 信息
        if self.verbose:
            input_ids = inputs.get("input_ids", None)
            if input_ids is not None:
                print(f"\n{'='*80}")
                print(f"[Token Information]")
                print(f"  Input IDs shape: {input_ids.shape}")
                print(f"  Total tokens: {input_ids.shape[1]}")
                
                # 解码 tokens
                if self.tokenizer is not None:
                    for batch_idx in range(input_ids.shape[0]):
                        # 完整解码文本
                        decoded_text = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=False)
                        print(f"\n  [Batch {batch_idx}] Decoded tokens (full text):")
                        print(f"    {decoded_text[:500]}..." if len(decoded_text) > 500 else f"    {decoded_text}")
                        
                        # Token-by-token breakdown (ALL tokens)
                        print(f"\n  [Batch {batch_idx}] Token-by-token breakdown (ALL {input_ids.shape[1]} tokens):")
                        tokens = input_ids[batch_idx].tolist()
                        for i, token_id in enumerate(tokens):
                            token_str = self.tokenizer.decode([token_id])
                            # 格式化特殊字符显示
                            token_display = repr(token_str) if token_str in ['', ' ', '\n', '\t'] or not token_str.isprintable() else f"'{token_str}'"
                            print(f"    [{i:4d}] ID={token_id:6d}: {token_display}")
                else:
                    print(f"  Input IDs: {input_ids.tolist()}")
                print(f"{'='*80}\n")
        
        # 前向传播
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        
        # 提取指定层的隐藏状态
        hidden_states_list = []
        for layer in self.layers:
            hidden_state = outputs.hidden_states[layer]  # (batch, seq_len, hidden_dim)
            hidden_states_list.append(hidden_state.float())
        
        if self.verbose:
            print(f"[Qwen VLM] 提取了 {len(self.layers)} 层隐藏状态")
            for i, hs in enumerate(hidden_states_list):
                print(f"  Layer {self.layers[i]}: shape={hs.shape}")
        
        return hidden_states_list


# ============================================================================
# OFT Policy 封装
# ============================================================================

class OFTPolicy:
    """
    OFT Policy 封装
    
    与训练代码的数据处理流程完全对齐
    
    ⚠️ State 格式与训练代码一致:
    [gripper1, gripper2, x, y, z, ax, ay, az] (8维)
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        vlm_backbone: QwenVLMBackbone = None,
        normalizers: Optional[Dict] = None,
        device: str = "cuda:0",
        num_vlm_layers: int = 1,
        vlm_output_dim: int = 1536,
        num_transformer_blocks: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        action_head_hidden_dim: int = 4096,
        action_chunk_size: int = 1,
        flip_images: bool = False,
        headless: bool = False,
    ):
        """
        初始化 OFT Policy
        
        Args:
            checkpoint_path: action head checkpoint 路径
            vlm_backbone: VLM backbone
            normalizers: 归一化器字典
            device: 设备
            num_vlm_layers: VLM 隐藏层数量
            vlm_output_dim: VLM 输出维度
            num_transformer_blocks: Transformer 块数量
            num_attention_heads: 注意力头数量
            dropout: Dropout 比率
            action_head_hidden_dim: Action head 隐藏层维度
            action_chunk_size: 每次执行的动作数
            flip_images: 是否翻转图像
            headless: 是否无头模式
        """
        self.device = device
        self.vlm_backbone = vlm_backbone
        self.normalizers = normalizers
        self.action_chunk_size = action_chunk_size
        self.flip_images = flip_images
        self.headless = headless
        self.num_vlm_layers = num_vlm_layers
        self.vlm_output_dim = vlm_output_dim
        
        self.action_queue = []
        
        # 加载 OFT 模型
        print(f"Loading OFT Action Head from: {checkpoint_path}")
        
        # 先加载 checkpoint 来检测维度
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # 从 checkpoint 自动检测维度
        detected_dim = None
        if 'proprio_projector.layer_norm.weight' in state_dict:
            detected_dim = state_dict['proprio_projector.layer_norm.weight'].shape[0]
            print(f"  - 从 checkpoint 检测到 VLM output dim: {detected_dim}")
            if detected_dim != vlm_output_dim:
                print(f"  ⚠️ 警告: 参数指定的 vlm_output_dim={vlm_output_dim} 与 checkpoint 不匹配!")
                print(f"  ⚠️ 自动使用 checkpoint 的维度: {detected_dim}")
                vlm_output_dim = detected_dim
        
        self.model = create_vlm2oft_pipeline(
            num_transformer_blocks=num_transformer_blocks,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            action_head_hidden_dim=action_head_hidden_dim,
            num_vlm_layers=num_vlm_layers,
            vlm_output_dim=vlm_output_dim,
        ).to(device)
        
        # 加载权重
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"✓ OFT Action Head loaded!")
        print(f"  - VLM layers: {num_vlm_layers}")
        print(f"  - VLM output dim: {vlm_output_dim}")
        print(f"  - Action chunk size: {action_chunk_size}")
    
    def reset_action_queue(self):
        """重置动作队列"""
        self.action_queue = []
    
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
        rpy = quat2axisangle(observation_dict["robot0_eef_quat"])
        gripper = observation_dict["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(observation_dict, flip=self.flip_images)
        
        # 获取 VLM 隐藏状态
        if self.vlm_backbone is not None:
            images = [img, wrist_img]
            vlm_hidden_states = self.vlm_backbone.get_hidden_states(images, lang)
        else:
            raise ValueError("VLM backbone is required for inference")
        
        # 准备 state - 与训练代码格式一致
        # [gripper1, gripper2, x, y, z, ax, ay, az] (8维)
        state = np.array([gripper[0], gripper[1], xyz[0], xyz[1], xyz[2], 
                         rpy[0], rpy[1], rpy[2]], dtype=np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 8)
        
        # 归一化 state
        if self.normalizers is not None and 'state' in self.normalizers:
            state_tensor = self.normalizers['state'].normalize(state_tensor)
        
        # 模型推理
        with torch.no_grad():
            # VLM hidden states 已经是列表格式
            vlm_hidden_states_device = [v.to(self.device) for v in vlm_hidden_states]
            
            # 前向传播
            pred_actions = self.model(vlm_hidden_states_device, state_tensor)
            # pred_actions: (1, 1, NUM_ACTIONS_CHUNK * ACTION_DIM)
        
        # 解析预测的动作
        pred_actions = pred_actions[0, 0].cpu().numpy()  # (NUM_ACTIONS_CHUNK * ACTION_DIM,)
        pred_actions = pred_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)  # (16, 7)
        
        # 反归一化
        if self.normalizers is not None and 'action' in self.normalizers:
            pred_actions_tensor = torch.from_numpy(pred_actions)
            pred_actions = self.normalizers['action'].denormalize(pred_actions_tensor).numpy()
        
        # 将动作加入队列
        for idx in range(min(self.action_chunk_size, NUM_ACTIONS_CHUNK)):
            action = pred_actions[idx].astype(np.float32)
            # 二值化 gripper
            action[-1] = 1.0 if action[-1] > 0 else -1.0
            self.action_queue.append(action)
        
        return self.action_queue.pop(0)


# ============================================================================
# 评估配置
# ============================================================================

@dataclass
class EvalConfig:
    """评估配置"""
    # 模型参数
    checkpoint_path: str = ""
    vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    data_path: str = ""
    
    # VLM 相关
    vlm_layers: List[int] = None
    vlm_output_dim: int = 1536
    add_action_prompt: bool = True
    
    # OFT 模型参数
    num_transformer_blocks: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.1
    action_head_hidden_dim: int = 4096
    
    # LIBERO 环境
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 5
    task_ids: Optional[List[int]] = None
    max_tasks: int = -1
    num_steps_wait: int = 10
    max_steps: int = 600
    
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
    
    def __post_init__(self):
        if self.vlm_layers is None:
            self.vlm_layers = [14]


# ============================================================================
# 训练数据测试函数
# ============================================================================

def eval_with_training_data(cfg: EvalConfig):
    """
    使用训练数据进行测试
    
    计算模型在训练数据上的预测误差，用于验证：
    1. 数据处理流程是否正确
    2. 模型是否正确加载
    3. 模型是否过拟合/欠拟合
    """
    from lerobot_dataset_loader import LeRobotDataset, collate_fn as lerobot_default_collate_fn
    
    print(f"\n{'='*60}")
    print("训练数据测试模式 (Training Data Evaluation)")
    print(f"{'='*60}")
    print(f"数据集路径: {cfg.data_path}")
    print(f"测试样本数: {cfg.num_test_samples}")
    print(f"{'='*60}\n")
    
    if not cfg.data_path:
        raise ValueError("必须指定 data_path 才能使用训练数据测试模式")
    
    # 加载数据集
    print("📂 加载训练数据集...")
    dataset = LeRobotDataset(
        dataset_path=cfg.data_path,
        num_action_chunks=NUM_ACTIONS_CHUNK,
        enable_chunking=True,
        verbose=True,
    )
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
    # 加载归一化统计量
    normalizers = load_normalization_stats(cfg.data_path, convert_quat_to_axisangle=True)
    
    # 加载 OFT 模型
    print(f"\n📦 加载 OFT Action Head: {cfg.checkpoint_path}")
    
    model = create_vlm2oft_pipeline(
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_attention_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
        action_head_hidden_dim=cfg.action_head_hidden_dim,
        num_vlm_layers=len(cfg.vlm_layers),
        vlm_output_dim=cfg.vlm_output_dim,
    ).to(cfg.device)
    
    state_dict = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ OFT Action Head 加载完成")
    
    # 测试
    num_samples = min(cfg.num_test_samples, len(dataset))
    print(f"\n🔍 开始测试 {num_samples} 个样本...")
    
    total_action_mse = 0.0
    total_samples = 0
    action_errors = []
    
    # 随机采样
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = list(range(num_samples))
    
    for i, idx in enumerate(tqdm.tqdm(indices, desc="Testing")):
        sample = dataset[idx]
        
        # 获取 VLM hidden states
        vlm_tensor_raw = sample['vlm_hidden_states']  # (num_layers, seq_len, hidden_dim)
        observation_state = sample['observation_state']  # (state_dim,)
        actions = sample['actions']  # (num_chunks, action_dim)
        
        # 转换为 batch 格式
        vlm_tensor = vlm_tensor_raw.unsqueeze(0)  # (1, num_layers, seq_len, hidden_dim)
        observation_state = observation_state.unsqueeze(0)  # (1, state_dim)
        actions = actions.unsqueeze(0)  # (1, num_chunks, action_dim)
        
        batch_size = 1
        num_layers = vlm_tensor.size(1)
        num_chunks = actions.size(1)
        actual_action_dim = actions.size(2)
        
        # State 四元数转轴角 (如果需要)
        original_state_dim = observation_state.size(1)
        if original_state_dim == 9:
            observation_state = convert_state_quat_to_axisangle(observation_state)
        
        # 归一化
        if normalizers is not None:
            if 'state' in normalizers:
                observation_state = normalizers['state'].normalize(observation_state)
            if 'action' in normalizers:
                actions_flat = actions.reshape(batch_size * num_chunks, actual_action_dim)
                actions_normalized = normalizers['action'].normalize(actions_flat)
                actions = actions_normalized.reshape(batch_size, num_chunks, actual_action_dim)
        
        # 准备模型输入
        vlm_hidden_states = [vlm_tensor[:, i, :, :].to(cfg.device) for i in range(num_layers)]
        proprioception = observation_state.to(cfg.device)
        gt_actions = actions.view(batch_size, 1, num_chunks * actual_action_dim).to(cfg.device)
        
        # 模型推理
        with torch.no_grad():
            pred_actions = model(vlm_hidden_states, proprioception)
        
        # 计算 MSE
        mse = ((pred_actions - gt_actions) ** 2).mean().item()
        action_errors.append(mse)
        total_action_mse += mse
        total_samples += 1
        
        if i < 5 and cfg.verbose:
            print(f"\n样本 {idx}:")
            print(f"  GT actions[:7]: {gt_actions[0, 0, :7].cpu().numpy()}")
            print(f"  Pred actions[:7]: {pred_actions[0, 0, :7].cpu().numpy()}")
            print(f"  MSE: {mse:.6f}")
    
    # 统计结果
    avg_mse = total_action_mse / total_samples if total_samples > 0 else 0
    action_errors = np.array(action_errors)
    
    print(f"\n{'='*60}")
    print("📊 测试结果")
    print(f"{'='*60}")
    print(f"测试样本数: {total_samples}")
    print(f"平均 MSE: {avg_mse:.6f}")
    print(f"MSE 标准差: {np.std(action_errors):.6f}")
    print(f"MSE 最小值: {np.min(action_errors):.6f}")
    print(f"MSE 最大值: {np.max(action_errors):.6f}")
    print(f"MSE 中位数: {np.median(action_errors):.6f}")
    print(f"\n平均 RMSE: {np.sqrt(avg_mse):.6f}")
    print(f"{'='*60}\n")
    
    # 保存结果
    results = {
        "num_samples": total_samples,
        "avg_mse": float(avg_mse),
        "avg_rmse": float(np.sqrt(avg_mse)),
        "std_mse": float(np.std(action_errors)),
        "min_mse": float(np.min(action_errors)),
        "max_mse": float(np.max(action_errors)),
        "median_mse": float(np.median(action_errors)),
    }
    
    os.makedirs(cfg.video_dir, exist_ok=True)
    result_path = Path(cfg.video_dir) / "training_data_eval_results.json"
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 结果已保存到: {result_path}")
    
    print(f"\n✅ 训练数据测试完成!")


# ============================================================================
# LIBERO 评估函数
# ============================================================================

def eval_libero(cfg: EvalConfig):
    """
    LIBERO 评估主函数
    """
    # 延迟初始化 LIBERO
    if not _init_libero():
        raise RuntimeError("LIBERO 未安装，无法进行仿真评估。"
                          "请运行: pip install robosuite==1.4.0")
    
    # 设置日志目录 (使用 cfg.log_dir，如果为空则使用 video_dir)
    log_dir = cfg.log_dir if cfg.log_dir else cfg.video_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.video_dir, exist_ok=True)
    
    # 加载归一化统计量
    normalizers = None
    if cfg.data_path:
        normalizers = load_normalization_stats(cfg.data_path, convert_quat_to_axisangle=True)
    
    # 创建 VLM backbone
    print(f"\n📦 加载 Qwen VLM: {cfg.vlm_model_path}")
    vlm_backbone = QwenVLMBackbone(
        model_path=cfg.vlm_model_path,
        device=cfg.device,
        layers=cfg.vlm_layers,
        add_action_prompt=cfg.add_action_prompt,
        verbose=cfg.verbose,
    )
    
    # 创建 Policy
    policy = OFTPolicy(
        checkpoint_path=cfg.checkpoint_path,
        vlm_backbone=vlm_backbone,
        normalizers=normalizers,
        device=cfg.device,
        num_vlm_layers=len(cfg.vlm_layers),
        vlm_output_dim=cfg.vlm_output_dim,
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_attention_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
        action_head_hidden_dim=cfg.action_head_hidden_dim,
        action_chunk_size=NUM_ACTIONS_CHUNK if cfg.execute_all_chunks else cfg.action_chunk_size,
        flip_images=cfg.flip_images,
        headless=cfg.headless,
    )
    
    # 初始化 LIBERO 任务
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    
    print(f"\n{'='*60}")
    print(f"LIBERO 评估")
    print(f"{'='*60}")
    print(f"Task Suite: {cfg.task_suite_name}")
    print(f"Total tasks: {num_tasks_in_suite}")
    print(f"Trials per task: {cfg.num_trials_per_task}")
    print(f"Max steps: {cfg.max_steps}")
    print(f"Execute all chunks: {cfg.execute_all_chunks}")
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
        print(f"  [{tid}] {task.language}")
    print()
    
    # 打开日志文件
    log_file = open(f"{log_dir}/eval_{cfg.task_suite_name}_{DATE_TIME}.log", "w")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    log_file.write(f"Checkpoint: {cfg.checkpoint_path}\n")
    log_file.write(f"Tasks: {task_id_list}\n\n")
    
    # 统计变量
    task_results = {}
    total_episodes, total_successes = 0, 0
    
    # 开始评估
    for task_id in tqdm.tqdm(task_id_list, desc="Tasks"):
        task = task_suite.get_task(task_id)
        task_description = task.language
        
        print(f"\n[Task {task_id}] {task_description}")
        log_file.write(f"\n[Task {task_id}] {task_description}\n")
        
        # 初始化环境
        env, _ = get_libero_env(task, resolution=256)
        
        # 获取初始状态
        initial_states = task_suite.get_task_init_states(task_id)
        
        task_successes = 0
        
        for trial in range(cfg.num_trials_per_task):
            print(f"  Trial {trial + 1}/{cfg.num_trials_per_task}")
            
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
                
                # 先获取图像（翻转+resize），用于视频保存和模型输入
                img, wrist_img = get_libero_image(obs, flip=cfg.flip_images)
                top_view.append(img)
                wrist_view.append(wrist_img)
                
                # 获取动作
                action = policy.get_action(obs, task_description)
                
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
            
            # 打印进度 - 与参考文件格式一致
            print(f"    Success: {success}")
            print(f"    Episodes: {total_episodes}, Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)")
            log_file.write(f"  Trial {trial + 1}: Success={success}, Steps={step + 1}\n")
            log_file.write(f"  Episodes: {total_episodes}, Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)\n")
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
        
        print(f"  Task {task_id} Success Rate: {task_success_rate:.1%} ({task_successes}/{cfg.num_trials_per_task})")
        log_file.write(f"  Task Success Rate: {task_success_rate:.1%} ({task_successes}/{cfg.num_trials_per_task})\n")
    
    # 打印最终结果
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Task Suite: {cfg.task_suite_name}")
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {total_successes/total_episodes*100:.1f}%")
    print(f"\nPer-Task Results:")
    print("-"*60)
    for tid, result in task_results.items():
        print(f"  Task {tid}: {result['success_rate']:.1%} ({result['successes']}/{result['episodes']}) - {result['task_description'][:50]}...")
    print(f"{'='*60}")
    
    # 写入日志
    log_file.write(f"\n{'='*60}\n")
    log_file.write("EVALUATION SUMMARY\n")
    log_file.write(f"{'='*60}\n")
    log_file.write(f"Task Suite: {cfg.task_suite_name}\n")
    log_file.write(f"Total Episodes: {total_episodes}\n")
    log_file.write(f"Total Successes: {total_successes}\n")
    log_file.write(f"Overall Success Rate: {total_successes/total_episodes*100:.1f}%\n")
    log_file.write(f"\nPer-Task Results:\n")
    for tid, result in task_results.items():
        log_file.write(f"  Task {tid}: {result['success_rate']:.1%} ({result['successes']}/{result['episodes']}) - {result['task_description']}\n")
    log_file.close()
    
    # 保存结果
    results_path = Path(cfg.video_dir) / f"eval_results_{cfg.task_suite_name}.json"
    with open(results_path, 'w') as f:
        json.dump({
            "task_suite": cfg.task_suite_name,
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
        description='LIBERO Evaluation with OFT Action Head (Qwen VLM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 Examples:
  # 基本用法 - LIBERO 仿真评估
  python eval_new.py \\
    --checkpoint_path ./experiments/checkpoints/step_5000/action_head.pt \\
    --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \\
    --data_path /path/to/dataset \\
    --task_suite_name libero_spatial \\
    --vlm_layers 14

  # 多层 VLM (Qwen4B)
  python eval_new.py \\
    --checkpoint_path ./experiments/checkpoints/step_5000/action_head.pt \\
    --vlm_model_path Qwen/Qwen3-VL-4B-Instruct \\
    --data_path /path/to/dataset \\
    --vlm_layers 16,17,18 \\
    --vlm_output_dim 2560 \\
    --task_suite_name libero_spatial

  # 使用训练数据测试
  python eval_new.py \\
    --checkpoint_path ./experiments/checkpoints/step_5000/action_head.pt \\
    --data_path /path/to/dataset \\
    --use_training_data \\
    --num_test_samples 100
        """
    )
    
    # 模型相关
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="训练好的 action_head.pt 文件路径")
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="VLM 模型路径")
    parser.add_argument("--data_path", type=str, default="",
                        help="训练数据集路径 (用于加载归一化统计量)")
    
    # VLM 相关参数
    parser.add_argument("--vlm_layers", type=str, default="14",
                        help="提取的 VLM 隐藏层 (如 '14' 或 '16,17,18')")
    parser.add_argument("--vlm_output_dim", type=int, default=1536,
                        help="VLM backbone 输出维度 (Qwen2B=1536, Qwen4B=2560)")
    parser.add_argument("--add_action_prompt", action="store_true", default=True,
                        help="添加 action prompt")
    parser.add_argument("--no_action_prompt", action="store_false", dest="add_action_prompt",
                        help="不添加 action prompt")
    
    # OFT 模型参数
    parser.add_argument("--num_transformer_blocks", type=int, default=2,
                        help="Transformer 块数量")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="注意力头数量")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比率")
    parser.add_argument("--action_head_hidden_dim", type=int, default=4096,
                        help="Action head 隐藏层维度")
    
    # LIBERO 环境
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", 
                                "libero_10", "libero_90"],
                        help="LIBERO 任务套件")
    parser.add_argument("--num_trials_per_task", type=int, default=5,
                        help="每个任务的评估次数")
    parser.add_argument("--task_ids", type=str, default=None,
                        help="指定要评估的任务ID (逗号分隔)")
    parser.add_argument("--max_tasks", type=int, default=-1,
                        help="最大评估任务数 (-1 表示全部)")
    parser.add_argument("--num_steps_wait", type=int, default=10,
                        help="等待物体稳定的步数")
    parser.add_argument("--max_steps", type=int, default=600,
                        help="每个 episode 的最大步数")
    
    # 推理参数
    parser.add_argument("--action_chunk_size", type=int, default=1,
                        help="每次执行的动作数")
    parser.add_argument("--execute_all_chunks", action="store_true",
                        help="执行所有 16 步预测动作")
    
    # 系统配置
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式 (不显示图像)")
    parser.add_argument("--flip_images", action="store_true", default=True,
                        help="是否翻转图像 180 度 (默认 True，与训练数据一致)")
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images",
                        help="不翻转图像")
    parser.add_argument("--video_dir", type=str, default="./eval_rollouts",
                        help="视频保存目录")
    parser.add_argument("--log_dir", type=str, default="",
                        help="日志保存目录，为空则使用 video_dir")
    parser.add_argument("--verbose", action="store_true",
                        help="详细模式")
    
    # 训练数据测试模式
    parser.add_argument("--use_training_data", action="store_true",
                        help="使用训练数据进行测试 (而非 LIBERO 环境)")
    parser.add_argument("--num_test_samples", type=int, default=100,
                        help="测试样本数量 (use_training_data=True 时有效)")
    
    args = parser.parse_args()
    
    # 解析 vlm_layers
    vlm_layers = [int(x.strip()) for x in args.vlm_layers.split(",")]
    
    # 解析 task_ids
    task_ids = None
    if args.task_ids:
        task_ids = [int(x.strip()) for x in args.task_ids.split(",")]
    
    # 创建配置
    cfg = EvalConfig(
        checkpoint_path=args.checkpoint_path,
        vlm_model_path=args.vlm_model_path,
        data_path=args.data_path,
        vlm_layers=vlm_layers,
        vlm_output_dim=args.vlm_output_dim,
        add_action_prompt=args.add_action_prompt,
        num_transformer_blocks=args.num_transformer_blocks,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        action_head_hidden_dim=args.action_head_hidden_dim,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        task_ids=task_ids,
        max_tasks=args.max_tasks,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        action_chunk_size=args.action_chunk_size,
        execute_all_chunks=args.execute_all_chunks,
        device=args.device,
        headless=args.headless,
        flip_images=args.flip_images,
        video_dir=args.video_dir,
        verbose=args.verbose,
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
