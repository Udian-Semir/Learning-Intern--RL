#!/usr/bin/env python3
"""
LIBERO 评估脚本 - 基于自定义训练的 Flow Matching Action Head (Qwen VLM 版本)

该脚本与 train_with_pretrained_action_head_weight_qwen2b_multigpu.py 训练方式对齐，
使用相同的:
- 数据处理流程 (四元数→轴角转换)
- 归一化方式 (min_max 到 [-1, 1])
- 模型配置 (FlowmatchingActionHead)
- VLM backbone (Qwen3-VL-2B-Instruct)

使用方法:
    # 基本使用
    python eval_libero_image_flip_for_qwen.py \
        --checkpoint_path ./experiments/fm0_pretrained_finetuning/checkpoints/best/action_head.pt \
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
        --data_path /path/to/dataset \
        --task_suite_name libero_spatial

    # 完整参数
    python eval_libero_image_flip_for_qwen.py \
        --checkpoint_path ./experiments/fm0_pretrained_finetuning/checkpoints/best/action_head.pt \
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
        --data_path /path/to/dataset \
        --task_suite_name libero_spatial \
        --num_trials_per_task 5 \
        --denoising_steps 4 \
        --action_chunk_size 1 \
        --device cuda:0 \
        --vlm_layer 14

参数说明:
    --checkpoint_path: 训练好的 action_head.pt 文件路径
    --vlm_model_path: VLM 模型路径 (Qwen/Qwen3-VL-2B-Instruct)
    --data_path: 训练数据集路径 (用于加载归一化统计量)
    --task_suite_name: LIBERO 任务套件名称
    --vlm_layer: 提取的 VLM 隐藏层 (1-28，默认14)
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers.feature_extraction_utils import BatchFeature

# 添加模型路径
sys.path.insert(0, str(Path(__file__).parent))
# 添加 Action_Heads/Flow_Matching_0 路径以导入 models 和 config
# eval 文件路径: sai0-vla/eval/libero/Flow_Matching_0/qwen/
# 目标路径:      sai0-vla/Action_Heads/Flow_Matching_0/
# parent = qwen, parent*2 = Flow_Matching_0, parent*3 = libero, parent*4 = eval, parent*5 = sai0-vla
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / 'Action_Heads' / 'Flow_Matching_0'))
# 添加 Isaac-GR00T 路径（可选，用于兼容）
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent.parent / 'Isaac-GR00T'))

from models.action_head.flow_matching_action_head import FlowmatchingActionHead
from config import get_flowmatching_action_head_config_original

# LIBERO 环境
try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except ImportError:
    LIBERO_AVAILABLE = False
    print("⚠️ LIBERO 未安装，请运行: pip install robosuite==1.4.0")

# Qwen VLM - 使用 transformers 加载
try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ transformers 未正确安装，VLM 功能不可用: {e}")

# 保留旧的导入用于兼容性（可选）
try:
    from gr00t.model.backbone.eagle_backbone import DEFAULT_EAGLE_PATH, EagleBackbone
    from gr00t.model.transforms import build_eagle_processor, collate, GR00TTransform
    from gr00t.model.gr00t_n1 import GR00T_N1_5
    from gr00t.model.policy import Gr00tPolicy
    GROOT_AVAILABLE = True
except ImportError:
    GROOT_AVAILABLE = False
    DEFAULT_EAGLE_PATH = None


# ============================================================================
# 常量定义
# ============================================================================
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEFAULT_EMBODIMENT_ID = 31


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
    # clip quaternion
    qw = quat[3]
    if qw > 1.0:
        qw = 1.0
    elif qw < -1.0:
        qw = -1.0
    
    den = np.sqrt(1.0 - qw * qw)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    
    return (quat[:3] * 2.0 * np.arccos(qw)) / den


def axisangle2quat(axis_angle: np.ndarray) -> np.ndarray:
    """
    轴角转四元数 (用于反归一化后转换)
    
    Args:
        axis_angle: (3,) 轴角 [ax, ay, az]
    
    Returns:
        quat: (4,) 四元数 [qx, qy, qz, qw]
    """
    angle = np.linalg.norm(axis_angle)
    if np.isclose(angle, 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0])
    
    axis = axis_angle / angle
    half_angle = angle / 2.0
    qw = np.cos(half_angle)
    qxyz = axis * np.sin(half_angle)
    
    return np.array([qxyz[0], qxyz[1], qxyz[2], qw])


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
    # 训练数据格式 (9维): [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
    # 转换后格式 (8维):   [gripper1, gripper2, x, y, z, ax, ay, az]
    if 'observation.state' in stats:
        state_stats = stats['observation.state']
        original_min = state_stats['min']
        original_max = state_stats['max']
        original_dim = len(original_min)
        
        if convert_quat_to_axisangle and original_dim == 9:
            # 9维 → 8维 转换
            # 用户数据顺序: [gripper1, gripper2, x, y, z, qx, qy, qz, qw]
            #                  0         1      2  3  4   5   6   7   8
            # 转换后顺序:   [gripper1, gripper2, x, y, z, ax, ay, az]
            #                  0         1      2  3  4   5   6   7
            state_min = torch.tensor([
                original_min[0], original_min[1],                   # gripper1, gripper2
                original_min[2], original_min[3], original_min[4],  # x, y, z
                -math.pi, -math.pi, -math.pi,                       # ax, ay, az (固定范围)
            ], dtype=torch.float32)
            state_max = torch.tensor([
                original_max[0], original_max[1],                   # gripper1, gripper2
                original_max[2], original_max[3], original_max[4],  # x, y, z
                math.pi, math.pi, math.pi,                          # ax, ay, az (固定范围)
            ], dtype=torch.float32)
            print(f"✓ State 归一化统计量: {original_dim}维 → 8维 (四元数→轴角转换)")
            print(f"  - Gripper1 min/max: [{state_min[0].item():.4f}, {state_max[0].item():.4f}]")
            print(f"  - Gripper2 min/max: [{state_min[1].item():.4f}, {state_max[1].item():.4f}]")
            print(f"  - 位置 min: {state_min[2:5].tolist()}")
            print(f"  - 位置 max: {state_max[2:5].tolist()}")
            print(f"  - 轴角范围: [-π, π]")
        else:
            state_min = torch.tensor(original_min, dtype=torch.float32)
            state_max = torch.tensor(original_max, dtype=torch.float32)
        
        normalizers['state'] = MinMaxNormalizer(state_min, state_max)
    
    # 加载 action 归一化统计量
    if 'action' in stats:
        action_stats = stats['action']
        original_min = action_stats['min']
        original_max = action_stats['max']
        original_dim = len(original_min)
        
        if convert_quat_to_axisangle and original_dim == 9:
            # 9维 → 7维 转换
            action_min = torch.tensor([
                original_min[0], original_min[1], original_min[2],
                -math.pi, -math.pi, -math.pi,
                original_min[7]
            ], dtype=torch.float32)
            action_max = torch.tensor([
                original_max[0], original_max[1], original_max[2],
                math.pi, math.pi, math.pi,
                original_max[7]
            ], dtype=torch.float32)
            print(f"✓ Action 归一化统计量: {original_dim}维 → 7维 (四元数→轴角转换)")
        else:
            action_min = torch.tensor(original_min, dtype=torch.float32)
            action_max = torch.tensor(original_max, dtype=torch.float32)
        
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
        flip: 是否翻转图像 180 度 (默认 False)，两张图像都翻转
        resize_to: resize 目标尺寸 (默认 128x128，与训练数据一致)
    """
    img = obs["agentview_image"].copy()  # 确保复制
    wrist_img = obs["robot0_eye_in_hand_image"].copy()
    
    # 两张图像都翻转 180 度（与训练时一致）
    if flip:
        img = img[::-1, ::-1].copy()  # 主视角 180 度翻转
        wrist_img = wrist_img[::-1, ::-1].copy()  # 手部图像也翻转
    
    # 确保是 uint8 格式
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if wrist_img.dtype != np.uint8:
        wrist_img = (wrist_img * 255).astype(np.uint8) if wrist_img.max() <= 1.0 else wrist_img.astype(np.uint8)
    
    # Resize 到训练数据尺寸 (128x128)
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
    
    # 检查是否有帧
    if len(top_view) == 0 or len(wrist_view) == 0:
        print(f"⚠️ 警告: 视频帧为空! top_view: {len(top_view)}, wrist_view: {len(wrist_view)}")
        if log_file is not None:
            log_file.write(f"⚠️ 警告: 视频帧为空! top_view: {len(top_view)}, wrist_view: {len(wrist_view)}\n")
        return None
    
    print(f"保存视频: {len(top_view)} 帧, 图像尺寸: {top_view[0].shape}")
    
    try:
        # 使用 ffmpeg 编码器，更稳定
        video_writer = imageio.get_writer(
            mp4_path, 
            fps=10,  # 与训练数据帧率一致
            codec='libx264',  # 使用 H.264 编码
            quality=8,  # 质量 (0-10, 10最好)
            pixelformat='yuv420p',  # 兼容性更好的像素格式
        )
        
        for img1, img2 in zip(top_view, wrist_view):
            # 确保图像是 uint8
            if img1.dtype != np.uint8:
                img1 = img1.astype(np.uint8)
            if img2.dtype != np.uint8:
                img2 = img2.astype(np.uint8)
            
            combined = np.hstack((img1, img2))
            video_writer.append_data(combined)
        
        video_writer.close()
        
        # 验证文件大小
        file_size = os.path.getsize(mp4_path)
        if file_size < 1024:  # 小于 1KB
            print(f"⚠️ 警告: 视频文件过小 ({file_size} bytes)，可能保存失败")
        else:
            print(f"✓ 视频保存成功: {mp4_path} ({file_size / 1024:.1f} KB)")
        
        if log_file is not None:
            log_file.write(f"Saved rollout MP4 at path {mp4_path} ({file_size / 1024:.1f} KB)\n")
        
    except Exception as e:
        print(f"❌ 视频保存失败: {e}")
        if log_file is not None:
            log_file.write(f"❌ 视频保存失败: {e}\n")
        
        # 尝试使用备用方法保存
        try:
            print("尝试使用备用方法保存...")
            avi_path = mp4_path.replace('.mp4', '.avi')
            video_writer = imageio.get_writer(avi_path, fps=10, format='FFMPEG')
            for img1, img2 in zip(top_view, wrist_view):
                combined = np.hstack((img1.astype(np.uint8), img2.astype(np.uint8)))
                video_writer.append_data(combined)
            video_writer.close()
            print(f"✓ 备用视频保存成功: {avi_path}")
            return avi_path
        except Exception as e2:
            print(f"❌ 备用方法也失败: {e2}")
        
        return None
    
    return mp4_path


def normalize_gripper_action(action, binarize=True):
    """归一化 gripper 动作到 [-1, 1]"""
    orig_low, orig_high = 0.0, 1.0
    action[-1]= 1 - action[-1] # ! 反转 gripper 动作
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)
    
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    
    return action


def save_training_data_comparison_video(
    images_list: List[List[np.ndarray]],  # 每个样本的图像列表
    gt_actions_list: List[np.ndarray],     # GT actions
    pred_actions_list: List[np.ndarray],   # 预测的 actions
    mse_list: List[float],                  # 每个样本的 MSE
    task_descriptions: List[str],           # 任务描述
    sample_indices: List[int],              # 样本索引
    video_dir: str = "./rollouts",
    fps: int = 1,
):
    """
    保存训练数据测试的对比视频/图像
    
    为每个样本创建可视化，展示输入图像、GT action 和预测 action 的对比
    """
    rollout_dir = f"{video_dir}/{DATE}/training_data_comparison"
    os.makedirs(rollout_dir, exist_ok=True)
    
    # 为每个样本保存对比图像
    for i, (images, gt_action, pred_action, mse, task_desc, sample_idx) in enumerate(
        zip(images_list, gt_actions_list, pred_actions_list, mse_list, task_descriptions, sample_indices)
    ):
        # 创建对比可视化图像
        fig_path = f"{rollout_dir}/sample_{sample_idx:04d}_mse_{mse:.4f}.png"
        
        # 准备图像
        img1 = images[0] if len(images) > 0 else np.zeros((224, 224, 3), dtype=np.uint8)
        img2 = images[1] if len(images) > 1 else np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 确保图像是 uint8 格式
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1.0 else img1.astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = (img2 * 255).astype(np.uint8) if img2.max() <= 1.0 else img2.astype(np.uint8)
        
        # 调整图像大小以便显示
        target_size = (256, 256)
        img1_resized = cv2.resize(img1, target_size)
        img2_resized = cv2.resize(img2, target_size)
        
        # 创建画布
        canvas_height = 480
        canvas_width = 700
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # 放置图像
        canvas[10:10+256, 10:10+256] = img1_resized  # 左上：agent view
        canvas[10:10+256, 276:276+256] = img2_resized  # 右上：wrist view
        
        # 添加文字信息
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_color = (0, 0, 0)
        line_height = 18
        
        # 标题
        cv2.putText(canvas, f"Sample: {sample_idx}", (10, 280), font, 0.5, font_color, 1)
        cv2.putText(canvas, f"MSE: {mse:.6f}", (200, 280), font, 0.5, (0, 0, 255) if mse > 0.1 else (0, 128, 0), 1)
        
        # 任务描述 (截断)
        task_short = task_desc[:60] + "..." if len(task_desc) > 60 else task_desc
        cv2.putText(canvas, f"Task: {task_short}", (10, 300), font, 0.35, font_color, 1)
        
        # GT vs Pred action 对比
        y_start = 325
        cv2.putText(canvas, "GT Action (first step):", (10, y_start), font, 0.45, (0, 100, 0), 1)
        cv2.putText(canvas, "Pred Action (first step):", (350, y_start), font, 0.45, (0, 0, 180), 1)
        
        # Action 值对比
        gt_first = gt_action[0] if gt_action.ndim > 1 else gt_action
        pred_first = pred_action[0] if pred_action.ndim > 1 else pred_action
        
        action_labels = ["x", "y", "z", "ax", "ay", "az", "g1", "g2"]
        y_pos = y_start + 20
        
        for j in range(min(len(gt_first), 8)):
            label = action_labels[j] if j < len(action_labels) else f"a{j}"
            gt_val = gt_first[j]
            pred_val = pred_first[j]
            diff = abs(gt_val - pred_val)
            
            # GT 值
            cv2.putText(canvas, f"{label}: {gt_val:+.4f}", (10, y_pos), font, font_scale, (0, 100, 0), 1)
            # Pred 值
            cv2.putText(canvas, f"{label}: {pred_val:+.4f}", (350, y_pos), font, font_scale, (0, 0, 180), 1)
            # Diff 值
            diff_color = (0, 0, 255) if diff > 0.1 else (128, 128, 128)
            cv2.putText(canvas, f"diff: {diff:.4f}", (550, y_pos), font, font_scale, diff_color, 1)
            
            y_pos += line_height
        
        # 添加图像标签
        cv2.putText(canvas, "Agent View", (80, 270), font, 0.4, (100, 100, 100), 1)
        cv2.putText(canvas, "Wrist View", (350, 270), font, 0.4, (100, 100, 100), 1)
        
        # 保存图像
        cv2.imwrite(fig_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    
    print(f"✅ 保存了 {len(images_list)} 个对比图像到: {rollout_dir}")
    
    # 如果样本数量较多，创建一个汇总视频
    if len(images_list) >= 5:
        video_path = f"{rollout_dir}/{DATE_TIME}_comparison_summary.mp4"
        try:
            # 重新读取保存的图像并创建视频
            all_frames = []
            for i, sample_idx in enumerate(sample_indices):
                mse = mse_list[i]
                fig_path = f"{rollout_dir}/sample_{sample_idx:04d}_mse_{mse:.4f}.png"
                if os.path.exists(fig_path):
                    frame = cv2.imread(fig_path)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    all_frames.append(frame)
            
            if all_frames:
                video_writer = imageio.get_writer(
                    video_path,
                    fps=fps,
                    codec='libx264',
                    quality=8,
                    pixelformat='yuv420p',
                )
                for frame in all_frames:
                    video_writer.append_data(frame)
                video_writer.close()
                print(f"✅ 汇总视频已保存: {video_path}")
        except Exception as e:
            print(f"⚠️ 汇总视频保存失败: {e}")
    
    return rollout_dir


def save_episode_video_with_predictions(
    dataset,
    episode_idx: int,
    vlm_backbone,
    action_head,
    normalizers: Dict,
    cfg,
    video_dir: str = "./rollouts",
    fps: int = 10,
):
    """
    保存完整 episode 的对比视频：
    - 上方: 原始数据集的 GT 视频轨迹
    - 下方: 从第一帧初始化，用模型预测动作在仿真环境中执行的轨迹
    
    需要 LIBERO 仿真环境
    
    Args:
        dataset: LeRobotDataset 实例
        episode_idx: Episode 索引
        vlm_backbone: VLM backbone 实例
        action_head: Action head 模型
        normalizers: 归一化器字典
        cfg: 配置
        video_dir: 视频保存目录
        fps: 视频帧率
    """
    import pandas as pd
    
    if not LIBERO_AVAILABLE:
        print(f"    ⚠️ LIBERO 未安装，跳过 episode {episode_idx} 的仿真视频")
        return None, 0.0
    
    rollout_dir = f"{video_dir}/{DATE}/episode_comparison_videos"
    os.makedirs(rollout_dir, exist_ok=True)
    
    # 获取 episode 的 parquet 文件
    chunk_idx = episode_idx // dataset.chunks_size
    parquet_path = dataset.dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    df = pd.read_parquet(parquet_path)
    episode_length = len(df)
    
    print(f"  处理 Episode {episode_idx}: {episode_length} 帧")
    
    # 获取 task description
    task_description = "Task"
    if 'annotation.human.action.task_description' in df.columns:
        task_idx = int(df.iloc[0]['annotation.human.action.task_description'])
        task_description = dataset.task_descriptions.get(task_idx, f"Task {task_idx}")
    
    print(f"    Task: {task_description}")
    
    # ========== 1. 加载 GT 视频帧 ==========
    gt_frames_agent = []
    gt_frames_wrist = []
    
    for frame_idx in range(episode_length):
        for i, image_key in enumerate(dataset.image_keys[:2]):
            video_path = dataset.dataset_path / "videos" / f"chunk-{chunk_idx:03d}" / image_key / f"episode_{episode_idx:06d}.mp4"
            img = dataset._load_image_from_video(video_path, frame_idx)
            if i == 0:
                gt_frames_agent.append(img)
            else:
                gt_frames_wrist.append(img)
    
    print(f"    GT 视频: {len(gt_frames_agent)} 帧")
    
    # ========== 2. 使用 LIBERO 仿真环境执行预测动作 ==========
    try:
        from libero.libero import benchmark
        
        # 尝试从 task description 找到对应的 LIBERO task
        # 这里我们需要初始化环境 - 使用 libero_spatial 作为默认
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict.get("libero_spatial", benchmark_dict.get("libero_10"))()
        
        # 找到匹配的 task
        matched_task = None
        matched_task_id = 0
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            if task.language.lower() in task_description.lower() or task_description.lower() in task.language.lower():
                matched_task = task
                matched_task_id = task_id
                break
        
        if matched_task is None:
            # 如果找不到匹配的 task，使用第一个
            matched_task = task_suite.get_task(0)
            matched_task_id = 0
            print(f"    ⚠️ 未找到匹配的 task，使用默认 task: {matched_task.language}")
        else:
            print(f"    ✓ 匹配到 task: {matched_task.language}")
        
        # 初始化环境
        env, _ = get_libero_env(matched_task, resolution=256)
        
        # 获取初始状态
        initial_states = task_suite.get_task_init_states(matched_task_id)
        
        # Reset 并设置初始状态
        env.reset()
        obs = env.set_init_state(initial_states[0])
        
        # 等待物体稳定
        for _ in range(10):
            obs, _, _, _ = env.step(get_libero_dummy_action())
        
        # 执行预测动作
        pred_frames_agent = []
        pred_frames_wrist = []
        max_steps = min(episode_length, 300)  # 限制最大步数
        action_horizon = cfg.action_horizon  # 16 步动作
        
        print(f"    开始仿真推理 (最大 {max_steps} 步, action_horizon={action_horizon})...")
        
        step = 0
        action_queue = []  # 动作队列，用于缓存预测的多步动作
        
        while step < max_steps:
            # 获取当前图像 (使用 cfg.flip_images 控制是否翻转)
            img_agent, img_wrist = get_libero_image(obs, flip=cfg.flip_images)
            pred_frames_agent.append(img_agent)
            pred_frames_wrist.append(img_wrist)
            
            # 如果动作队列为空，进行新的预测
            if len(action_queue) == 0:
                # 获取当前 state
                xyz = obs["robot0_eef_pos"]
                rpy = quat2axisangle(obs["robot0_eef_quat"])
                gripper = obs["robot0_gripper_qpos"]
                state = np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], 
                                 gripper[0], gripper[1]], dtype=np.float32)
                
                # VLM 推理
                images_for_vlm = [img_agent, img_wrist]
                with torch.no_grad():
                    backbone_features = vlm_backbone.get_hidden_states(images_for_vlm, task_description)
                
                # 准备 state tensor
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(cfg.device)
                if normalizers is not None and 'state' in normalizers:
                    state_tensor = normalizers['state'].normalize(state_tensor)
                
                # Padding state
                if state_tensor.shape[1] < cfg.max_state_dim:
                    padding = torch.zeros(1, cfg.max_state_dim - state_tensor.shape[1], 
                                        device=cfg.device, dtype=state_tensor.dtype)
                    state_tensor = torch.cat([state_tensor, padding], dim=1)
                state_tensor = state_tensor.unsqueeze(1)
                
                # Action head 推理
                with torch.no_grad():
                    backbone_attention_mask = torch.ones(1, backbone_features.shape[1], dtype=torch.long, device=cfg.device)
                    embodiment_id = torch.full((1,), DEFAULT_EMBODIMENT_ID, dtype=torch.long, device=cfg.device)
                    
                    backbone_output = BatchFeature(data={
                        "backbone_features": backbone_features,
                        "backbone_attention_mask": backbone_attention_mask,
                    })
                    action_input = BatchFeature(data={
                        "state": state_tensor,
                        "embodiment_id": embodiment_id,
                    })
                    
                    original_steps = action_head.num_inference_timesteps
                    action_head.num_inference_timesteps = cfg.denoising_steps
                    output = action_head.get_action(backbone_output, action_input)
                    action_head.num_inference_timesteps = original_steps
                
                # 获取所有 16 步动作并加入队列
                all_actions = output["action_pred"][0].cpu().numpy()  # (action_horizon, action_dim)
                for action_idx in range(action_horizon):
                    action = all_actions[action_idx]
                    # 反归一化
                    if normalizers is not None and 'action' in normalizers:
                        action = normalizers['action'].denormalize_numpy(action[:7].reshape(1, -1))[0]
                    else:
                        action = action[:7]
                    
                    # 简单粗暴处理 gripper: >0 则为 1 (打开), <0 则为 -1 (闭合)
                    action = action.astype(np.float32)
                    action[-1] = 1.0 if action[-1] > 0 else -1.0
                    
                    action_queue.append(action)
                
                print(f"      步骤 {step}: 预测了 {len(action_queue)} 步动作")
            
            # 从队列中取出一个动作执行
            pred_action = action_queue.pop(0)
            
            # 执行动作
            obs, reward, done, info = env.step(pred_action.tolist())
            step += 1
            
            if done:
                print(f"    ✓ 任务完成于步骤 {step}")
                break
        
        env.close()
        print(f"    仿真视频: {len(pred_frames_agent)} 帧")
        
    except Exception as e:
        print(f"    ⚠️ 仿真失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果仿真失败，使用 GT 视频作为 pred 视频
        pred_frames_agent = gt_frames_agent.copy()
        pred_frames_wrist = gt_frames_wrist.copy()
    
    # ========== 3. 创建对比视频 ==========
    # 统一帧数
    min_frames = min(len(gt_frames_agent), len(pred_frames_agent))
    
    all_frames = []
    for i in range(min_frames):
        frame = create_gt_vs_pred_frame(
            gt_agent=gt_frames_agent[i],
            gt_wrist=gt_frames_wrist[i] if i < len(gt_frames_wrist) else gt_frames_agent[i],
            pred_agent=pred_frames_agent[i],
            pred_wrist=pred_frames_wrist[i] if i < len(pred_frames_wrist) else pred_frames_agent[i],
            task_description=task_description,
            frame_idx=i,
            total_frames=min_frames,
        )
        all_frames.append(frame)
    
    # 保存视频
    if len(all_frames) > 0:
        video_path = f"{rollout_dir}/episode_{episode_idx:04d}_comparison.mp4"
        
        try:
            video_writer = imageio.get_writer(
                video_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p'
            )
            for frame in all_frames:
                video_writer.append_data(frame)
            video_writer.close()
            print(f"    ✅ 对比视频已保存: {video_path} ({len(all_frames)} 帧)")
            return video_path, 0.0
        except Exception as e:
            print(f"    ❌ 视频保存失败: {e}")
            return None, 0.0
    
    return None, 0.0


def create_gt_vs_pred_frame(
    gt_agent: np.ndarray,
    gt_wrist: np.ndarray,
    pred_agent: np.ndarray,
    pred_wrist: np.ndarray,
    task_description: str,
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """
    创建 GT vs Pred 对比帧
    
    布局:
    +------------------+------------------+
    |   GT Agent View  |   GT Wrist View  |  <- 上方: Ground Truth
    +------------------+------------------+
    |  Pred Agent View | Pred Wrist View  |  <- 下方: 模型预测
    +------------------+------------------+
    """
    # 确保 uint8
    def to_uint8(img):
        if img.dtype != np.uint8:
            return (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        return img
    
    gt_agent = to_uint8(gt_agent)
    gt_wrist = to_uint8(gt_wrist)
    pred_agent = to_uint8(pred_agent)
    pred_wrist = to_uint8(pred_wrist)
    
    # 调整大小为统一尺寸
    target_size = (256, 256)
    gt_agent = cv2.resize(gt_agent, target_size)
    gt_wrist = cv2.resize(gt_wrist, target_size)
    pred_agent = cv2.resize(pred_agent, target_size)
    pred_wrist = cv2.resize(pred_wrist, target_size)
    
    # 创建画布
    canvas_height = 256 * 2 + 80  # 两行图像 + 文字区域
    canvas_width = 256 * 2 + 20   # 两列图像 + 间隔
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 40  # 深灰色背景
    
    # 放置图像
    # 上方: GT
    canvas[40:40+256, 5:5+256] = gt_agent
    canvas[40:40+256, 261:261+256] = gt_wrist
    
    # 下方: Pred
    canvas[40+256+10:40+256+10+256, 5:5+256] = pred_agent
    canvas[40+256+10:40+256+10+256, 261:261+256] = pred_wrist
    
    # 添加文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    green = (0, 255, 0)
    blue = (255, 165, 0)  # 橙色 (BGR)
    
    # 标题
    task_short = task_description[:50] + "..." if len(task_description) > 50 else task_description
    cv2.putText(canvas, f"Frame {frame_idx}/{total_frames}: {task_short}", (10, 25), font, 0.45, white, 1)
    
    # 行标签
    cv2.putText(canvas, "GT (Ground Truth)", (10, 55), font, 0.4, green, 1)
    cv2.putText(canvas, "Agent View", (90, 40+256-5), font, 0.35, white, 1)
    cv2.putText(canvas, "Wrist View", (340, 40+256-5), font, 0.35, white, 1)
    
    cv2.putText(canvas, "Pred (Model Prediction)", (10, 40+256+25), font, 0.4, blue, 1)
    cv2.putText(canvas, "Agent View", (90, canvas_height-5), font, 0.35, white, 1)
    cv2.putText(canvas, "Wrist View", (340, canvas_height-5), font, 0.35, white, 1)
    
    return canvas


def show_obs_images_cv2(new_obs):
    """显示观测图像"""
    img_agent = new_obs["video.image"][0]
    img_wrist = new_obs["video.wrist_image"][0]
    img_agent_bgr = cv2.cvtColor(img_agent, cv2.COLOR_RGB2BGR)
    img_wrist_bgr = cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR)
    cv2.imshow("Agent View", img_agent_bgr)
    cv2.imshow("Wrist View", img_wrist_bgr)
    cv2.waitKey(1)


# ============================================================================
# VLM Backbone 封装
# ============================================================================

class EagleVLMBackbone:
    """
    Eagle VLM Backbone 封装
    
    完全参照 extract_vlm_hidden_states.py 的方式:
    1. 加载 GR00T_N1_5 完整模型
    2. 使用 GR00TTransform 处理数据
    3. 使用 collate 将 eagle_content 转换为 tensor
    4. 使用 model.prepare_input 准备输入
    5. 使用 model.backbone 提取特征
    """
    
    def __init__(self, model_path: str = None, device: str = "cuda:0", project_to_dim: int = 1536, verbose: bool = False):
        """
        初始化 Eagle VLM
        
        Args:
            model_path: 模型路径 (如 "nvidia/GR00T-N1.5-3B")
            device: 设备
            project_to_dim: 投影维度 (默认 1536，与训练一致)
            verbose: 是否打印详细信息 (input_ids, decoded tokens 等)
        """
        from gr00t.model.policy import COMPUTE_DTYPE
        from transformers import AutoTokenizer
        
        self.device = device
        self.project_to_dim = project_to_dim
        self.compute_dtype = COMPUTE_DTYPE  # bfloat16
        self.verbose = verbose
        
        # 加载完整 GR00T 模型
        model_path = model_path or "nvidia/GR00T-N1.5-3B"
        print(f"Loading GR00T model from: {model_path}")
        
        self.model = GR00T_N1_5.from_pretrained(
            model_path,
            torch_dtype=self.compute_dtype,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 获取 processor
        self.processor = build_eagle_processor(DEFAULT_EAGLE_PATH)
        
        # 加载 tokenizer (用于 verbose 模式下解码 tokens)
        self.tokenizer = None
        if self.verbose:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_EAGLE_PATH, trust_remote_code=True)
                print(f"✓ Tokenizer loaded from {DEFAULT_EAGLE_PATH}")
            except Exception as e:
                print(f"⚠️ Could not load tokenizer: {e}")
                print("  Token decoding will be disabled.")
        
        print(f"✓ GR00T model loaded successfully!")
        if self.verbose:
            print(f"  Verbose mode: ENABLED (will print input_ids and decoded tokens)")
    
    def _print_verbose_info(self, normalized_input: dict, text: str, images: List[np.ndarray]):
        """打印详细的输入信息 (参照 extract_vlm_hidden_states.py)"""
        print("\n" + "="*80)
        print("VERBOSE: VLM INPUT INFORMATION")
        print("="*80)
        
        # 打印文本输入
        print(f"\n[Task Description]")
        print(f"  {text}")
        
        # 打印图像信息
        print(f"\n[Image Inputs]")
        for i, img in enumerate(images):
            print(f"  Image {i}: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
        
        # 打印 normalized_input 的 keys 和 shapes
        print(f"\n[Normalized Input Keys (after collate)]")
        for k, v in normalized_input.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__}")
        
        # 打印 token 信息
        if "eagle_input_ids" in normalized_input:
            input_ids = normalized_input["eagle_input_ids"]
            print(f"\n[Token Information]")
            print(f"  Input IDs shape: {input_ids.shape}")
            print(f"  Input IDs: {input_ids.tolist()}")
            
            # 解码 tokens
            if self.tokenizer is not None:
                for batch_idx in range(input_ids.shape[0]):
                    decoded_text = self.tokenizer.decode(input_ids[batch_idx], skip_special_tokens=False)
                    print(f"\n  [Batch {batch_idx}] Decoded tokens:")
                    print(f"    {decoded_text}")
                    
                    # Token-by-token breakdown (前 50 个)
                    print(f"\n  [Batch {batch_idx}] Token-by-token breakdown:")
                    tokens = input_ids[batch_idx].tolist()
                    for i, token_id in enumerate(tokens[:50]):
                        token_str = self.tokenizer.decode([token_id])
                        print(f"    [{i}] ID={token_id}: '{token_str}'")
                    if len(tokens) > 50:
                        print(f"    ... ({len(tokens) - 50} more tokens)")
        
        print("="*80 + "\n")
    
    @torch.no_grad()
    def get_hidden_states(
        self,
        images: List[np.ndarray],
        text: str,
    ) -> torch.Tensor:
        """
        提取 VLM 隐藏状态 - 完全参照 extract_vlm_hidden_states.py
        
        Args:
            images: 图像列表 [agent_view, wrist_view], 每个是 (H, W, C) uint8
            text: 任务描述文本
        
        Returns:
            hidden_states: (1, seq_len, hidden_dim) 隐藏状态张量
        """
        from PIL import Image
        from einops import rearrange
        
        # ========== 关键：图像预处理与训练时保持一致 ==========
        # 训练时使用 VideoResize(height=224, width=224) 进行 resize
        # 必须在这里也做同样的 resize
        
        processed_images = []
        for img in images:
            # 确保是 uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            
            # Resize 到 224x224 (与训练时一致)
            if img.shape[0] != 224 or img.shape[1] != 224:
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((224, 224), Image.BILINEAR)
                img = np.array(pil_img)
            
            processed_images.append(img)
        
        if self.verbose:
            print(f"[Image Preprocessing] Resized images to 224x224 (matching training)")
            for i, img in enumerate(processed_images):
                print(f"  Image {i}: shape={img.shape}, dtype={img.dtype}")
        
        # 1. 准备图像 - 转换为 GR00TTransform 期望的格式
        # GR00TTransform._prepare_video 期望 video: [T, V, H, W, C]
        # 然后转换为 images: [V, T, C, H, W]
        
        # 我们有 images: list of [H, W, C], 假设是 [view1, view2]
        # 需要构造成: video [T=1, V=num_views, H, W, C]
        num_views = len(processed_images)
        h, w, c = processed_images[0].shape
        
        # 堆叠成 [T=1, V, H, W, C]
        video = np.stack(processed_images, axis=0)  # [V, H, W, C]
        video = np.expand_dims(video, axis=0)  # [T=1, V, H, W, C]
        
        # 转换为 [V, T, C, H, W] (与 GR00TTransform._prepare_video 一致)
        images_tensor = rearrange(video, "t v h w c -> v t c h w")
        images_tensor = images_tensor.astype(np.uint8)
        
        # 2. 应用 VLM 处理 - 与 GR00TTransform._apply_vlm_processing 一致
        # 重排为 (t v) c h w
        np_images = rearrange(images_tensor, "v t c h w -> (t v) c h w")
        
        # 转换为 PIL Image
        eagle_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in np_images]
        
        # 构建 eagle conversation
        eagle_image_content = [{"type": "image", "image": img} for img in eagle_images]
        text_content = [{"type": "text", "text": text}]
        eagle_conversation = [
            {
                "role": "user",
                "content": eagle_image_content + text_content,
            }
        ]
        
        # 使用 apply_chat_template 处理文本
        text_list = [
            self.processor.apply_chat_template(
                eagle_conversation, tokenize=False, add_generation_prompt=True
            )
        ]
        
        # 使用 process_vision_info 处理图像
        image_inputs, video_inputs = self.processor.process_vision_info(eagle_conversation)
        
        # 3. 构建 eagle_content (与 GR00TTransform 输出一致)
        eagle_content = {
            "image_inputs": image_inputs,
            "video_inputs": video_inputs,
            "text_list": text_list,
        }
        
        # 4. 使用 collate 处理 - 与 extract_vlm_hidden_states.py 一致
        # collate 期望一个 sample list，每个 sample 包含 eagle_content
        sample = {"eagle_content": eagle_content}
        normalized_input = collate([sample], self.processor)
        
        # Verbose 模式：打印详细信息
        if self.verbose:
            self._print_verbose_info(normalized_input, text, processed_images)
        
        # 5. 提取 backbone 特征 - 与 extract_vlm_hidden_states.py 完全一致
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.compute_dtype):
            # 准备 backbone 输入 (与 model.prepare_input 类似，但只处理 backbone 部分)
            # 将 tensor 移动到设备
            backbone_inputs = {}
            for k, v in normalized_input.items():
                if isinstance(v, torch.Tensor):
                    if torch.is_floating_point(v):
                        backbone_inputs[k] = v.to(self.device, dtype=self.compute_dtype)
                    else:
                        backbone_inputs[k] = v.to(self.device)
                else:
                    backbone_inputs[k] = v
            
            backbone_inputs = BatchFeature(data=backbone_inputs)
            
            # 使用 model.backbone 提取特征
            backbone_outputs = self.model.backbone(backbone_inputs)
            
            # 获取隐藏状态
            hidden_states = backbone_outputs["backbone_features"]  # (1, seq_len, hidden_dim)
        
        # Verbose 模式：打印输出信息
        if self.verbose:
            print(f"[VLM Output] hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        
        # 转换为 float32 (与训练时一致，extract_vlm_hidden_states.py 使用 .float() 保存)
        hidden_states = hidden_states.float()
        
        if self.verbose:
            print(f"[VLM Output] After .float(): shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
        
        return hidden_states  # (1, seq_len, hidden_dim)


# ============================================================================
# Qwen VLM Backbone 封装
# ============================================================================

class QwenVLMBackbone:
    """
    Qwen VLM Backbone 封装
    
    参照 extract_vlm_hidden_states.py 的方式:
    1. 加载 Qwen3-VL-2B-Instruct 模型
    2. 使用 AutoProcessor 处理数据
    3. 使用 model(**inputs, output_hidden_states=True) 提取特征
    4. 从 outputs.hidden_states 获取指定层的隐藏状态
    
    Qwen3-VL-2B-Instruct 的 language_model 有 28 层 transformer:
    - hidden_states[0]: embedding 层输出
    - hidden_states[1-28]: transformer 第 1-28 层输出
    """
    
    def __init__(
        self, 
        model_path: str = "Qwen/Qwen3-VL-2B-Instruct", 
        device: str = "cuda:0", 
        layer: int = 14,
        add_action_prompt: bool = True,
        verbose: bool = False
    ):
        """
        初始化 Qwen VLM
        
        Args:
            model_path: 模型路径 (默认 "Qwen/Qwen3-VL-2B-Instruct")
            device: 设备 (如 "cuda:0", "cuda:1")
            layer: 要提取的 transformer 层号 (1-28)
            add_action_prompt: 是否添加 action prompt
            verbose: 是否打印详细信息
        """
        self.device = device
        self.layer = layer
        self.add_action_prompt = add_action_prompt
        self.verbose = verbose
        
        # 验证层号
        if layer < 1 or layer > 28:
            raise ValueError(f"层号必须在 [1, 28] 范围内，当前: {layer}")
        
        print(f"Loading Qwen VLM from: {model_path}")
        print(f"  - 提取层: {layer}")
        print(f"  - 设备: {device}")
        
        # 解析 GPU 索引，设置 CUDA_VISIBLE_DEVICES 以锁定单个 GPU
        if device.startswith("cuda:"):
            gpu_id = device.split(":")[1]
            import os
            # 注意: 设置后，device 应该用 "cuda:0" 因为只能看到一个 GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
            actual_device = "cuda:0"
            print(f"  - 锁定 GPU: {gpu_id} (CUDA_VISIBLE_DEVICES={gpu_id})")
        else:
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
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"✓ Qwen VLM 加载成功!")
        if self.verbose:
            print(f"  Verbose mode: ENABLED")
    
    def _print_verbose_info(self, inputs: dict, text: str, images: List[np.ndarray]):
        """打印详细的输入信息"""
        print("\n" + "="*80)
        print("VERBOSE: QWEN VLM INPUT INFORMATION")
        print("="*80)
        
        # 打印文本输入
        print(f"\n[Task Description]")
        print(f"  {text}")
        
        # 打印图像信息
        print(f"\n[Image Inputs]")
        for i, img in enumerate(images):
            print(f"  Image {i}: shape={img.shape}, dtype={img.dtype}, range=[{img.min()}, {img.max()}]")
        
        # 打印 input 的 keys 和 shapes
        print(f"\n[Model Inputs Keys]")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            else:
                print(f"  {k}: {type(v).__name__}")
        
        # 打印 token 信息
        if "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            print(f"\n[Token Information]")
            print(f"  Input IDs shape: {input_ids.shape}")
            print(f"  Total tokens: {input_ids.shape[1]}")
            
            # 解码完整 tokens
            for batch_idx in range(input_ids.shape[0]):
                # 完整解码文本
                decoded_text = self.processor.decode(input_ids[batch_idx], skip_special_tokens=False)
                print(f"\n  [Batch {batch_idx}] Full decoded text:")
                print(f"    {decoded_text}")
                
                # 打印所有 token ID 和对应的 token
                print(f"\n  [Batch {batch_idx}] Token-by-token breakdown (ALL {input_ids.shape[1]} tokens):")
                tokens = input_ids[batch_idx].tolist()
                for i, token_id in enumerate(tokens):
                    token_str = self.processor.decode([token_id])
                    # 处理特殊字符的显示
                    token_display = repr(token_str) if token_str in ['', ' ', '\n', '\t'] or not token_str.isprintable() else f"'{token_str}'"
                    print(f"    [{i:4d}] ID={token_id:6d}: {token_display}")
        
        print("="*80 + "\n")
    
    @torch.no_grad()
    def get_hidden_states(
        self,
        images: List[np.ndarray],
        text: str,
    ) -> torch.Tensor:
        """
        提取 VLM 隐藏状态 - 参照 extract_vlm_hidden_states.py
        
        Args:
            images: 图像列表 [agent_view, wrist_view], 每个是 (H, W, C) uint8
            text: 任务描述文本
        
        Returns:
            hidden_states: (1, seq_len, hidden_dim) 隐藏状态张量
        """
        # 处理图像 - 转换为 PIL Image
        pil_images = []
        for img in images:
            # 确保是 uint8
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            pil_images.append(Image.fromarray(img))
        
        # 构建文本内容
        if self.add_action_prompt:
            text_content = f"What action should the robot take to {text.lower()}?"
        else:
            text_content = text
        
        # 构建 messages - 参照 extract_vlm_hidden_states.py
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_images[0]},  # agentview
                    {"type": "image", "image": pil_images[1]},  # wrist
                    {"type": "text", "text": text_content}
                ]
            }
        ]
        
        # 使用 processor 处理输入
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Verbose 模式：打印详细信息
        if self.verbose:
            self._print_verbose_info(inputs, text, [np.array(img) for img in pil_images])
        
        # 移动到设备
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # 前向传播获取隐藏状态
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        
        # 提取指定层的隐藏状态
        # hidden_states 是一个 tuple: (embedding层, transformer层1, ..., transformer层28)
        # layer=14 对应 hidden_states[14]
        hidden_state = outputs.hidden_states[self.layer]  # (batch, seq_len, hidden_dim)
        
        # Verbose 模式：打印输出信息
        if self.verbose:
            print(f"[Qwen VLM Output] hidden_states layer {self.layer} shape: {hidden_state.shape}, dtype: {hidden_state.dtype}")
        
        # 转换为 float32
        hidden_state = hidden_state.float()
        
        if self.verbose:
            print(f"[Qwen VLM Output] After .float(): shape: {hidden_state.shape}, dtype: {hidden_state.dtype}")
        
        return hidden_state  # (1, seq_len, hidden_dim)


# ============================================================================
# Policy 封装
# ============================================================================

class FlowMatchingPolicy:
    """
    Flow Matching Policy 封装
    
    与训练代码的数据处理流程完全对齐
    
    ⚠️ State 格式与训练代码一致:
    [gripper1, gripper2, x, y, z, ax, ay, az] (8维)
    """
    
    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            # 训练数据顺序: [gripper1, gripper2, x, y, z, ax, ay, az]
            "gripper": (0, 2),
            "x": 2, "y": 3, "z": 4,
            "roll": 5, "pitch": 6, "yaw": 7,
        },
    }
    
    def __init__(
        self,
        checkpoint_path: str,
        vlm_backbone = None,  # QwenVLMBackbone 或 EagleVLMBackbone
        normalizers: Optional[Dict] = None,
        device: str = "cuda:0",
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        action_horizon: int = 16,
        denoising_steps: int = 4,
        action_chunk_size: int = 1,
        flip_images: bool = False,
        headless: bool = False,
        backbone_dim: int = 1536,  # Qwen3-VL-2B hidden_dim=1536
    ):
        """
        初始化 Policy
        
        Args:
            checkpoint_path: action head checkpoint 路径
            vlm_backbone: VLM backbone (QwenVLMBackbone 或 EagleVLMBackbone)
            normalizers: 归一化器字典
            device: 设备
            max_state_dim: 最大状态维度
            max_action_dim: 最大动作维度
            action_horizon: 动作预测步数
            denoising_steps: 去噪步数
            action_chunk_size: 每次执行的动作数
            flip_images: 是否翻转图像
            headless: 是否无头模式
            backbone_dim: VLM backbone 输出维度 (Qwen3-VL-2B=1536)
        """
        self.device = device
        self.vlm_backbone = vlm_backbone
        self.normalizers = normalizers
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.denoising_steps = denoising_steps
        self.action_chunk_size = action_chunk_size
        self.flip_images = flip_images
        self.headless = headless
        self.backbone_dim = backbone_dim
        
        self.action_queue = []
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        
        # 加载 Action Head
        print(f"Loading Action Head from: {checkpoint_path}")
        
        # 创建配置
        cfg = get_flowmatching_action_head_config_original(
            action_backbone_dim=backbone_dim,
            action_dim=max_action_dim,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
        )
        
        # 创建并加载模型
        self.action_head = FlowmatchingActionHead(cfg).to(device)
        
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.action_head.load_state_dict(state_dict)
        # 确保模型完全在指定设备上
        self.action_head = self.action_head.to(device)
        self.action_head.eval()
        
        print(f"✓ Action Head loaded successfully!")
        print(f"  - Backbone dim: {backbone_dim}")
        print(f"  - Action horizon: {action_horizon}")
        print(f"  - Denoising steps: {denoising_steps}")
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
        obs_dict = self._process_observation(observation_dict, lang)
        
        # 获取 VLM 隐藏状态
        if self.vlm_backbone is not None:
            # 使用 VLM 提取特征
            images = [obs_dict["video.image"][0], obs_dict["video.wrist_image"][0]]
            backbone_features = self.vlm_backbone.get_hidden_states(images, lang)
            # get_hidden_states 已经返回 (1, seq_len, hidden_dim)
        else:
            raise ValueError("VLM backbone is required for inference")
        
        # 准备 state
        state_tensor = self._prepare_state(obs_dict)
        
        # 模型推理
        with torch.no_grad():
            action_chunk = self._inference(backbone_features, state_tensor)
        
        # 转换动作格式并加入队列
        for idx in range(min(self.action_chunk_size, self.action_horizon)):
            action = self._convert_to_libero_action(action_chunk, idx)
            self.action_queue.append(action)
        
        return self.action_queue.pop(0)
    
    def _process_observation(self, obs: dict, lang: str) -> dict:
        """处理观测数据"""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs, flip=self.flip_images)
        
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        
        if not self.headless:
            show_obs_images_cv2(new_obs)
        
        return new_obs
    
    def _prepare_state(self, obs_dict: dict) -> torch.Tensor:
        """
        准备 state tensor - 与训练代码对齐
        
        ⚠️ 重要: state 顺序必须与训练代码一致!
        训练数据格式: [gripper1, gripper2, x, y, z, ax, ay, az] (8维)
        
        Args:
            obs_dict: 处理后的观测字典
        
        Returns:
            state: (1, 1, max_state_dim) 归一化后的 state tensor
        """
        # 从观测中提取数据
        x = obs_dict["state.x"][0, 0]
        y = obs_dict["state.y"][0, 0]
        z = obs_dict["state.z"][0, 0]
        roll = obs_dict["state.roll"][0, 0]   # ax
        pitch = obs_dict["state.pitch"][0, 0] # ay
        yaw = obs_dict["state.yaw"][0, 0]     # az
        gripper = obs_dict["state.gripper"][0]  # (2,)
        
        # ⚠️ 关键: 组装 state 顺序必须与训练代码一致
        # 训练代码顺序: [gripper1, gripper2, x, y, z, ax, ay, az]
        state = np.array([gripper[0], gripper[1], x, y, z, roll, pitch, yaw], dtype=np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 8)
        
        # 归一化
        if self.normalizers is not None and 'state' in self.normalizers:
            state_tensor = self.normalizers['state'].normalize(state_tensor)
        
        # Padding 到 max_state_dim
        actual_dim = state_tensor.shape[1]
        if actual_dim < self.max_state_dim:
            padding = torch.zeros(1, self.max_state_dim - actual_dim, 
                                device=self.device, dtype=state_tensor.dtype)
            state_tensor = torch.cat([state_tensor, padding], dim=1)
        
        # 添加时间维度
        state_tensor = state_tensor.unsqueeze(1)  # (1, 1, max_state_dim)
        
        return state_tensor
    
    @torch.no_grad()
    def _inference(self, backbone_features: torch.Tensor, state: torch.Tensor) -> np.ndarray:
        """
        模型推理
        
        Args:
            backbone_features: (1, seq_len, hidden_dim) VLM 特征
            state: (1, 1, max_state_dim) 归一化后的 state
        
        Returns:
            action_chunk: (action_horizon, 7) 预测的动作序列
        """
        batch_size = 1
        
        # 确保 backbone_features 在正确的设备上
        backbone_features = backbone_features.to(self.device)
        
        # 准备输入
        backbone_attention_mask = torch.ones(
            batch_size, backbone_features.shape[1],
            dtype=torch.long, device=self.device
        )
        
        embodiment_id = torch.full(
            (batch_size,), DEFAULT_EMBODIMENT_ID,
            dtype=torch.long, device=self.device
        )
        
        # 构建 BatchFeature
        backbone_output = BatchFeature(data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_attention_mask,
        })
        
        action_head_inputs = BatchFeature(data={
            "state": state,
            "embodiment_id": embodiment_id,
        })
        
        # 临时设置推理步数
        original_num_inference_timesteps = self.action_head.num_inference_timesteps
        self.action_head.num_inference_timesteps = self.denoising_steps
        
        # 推理 (使用 get_action 方法)
        output = self.action_head.get_action(
            backbone_output=backbone_output,
            action_input=action_head_inputs,
        )
        
        # 恢复原始推理步数
        self.action_head.num_inference_timesteps = original_num_inference_timesteps
        
        # 获取预测的动作
        pred_actions = output["action_pred"]  # (1, action_horizon, max_action_dim)
        
        # 提取实际动作维度 (7维)
        pred_actions = pred_actions[0, :, :7].cpu().numpy()  # (action_horizon, 7)
        
        # 反归一化
        if self.normalizers is not None and 'action' in self.normalizers:
            pred_actions_tensor = torch.from_numpy(pred_actions)
            pred_actions = self.normalizers['action'].denormalize(pred_actions_tensor).numpy()
        
        return pred_actions
    
    def _convert_to_libero_action(self, action_chunk: np.ndarray, idx: int) -> np.ndarray:
        """转换为 LIBERO 动作格式"""
        action = action_chunk[idx].astype(np.float32)
        # action = normalize_gripper_action(action, binarize=True)
        # 简单二值化：gripper > 0 则为 1 (张开)，<= 0 则为 -1 (闭合)
        action[-1] = 1.0 if action[-1] > 0 else -1.0
        return action


# ============================================================================
# 简化版 Policy (使用预计算的 VLM 特征)
# ============================================================================

class FlowMatchingPolicySimple:
    """
    简化版 Policy - 不需要 VLM，使用固定特征
    
    主要用于快速测试或当 VLM 不可用时
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        normalizers: Optional[Dict] = None,
        device: str = "cuda:0",
        max_state_dim: int = 64,
        max_action_dim: int = 32,
        action_horizon: int = 16,
        denoising_steps: int = 4,
        action_chunk_size: int = 1,
        flip_images: bool = False,
        headless: bool = False,
        dummy_backbone_seq_len: int = 100,
    ):
        """
        初始化简化版 Policy
        
        注意: 这个版本使用随机 backbone 特征，仅用于测试
        """
        self.device = device
        self.normalizers = normalizers
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.action_horizon = action_horizon
        self.denoising_steps = denoising_steps
        self.action_chunk_size = action_chunk_size
        self.flip_images = flip_images
        self.headless = headless
        self.dummy_backbone_seq_len = dummy_backbone_seq_len
        
        self.action_queue = []
        
        # 加载 Action Head
        print(f"Loading Action Head from: {checkpoint_path}")
        
        backbone_dim = 1536
        cfg = get_flowmatching_action_head_config_original(
            backbone_dim=backbone_dim,
            action_dim=max_action_dim,
            action_horizon=action_horizon,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
        )
        
        self.action_head = FlowmatchingActionHead(cfg).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.action_head.load_state_dict(state_dict)
        self.action_head.eval()
        
        print(f"✓ Action Head loaded (Simple mode - dummy backbone)")
    
    def reset_action_queue(self):
        self.action_queue = []
    
    def get_action(self, observation_dict: dict, lang: str) -> np.ndarray:
        if len(self.action_queue) > 0:
            return self.action_queue.pop(0)
        
        # 处理 state
        xyz = observation_dict["robot0_eef_pos"]
        rpy = quat2axisangle(observation_dict["robot0_eef_quat"])
        gripper = observation_dict["robot0_gripper_qpos"]
        
        state = np.array([xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2], 
                         gripper[0], gripper[1]], dtype=np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
        
        # 归一化
        if self.normalizers is not None and 'state' in self.normalizers:
            state_tensor = self.normalizers['state'].normalize(state_tensor)
        
        # Padding
        if state_tensor.shape[1] < self.max_state_dim:
            padding = torch.zeros(1, self.max_state_dim - state_tensor.shape[1],
                                device=self.device, dtype=state_tensor.dtype)
            state_tensor = torch.cat([state_tensor, padding], dim=1)
        
        state_tensor = state_tensor.unsqueeze(1)  # (1, 1, max_state_dim)
        
        # Dummy backbone features
        backbone_features = torch.randn(
            1, self.dummy_backbone_seq_len, 1536,
            device=self.device, dtype=torch.float32
        )
        backbone_attention_mask = torch.ones(
            1, self.dummy_backbone_seq_len,
            dtype=torch.long, device=self.device
        )
        
        embodiment_id = torch.full((1,), DEFAULT_EMBODIMENT_ID, 
                                   dtype=torch.long, device=self.device)
        
        # 推理
        with torch.no_grad():
            # 临时设置推理步数
            original_num_inference_timesteps = self.action_head.num_inference_timesteps
            self.action_head.num_inference_timesteps = self.denoising_steps
            
            backbone_output = BatchFeature(data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": backbone_attention_mask,
            })
            
            action_head_inputs = BatchFeature(data={
                "state": state_tensor,
                "embodiment_id": embodiment_id,
            })
            
            output = self.action_head.get_action(
                backbone_output=backbone_output,
                action_input=action_head_inputs,
            )
            
            # 恢复原始推理步数
            self.action_head.num_inference_timesteps = original_num_inference_timesteps
        
        # 处理输出
        pred_actions = output["action_pred"]
        pred_actions = pred_actions[0, :, :7].cpu().numpy()
        
        if self.normalizers is not None and 'action' in self.normalizers:
            pred_actions_tensor = torch.from_numpy(pred_actions)
            pred_actions = self.normalizers['action'].denormalize(pred_actions_tensor).numpy()
        
        for idx in range(min(self.action_chunk_size, self.action_horizon)):
            action = pred_actions[idx].astype(np.float32)
            action = normalize_gripper_action(action, binarize=True)
            self.action_queue.append(action)
        
        return self.action_queue.pop(0)


# ============================================================================
# 评估配置
# ============================================================================

@dataclass
class EvalConfig:
    """评估配置"""
    # 模型相关
    checkpoint_path: str = ""
    """训练好的 action_head.pt 文件路径"""
    
    vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    """VLM 模型路径 (默认 Qwen3-VL-2B-Instruct)"""
    
    data_path: str = ""
    """训练数据集路径 (用于加载归一化统计量)"""
    
    # VLM 相关参数
    vlm_layer: int = 14
    """提取的 VLM 隐藏层 (1-28, 默认14)"""
    
    add_action_prompt: bool = True
    """是否添加 action prompt (训练时使用了 prompt，评估时也应使用)"""
    
    backbone_dim: int = 1536
    """VLM backbone 输出维度 (Qwen3-VL-2B=1536)"""
    
    # LIBERO 环境参数
    task_suite_name: str = "libero_spatial"
    """任务套件: libero_spatial, libero_object, libero_goal, libero_10, libero_90"""
    
    num_trials_per_task: int = 5
    """每个任务的评估次数"""
    
    task_ids: Optional[str] = None
    """指定要评估的任务ID (逗号分隔)"""
    
    max_tasks: int = -1
    """最大评估任务数 (-1 表示全部)"""
    
    num_steps_wait: int = 10
    """等待物体稳定的步数"""
    
    # 推理参数
    denoising_steps: int = 4
    """去噪步数"""
    
    action_chunk_size: int = 1
    """每次执行的动作数 (1=逐步推理, >1=chunk执行)"""
    
    # 系统配置
    device: str = "cuda:0"
    """设备"""
    
    headless: bool = False
    """无头模式 (不显示图像)"""
    
    flip_images: bool = True
    """是否翻转图像 180 度 ([::-1, ::-1])，在 VLM 推理前进行翻转"""
    
    video_dir: str = "./eval_rollouts"
    """视频保存目录"""
    
    log_dir: str = ""
    """日志保存目录，为空则使用 video_dir"""
    
    use_simple_policy: bool = False
    """使用简化版 Policy (不需要 VLM)"""
    
    verbose: bool = False
    """详细模式 - 打印 input_ids 和 decoded tokens"""
    
    # 训练数据测试模式
    use_training_data: bool = False
    """使用训练数据进行测试 (而非 LIBERO 环境)"""
    
    num_test_samples: int = 100
    """测试样本数量 (use_training_data=True 时有效)"""
    
    num_episode_videos: int = 5
    """保存完整 episode 视频的数量"""
    
    # 模型维度
    max_state_dim: int = 64
    max_action_dim: int = 32
    action_horizon: int = 16


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
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "utils"))
    from lerobot_dataset_loader import LeRobotDataset
    
    print(f"\n{'='*60}")
    print("训练数据测试模式 (Training Data Evaluation)")
    print(f"{'='*60}")
    print(f"数据集路径: {cfg.data_path}")
    print(f"测试样本数: {cfg.num_test_samples}")
    print(f"{'='*60}\n")
    
    if not cfg.data_path:
        print("❌ 错误: 必须提供 --data_path 参数!")
        return
    
    # 加载数据集
    print("📂 加载训练数据集...")
    dataset = LeRobotDataset(
        dataset_path=cfg.data_path,
        split="train",
        num_action_chunks=cfg.action_horizon,
        enable_chunking=True,
        verbose=True,
    )
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
    # 加载归一化统计量
    normalizers = load_normalization_stats(cfg.data_path, convert_quat_to_axisangle=True)
    
    # 创建 Qwen VLM backbone
    print(f"\n📦 加载 Qwen VLM: {cfg.vlm_model_path}")
    vlm_backbone = QwenVLMBackbone(
        model_path=cfg.vlm_model_path,
        device=cfg.device,
        layer=cfg.vlm_layer,
        add_action_prompt=cfg.add_action_prompt,
        verbose=cfg.verbose,
    )
    
    # 加载 action head
    print(f"\n📦 加载 Action Head: {cfg.checkpoint_path}")
    from models.action_head.flow_matching_action_head import FlowmatchingActionHead
    from config import get_flowmatching_action_head_config_original
    
    # 创建配置
    action_head_cfg = get_flowmatching_action_head_config_original(
        backbone_dim=cfg.backbone_dim,
        action_dim=cfg.max_action_dim,
        action_horizon=cfg.action_horizon,
        max_state_dim=cfg.max_state_dim,
        max_action_dim=cfg.max_action_dim,
    )
    
    action_head = FlowmatchingActionHead(action_head_cfg)
    
    checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
    if "action_head" in checkpoint:
        action_head.load_state_dict(checkpoint["action_head"])
    elif "model_state_dict" in checkpoint:
        action_head.load_state_dict(checkpoint["model_state_dict"])
    else:
        action_head.load_state_dict(checkpoint)
    
    action_head = action_head.to(cfg.device)
    action_head.eval()
    print("✅ Action Head 加载完成")
    
    # 测试
    num_samples = min(cfg.num_test_samples, len(dataset))
    print(f"\n🔍 开始测试 {num_samples} 个样本...")
    
    total_action_mse = 0.0
    total_samples = 0
    action_errors = []
    
    # 用于保存视频的数据
    video_images_list = []
    video_gt_actions_list = []
    video_pred_actions_list = []
    video_mse_list = []
    video_task_descriptions = []
    video_sample_indices = []
    max_video_samples = min(50, num_samples)  # 最多保存 50 个样本的可视化
    
    # 随机采样或顺序采样
    if num_samples < len(dataset):
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    else:
        indices = range(len(dataset))
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # 获取 GT action (从数据集) - 注意键名是 'actions' (复数)
        gt_action = sample.get('actions', sample.get('action', None))
        if gt_action is None:
            print(f"⚠️ 样本 {idx} 无 action，跳过")
            continue
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.numpy()
        
        # 获取图像 - 数据集返回 sample['images'] 是一个字典
        images = []
        sample_images = sample.get('images', {})
        if isinstance(sample_images, dict):
            # 按照 image_keys 顺序获取图像
            for img_key in dataset.image_keys:
                if img_key in sample_images:
                    img = sample_images[img_key]
                    if isinstance(img, torch.Tensor):
                        img = img.numpy()
                    # 确保 uint8 和 (H, W, C) 格式
                    if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
                        img = np.transpose(img, (1, 2, 0))  # -> (H, W, C)
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    images.append(img)
        else:
            # 兼容旧格式
            for img_key in dataset.image_keys:
                if img_key in sample:
                    img = sample[img_key]
                    if isinstance(img, torch.Tensor):
                        img = img.numpy()
                    if img.ndim == 4:  # (T, C, H, W)
                        img = img[0]
                    if img.ndim == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
                        img = np.transpose(img, (1, 2, 0))
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    images.append(img)
        
        if len(images) < 2:
            print(f"⚠️ 样本 {idx} 图像不足 ({len(images)} 个)，跳过")
            continue
        
        # 获取 state - 注意键名是 'observation_state'
        state = sample.get('observation_state', sample.get('state', sample.get('observation.state', None)))
        if state is None:
            print(f"⚠️ 样本 {idx} 无 state，跳过")
            continue
        if isinstance(state, torch.Tensor):
            state = state.numpy()
        
        # state 可能是 (T, dim) 或 (dim,)
        if state.ndim == 2:
            state = state[0]  # 取第一帧
        
        # 转换 state 格式 (如果是 9 维四元数格式，转为 8 维轴角格式)
        if len(state) == 9:
            # [x, y, z, qx, qy, qz, qw, g1, g2] -> [x, y, z, ax, ay, az, g1, g2]
            pos = state[:3]
            quat = state[3:7]
            gripper = state[7:9]
            axisangle = quat2axisangle(quat)
            state = np.concatenate([pos, axisangle, gripper])
        
        # 获取语言描述 - 注意键名是 'task_description'
        lang = sample.get('task_description', 
                         sample.get('annotation.human.action.task_description', 
                         sample.get('language', 'Perform the task')))
        if isinstance(lang, list):
            lang = lang[0]
        if lang is None:
            lang = 'Perform the task'
        
        # 使用 VLM 提取特征
        with torch.no_grad():
            backbone_features = vlm_backbone.get_hidden_states(images[:2], lang)
        
        # 准备 state tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(cfg.device)
        
        # 归一化 state
        if normalizers is not None and 'state' in normalizers:
            state_tensor = normalizers['state'].normalize(state_tensor)
        
        # Padding state
        if state_tensor.shape[1] < cfg.max_state_dim:
            padding = torch.zeros(1, cfg.max_state_dim - state_tensor.shape[1], 
                                device=cfg.device, dtype=state_tensor.dtype)
            state_tensor = torch.cat([state_tensor, padding], dim=1)
        
        state_tensor = state_tensor.unsqueeze(1)  # (1, 1, max_state_dim)
        
        # 模型推理
        with torch.no_grad():
            backbone_attention_mask = torch.ones(
                1, backbone_features.shape[1],
                dtype=torch.long, device=cfg.device
            )
            embodiment_id = torch.full((1,), DEFAULT_EMBODIMENT_ID, dtype=torch.long, device=cfg.device)
            
            backbone_output = BatchFeature(data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": backbone_attention_mask,
            })
            
            action_input = BatchFeature(data={
                "state": state_tensor,
                "embodiment_id": embodiment_id,
            })
            
            # 临时设置推理步数
            original_num_inference_timesteps = action_head.num_inference_timesteps
            action_head.num_inference_timesteps = cfg.denoising_steps
            
            # 预测
            output = action_head.get_action(backbone_output, action_input)
            
            # 恢复原始推理步数
            action_head.num_inference_timesteps = original_num_inference_timesteps
        
        # 从 BatchFeature 中提取 action_pred
        pred_action = output["action_pred"]  # (1, action_horizon, max_action_dim)
        pred_action = pred_action[0].cpu().numpy()  # (action_horizon, max_action_dim)
        
        # 只取有效的 7 维 action (或根据实际 action_dim)
        actual_action_dim = gt_action.shape[-1] if gt_action.ndim > 1 else len(gt_action)
        
        # 反归一化预测的 action
        if normalizers is not None and 'action' in normalizers:
            pred_action_valid = pred_action[:, :actual_action_dim]
            pred_action_valid = normalizers['action'].denormalize_numpy(pred_action_valid)
        else:
            pred_action_valid = pred_action[:, :actual_action_dim]
        
        # 计算误差
        if gt_action.ndim == 1:
            gt_action = gt_action.reshape(1, -1)
        
        min_horizon = min(pred_action_valid.shape[0], gt_action.shape[0])
        mse = np.mean((pred_action_valid[:min_horizon] - gt_action[:min_horizon]) ** 2)
        
        total_action_mse += mse
        action_errors.append(mse)
        total_samples += 1
        
        # 保存用于视频的数据
        if len(video_images_list) < max_video_samples:
            video_images_list.append(images.copy())
            video_gt_actions_list.append(gt_action.copy())
            video_pred_actions_list.append(pred_action_valid.copy())
            video_mse_list.append(mse)
            video_task_descriptions.append(lang)
            video_sample_indices.append(idx)
        
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{num_samples}] MSE: {mse:.6f}")
        
        if cfg.verbose and i < 5:
            print(f"\n  Sample {idx} details:")
            print(f"    GT action (first step): {gt_action[0]}")
            print(f"    Pred action (first step): {pred_action_valid[0]}")
            print(f"    Diff: {np.abs(gt_action[0] - pred_action_valid[0])}")
    
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
    
    # RMSE (更直观)
    avg_rmse = np.sqrt(avg_mse)
    print(f"\n平均 RMSE: {avg_rmse:.6f}")
    print(f"{'='*60}\n")
    
    # 保存结果
    results = {
        "num_samples": total_samples,
        "avg_mse": float(avg_mse),
        "avg_rmse": float(avg_rmse),
        "std_mse": float(np.std(action_errors)),
        "min_mse": float(np.min(action_errors)),
        "max_mse": float(np.max(action_errors)),
        "median_mse": float(np.median(action_errors)),
    }
    
    result_path = Path(cfg.video_dir) / "training_data_eval_results.json"
    os.makedirs(cfg.video_dir, exist_ok=True)
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 结果已保存到: {result_path}")
    
    # 保存对比可视化视频/图像
    if len(video_images_list) > 0:
        print(f"\n📹 保存对比可视化 ({len(video_images_list)} 个样本)...")
        save_training_data_comparison_video(
            images_list=video_images_list,
            gt_actions_list=video_gt_actions_list,
            pred_actions_list=video_pred_actions_list,
            mse_list=video_mse_list,
            task_descriptions=video_task_descriptions,
            sample_indices=video_sample_indices,
            video_dir=cfg.video_dir,
            fps=1,  # 每秒1帧，方便查看
        )
    else:
        print("⚠️ 没有收集到用于视频的数据")
    
    # 保存完整 episode 视频
    print(f"\n🎬 保存完整 Episode 视频...")
    
    # 获取测试样本涉及的 unique episodes
    tested_episodes = set()
    for sample_idx in video_sample_indices:
        sample = dataset[sample_idx]
        ep_idx = sample.get('episode_index', -1)
        if ep_idx >= 0:
            tested_episodes.add(ep_idx)
    
    # 限制保存的 episode 数量
    max_episode_videos = min(cfg.num_episode_videos, len(tested_episodes))
    episodes_to_save = list(tested_episodes)[:max_episode_videos]
    
    print(f"  将保存 {len(episodes_to_save)} 个 episode 视频")
    
    for ep_idx in episodes_to_save:
        try:
            save_episode_video_with_predictions(
                dataset=dataset,
                episode_idx=ep_idx,
                vlm_backbone=vlm_backbone,
                action_head=action_head,
                normalizers=normalizers,
                cfg=cfg,
                video_dir=cfg.video_dir,
                fps=10,  # 10 FPS
            )
        except Exception as e:
            print(f"    ⚠️ Episode {ep_idx} 视频保存失败: {e}")
    
    print(f"\n✅ 训练数据测试完成!")


# ============================================================================
# 主评估函数
# ============================================================================

def eval_libero(cfg: EvalConfig):
    """
    LIBERO 评估主函数
    """
    if not LIBERO_AVAILABLE:
        print("Error: LIBERO is not installed!")
        print("Please install: pip install robosuite==1.4.0")
        return
    
    # 设置日志目录 (使用 cfg.log_dir，如果为空则使用 video_dir)
    log_dir = cfg.log_dir if cfg.log_dir else cfg.video_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.video_dir, exist_ok=True)
    
    # 加载归一化统计量
    normalizers = None
    if cfg.data_path:
        print(f"\n加载归一化统计量 from: {cfg.data_path}")
        normalizers = load_normalization_stats(cfg.data_path, convert_quat_to_axisangle=True)
    else:
        print("⚠️ 未提供 data_path，将不使用归一化")
    
    # 创建 Policy
    if cfg.use_simple_policy:
        print("\n使用简化版 Policy (无 VLM)")
        policy = FlowMatchingPolicySimple(
            checkpoint_path=cfg.checkpoint_path,
            normalizers=normalizers,
            device=cfg.device,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            action_horizon=cfg.action_horizon,
            denoising_steps=cfg.denoising_steps,
            action_chunk_size=cfg.action_chunk_size,
            flip_images=cfg.flip_images,
            headless=cfg.headless,
        )
    else:
        # 加载 Qwen VLM
        print(f"\n加载 Qwen VLM: {cfg.vlm_model_path}")
        print(f"  - 提取层: {cfg.vlm_layer}")
        if cfg.verbose:
            print(f"  - Verbose mode: ENABLED")
        vlm_backbone = QwenVLMBackbone(
            model_path=cfg.vlm_model_path,
            device=cfg.device,
            layer=cfg.vlm_layer,
            add_action_prompt=cfg.add_action_prompt,
            verbose=cfg.verbose,
        )
        
        policy = FlowMatchingPolicy(
            checkpoint_path=cfg.checkpoint_path,
            vlm_backbone=vlm_backbone,
            normalizers=normalizers,
            device=cfg.device,
            max_state_dim=cfg.max_state_dim,
            max_action_dim=cfg.max_action_dim,
            action_horizon=cfg.action_horizon,
            denoising_steps=cfg.denoising_steps,
            action_chunk_size=cfg.action_chunk_size,
            flip_images=cfg.flip_images,
            headless=cfg.headless,
            backbone_dim=cfg.backbone_dim,
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
    print(f"{'='*60}\n")
    
    # 确定要评估的任务
    if cfg.task_ids is not None:
        task_id_list = [int(x.strip()) for x in cfg.task_ids.split(",")]
    else:
        task_id_list = list(range(num_tasks_in_suite))
        if cfg.max_tasks > 0:
            task_id_list = task_id_list[:cfg.max_tasks]
    
    # 打印任务列表
    print("Tasks to evaluate:")
    for tid in task_id_list:
        task = task_suite.get_task(tid)
        print(f"  Task {tid}: {task.language}")
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
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)
        
        task_episodes, task_successes = 0, 0
        
        for episode_idx in range(cfg.num_trials_per_task):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            
            # 重置环境
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])
            policy.reset_action_queue()
            
            # 设置最大步数
            max_steps_dict = {
                "libero_spatial": 220,
                "libero_object": 280,
                "libero_goal": 600,
                "libero_10": 1000,
                "libero_90": 400,
            }
            max_steps = max_steps_dict.get(cfg.task_suite_name, 300)
            
            # Rollout
            t = 0
            top_view, wrist_view = [], []
            done = False
            
            while t < max_steps + cfg.num_steps_wait:
                try:
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue
                    
                    img, wrist_img = get_libero_image(obs, flip=cfg.flip_images)
                    top_view.append(img)
                    wrist_view.append(wrist_img)
                    
                    action = policy.get_action(obs, task.language)
                    obs, reward, done, info = env.step(action.tolist())
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    t += 1
                    
                except Exception as e:
                    import traceback
                    error_msg = f"Error: {e}\n{traceback.format_exc()}"
                    print(error_msg)
                    log_file.write(f"{error_msg}\n")
                    break
            
            task_episodes += 1
            total_episodes += 1
            
            # 保存视频
            save_rollout_video(
                top_view, wrist_view,
                total_episodes, success=done,
                task_description=task_description,
                log_file=log_file,
                video_dir=cfg.video_dir,
            )
            
            # 打印进度
            print(f"Success: {done}")
            print(f"Episodes: {total_episodes}, Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Episodes: {total_episodes}, Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)\n")
            log_file.flush()
        
        # 记录任务结果
        task_success_rate = task_successes / task_episodes
        task_results[task_id] = {
            "task_description": task_description,
            "successes": task_successes,
            "episodes": task_episodes,
            "success_rate": task_success_rate,
        }
    
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
    
    return task_results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LIBERO Evaluation with Custom Flow Matching Action Head (Qwen VLM)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例 Examples:
  # 基本用法 (Qwen VLM)
  python eval_libero_image_flip_for_qwen.py \\
    --checkpoint_path ./experiments/checkpoints/best/action_head.pt \\
    --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \\
    --data_path /path/to/training/dataset \\
    --task_suite_name libero_spatial \\
    --vlm_layer 14

  # 使用简化版 Policy (不需要 VLM)
  python eval_libero_image_flip_for_qwen.py \\
    --checkpoint_path ./experiments/checkpoints/best/action_head.pt \\
    --data_path /path/to/training/dataset \\
    --task_suite_name libero_spatial \\
    --use_simple_policy

  # 评估指定任务
  python eval_libero_image_flip_for_qwen.py \\
    --checkpoint_path ./checkpoint.pt \\
    --task_suite_name libero_spatial \\
    --task_ids 0,1,2,3,4

  # 使用训练数据测试
  python eval_libero_image_flip_for_qwen.py \\
    --checkpoint_path ./checkpoint.pt \\
    --data_path /path/to/dataset \\
    --use_training_data \\
    --num_test_samples 100
        """
    )
    
    # 模型相关
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="训练好的 action_head.pt 文件路径")
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="VLM 模型路径 (默认 Qwen/Qwen3-VL-2B-Instruct)")
    parser.add_argument("--data_path", type=str, default="",
                        help="训练数据集路径 (用于加载归一化统计量)")
    
    # VLM 相关参数
    parser.add_argument("--vlm_layer", type=int, default=14,
                        help="提取的 VLM 隐藏层 (1-28, 默认14)")
    parser.add_argument("--add_action_prompt", action="store_true",
                        help="添加 action prompt")
    parser.add_argument("--backbone_dim", type=int, default=1536,
                        help="VLM backbone 输出维度 (Qwen3-VL-2B=1536)")
    
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
    
    # 推理参数
    parser.add_argument("--denoising_steps", type=int, default=4,
                        help="去噪步数")
    parser.add_argument("--action_chunk_size", type=int, default=1,
                        help="每次执行的动作数")
    
    # 系统配置
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="设备")
    parser.add_argument("--headless", action="store_true",
                        help="无头模式 (不显示图像)")
    parser.add_argument("--flip_images", action="store_true", default=True,
                        help="是否翻转图像 (默认翻转)")
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images",
                        help="不翻转图像")
    parser.add_argument("--video_dir", type=str, default="./eval_rollouts",
                        help="视频保存目录")
    parser.add_argument("--log_dir", type=str, default="",
                        help="日志保存目录，为空则使用 video_dir")
    parser.add_argument("--use_simple_policy", action="store_true",
                        help="使用简化版 Policy (不需要 VLM)")
    parser.add_argument("--verbose", action="store_true",
                        help="详细模式 - 打印 input_ids 和 decoded tokens")
    
    # 训练数据测试模式
    parser.add_argument("--use_training_data", action="store_true",
                        help="使用训练数据进行测试 (而非 LIBERO 环境)")
    parser.add_argument("--num_test_samples", type=int, default=100,
                        help="测试样本数量 (use_training_data=True 时有效)")
    parser.add_argument("--num_episode_videos", type=int, default=5,
                        help="保存完整 episode 视频的数量")
    
    # 模型维度
    parser.add_argument("--max_state_dim", type=int, default=64,
                        help="最大状态维度")
    parser.add_argument("--max_action_dim", type=int, default=32,
                        help="最大动作维度")
    parser.add_argument("--action_horizon", type=int, default=16,
                        help="动作预测步数")
    
    args = parser.parse_args()
    
    # 创建配置
    cfg = EvalConfig(
        checkpoint_path=args.checkpoint_path,
        vlm_model_path=args.vlm_model_path,
        data_path=args.data_path,
        vlm_layer=args.vlm_layer,
        add_action_prompt=args.add_action_prompt,
        backbone_dim=args.backbone_dim,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        task_ids=args.task_ids,
        max_tasks=args.max_tasks,
        num_steps_wait=args.num_steps_wait,
        denoising_steps=args.denoising_steps,
        action_chunk_size=args.action_chunk_size,
        device=args.device,
        headless=args.headless,
        flip_images=args.flip_images,
        video_dir=args.video_dir,
        use_simple_policy=args.use_simple_policy,
        verbose=args.verbose,
        use_training_data=args.use_training_data,
        num_test_samples=args.num_test_samples,
        num_episode_videos=args.num_episode_videos,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        action_horizon=args.action_horizon,
    )
    
    # 打印配置
    print(f"\n{'='*60}")
    print("Evaluation Configuration (Qwen VLM)")
    print(f"{'='*60}")
    for key, value in vars(cfg).items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # 根据模式选择评估函数
    if cfg.use_training_data:
        # 使用训练数据进行测试
        eval_with_training_data(cfg)
    else:
        # 使用 LIBERO 环境进行评估
        eval_libero(cfg)


if __name__ == "__main__":
    main()
