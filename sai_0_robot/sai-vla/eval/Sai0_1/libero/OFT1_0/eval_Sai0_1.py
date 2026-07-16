#!/usr/bin/env python3
"""
LIBERO 评估脚本 - 基于 Sai0_1 模块化架构

使用 Sai0_1 模块化结构评估 OFT Action Head 在 LIBERO 环境中的表现。

使用方法:
    # 基本用法 - LIBERO 环境评估
    CUDA_VISIBLE_DEVICES=0 python eval_Sai0_1.py \
        --checkpoint_path ./checkpoints/step_5000/action_head.pt \
        --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
        --dataset_path /path/to/dataset \
        --task_suite_name libero_spatial \
        --vlm_layers 14

    # 多层 VLM (Qwen4B)
    CUDA_VISIBLE_DEVICES=1 python eval_Sai0_1.py \
        --checkpoint_path ./checkpoints/step_5000/action_head.pt \
        --vlm_model_path Qwen/Qwen3-VL-4B-Instruct \
        --dataset_path /path/to/dataset \
        --vlm_layers 16,17,18 \
        --vlm_output_dim 2560 \
        --task_suite_name libero_spatial

    # 使用训练数据测试
    CUDA_VISIBLE_DEVICES=0 python eval_Sai0_1.py \
        --checkpoint_path ./checkpoints/step_5000/action_head.pt \
        --dataset_path /path/to/dataset \
        --use_training_data \
        --num_test_samples 100

参数说明:
    --checkpoint_path: 训练好的 action_head.pt 文件路径
    --vlm_model_path: VLM 模型路径 (Qwen/Qwen3-VL-2B-Instruct 或 Qwen/Qwen3-VL-4B-Instruct)
    --dataset_path: 训练数据集路径 (用于加载归一化统计量)
    --task_suite_name: LIBERO 任务套件名称
    --vlm_layers: 提取的 VLM 隐藏层 (如 "14" 或 "16,17,18")
    --vlm_output_dim: VLM backbone 输出维度 (Qwen2B=1536, Qwen4B=2560)
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
SAI0_ROOT = EVAL_DIR.parents[3]  # eval/Sai0_1/libero/OFT1_0 -> sai0-vla
if str(SAI0_ROOT) not in sys.path:
    sys.path.insert(0, str(SAI0_ROOT))

# 导入 Sai0_1 模块
from VLAs.Sai0_1 import (
    Sai0Config, VLMConfig, ActionHeadConfig, DataConfig, TrainingConfig,
    Sai0Inference,
    load_normalization_stats, quat2axisangle_torch, convert_state_quat_to_axisangle,
    MinMaxNormalizer,
)

# 导入 VLM Backbone
from VLMs.S0_1.backbone import create_vlm_backbone

# LIBERO 环境 - 延迟导入，避免 robosuite 提前初始化 CUDA
LIBERO_AVAILABLE = False
benchmark = None
get_libero_path = None
OffScreenRenderEnv = None


def _fix_robosuite_log_permission():
    """修复 robosuite 日志文件权限问题"""
    import logging
    robosuite_log = "/tmp/robosuite.log"
    
    # 检查是否可以写入
    try:
        with open(robosuite_log, 'a') as f:
            pass
        return  # 可以写入，无需修复
    except PermissionError:
        pass
    
    # 无法写入，使用 monkey patch 修改 FileHandler
    user_log = os.path.expanduser("~/.robosuite/robosuite.log")
    os.makedirs(os.path.dirname(user_log), exist_ok=True)
    
    # Monkey patch logging.FileHandler 来拦截 robosuite 的日志创建
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
    
    # 修复日志权限问题（在导入 robosuite 之前）
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
# 工具函数
# ============================================================================

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


# ============================================================================
# LIBERO 环境工具函数
# ============================================================================

def get_libero_env(task, resolution=256, seed=None):
    """初始化 LIBERO 环境
    
    Args:
        task: LIBERO 任务对象
        resolution: 图像分辨率
        seed: 环境随机种子，None 表示使用随机数
    """
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
    
    # 设置环境随机种子
    if seed is not None:
        env.seed(seed)
    else:
        # 使用随机数作为种子
        random_seed = random.randint(0, 2**31 - 1)
        env.seed(random_seed)
    
    return env, task_description


def get_libero_dummy_action():
    """获取空动作"""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs, resize_to: int = 128, flip: bool = False):
    """
    提取 LIBERO 图像
    
    Args:
        obs: LIBERO 观测字典
        resize_to: 缩放目标尺寸
        flip: 是否翻转图像 (180度)
    
    Returns:
        img, wrist_img: 两张图像
    """
    img = obs["agentview_image"].copy()
    wrist_img = obs["robot0_eye_in_hand_image"].copy()
    
    # 转换数据类型
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if wrist_img.dtype != np.uint8:
        wrist_img = (wrist_img * 255).astype(np.uint8) if wrist_img.max() <= 1.0 else wrist_img.astype(np.uint8)
    
    # 缩放图像
    if resize_to is not None:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        wrist_img = cv2.resize(wrist_img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    
    # 翻转图像 (180度)
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
# Sai0_1 Policy 封装 (使用模块化架构)
# ============================================================================

class Sai0Policy:
    """
    Sai0 Policy 封装
    
    使用 Sai0_1 模块化架构进行 LIBERO 评估
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        vlm_type: str = "qwen3_vl",
        vlm_layers: List[int] = None,
        vlm_output_dim: int = 1536,
        dataset_path: str = None,
        device: str = "cuda:0",
        num_transformer_blocks: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        action_head_hidden_dim: int = 4096,
        action_chunk_size: int = 1,
        num_action_chunks: int = 16,
        action_dim: int = 7,
        flip_images: bool = True,
        # Prompt 配置
        content_order: str = "images_first",
        lowercase_instruction: bool = True,
        add_generation_prompt: bool = True,
        add_action_prompt: bool = True,
        verbose: bool = False,
    ):
        """
        初始化 Sai0 Policy
        
        Args:
            checkpoint_path: Action Head checkpoint 路径
            vlm_model_path: VLM 模型路径
            vlm_type: VLM 类型 ("qwen3_vl" 或 "eagle2_5_vl")
            vlm_layers: 提取的 VLM 隐藏层
            vlm_output_dim: VLM 输出维度
            dataset_path: 数据集路径 (用于加载归一化统计)
            device: 设备
            num_transformer_blocks: Transformer 块数量
            num_attention_heads: 注意力头数量
            dropout: Dropout 比率
            action_head_hidden_dim: Action head 隐藏层维度
            action_chunk_size: 每次执行的动作数
            num_action_chunks: 预测的动作块数量
            action_dim: 动作维度
            flip_images: 是否翻转图像
            content_order: 内容顺序 (images_first, text_first, interleaved, single_image)
            lowercase_instruction: 是否将指令转为小写
            add_generation_prompt: 是否添加 generation prompt
            add_action_prompt: 是否添加 "What action should the robot take to {instruction}?" 前缀
            verbose: 详细输出
        """
        self.device = device
        self.vlm_layers = vlm_layers or [14]
        self.vlm_output_dim = vlm_output_dim
        self.action_chunk_size = action_chunk_size
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim
        self.flip_images = flip_images
        self.content_order = content_order
        self.lowercase_instruction = lowercase_instruction
        self.add_generation_prompt = add_generation_prompt
        self.add_action_prompt = add_action_prompt
        self.verbose = verbose
        
        self.action_queue = []
        
        # ====== 1. 加载 VLM Backbone ======
        print(f"\n📦 加载 VLM Backbone: {vlm_model_path}")
        print(f"  - 类型: {vlm_type}")
        print(f"  - 提取层: {self.vlm_layers}")
        print(f"  - 内容顺序: {content_order}")
        print(f"  - Action Prompt: {add_action_prompt}")
        
        # 注意: 如果 add_action_prompt=True，我们在 get_action() 中手动添加前缀
        # 所以 VLM backbone 应该使用简单模板，避免重复
        # 如果 add_action_prompt=False，VLM backbone 也使用简单模板，只传任务描述
        # lowercase_instruction=False: 我们在 get_action() 中已经手动处理了小写转换
        # flip_images: 由 VLM backbone 统一处理图像翻转
        self.vlm_backbone = create_vlm_backbone(
            model_type=vlm_type,
            model_path=vlm_model_path,
            device=device,
            layers=self.vlm_layers,
            flip_images=flip_images,  # VLM backbone 统一处理图像翻转
            content_order=content_order,
            prompt_template="simple",  # 使用简单模板 "{instruction}"，避免重复添加前缀
            lowercase_instruction=False,  # 禁用 backbone 的小写转换，我们已经在 get_action() 中处理
            verbose=verbose,
        )
        print(f"✓ VLM Backbone 加载成功!")
        
        # ====== 2. 加载 OFT Action Head ======
        print(f"\n📦 加载 OFT Action Head: {checkpoint_path}")
        
        # 导入 OFT 模型
        from Action_Heads.OFT1_0 import create_vlm2oft_pipeline
        
        # 先加载 checkpoint 检测维度
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # 从 checkpoint 自动检测维度
        if 'proprio_projector.layer_norm.weight' in state_dict:
            detected_dim = state_dict['proprio_projector.layer_norm.weight'].shape[0]
            print(f"  - 从 checkpoint 检测到 VLM output dim: {detected_dim}")
            if detected_dim != vlm_output_dim:
                print(f"  ⚠️ 自动使用 checkpoint 的维度: {detected_dim}")
                vlm_output_dim = detected_dim
        
        self.model = create_vlm2oft_pipeline(
            num_transformer_blocks=num_transformer_blocks,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            action_head_hidden_dim=action_head_hidden_dim,
            num_vlm_layers=len(self.vlm_layers),
            vlm_output_dim=vlm_output_dim,
        ).to(device)
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"✓ OFT Action Head 加载成功!")
        
        # ====== 3. 加载归一化统计量 ======
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
        img, wrist_img = get_libero_image(observation_dict, flip=False)  # 不翻转，VLM backbone 内部会根据配置翻转
        
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
        
        # 准备 state - [gripper1, gripper2, x, y, z, ax, ay, az] (8维)
        state = np.array([
            gripper[0], gripper[1], 
            xyz[0], xyz[1], xyz[2], 
            rpy[0], rpy[1], rpy[2]
        ], dtype=np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 8)
        
        # 归一化 state
        if self.normalizers and 'state' in self.normalizers:
            state_tensor = self.normalizers['state'].normalize(state_tensor)
        
        # 模型推理
        with torch.no_grad():
            vlm_hidden_states_device = [v.to(self.device) for v in vlm_hidden_states]
            pred_actions = self.model(vlm_hidden_states_device, state_tensor)
            # pred_actions: (1, 1, num_action_chunks * action_dim)
        
        # 解析预测的动作
        pred_actions = pred_actions[0, 0].cpu().numpy()  # (num_action_chunks * action_dim,)
        pred_actions = pred_actions.reshape(self.num_action_chunks, self.action_dim)
        
        # 反归一化
        if self.normalizers and 'action' in self.normalizers:
            pred_actions_tensor = torch.from_numpy(pred_actions)
            pred_actions = self.normalizers['action'].denormalize(pred_actions_tensor).numpy()
        
        # 将动作加入队列
        for idx in range(min(self.action_chunk_size, self.num_action_chunks)):
            action = pred_actions[idx].astype(np.float32)
            # 二值化 gripper
            # action[-1] = 1.0 if action[-1] > 0 else -1.0
            action[-1] = np.sign(action[-1]) # ! 二值化 gripper
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
    vlm_type: str = "qwen3_vl"
    dataset_path: str = ""
    
    # VLM 相关
    vlm_layers: List[int] = field(default_factory=lambda: [14])
    vlm_output_dim: int = 1536
    
    # Prompt 配置
    content_order: str = "images_first"  # images_first, text_first, interleaved, single_image
    lowercase_instruction: bool = True  # 是否将指令转为小写
    add_generation_prompt: bool = True  # 是否添加 generation prompt
    add_action_prompt: bool = True  # 是否添加 "What action should the robot take to {instruction}?" 前缀
    
    # OFT 模型参数
    num_transformer_blocks: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.1
    action_head_hidden_dim: int = 4096
    num_action_chunks: int = 16
    action_dim: int = 7
    
    # LIBERO 环境
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 5
    task_ids: Optional[List[int]] = None
    max_tasks: int = -1
    num_steps_wait: int = 10
    max_steps: int = 600
    env_seed: Optional[int] = None  # 环境随机种子，None 表示使用随机数
    
    # 任务指令替换配置
    # 格式: {task_id: new_instruction}
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
    from utils.lerobot_dataset_loader import LeRobotDataset
    from Action_Heads.OFT1_0 import create_vlm2oft_pipeline
    
    print(f"\n{'='*60}")
    print("训练数据测试模式 (Training Data Evaluation)")
    print(f"{'='*60}")
    print(f"数据集路径: {cfg.dataset_path}")
    print(f"测试样本数: {cfg.num_test_samples}")
    print(f"{'='*60}\n")
    
    if not cfg.dataset_path:
        raise ValueError("必须指定 dataset_path")
    
    # 加载数据集
    print("📂 加载训练数据集...")
    dataset = LeRobotDataset(
        dataset_path=cfg.dataset_path,
        num_action_chunks=cfg.num_action_chunks,
        enable_chunking=True,
        verbose=True,
    )
    print(f"✅ 数据集加载完成: {len(dataset)} 个样本")
    
    # 加载归一化统计量
    normalizers = load_normalization_stats(cfg.dataset_path, convert_quat_to_axisangle=True)
    
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
        
        # 获取数据
        vlm_tensor_raw = torch.from_numpy(sample['vlm_hidden_states']).float()
        observation_state = torch.from_numpy(sample['observation_state']).float()
        actions = torch.from_numpy(sample['actions']).float()
        
        # 转换为 batch 格式
        vlm_tensor = vlm_tensor_raw.unsqueeze(0)
        observation_state = observation_state.unsqueeze(0)
        actions = actions.unsqueeze(0)
        
        batch_size = 1
        num_layers = vlm_tensor.size(1)
        num_chunks = actions.size(1)
        actual_action_dim = actions.size(2)
        
        # State 四元数转轴角
        original_state_dim = observation_state.size(1)
        if original_state_dim == 9:
            observation_state = convert_state_quat_to_axisangle(observation_state)
        
        # 归一化
        if normalizers:
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
    print(f"平均 RMSE: {np.sqrt(avg_mse):.6f}")
    print(f"{'='*60}\n")
    
    # 保存结果
    results = {
        "num_samples": total_samples,
        "avg_mse": float(avg_mse),
        "avg_rmse": float(np.sqrt(avg_mse)),
        "std_mse": float(np.std(action_errors)),
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
    policy = Sai0Policy(
        checkpoint_path=cfg.checkpoint_path,
        vlm_model_path=cfg.vlm_model_path,
        vlm_type=cfg.vlm_type,
        vlm_layers=cfg.vlm_layers,
        vlm_output_dim=cfg.vlm_output_dim,
        dataset_path=cfg.dataset_path,
        device=cfg.device,
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_attention_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
        action_head_hidden_dim=cfg.action_head_hidden_dim,
        action_chunk_size=cfg.num_action_chunks if cfg.execute_all_chunks else cfg.action_chunk_size,
        num_action_chunks=cfg.num_action_chunks,
        action_dim=cfg.action_dim,
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
    print(f"LIBERO 评估 (Sai0_1 架构)")
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
        # 检查是否有指令替换
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
    log_file.write(f"Checkpoint: {cfg.checkpoint_path}\n")
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
            trial_start_time = time.time()  # 记录 trial 开始时间
            
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
                
                # 获取图像（用于保存视频，根据 flip_images 参数决定是否翻转）
                img, wrist_img = get_libero_image(obs, flip=cfg.flip_images)
                top_view.append(img)
                wrist_view.append(wrist_img)
                
                # 获取动作（记录推理时间）
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
            
            trial_time = time.time() - trial_start_time  # 计算 trial 总时间
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
            "checkpoint": cfg.checkpoint_path,
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
        description='LIBERO Evaluation with Sai0_1 (OFT Action Head)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 模型相关
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="训练好的 action_head.pt 文件路径")
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="VLM 模型路径")
    parser.add_argument("--vlm_type", type=str, default="qwen3_vl",
                        choices=["qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"],
                        help="VLM 类型")
    parser.add_argument("--dataset_path", type=str, default="",
                        help="训练数据集路径 (用于加载归一化统计量)")
    
    # VLM 相关参数
    parser.add_argument("--vlm_layers", type=str, default="14",
                        help="提取的 VLM 隐藏层 (如 '14' 或 '16,17,18')")
    parser.add_argument("--vlm_output_dim", type=int, default=1536,
                        help="VLM backbone 输出维度")
    
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
    
    # OFT 模型参数
    parser.add_argument("--num_transformer_blocks", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action_head_hidden_dim", type=int, default=4096)
    parser.add_argument("--num_action_chunks", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    
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
                             "'task_id1:instruction1|task_id2:instruction2' 用于多个替换。"
                             "例如: '1:put the bowl on the white square plate'")
    
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
    # 格式: "task_id:instruction" 或 "task_id1:instruction1|task_id2:instruction2"
    task_instruction_override = None
    if args.task_instruction_override:
        task_instruction_override = {}
        # 按 | 分割多个替换
        overrides = args.task_instruction_override.split("|")
        for override in overrides:
            override = override.strip()
            if ":" in override:
                # 只分割第一个 : ，因为指令中可能包含 :
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
        checkpoint_path=args.checkpoint_path,
        vlm_model_path=args.vlm_model_path,
        vlm_type=args.vlm_type,
        dataset_path=args.dataset_path,
        vlm_layers=vlm_layers,
        vlm_output_dim=args.vlm_output_dim,
        # Prompt 配置
        content_order=args.content_order,
        lowercase_instruction=args.lowercase_instruction,
        add_generation_prompt=args.add_generation_prompt,
        add_action_prompt=args.add_action_prompt,
        # OFT 模型参数
        num_transformer_blocks=args.num_transformer_blocks,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        action_head_hidden_dim=args.action_head_hidden_dim,
        num_action_chunks=args.num_action_chunks,
        action_dim=args.action_dim,
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

