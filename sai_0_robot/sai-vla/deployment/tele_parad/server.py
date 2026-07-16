"""
OFT 遥操作推理服务器

基于 FastAPI 的高性能推理服务，支持：
- VLM Backbone (Qwen/Eagle) 实时 hidden state 提取
- VLM2OFT Pipeline (TransformerBlocks + L1RegressionActionHead)
- 连续动作预测

使用方法:
    python server.py --config config.yaml

API 端点:
    POST /predict - 单次预测
    GET /health - 健康检查
    GET /info - 模型信息
"""

import os
import sys
import time
import math
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import base64
from io import BytesIO

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yaml
import argparse

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from VLMs.S0_1.backbone import create_vlm_backbone
from Action_Heads.OFT1_0.vlm2oft_pipeline import create_vlm2oft_pipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== State 预处理函数 ====================

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
    """按配置顺序应用 state 预处理"""
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
                            [1.0 if np.mean(hand_data) > hand_binary_threshold else -1.0], 
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
            # 简化实现，具体转换逻辑省略
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
            # Gripper 二值化: > threshold 为 1, <= threshold 为 -1
            for col in gripper_binarize_columns:
                if col < len(observation_state):
                    if observation_state[col] > gripper_binarize_threshold:
                        observation_state[col] = 1
                    else:
                        observation_state[col] = -1
    
    return observation_state


# ==================== 图像预处理 ====================

def preprocess_image(
    image: Image.Image,
    resize: Optional[List[int]] = None,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotate_180: bool = False,
) -> Image.Image:
    """
    预处理图像
    
    Args:
        image: PIL 图像
        resize: [width, height] 或 None
        flip_horizontal: 水平翻转
        flip_vertical: 垂直翻转
        rotate_180: 旋转 180 度
        
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


# ==================== 请求/响应模型 ====================

class PredictRequest(BaseModel):
    """预测请求"""
    # 图像输入 (三选一)
    images: Optional[List[str]] = Field(None, description="Base64 编码的图像列表")
    image_paths: Optional[List[str]] = Field(None, description="图像本地路径列表")
    image_arrays: Optional[List[List[List[List[float]]]]] = Field(None, description="图像数组列表 [N, H, W, C]")
    state: Optional[List[float]] = Field(None, description="机器人状态向量 (可选)")


class TimingInfo(BaseModel):
    """时间统计"""
    vlm_time: float = Field(..., description="VLM 推理时间 (秒)")
    oft_time: float = Field(..., description="OFT Pipeline 推理时间 (秒)")
    total_time: float = Field(..., description="总时间 (秒)")


class PredictResponse(BaseModel):
    """预测响应"""
    actions: List[List[float]] = Field(..., description="预测的动作序列 [chunk_size, action_dim]")
    timing: TimingInfo
    chunk_size: int
    action_dim: int


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    models_loaded: Dict[str, bool]
    device: str


class InfoResponse(BaseModel):
    """模型信息响应"""
    vlm_type: str
    vlm_model_path: str
    chunk_size: int
    action_dim: int
    device: str
    config: Dict[str, Any]


# ==================== 推理引擎 ====================

class OFTInferenceEngine:
    """OFT 推理引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda:0')
        
        # 模型组件
        self.vlm_backbone = None
        self.oft_pipeline = None
        
        # 配置
        self.chunk_size = config.get('chunk_size', 50)
        self.action_dim = config.get('action_dim', 14)
        self.proprio_dim = config.get('proprio_dim', 14)
        
        # State 索引提取配置 (从原始 state 中提取指定索引组成新的 state)
        self.state_select_indices = config.get('state_select_indices', None)
        
        # State 预处理配置
        self.state_process_order = config.get('state_process_order', [])
        self.hand_binary_columns = config.get('hand_binary_columns', [])
        self.hand_binary_threshold = config.get('hand_binary_threshold', 442.0)
        self.state_euler_to_axisangle_columns = config.get('state_euler_to_axisangle_columns', [])
        self.state_dim = config.get('state_dim', 14)
        
        # 验证: 提取索引数量必须等于 state_dim
        if self.state_select_indices is not None:
            if len(self.state_select_indices) != self.state_dim:
                raise ValueError(
                    f"state_select_indices 长度 ({len(self.state_select_indices)}) "
                    f"与 state_dim ({self.state_dim}) 不一致"
                )
            logger.info(f"✓ State 索引提取配置")
            logger.info(f"   提取索引: {self.state_select_indices}")
            logger.info(f"   提取后维度: {len(self.state_select_indices)} (应等于 state_dim={self.state_dim})")
        
        # State MinMax 归一化配置
        state_normalize_config = config.get('state_normalize', {})
        self.minmax_columns = state_normalize_config.get('columns', [])
        self.minmax_min = None
        self.minmax_max = None
        
        # 从 stats.json 加载 min/max 值
        stats_file = state_normalize_config.get('stats_file', '')
        stats_key = state_normalize_config.get('stats_key', 'observation.state')
        
        if stats_file and os.path.exists(stats_file) and self.minmax_columns:
            try:
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)
                
                if stats_key in stats_data:
                    state_stats = stats_data[stats_key]
                    if 'min' in state_stats and 'max' in state_stats:
                        self.minmax_min = np.array(state_stats['min'], dtype=np.float32)
                        self.minmax_max = np.array(state_stats['max'], dtype=np.float32)
                        logger.info(f"✓ 加载 stats.json: {stats_file}")
                        logger.info(f"   Key: {stats_key}")
                        logger.info(f"   归一化列: {self.minmax_columns}")
                        logger.info(f"   Min: [{', '.join([f'{v:.4f}' for v in self.minmax_min[:6]])}{'...' if len(self.minmax_min) > 6 else ''}]")
                        logger.info(f"   Max: [{', '.join([f'{v:.4f}' for v in self.minmax_max[:6]])}{'...' if len(self.minmax_max) > 6 else ''}]")
                    else:
                        logger.warning(f"stats.json 中 {stats_key} 缺少 min/max 字段")
                else:
                    logger.warning(f"stats.json 中找不到 key: {stats_key}")
            except Exception as e:
                logger.error(f"加载 stats.json 失败: {e}")
        
        # State Gripper 二值化配置
        gripper_binarize_config = config.get('state_gripper_binarize', {})
        self.gripper_binarize_columns = gripper_binarize_config.get('columns', [])
        self.gripper_binarize_threshold = gripper_binarize_config.get('threshold', 0.5)
        if self.gripper_binarize_columns:
            logger.info(f"✓ State Gripper 二值化配置")
            logger.info(f"   列: {self.gripper_binarize_columns}")
            logger.info(f"   阈值: {self.gripper_binarize_threshold}")
        
        # 图像预处理配置
        image_preprocess = config.get('image_preprocess', {})
        self.image_resize = image_preprocess.get('resize', None)
        self.flip_horizontal = image_preprocess.get('flip_horizontal', False)
        self.flip_vertical = image_preprocess.get('flip_vertical', False)
        self.rotate_180 = image_preprocess.get('rotate_180', False)
        self.bgr_to_rgb = image_preprocess.get('bgr_to_rgb', False)
        
        # 输出动作后处理配置
        action_postprocess = config.get('action_postprocess', {})
        # Action 反归一化：与训练时一致，将模型输出的 [-1,1] 还原为物理量纲
        action_denorm_config = action_postprocess.get('action_denormalize', {})
        self.action_denormalize_enabled = action_denorm_config.get('enabled', False)
        self.action_min = None
        self.action_max = None
        if self.action_denormalize_enabled:
            denorm_stats_file = action_denorm_config.get('stats_file', '')
            denorm_stats_key = action_denorm_config.get('stats_key', 'action')
            if denorm_stats_file and os.path.exists(denorm_stats_file):
                try:
                    with open(denorm_stats_file, 'r') as f:
                        stats_data = json.load(f)
                    if denorm_stats_key in stats_data:
                        action_stats = stats_data[denorm_stats_key]
                        if 'min' in action_stats and 'max' in action_stats:
                            self.action_min = np.array(action_stats['min'], dtype=np.float32)
                            self.action_max = np.array(action_stats['max'], dtype=np.float32)
                            logger.info(f"✓ Action 反归一化: 已加载 {denorm_stats_file} key={denorm_stats_key}")
                            logger.info(f"   Action 维度: {len(self.action_min)}, 前3维 min: {self.action_min[:3].tolist()}, max: {self.action_max[:3].tolist()}")
                        else:
                            logger.warning(f"stats.json 中 {denorm_stats_key} 缺少 min/max，关闭 action 反归一化")
                            self.action_denormalize_enabled = False
                    else:
                        logger.warning(f"stats.json 中找不到 key: {denorm_stats_key}，关闭 action 反归一化")
                        self.action_denormalize_enabled = False
                except Exception as e:
                    logger.error(f"加载 action 反归一化 stats 失败: {e}")
                    self.action_denormalize_enabled = False
            else:
                logger.warning(f"Action 反归一化 stats_file 不存在或未配置: {denorm_stats_file}")
                self.action_denormalize_enabled = False
        if not self.action_denormalize_enabled:
            logger.info(f"  Action 反归一化: 未启用，输出为模型原始 [-1,1] 空间")
        gripper_binarize_output = action_postprocess.get('gripper_binarize', {})
        self.action_gripper_binarize_enabled = gripper_binarize_output.get('enabled', False)
        self.action_gripper_binarize_columns = gripper_binarize_output.get('columns', [])
        self.action_gripper_binarize_threshold = gripper_binarize_output.get('threshold', 0.0)
        # BCE 模式: 模型输出为 logit，需先 sigmoid 再阈值判断
        self.action_gripper_bce_mode = gripper_binarize_output.get('bce_mode', False)
        if self.action_gripper_binarize_enabled and self.action_gripper_binarize_columns:
            logger.info(f"✓ 输出动作 Gripper 二值化配置")
            logger.info(f"   启用: {self.action_gripper_binarize_enabled}")
            logger.info(f"   列: {self.action_gripper_binarize_columns}")
            logger.info(f"   阈值: {self.action_gripper_binarize_threshold}")
            if self.action_gripper_bce_mode:
                logger.info(f"   BCE 模式: 启用 (logit → sigmoid → 阈值 → -1/1)")
            else:
                logger.info(f"   规则: > {self.action_gripper_binarize_threshold} 变为 1, <= {self.action_gripper_binarize_threshold} 变为 0")
        
        # 零填充配置
        padding_zeros_config = action_postprocess.get('padding_zeros', {})
        self.padding_zeros_enabled = padding_zeros_config.get('enabled', False)
        self.padding_zeros_prepend_count = padding_zeros_config.get('prepend_count', 0)
        self.padding_zeros_insert_before_last = padding_zeros_config.get('insert_before_last', 0)
        if self.padding_zeros_enabled:
            logger.info(f"✓ 输出动作零填充配置")
            logger.info(f"   启用: {self.padding_zeros_enabled}")
            logger.info(f"   开头插入零个数: {self.padding_zeros_prepend_count}")
            logger.info(f"   最后元素前插入零个数: {self.padding_zeros_insert_before_last}")
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载所有模型组件"""
        config = self.config
        
        # ====== 1. 加载 VLM Backbone ======
        vlm_config = config.get('vlm', {})
        logger.info(f"加载 VLM Backbone: {vlm_config.get('model_path')}")
        logger.info("  (进度条 100% 后可能还需 1~2 分钟：Materializing / 搬移到 GPU，属正常)")
        
        self.vlm_backbone = create_vlm_backbone(
            model_type=vlm_config.get('type', 'eagle2_5_vl'),
            model_path=vlm_config.get('model_path'),
            device=self.device,
            layers=vlm_config.get('layers', [-1]),
            flip_images=vlm_config.get('flip_images', False),
            content_order=vlm_config.get('content_order', 'images_first'),
            prompt_template=vlm_config.get('prompt_template', 'simple'),
            lowercase_instruction=vlm_config.get('lowercase_instruction', True),
            add_generation_prompt=vlm_config.get('add_generation_prompt', True),
        )
        logger.info("✓ VLM Backbone 加载成功")
        logger.info(f"   add_generation_prompt: {vlm_config.get('add_generation_prompt', True)}")
        
        # 获取 hidden_dim
        hidden_dim = config.get('hidden_dim', 2048)
        
        # ====== 2. 加载 OFT Pipeline ======
        oft_config = config.get('oft', {})
        oft_checkpoint = oft_config.get('checkpoint')
        
        if not oft_checkpoint or not os.path.exists(oft_checkpoint):
            raise ValueError(f"OFT checkpoint 不存在: {oft_checkpoint}")
        
        logger.info(f"加载 OFT Pipeline: {oft_checkpoint}")
        
        # 尝试从 config.json 加载配置
        ckpt_dir = Path(oft_checkpoint).parent
        config_path = ckpt_dir / "config.json"
        
        num_transformer_blocks = oft_config.get('num_transformer_blocks', 2)
        num_attention_heads = oft_config.get('num_attention_heads', 8)
        num_vlm_layers = oft_config.get('num_vlm_layers', 1)
        action_head_hidden_dim = oft_config.get('action_head_hidden_dim', 4096)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                ckpt_config = json.load(f)
            
            # 从 checkpoint config 更新参数
            self.chunk_size = ckpt_config.get('chunk_size', self.chunk_size)
            self.action_dim = ckpt_config.get('action_dim', self.action_dim)
            hidden_dim = ckpt_config.get('vlm_output_dim', hidden_dim)
            num_transformer_blocks = ckpt_config.get('num_transformer_blocks', num_transformer_blocks)
            num_attention_heads = ckpt_config.get('num_attention_heads', num_attention_heads)
            num_vlm_layers = ckpt_config.get('num_vlm_layers', num_vlm_layers)
            
            logger.info(f"   从 config.json 加载配置:")
            logger.info(f"     chunk_size: {self.chunk_size}")
            logger.info(f"     action_dim: {self.action_dim}")
            logger.info(f"     hidden_dim: {hidden_dim}")
            logger.info(f"     num_transformer_blocks: {num_transformer_blocks}")
            logger.info(f"     num_attention_heads: {num_attention_heads}")
            logger.info(f"     num_vlm_layers: {num_vlm_layers}")
        
        # 创建 OFT Pipeline
        self.oft_pipeline = create_vlm2oft_pipeline(
            num_transformer_blocks=num_transformer_blocks,
            num_attention_heads=num_attention_heads,
            num_vlm_layers=num_vlm_layers,
            vlm_output_dim=hidden_dim,
            action_head_hidden_dim=action_head_hidden_dim,
        ).to(self.device)
        
        # 加载权重
        oft_state = torch.load(oft_checkpoint, map_location=self.device)
        self.oft_pipeline.load_state_dict(oft_state)
        self.oft_pipeline.eval()
        logger.info(f"✓ OFT Pipeline 加载成功 (chunk={self.chunk_size}, action_dim={self.action_dim})")
        
        logger.info("所有模型加载完成")
    
    def predict(
        self,
        images: List[Image.Image],
        instruction: str = "",
        observation_state: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        执行预测
        
        Args:
            images: PIL 图像列表
            instruction: 任务指令
            observation_state: 观测状态 (原始未预处理)
            verbose: 是否打印详细信息
            
        Returns:
            预测结果字典
        """
        timing = {}
        total_start = time.time()
        
        # 使用 config 中的图像预处理参数
        use_resize = self.image_resize
        use_flip_h = self.flip_horizontal
        use_flip_v = self.flip_vertical
        use_rotate = self.rotate_180
        
        # ========== Verbose: 输入信息 ==========
        if verbose:
            logger.info("=" * 60)
            logger.info("📥 收到推理请求")
            logger.info(f"  📷 图像数量: {len(images)}")
            for i, img in enumerate(images):
                logger.info(f"     图像 {i} (原始): {img.size[0]}x{img.size[1]} ({img.mode})")
            
            # 显示预处理配置
            preprocess_info = []
            if use_resize:
                preprocess_info.append(f"resize={use_resize}")
            if use_flip_h:
                preprocess_info.append("flip_h")
            if use_flip_v:
                preprocess_info.append("flip_v")
            if use_rotate:
                preprocess_info.append("rotate_180")
            if preprocess_info:
                logger.info(f"  🔧 图像预处理: {', '.join(preprocess_info)}")
            
            logger.info(f"  📝 指令: \"{instruction}\"")
            if observation_state is not None:
                logger.info(f"  🎮 State: dim={len(observation_state)}")
                logger.info(f"     值: [{', '.join([f'{v:.4f}' for v in observation_state[:6]])}{'...' if len(observation_state) > 6 else ''}]")
            else:
                logger.info(f"  🎮 State: None")
        
        # ========== 图像预处理 ==========
        processed_images = []
        for img in images:
            processed_img = preprocess_image(
                img,
                resize=use_resize,
                flip_horizontal=use_flip_h,
                flip_vertical=use_flip_v,
                rotate_180=use_rotate,
            )
            processed_images.append(processed_img)
        
        if verbose and (use_resize or use_flip_h or use_flip_v or use_rotate):
            for i, img in enumerate(processed_images):
                logger.info(f"     图像 {i} (处理后): {img.size[0]}x{img.size[1]} ({img.mode})")
        
        # 使用处理后的图像
        images = processed_images
        
        # State 预处理
        processed_state = None
        if observation_state is not None:
            # ========== DEBUG: 打印原始 state ==========
            if verbose:
                logger.info(f"  [DEBUG] 原始 State (接收到的):")
                logger.info(f"     dim={len(observation_state)}")
                logger.info(f"     完整值: {observation_state.tolist()}")
            
            # ========== State 索引提取: 从原始 state 中提取指定维度 ==========
            if self.state_select_indices is not None:
                observation_state = observation_state[self.state_select_indices]
                if verbose:
                    logger.info(f"  [DEBUG] 索引提取后 State:")
                    logger.info(f"     indices={self.state_select_indices}")
                    logger.info(f"     dim={len(observation_state)}")
                    logger.info(f"     完整值: {observation_state.tolist()}")
            
            processed_state = apply_state_preprocessing(
                observation_state.copy(),
                state_process_order=self.state_process_order,
                hand_binary_columns=self.hand_binary_columns,
                hand_binary_threshold=self.hand_binary_threshold,
                state_euler_to_axisangle_columns=self.state_euler_to_axisangle_columns,
                # MinMax 归一化参数
                minmax_columns=self.minmax_columns,
                minmax_min=self.minmax_min,
                minmax_max=self.minmax_max,
                # Gripper 二值化参数
                gripper_binarize_columns=self.gripper_binarize_columns,
                gripper_binarize_threshold=self.gripper_binarize_threshold,
            )
            
            # ========== DEBUG: 打印处理后 state ==========
            if verbose:
                logger.info(f"  [DEBUG] 处理后 State (传给模型的):")
                logger.info(f"     dim={len(processed_state)}")
                logger.info(f"     完整值: {processed_state.tolist()}")
                logger.info(f"     归一化列: {self.minmax_columns}")
                logger.info(f"     二值化列: {self.gripper_binarize_columns}")
        
        # VLM 推理
        vlm_start = time.time()
        vlm_output = self.vlm_backbone.get_hidden_states(
            images=images,
            instruction=instruction,
        )
        vlm_hidden_states = vlm_output.hidden_states
        timing['vlm_time'] = time.time() - vlm_start
        
        # ========== Verbose: VLM 输出信息 ==========
        if verbose:
            total_tokens = sum(h.shape[1] for h in vlm_hidden_states)
            logger.info(f"  🧠 VLM 推理完成 ({timing['vlm_time']:.3f}s)")
            logger.info(f"     Hidden states 数量: {len(vlm_hidden_states)}")
            for i, h in enumerate(vlm_hidden_states):
                logger.info(f"     Layer {i}: shape={list(h.shape)} (tokens={h.shape[1]}, dim={h.shape[2]})")
            logger.info(f"     📊 总 Token 数: {total_tokens}")
        
        with torch.no_grad():
            vlm_hidden_states_device = [v.to(self.device) for v in vlm_hidden_states]
            
            # 准备 proprioception
            if processed_state is not None:
                # 取前 proprio_dim 维
                proprio_state = processed_state[:self.proprio_dim]
                proprio_tensor = torch.tensor(
                    proprio_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
            else:
                # 如果没有 state，使用全零
                proprio_tensor = torch.zeros(
                    1, self.proprio_dim, dtype=torch.float32, device=self.device
                )
            
            # OFT Pipeline 推理
            oft_start = time.time()
            action_predictions = self.oft_pipeline(
                vlm_hidden_states_device, 
                proprio_tensor
            )
            timing['oft_time'] = time.time() - oft_start
        
        # 转换为 numpy
        # action_predictions shape: (batch_size, 1, chunk_size * action_dim)
        action_predictions_np = action_predictions[0, 0].cpu().numpy()  # (chunk_size * action_dim,)
        
        # 重塑为 (chunk_size, action_dim)
        continuous_actions = action_predictions_np.reshape(self.chunk_size, self.action_dim)
        
        # ========== 输出动作后处理: Action 反归一化（与训练时对齐） ==========
        if self.action_denormalize_enabled and self.action_min is not None and self.action_max is not None:
            d = min(self.action_dim, len(self.action_min), len(self.action_max))
            # BCE 模式下，BCE 列输出为 logit，不应做反归一化，需要跳过
            bce_skip_cols = set()
            if self.action_gripper_bce_mode and self.action_gripper_binarize_enabled and self.action_gripper_binarize_columns:
                bce_skip_cols = set(self.action_gripper_binarize_columns)
            # 公式与训练时 MinMaxNormalizer.denormalize 一致: (x+1)/2 * (max-min) + min
            denorm_cols = [c for c in range(d) if c not in bce_skip_cols]
            if denorm_cols:
                denorm_cols_arr = np.array(denorm_cols)
                continuous_actions[:, denorm_cols_arr] = (continuous_actions[:, denorm_cols_arr] + 1.0) / 2.0 * (
                    self.action_max[denorm_cols_arr] - self.action_min[denorm_cols_arr]
                ) + self.action_min[denorm_cols_arr]
            if verbose:
                if bce_skip_cols:
                    logger.info(f"  🔧 Action 反归一化已应用 (维度 {d}，跳过 BCE 列 {sorted(bce_skip_cols)})，输出为物理量纲")
                else:
                    logger.info(f"  🔧 Action 反归一化已应用 (维度 {d})，输出为物理量纲")
        
        # ========== 输出动作后处理: Gripper 二值化 ==========
        if self.action_gripper_binarize_enabled and self.action_gripper_binarize_columns:
            for col in self.action_gripper_binarize_columns:
                if col < self.action_dim:
                    if self.action_gripper_bce_mode:
                        # BCE 模式: logit → sigmoid → 阈值 → 1/0
                        prob = 1.0 / (1.0 + np.exp(-continuous_actions[:, col]))
                        continuous_actions[:, col] = np.where(prob > self.action_gripper_binarize_threshold, 1.0, 0.0)
                    else:
                        # 普通模式: > threshold 变为 1, <= threshold 变为 0
                        continuous_actions[:, col] = np.where(
                            continuous_actions[:, col] > self.action_gripper_binarize_threshold, 1.0, 0.0
                        )
            if verbose:
                if self.action_gripper_bce_mode:
                    logger.info(f"  🔧 Gripper 后处理 (BCE): 列 {self.action_gripper_binarize_columns} sigmoid→阈值 {self.action_gripper_binarize_threshold}→-1/1")
                else:
                    logger.info(f"  🔧 Gripper 后处理: 列 {self.action_gripper_binarize_columns} 已二值化 (阈值: {self.action_gripper_binarize_threshold})")
        
        # ========== 输出动作后处理: 零填充 ==========
        if self.padding_zeros_enabled:
            chunk_size, action_dim = continuous_actions.shape
            
            # 在开头插入零
            if self.padding_zeros_prepend_count > 0:
                prepend_zeros = np.zeros((chunk_size, self.padding_zeros_prepend_count), dtype=continuous_actions.dtype)
                continuous_actions = np.concatenate([prepend_zeros, continuous_actions], axis=1)
            
            # 在最后一个元素前插入零
            if self.padding_zeros_insert_before_last > 0:
                # 分割: 除了最后一列的部分 和 最后一列
                actions_except_last = continuous_actions[:, :-1]
                last_col = continuous_actions[:, -1:]
                insert_zeros = np.zeros((chunk_size, self.padding_zeros_insert_before_last), dtype=continuous_actions.dtype)
                continuous_actions = np.concatenate([actions_except_last, insert_zeros, last_col], axis=1)
            
            if verbose:
                logger.info(f"  🔧 零填充后处理: 开头+{self.padding_zeros_prepend_count}零, 最后元素前+{self.padding_zeros_insert_before_last}零")
                logger.info(f"     新 shape: {continuous_actions.shape}")
        
        timing['total_time'] = time.time() - total_start
        
        # ========== Verbose: 输出信息 ==========
        if verbose:
            logger.info(f"  🎯 OFT Pipeline 推理完成 ({timing['oft_time']:.3f}s)")
            logger.info(f"     连续动作: shape={list(continuous_actions.shape)}")
            
            # 显示前几步动作
            logger.info(f"     连续动作 (前 3 步):")
            for t in range(min(3, len(continuous_actions))):
                action_str = ', '.join([f'{v:.4f}' for v in continuous_actions[t][:7]])
                logger.info(f"       Step {t}: [{action_str}...]")
            
            logger.info(f"  ⏱️  总耗时: {timing['total_time']:.3f}s")
            logger.info("=" * 60)
        
        return {
            'actions': continuous_actions.tolist(),
            'timing': timing,
            'chunk_size': self.chunk_size,
            'action_dim': self.action_dim,
        }


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="OFT 遥操作推理服务",
    description="VLM + OFT Pipeline 实时推理 API",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局推理引擎
inference_engine: Optional[OFTInferenceEngine] = None
server_config: Dict[str, Any] = {}


def decode_image(image_data: str) -> Image.Image:
    """解码 Base64 图像"""
    try:
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """执行单次预测"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 解码图像 (支持 base64、本地路径、numpy 数组)
        images = []
        bgr_to_rgb = server_config.get('image_preprocess', {}).get('bgr_to_rgb', False)
        
        if request.image_arrays:
            # 从 numpy 数组转换 [H, W, C]
            for idx, arr in enumerate(request.image_arrays):
                np_arr = np.array(arr, dtype=np.uint8)
                
                # ========== DEBUG: 打印 image array 前3列 ==========
                logger.info(f"[DEBUG] 接收到 image_array[{idx}]:")
                logger.info(f"  Shape: {np_arr.shape}, dtype: {np_arr.dtype}")
                
                if np_arr.max() <= 1.0:
                    np_arr = (np_arr * 255).astype(np.uint8)
                # BGR → RGB 转换
                if bgr_to_rgb:
                    np_arr = np_arr[:, :, ::-1].copy()
                images.append(Image.fromarray(np_arr, mode='RGB'))
        elif request.image_paths:
            # 从本地路径加载
            for path in request.image_paths:
                if not os.path.exists(path):
                    raise HTTPException(status_code=400, detail=f"图像路径不存在: {path}")
                images.append(Image.open(path).convert('RGB'))
        elif request.images:
            # 从 base64 解码
            images = [decode_image(img) for img in request.images]
        else:
            raise HTTPException(status_code=400, detail="需要提供 images、image_paths 或 image_arrays")
        
        # 解析 state
        observation_state = None
        if request.state is not None:
            observation_state = np.array(request.state, dtype=np.float32)
            
            # ========== DEBUG: 打印完整 state ==========
            logger.info(f"[DEBUG] 接收到 state:")
            logger.info(f"  Length: {len(observation_state)}")
            logger.info(f"  完整值: {observation_state.tolist()}")
        else:
            logger.info(f"[DEBUG] 未接收到 state (None)")
        
        # 获取指令 (使用 config 中的默认指令)
        instruction = server_config.get('default_instruction', '')
        
        # 执行预测
        verbose = server_config.get('verbose', True)
        result = inference_engine.predict(
            images=images,
            instruction=instruction,
            observation_state=observation_state,
            verbose=verbose,
        )
        
        # ========== DEBUG: 打印所有输出动作 ==========
        logger.info(f"[DEBUG] 输出动作:")
        logger.info(f"  Chunk size: {result['chunk_size']}, Action dim: {result['action_dim']}")
        
        logger.info(f"  === 连续动作 (continuous actions) ===")
        for t, action in enumerate(result['actions']):
            logger.info(f"    Step {t:2d}: {action}")
        
        return PredictResponse(
            actions=result['actions'],
            timing=TimingInfo(
                vlm_time=result['timing']['vlm_time'],
                oft_time=result['timing']['oft_time'],
                total_time=result['timing']['total_time'],
            ),
            chunk_size=result['chunk_size'],
            action_dim=result['action_dim'],
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("预测失败")
        raise HTTPException(status_code=500, detail=f"预测失败: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    if inference_engine is None:
        return HealthResponse(
            status="not_ready",
            models_loaded={
                "vlm": False,
                "oft_pipeline": False,
            },
            device="unknown",
        )
    
    return HealthResponse(
        status="ready",
        models_loaded={
            "vlm": inference_engine.vlm_backbone is not None,
            "oft_pipeline": inference_engine.oft_pipeline is not None,
        },
        device=inference_engine.device,
    )


@app.get("/info", response_model=InfoResponse)
async def info():
    """获取模型信息"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    vlm_config = server_config.get('vlm', {})
    
    return InfoResponse(
        vlm_type=vlm_config.get('type', 'unknown'),
        vlm_model_path=vlm_config.get('model_path', 'unknown'),
        chunk_size=inference_engine.chunk_size,
        action_dim=inference_engine.action_dim,
        device=inference_engine.device,
        config={
            'proprio_dim': inference_engine.proprio_dim,
            'state_dim': inference_engine.state_dim,
        },
    )


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global inference_engine, server_config
    
    # 从命令行或环境变量获取配置文件路径
    config_path = os.environ.get('CONFIG_PATH', 'config.yaml')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            server_config = yaml.safe_load(f)
        
        logger.info(f"加载配置文件: {config_path}")
        inference_engine = OFTInferenceEngine(server_config)
    else:
        logger.warning(f"配置文件不存在: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='OFT 推理服务器')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8000, help='监听端口')
    parser.add_argument('--offline', action='store_true',
                        help='离线模式：设置 TRANSFORMERS_OFFLINE=1 和 HF_HUB_OFFLINE=1，仅从本地加载模型')
    args = parser.parse_args()
    
    # 离线模式：在加载模型前设置，避免 Hugging Face 访问网络
    if args.offline:
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        logger.info('已启用离线模式 (TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1)')
    
    # 设置环境变量
    os.environ['CONFIG_PATH'] = args.config
    
    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
