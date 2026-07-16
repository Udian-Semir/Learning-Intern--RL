"""
ParaCAT 遥操作推理服务器

基于 FastAPI 的高性能推理服务，支持：
- VLM Backbone (Qwen/Eagle) 实时 hidden state 提取
- Pons Adapter 特征聚合
- ParaCAT Action Head 离散动作预测
- State Mapper (可选) 状态嵌入

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
from Action_Heads.ParaCAT.model.action_head.paracat_action_head import create_paracat_action_head
from Adapter.Pons.pons_adapter import create_pons_adapter
from utils.discrete import undiscrete_constrain_delta

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== ObservationStateMapper ====================

class ObservationStateMapper(nn.Module):
    """将 observation_state 归一化并映射到 VLM 隐藏空间"""
    
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
            # 简化实现，具体转换逻辑省略
            pass
        
        elif processor_name == "minmax_normalize" and minmax_columns is not None:
            # 最大最小值归一化: (x - min) / (max - min) -> [0, 1]
            if minmax_min is not None and minmax_max is not None:
                for col in minmax_columns:
                    if col < len(observation_state):
                        col_min = minmax_min[col]
                        col_max = minmax_max[col]
                        col_range = col_max - col_min
                        if col_range > 1e-8:
                            observation_state[col] = (observation_state[col] - col_min) / col_range
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
    pons_time: float = Field(..., description="Pons 推理时间 (秒)")
    paracat_time: float = Field(..., description="ParaCAT 推理时间 (秒)")
    total_time: float = Field(..., description="总时间 (秒)")


class PredictResponse(BaseModel):
    """预测响应"""
    actions: List[List[float]] = Field(..., description="预测的动作序列 [chunk_size, action_dim]")
    discrete_actions: List[List[int]] = Field(..., description="离散动作 {-1, 0, 1}")
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
    use_pons: bool
    use_state_mapper: bool
    device: str
    config: Dict[str, Any]


# ==================== 推理引擎 ====================

class ParaCATInferenceEngine:
    """ParaCAT 推理引擎"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda:0')
        
        # 模型组件
        self.vlm_backbone = None
        self.pons = None
        self.paracat = None
        self.state_mapper = None
        
        # 标志位
        self.use_pons = False
        self.use_state_mapper = False
        
        # 配置
        self.chunk_size = config.get('chunk_size', 25)
        self.action_dim = config.get('action_dim', 14)
        self.undiscrete_columns = config.get('undiscrete_columns', [])
        self.undiscrete_deltas = config.get('undiscrete_deltas', [])
        self.gripper_columns = config.get('gripper_columns', [])
        self.gripper_clamp_negative = config.get('gripper_clamp_negative', False)
        
        # State 预处理配置
        self.state_process_order = config.get('state_process_order', [])
        self.hand_binary_columns = config.get('hand_binary_columns', [])
        self.hand_binary_threshold = config.get('hand_binary_threshold', 442.0)
        self.state_euler_to_axisangle_columns = config.get('state_euler_to_axisangle_columns', [])
        self.state_dim = config.get('state_dim', 14)
        
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
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载所有模型组件"""
        config = self.config
        
        # ====== 1. 加载 VLM Backbone ======
        vlm_config = config.get('vlm', {})
        logger.info(f"加载 VLM Backbone: {vlm_config.get('model_path')}")
        
        self.vlm_backbone = create_vlm_backbone(
            model_type=vlm_config.get('type', 'eagle2_5_vl'),
            model_path=vlm_config.get('model_path'),
            device=self.device,
            layers=vlm_config.get('layers', [-1]),
            flip_images=vlm_config.get('flip_images', False),
            content_order=vlm_config.get('content_order', 'images_first'),
            prompt_template="simple",
            lowercase_instruction=vlm_config.get('lowercase_instruction', True),
            add_generation_prompt=vlm_config.get('add_generation_prompt', True),
        )
        logger.info("✓ VLM Backbone 加载成功")
        logger.info(f"   add_generation_prompt: {vlm_config.get('add_generation_prompt', True)}")
        
        # 获取 hidden_dim
        hidden_dim = config.get('hidden_dim', 2048)
        
        # ====== 2. 加载 Pons Adapter (可选) ======
        pons_config = config.get('pons', {})
        pons_checkpoint = pons_config.get('checkpoint', '')
        
        if pons_checkpoint and os.path.exists(pons_checkpoint):
            logger.info(f"加载 Pons Adapter: {pons_checkpoint}")
            
            self.pons = create_pons_adapter(
                q_seq_len=pons_config.get('q_seq_len', 128),
                hidden_dim=hidden_dim,
                num_blocks=pons_config.get('num_blocks', 2),
                num_heads=pons_config.get('num_heads', 8),
            ).to(self.device)
            
            pons_state = torch.load(pons_checkpoint, map_location=self.device)
            self.pons.load_state_dict(pons_state)
            self.pons.eval()
            self.use_pons = True
            logger.info("✓ Pons Adapter 加载成功")
        
        # ====== 3. 加载 ParaCAT Action Head ======
        paracat_config = config.get('paracat', {})
        paracat_checkpoint = paracat_config.get('checkpoint')
        
        if not paracat_checkpoint or not os.path.exists(paracat_checkpoint):
            raise ValueError(f"ParaCAT checkpoint 不存在: {paracat_checkpoint}")
        
        logger.info(f"加载 ParaCAT Action Head: {paracat_checkpoint}")
        
        # 尝试从 config.json 加载配置
        ckpt_dir = Path(paracat_checkpoint).parent
        config_path = ckpt_dir / "config.json"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                ckpt_config = json.load(f)
            
            self.chunk_size = ckpt_config.get('chunk_size', self.chunk_size)
            self.action_dim = ckpt_config.get('action_dim', self.action_dim)
            hidden_dim = ckpt_config.get('hidden_dim', hidden_dim)
            
            if not self.undiscrete_columns and 'undiscrete_columns' in ckpt_config:
                self.undiscrete_columns = ckpt_config['undiscrete_columns']
            if not self.undiscrete_deltas and 'undiscrete_deltas' in ckpt_config:
                self.undiscrete_deltas = ckpt_config['undiscrete_deltas']
            if not self.gripper_columns and 'gripper_columns' in ckpt_config:
                self.gripper_columns = ckpt_config['gripper_columns']
        
        self.paracat = create_paracat_action_head(
            chunk_size=self.chunk_size,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            num_transformer_blocks=paracat_config.get('num_transformer_blocks', 2),
            num_mlp_layers=paracat_config.get('num_mlp_layers', 2),
            mlp_expand_dim=paracat_config.get('mlp_expand_dim', 1024),
            num_heads=paracat_config.get('num_heads', 8),
        ).to(self.device)
        
        paracat_state = torch.load(paracat_checkpoint, map_location=self.device)
        self.paracat.load_state_dict(paracat_state)
        self.paracat.eval()
        logger.info(f"✓ ParaCAT Action Head 加载成功 (chunk={self.chunk_size}, action_dim={self.action_dim})")
        
        # ====== 4. 加载 State Mapper (可选) ======
        state_mapper_config = config.get('state_mapper', {})
        state_mapper_checkpoint = state_mapper_config.get('checkpoint', '')
        
        # 自动检测
        if not state_mapper_checkpoint:
            auto_path = ckpt_dir / "state_mapper.pt"
            if auto_path.exists():
                state_mapper_checkpoint = str(auto_path)
        
        if state_mapper_checkpoint and os.path.exists(state_mapper_checkpoint):
            logger.info(f"加载 State Mapper: {state_mapper_checkpoint}")
            
            # 从 config.json 获取配置
            sm_state_dim = self.state_dim
            sm_minmax_cols = state_mapper_config.get('minmax_columns', [])
            sm_axisangle_cols = state_mapper_config.get('axisangle_columns', [])
            sm_state_min = None
            sm_state_max = None
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    ckpt_config = json.load(f)
                if ckpt_config.get('enable_state_mapper', False):
                    sm_state_dim = ckpt_config.get('state_dim', sm_state_dim)
                    sm_minmax_cols = ckpt_config.get('state_norm_columns_minmax', sm_minmax_cols)
                    sm_axisangle_cols = ckpt_config.get('state_norm_columns_axisangle', sm_axisangle_cols)
                    if 'state_min' in ckpt_config:
                        sm_state_min = torch.tensor(ckpt_config['state_min'], dtype=torch.float32)
                    if 'state_max' in ckpt_config:
                        sm_state_max = torch.tensor(ckpt_config['state_max'], dtype=torch.float32)
            
            self.state_mapper = ObservationStateMapper(
                state_dim=sm_state_dim,
                hidden_dim=hidden_dim,
                minmax_columns=sm_minmax_cols,
                axisangle_columns=sm_axisangle_cols,
                state_min=sm_state_min,
                state_max=sm_state_max,
            ).to(self.device)
            
            state_mapper_state = torch.load(state_mapper_checkpoint, map_location=self.device)
            self.state_mapper.load_state_dict(state_mapper_state)
            self.state_mapper.eval()
            self.use_state_mapper = True
            self.state_dim = sm_state_dim
            logger.info(f"✓ State Mapper 加载成功 (state_dim={sm_state_dim})")
        
        logger.info("所有模型加载完成")
    
    def undiscretize_action(self, discrete_action: np.ndarray) -> np.ndarray:
        """将离散动作转换为连续动作"""
        continuous_action = discrete_action.copy().astype(np.float32)
        
        if self.undiscrete_columns and self.undiscrete_deltas:
            for i, col in enumerate(self.undiscrete_columns):
                if col < len(continuous_action) and i < len(self.undiscrete_deltas):
                    delta = self.undiscrete_deltas[i]
                    continuous_action[col] = undiscrete_constrain_delta(
                        np.array([discrete_action[col]]), delta
                    )[0]
        
        if self.gripper_columns:
            for col in self.gripper_columns:
                if col < len(continuous_action):
                    if self.gripper_clamp_negative:
                        # 将负值变为 0: {-1, 0, 1} -> {0, 0, 1}
                        # continuous_action[col] = max(0.0, float(discrete_action[col]))
                        continuous_action[col] = 0
                    else:
                        # 直接使用离散值
                        # continuous_action[col] = float(discrete_action[col])
                        continuous_action[col] = 0
        
        return continuous_action
    
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
                # ========== DEBUG: 打印 resize 后图片前3列 ==========
                img_arr = np.array(img)
                logger.info(f"  [DEBUG] resize 后 image[{i}] 前3列:")
                if img_arr.shape[1] >= 3:
                    for row in range(min(5, img_arr.shape[0])):
                        logger.info(f"    Row {row}: {img_arr[row, :3, :].tolist()}")
                    if img_arr.shape[0] > 5:
                        logger.info(f"    ... (共 {img_arr.shape[0]} 行)")
        
        # 使用处理后的图像
        images = processed_images
        
        # State 预处理
        processed_state = None
        if observation_state is not None and self.use_state_mapper:
            # ========== DEBUG: 打印原始 state ==========
            if verbose:
                logger.info(f"  [DEBUG] 原始 State (接收到的):")
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
            
            # 打印 metadata 中的 token 详细信息
            if hasattr(vlm_output, 'metadata') and vlm_output.metadata:
                metadata = vlm_output.metadata
                if 'seq_len' in metadata:
                    logger.info(f"     序列长度: {metadata['seq_len']}")
                if 'hidden_dim' in metadata:
                    logger.info(f"     隐藏维度: {metadata['hidden_dim']}")
                if 'layer_indices' in metadata:
                    logger.info(f"     提取层: {metadata['layer_indices']}")
                if 'num_images' in metadata:
                    logger.info(f"     图像数量: {metadata['num_images']}")
                if 'instruction' in metadata:
                    logger.info(f"     Instruction: \"{metadata['instruction'][:50]}{'...' if len(metadata.get('instruction', '')) > 50 else ''}\"")
                
                # ========== DEBUG: 打印进入 VLM 的所有 tokens ==========
                if 'input_ids' in metadata and metadata['input_ids'] is not None:
                    input_ids = metadata['input_ids']
                    logger.info(f"  [DEBUG] VLM 输入 Tokens:")
                    logger.info(f"     Input IDs shape: {list(input_ids.shape)}")
                    logger.info(f"     Total tokens: {input_ids.shape[1]}")
                    
                    # 尝试解码 tokens
                    try:
                        if hasattr(self.vlm_backbone, 'processor'):
                            decoded = self.vlm_backbone.processor.decode(
                                input_ids[0], 
                                skip_special_tokens=False
                            )
                            logger.info(f"     Decoded tokens (完整, {len(decoded)} chars):")
                            logger.info("-" * 60)
                            logger.info(decoded)
                            logger.info("-" * 60)
                    except Exception as e:
                        logger.warning(f"     无法解码 tokens: {e}")
            
            # 打印 layer_indices
            if hasattr(vlm_output, 'layer_indices'):
                logger.info(f"     Layer indices: {vlm_output.layer_indices}")
        
        with torch.no_grad():
            vlm_hidden_states_device = [v.to(self.device) for v in vlm_hidden_states]
            
            # Pons 推理
            pons_start = time.time()
            if self.use_pons:
                pons_output = self.pons(vlm_hidden_states_device)
            else:
                pons_output = torch.cat(vlm_hidden_states_device, dim=1)
            
            pons_shape_before_state = list(pons_output.shape)
            
            # State Mapper
            if self.use_state_mapper and processed_state is not None:
                state_tensor = torch.tensor(
                    processed_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                state_embedding = self.state_mapper(state_tensor)
                pons_output = torch.cat([pons_output, state_embedding], dim=1)
            
            timing['pons_time'] = time.time() - pons_start
            
            # ========== Verbose: Pons 输出信息 ==========
            if verbose:
                logger.info(f"  🔗 Pons 推理完成 ({timing['pons_time']:.3f}s)")
                logger.info(f"     Pons 输出: shape={pons_shape_before_state}")
                if self.use_state_mapper and processed_state is not None:
                    logger.info(f"     + State Embedding: shape=[1, 1, {pons_shape_before_state[2]}]")
                    logger.info(f"     最终输入 ParaCAT: shape={list(pons_output.shape)}")
            
            # ParaCAT 推理
            paracat_start = time.time()
            discrete_actions = self.paracat.predict_discrete_action(pons_output)
            timing['paracat_time'] = time.time() - paracat_start
        
        # 转换为 numpy
        discrete_actions_np = discrete_actions[0].cpu().numpy()  # (chunk_size, action_dim)
        
        # 反离散化
        continuous_actions = np.zeros_like(discrete_actions_np, dtype=np.float32)
        for t in range(discrete_actions_np.shape[0]):
            continuous_actions[t] = self.undiscretize_action(discrete_actions_np[t])
        
        timing['total_time'] = time.time() - total_start
        
        # ========== Verbose: 输出信息 ==========
        if verbose:
            logger.info(f"  🎯 ParaCAT 推理完成 ({timing['paracat_time']:.3f}s)")
            logger.info(f"     离散动作: shape={list(discrete_actions_np.shape)}")
            
            # 统计离散动作分布
            unique, counts = np.unique(discrete_actions_np, return_counts=True)
            action_dist = dict(zip(unique.astype(int), counts))
            logger.info(f"     离散值分布: {action_dist}")
            
            # 显示前几步动作
            logger.info(f"     连续动作 (前 3 步):")
            for t in range(min(3, len(continuous_actions))):
                action_str = ', '.join([f'{v:.4f}' for v in continuous_actions[t][:7]])
                logger.info(f"       Step {t}: [{action_str}...]")
            
            logger.info(f"  ⏱️  总耗时: {timing['total_time']:.3f}s")
            logger.info("=" * 60)
        
        return {
            'actions': continuous_actions.tolist(),
            'discrete_actions': discrete_actions_np.astype(int).tolist(),
            'timing': timing,
            'chunk_size': self.chunk_size,
            'action_dim': self.action_dim,
        }


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="ParaCAT 遥操作推理服务",
    description="VLM + Pons + ParaCAT 实时推理 API",
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
inference_engine: Optional[ParaCATInferenceEngine] = None
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
                logger.info(f"  前3列 ([:, :3, :]):")
                if np_arr.shape[1] >= 3:
                    for row in range(min(5, np_arr.shape[0])):  # 只打印前5行
                        logger.info(f"    Row {row}: {np_arr[row, :3, :].tolist()}")
                    if np_arr.shape[0] > 5:
                        logger.info(f"    ... (共 {np_arr.shape[0]} 行)")
                
                if np_arr.max() <= 1.0:
                    np_arr = (np_arr * 255).astype(np.uint8)
                # BGR → RGB 转换
                if bgr_to_rgb:
                    np_arr = np_arr[:, :, ::-1].copy()
                    # ========== DEBUG: 打印 BGR→RGB 转换后的前3列 ==========
                    logger.info(f"[DEBUG] BGR→RGB 转换后 image_array[{idx}]:")
                    logger.info(f"  前3列 ([:, :3, :]) - RGB顺序:")
                    if np_arr.shape[1] >= 3:
                        for row in range(min(5, np_arr.shape[0])):
                            logger.info(f"    Row {row}: {np_arr[row, :3, :].tolist()}")
                        if np_arr.shape[0] > 5:
                            logger.info(f"    ... (共 {np_arr.shape[0]} 行)")
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
        
        # 执行预测 (图像预处理参数使用 config 配置)
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
        
        logger.info(f"  === 离散动作 (discrete actions) ===")
        for t, action in enumerate(result['discrete_actions']):
            logger.info(f"    Step {t:2d}: {action}")
        
        return PredictResponse(
            actions=result['actions'],
            discrete_actions=result['discrete_actions'],
            timing=TimingInfo(
                vlm_time=result['timing']['vlm_time'],
                pons_time=result['timing']['pons_time'],
                paracat_time=result['timing']['paracat_time'],
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
                "pons": False,
                "paracat": False,
                "state_mapper": False,
            },
            device="unknown",
        )
    
    return HealthResponse(
        status="ready",
        models_loaded={
            "vlm": inference_engine.vlm_backbone is not None,
            "pons": inference_engine.use_pons,
            "paracat": inference_engine.paracat is not None,
            "state_mapper": inference_engine.use_state_mapper,
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
        use_pons=inference_engine.use_pons,
        use_state_mapper=inference_engine.use_state_mapper,
        device=inference_engine.device,
        config={
            'undiscrete_columns': inference_engine.undiscrete_columns,
            'undiscrete_deltas': inference_engine.undiscrete_deltas,
            'gripper_columns': inference_engine.gripper_columns,
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
        inference_engine = ParaCATInferenceEngine(server_config)
    else:
        logger.warning(f"配置文件不存在: {config_path}")


def main():
    parser = argparse.ArgumentParser(description='ParaCAT 推理服务器')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='监听地址')
    parser.add_argument('--port', type=int, default=8000, help='监听端口')
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ['CONFIG_PATH'] = args.config
    
    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
