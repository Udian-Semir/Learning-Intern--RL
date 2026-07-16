"""
Sai0_1 模型模块

整合 VLM Backbone 和 Action Head 的统一模型接口。

使用示例:
    from VLAs.Sai0_1 import Sai0Model, Sai0Config
    
    config = Sai0Config.for_qwen3_libero(
        dataset_path="/path/to/dataset",
        pretrained_weights="./pretrained.pt"
    )
    
    model = Sai0Model(config)
    
    # 训练模式
    loss = model.compute_loss(backbone_output, action_head_inputs)
    
    # 推理模式
    actions = model.predict(images, instruction, state)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

# 添加路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# Sai0 Model
# ============================================================================

class Sai0Model(nn.Module):
    """
    Sai0 VLA 模型
    
    整合 VLM Backbone 和 Action Head：
    - VLM Backbone: 提取视觉-语言特征 (支持 Qwen3-VL, Eagle 2.5 VL)
    - Action Head: 预测机器人动作 (Flow Matching)
    
    工作流程:
    1. 图像 + 指令 → VLM → Hidden States
    2. Hidden States + State → Action Head → Actions
    
    支持两种模式:
    - 预计算模式 (训练): 使用预先提取的 VLM hidden states
    - 实时模式 (推理): 实时提取 VLM hidden states
    """
    
    def __init__(
        self,
        config: "Sai0Config" = None,
        vlm_backbone = None,
        action_head = None,
        device: str = "cuda:0",
    ):
        """
        初始化 Sai0 模型
        
        Args:
            config: Sai0Config 配置对象
            vlm_backbone: 预创建的 VLM backbone (可选)
            action_head: 预创建的 Action Head (可选)
            device: 设备
        """
        super().__init__()
        
        self.config = config
        self.device = device
        self._vlm_backbone = vlm_backbone
        self._action_head = action_head
        
        # 延迟初始化标志
        self._vlm_initialized = vlm_backbone is not None
        self._action_head_initialized = action_head is not None
    
    @property
    def vlm_backbone(self):
        """延迟加载 VLM Backbone"""
        if self._vlm_backbone is None and self.config is not None:
            self._vlm_backbone = self._create_vlm_backbone()
            self._vlm_initialized = True
        return self._vlm_backbone
    
    @property
    def action_head(self):
        """延迟加载 Action Head"""
        if self._action_head is None and self.config is not None:
            self._action_head = self._create_action_head()
            self._action_head_initialized = True
        return self._action_head
    
    def _create_vlm_backbone(self):
        """创建 VLM Backbone"""
        from VLMs.S0_1.backbone import create_vlm_backbone
        
        vlm_config = self.config.vlm
        backbone = create_vlm_backbone(
            model_type=vlm_config.model_type,
            model_path=vlm_config.model_path,
            device=vlm_config.device or self.device,
            layers=vlm_config.layers,
            prompt_template=vlm_config.prompt_template,
            content_order=vlm_config.content_order,
            flip_images=vlm_config.flip_images,
            dtype=vlm_config.dtype,
            verbose=vlm_config.verbose,
            lowercase_instruction=vlm_config.lowercase_instruction,
            add_generation_prompt=vlm_config.add_generation_prompt,
        )
        
        return backbone
    
    def _create_action_head(self):
        """创建 Action Head"""
        from .config import ActionHeadConfig
        
        ah_config = self.config.action_head
        head_type = ActionHeadConfig.normalize_head_type(ah_config.head_type)
        
        if head_type == "flow_matching_0":
            return self._create_flow_matching_0_head()
        elif head_type == "flow_matching_1":
            return self._create_flow_matching_1_head()
        elif head_type == "oft_1_0":
            return self._create_oft_head()
        else:
            raise ValueError(
                f"Unknown action head type: {ah_config.head_type}. "
                f"Supported types: flow_matching_0, flow_matching_1, oft_1_0"
            )
    
    def _create_flow_matching_0_head(self):
        """创建 Flow Matching 0 Action Head (GR00T N1.5 原始架构)"""
        from Action_Heads.Flow_Matching_0.models.action_head.flow_matching_action_head import (
            FlowmatchingActionHead
        )
        from Action_Heads.Flow_Matching_0.config import (
            get_flowmatching_action_head_config_original
        )
        
        ah_config = self.config.action_head
        
        # 获取配置
        action_head_config = get_flowmatching_action_head_config_original()
        
        # 创建 Action Head
        action_head = FlowmatchingActionHead(action_head_config)
        
        # 加载预训练权重
        self._load_pretrained_weights(action_head, ah_config.pretrained_weights)
        
        action_head = action_head.to(self.device)
        return action_head
    
    def _create_flow_matching_1_head(self):
        """创建 Flow Matching 1 Action Head (自定义配置，支持多层 VLM)"""
        from Action_Heads.Flow_Matching_1.models.action_head.flow_matching_action_head import (
            FlowmatchingActionHead
        )
        from Action_Heads.Flow_Matching_1.config import (
            get_flowmatching_action_head_config
        )
        
        ah_config = self.config.action_head
        
        # 获取配置 (支持自定义维度)
        action_head_config = get_flowmatching_action_head_config(
            action_backbone_dim=ah_config.action_backbone_dim,
            vlm_output_dim=ah_config.vlm_output_dim,
            action_dim=ah_config.action_dim if ah_config.action_dim else 16,
            action_horizon=ah_config.num_action_chunks,
            max_state_dim=ah_config.max_state_dim,
            max_action_dim=ah_config.max_action_dim,
        )
        
        # 创建 Action Head
        action_head = FlowmatchingActionHead(action_head_config)
        
        # 加载预训练权重
        self._load_pretrained_weights(action_head, ah_config.pretrained_weights)
        
        action_head = action_head.to(self.device)
        return action_head
    
    def _create_oft_head(self):
        """创建 OFT 1.0 Action Head (L1 Regression + Transformer)"""
        from Action_Heads.OFT1_0.vlm2oft_pipeline import VLM2OFTPipeline
        import Action_Heads.OFT1_0.constants as oft_constants
        
        ah_config = self.config.action_head
        
        # 更新 OFT constants (动态配置) - 必须在创建 Pipeline 之前设置
        oft_constants.LLM_OUTPUT_DIM_MLP_INPUT_DIM = ah_config.llm_output_dim
        oft_constants.NUM_VLM_HIDDEN_LAYERS = ah_config.num_vlm_hidden_layers
        oft_constants.ACTION_DIM = ah_config.action_dim
        oft_constants.NUM_ACTIONS_CHUNK = ah_config.num_action_chunks
        oft_constants.PROPRIO_DIM = ah_config.proprio_dim
        oft_constants.USE_DIFFUSION = ah_config.use_diffusion
        
        # 创建 OFT Pipeline
        # 注意: VLM2OFTPipeline 使用 constants 中的值，但也支持直接传递参数覆盖
        action_head = VLM2OFTPipeline(
            num_vlm_layers=ah_config.num_vlm_hidden_layers,
            vlm_output_dim=ah_config.llm_output_dim,
            num_transformer_blocks=ah_config.num_transformer_blocks,
            num_attention_heads=ah_config.num_attention_heads,
            dropout=ah_config.dropout,
            action_head_hidden_dim=ah_config.action_head_hidden_dim,
        )
        
        # 加载预训练权重
        self._load_pretrained_weights(action_head, ah_config.pretrained_weights)
        
        action_head = action_head.to(self.device)
        return action_head
    
    def _load_pretrained_weights(self, model: nn.Module, weights_path: Optional[str]):
        """加载预训练权重"""
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pretrained weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            
            # 处理 DDP 保存的权重
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            print("✓ Pretrained weights loaded")
    
    def forward(
        self,
        backbone_output: BatchFeature,
        action_head_inputs: BatchFeature,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 (训练模式，使用预计算的 VLM features)
        
        Args:
            backbone_output: VLM backbone 输出
                - backbone_features: (batch, seq_len, hidden_dim)
                - backbone_attention_mask: (batch, seq_len)
            action_head_inputs: Action Head 输入
                - state: (batch, 1, max_state_dim)
                - action: (batch, num_chunks, max_action_dim)
                - action_mask: (batch, num_chunks, max_action_dim)
                - embodiment_id: (batch,)
        
        Returns:
            Action Head 输出字典
        """
        return self.action_head(backbone_output, action_head_inputs)
    
    def compute_loss(
        self,
        backbone_output: BatchFeature,
        action_head_inputs: BatchFeature,
    ) -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            backbone_output: VLM backbone 输出
            action_head_inputs: Action Head 输入
        
        Returns:
            损失值
        """
        output = self.forward(backbone_output, action_head_inputs)
        return output["loss"]
    
    def predict(
        self,
        images: List,
        instruction: str,
        state: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        实时推理预测动作
        
        Args:
            images: 图像列表 [PIL.Image, ...]
            instruction: 任务指令
            state: 当前状态 (1, state_dim) 或 (state_dim,)
            num_samples: 采样次数
        
        Returns:
            预测的动作 (num_chunks, action_dim)
        """
        # 确保模型已初始化
        if not self._vlm_initialized:
            _ = self.vlm_backbone
        if not self._action_head_initialized:
            _ = self.action_head
        
        # 获取 VLM hidden states
        vlm_output = self.vlm_backbone.get_hidden_states(
            images=images,
            instruction=instruction,
        )
        
        # vlm_output.hidden_states: List[Tensor], 每个 (batch, seq_len, hidden_dim)
        hidden_states_list = vlm_output.hidden_states
        
        # 准备 state
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)
        state = state.to(self.device)
        batch_size = state.size(0)
        
        # 根据 action head 类型选择推理方式
        from .config import ActionHeadConfig
        head_type = self.config.action_head.head_type.lower()
        head_type = ActionHeadConfig.normalize_head_type(head_type)
        
        with torch.no_grad():
            if head_type == "oft_1_0":
                # OFT Pipeline: 直接使用 forward
                actions = self._predict_oft(hidden_states_list, state)
            else:
                # Flow Matching: 使用 get_action
                actions = self._predict_flow_matching(hidden_states_list, state, batch_size)
        
        return actions
    
    def _predict_flow_matching(
        self,
        hidden_states_list: List[torch.Tensor],
        state: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Flow Matching Action Head 的推理"""
        # 取第一层 (或拼接多层，取决于配置)
        backbone_features = hidden_states_list[0]
        if backbone_features.dim() == 2:
            backbone_features = backbone_features.unsqueeze(0)
        
        seq_len = backbone_features.size(1)
        
        backbone_output = BatchFeature(data={
            "backbone_features": backbone_features.to(self.device),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=self.device),
        })
        
        # Padding state
        actual_state_dim = state.size(1)
        max_state_dim = self.config.action_head.max_state_dim
        if actual_state_dim < max_state_dim:
            padding = torch.zeros(batch_size, max_state_dim - actual_state_dim, 
                                 dtype=state.dtype, device=self.device)
            state = torch.cat([state, padding], dim=1)
        
        state = state.unsqueeze(1)  # (batch, 1, max_state_dim)
        
        action_head_inputs = BatchFeature(data={
            "state": state,
            "embodiment_id": torch.full((batch_size,), self.config.action_head.embodiment_id, 
                                        dtype=torch.long, device=self.device),
        })
        
        # Flow Matching 使用 get_action 方法
        output = self.action_head.get_action(
            backbone_output=backbone_output,
            action_input=action_head_inputs,
        )
        
        # 返回动作
        actions = output["action_pred"]  # (batch, num_chunks, action_dim)
        return actions[0]  # (num_chunks, action_dim)
    
    def _predict_oft(
        self,
        hidden_states_list: List[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """OFT Action Head 的推理"""
        import Action_Heads.OFT1_0.constants as oft_constants
        
        # OFT 需要多层 hidden states 作为列表输入
        # 确保每层都在正确的设备上
        hidden_states_list = [h.to(self.device) for h in hidden_states_list]
        
        # Proprioception (state)
        # OFT 期望的是 (batch, proprio_dim)
        if state.dim() == 2 and state.size(1) > oft_constants.PROPRIO_DIM:
            # 只取前 proprio_dim 维
            proprio = state[:, :oft_constants.PROPRIO_DIM]
        else:
            proprio = state.squeeze(1) if state.dim() == 3 else state
        
        proprio = proprio.to(self.device)
        
        # OFT forward
        action_predictions = self.action_head(
            vlm_hidden_states=hidden_states_list,
            proprioception=proprio,
        )
        
        # 输出形状: (batch, 1, num_chunks * action_dim)
        # 需要 reshape 为 (num_chunks, action_dim)
        action_dim = oft_constants.ACTION_DIM
        num_chunks = oft_constants.NUM_ACTIONS_CHUNK
        
        actions = action_predictions.squeeze(1)  # (batch, num_chunks * action_dim)
        actions = actions.view(-1, num_chunks, action_dim)  # (batch, num_chunks, action_dim)
        
        return actions[0]  # (num_chunks, action_dim)
    
    def get_vlm_hidden_states(
        self,
        images: List,
        instruction: str,
    ) -> torch.Tensor:
        """
        提取 VLM Hidden States
        
        Args:
            images: 图像列表
            instruction: 任务指令
        
        Returns:
            Hidden states tensor
        """
        if not self._vlm_initialized:
            _ = self.vlm_backbone
        
        vlm_output = self.vlm_backbone.get_hidden_states(
            images=images,
            instruction=instruction,
        )
        
        return vlm_output.hidden_states
    
    def save_action_head(self, path: str):
        """保存 Action Head 权重"""
        if self._action_head is not None:
            torch.save(self._action_head.state_dict(), path)
            print(f"✓ Action Head saved to: {path}")
    
    def load_action_head(self, path: str):
        """加载 Action Head 权重"""
        if self._action_head is None:
            _ = self.action_head  # 初始化
        
        state_dict = torch.load(path, map_location=self.device)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        self._action_head.load_state_dict(state_dict)
        print(f"✓ Action Head loaded from: {path}")
    
    def to(self, device):
        """移动模型到指定设备"""
        self.device = device if isinstance(device, str) else str(device)
        if self._vlm_backbone is not None and hasattr(self._vlm_backbone, 'to'):
            self._vlm_backbone = self._vlm_backbone.to(device)
        if self._action_head is not None:
            self._action_head = self._action_head.to(device)
        return super().to(device)
    
    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        if self._vlm_backbone is not None and hasattr(self._vlm_backbone, 'eval'):
            self._vlm_backbone.eval()
        if self._action_head is not None:
            self._action_head.train(mode)
        return self
    
    def eval(self):
        """设置评估模式"""
        return self.train(False)
    
    def get_trainable_parameters(self):
        """获取可训练参数 (通常只有 Action Head)"""
        if self._action_head is not None:
            return self._action_head.parameters()
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            "config": self.config.to_dict() if self.config else None,
            "device": self.device,
            "vlm_initialized": self._vlm_initialized,
            "action_head_initialized": self._action_head_initialized,
        }
        
        if self._vlm_backbone is not None:
            info["vlm_info"] = self._vlm_backbone.get_model_info()
        
        if self._action_head is not None:
            info["action_head_params"] = sum(p.numel() for p in self._action_head.parameters())
        
        return info


# ============================================================================
# 工厂函数
# ============================================================================

def create_sai0_model(
    vlm_type: str = "qwen3_vl",
    vlm_model_path: str = None,
    action_head_type: str = "flow_matching",
    pretrained_weights: str = None,
    device: str = "cuda:0",
    **kwargs
) -> Sai0Model:
    """
    创建 Sai0 模型
    
    Args:
        vlm_type: VLM 类型 (qwen3_vl, eagle2_5_vl)
        vlm_model_path: VLM 模型路径
        action_head_type: Action Head 类型
        pretrained_weights: Action Head 预训练权重路径
        device: 设备
        **kwargs: 其他配置参数
    
    Returns:
        Sai0Model 实例
    """
    from .config import Sai0Config, VLMConfig, ActionHeadConfig
    
    # 创建配置
    vlm_config = VLMConfig(
        model_type=vlm_type,
        model_path=vlm_model_path or ("Qwen/Qwen3-VL-2B-Instruct" if vlm_type == "qwen3_vl" else None),
        device=device,
        **{k: v for k, v in kwargs.items() if k in VLMConfig.__dataclass_fields__}
    )
    
    ah_config = ActionHeadConfig(
        head_type=action_head_type,
        pretrained_weights=pretrained_weights,
        **{k: v for k, v in kwargs.items() if k in ActionHeadConfig.__dataclass_fields__}
    )
    
    config = Sai0Config(vlm=vlm_config, action_head=ah_config)
    
    return Sai0Model(config=config, device=device)


def create_model_from_config(config: "Sai0Config") -> Sai0Model:
    """从配置创建模型"""
    return Sai0Model(config=config, device=config.vlm.device)

