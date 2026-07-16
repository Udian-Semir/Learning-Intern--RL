"""
Cosmos Reason 2B VL Backbone Module

提供 Cosmos Reason 2B VL 模型的 hidden states 提取功能。
可独立使用，也可与 Action Head 组合形成完整的 VLA 系统。

核心组件:
- CosmosReason2BVLBackbone: VLM backbone 主类
- CosmosReason2BVLConfig: 配置类
- HiddenStateOutput: hidden states 输出封装

使用示例:
    # 1. 基本使用
    from VLMs.S0_1.backbone.cosmos_reason_2b_vl import CosmosReason2BVLBackbone
    
    backbone = CosmosReason2BVLBackbone(
        model_path="/path/to/cosmos_model",
        layers=[-1],
        device="cuda:0"
    )
    
    output = backbone.get_hidden_states(
        images=[agentview_img, wrist_img],
        instruction="pick up the apple"
    )
    
    # 2. 与 Action Head 组合
    vlm_features = output.hidden_states  # List[Tensor]
    actions = action_head(vlm_features, proprioception)
    
    # 3. 使用便捷函数
    backbone = create_backbone("/path/to/model", device="cuda:0")
    
    # 4. 自定义配置
    from VLMs.S0_1.backbone.cosmos_reason_2b_vl import CosmosReason2BVLConfig
    
    config = CosmosReason2BVLConfig(
        model_path="/path/to/model",
        layers=[-1],
        prompt_template="Robot action: {instruction}",
        content_preset="text_first"
    )
    backbone = CosmosReason2BVLBackbone(config=config)
"""

# 导入核心类
from .backbone import (
    CosmosReason2BVLBackbone,
    HiddenStateOutput,
    VLMInterface,
)

from .config import (
    CosmosReason2BVLConfig,
    load_config,
    load_yaml_config,
    get_default_config,
    get_config_for_model,
)

# 版本信息
__version__ = "0.1.0"
__author__ = "HuangWenlong"

# 导出列表
__all__ = [
    # 核心类
    "CosmosReason2BVLBackbone",
    "HiddenStateOutput",
    "VLMInterface",
    # 配置
    "CosmosReason2BVLConfig",
    "load_config",
    "load_yaml_config",
    "get_default_config",
    "get_config_for_model",
    # 便捷函数
    "create_backbone",
    "create_backbone_for_model",
]


# ============================================================================
# 便捷函数
# ============================================================================

def create_backbone(
    model_path: str,
    layers: list = None,
    device: str = "cuda",
    **kwargs
) -> CosmosReason2BVLBackbone:
    """
    创建 Cosmos Reason 2B VL Backbone 实例
    
    这是最简单的创建方式，适合快速开始。
    
    Args:
        model_path: 模型路径 (本地路径或 HuggingFace ID)
        layers: 要提取的层号，默认 [-1] (最后一层)
        device: 设备
        **kwargs: 其他配置参数
        
    Returns:
        CosmosReason2BVLBackbone 实例
        
    Example:
        >>> backbone = create_backbone("/path/to/model")
        >>> output = backbone.get_hidden_states(images, instruction)
    """
    if layers is None:
        layers = [-1]
    
    return CosmosReason2BVLBackbone(
        model_path=model_path,
        layers=layers,
        device=device,
        **kwargs
    )


def create_backbone_for_model(
    model_path: str,
    device: str = "cuda",
    **kwargs
) -> CosmosReason2BVLBackbone:
    """
    根据模型路径创建 Backbone
    
    提供预设配置，简化常用模型的初始化。
    
    Args:
        model_path: 模型路径
        device: 设备
        **kwargs: 覆盖配置
        
    Returns:
        CosmosReason2BVLBackbone 实例
        
    Example:
        >>> backbone = create_backbone_for_model("/path/to/model")
        >>> backbone = create_backbone_for_model("/path/to/model", layers=[-1, -2])
    """
    config = get_config_for_model(model_path, device=device, **kwargs)
    return CosmosReason2BVLBackbone(config=config)


# ============================================================================
# VLM + Action Head 组合工具
# ============================================================================

def connect_vlm_to_action_head(
    vlm_backbone: CosmosReason2BVLBackbone,
    action_head,
    proprioception_normalizer=None
):
    """
    将 VLM Backbone 与 Action Head 连接
    
    返回一个组合函数，输入图像和状态，输出动作。
    
    Args:
        vlm_backbone: VLM backbone 实例
        action_head: Action Head 实例（需要有 forward 方法）
        proprioception_normalizer: 本体感知归一化器（可选）
        
    Returns:
        组合函数 predict(images, instruction, proprioception) -> actions
        
    Example:
        >>> predict = connect_vlm_to_action_head(backbone, action_head)
        >>> actions = predict(images, "pick up apple", state)
    """
    import torch
    
    def predict(
        images,
        instruction: str,
        proprioception,
        **kwargs
    ):
        """
        端到端预测动作
        
        Args:
            images: 输入图像
            instruction: 任务指令
            proprioception: 本体感知数据 (batch, proprio_dim)
            
        Returns:
            预测的动作
        """
        # 1. 提取 VLM hidden states
        output = vlm_backbone.get_hidden_states(images, instruction)
        vlm_features = output.hidden_states
        
        # 2. 处理本体感知数据
        if proprioception_normalizer is not None:
            proprioception = proprioception_normalizer.normalize(proprioception)
        
        # 确保是 tensor 并在正确设备上
        if not isinstance(proprioception, torch.Tensor):
            proprioception = torch.from_numpy(proprioception)
        proprioception = proprioception.to(vlm_features[0].device)
        
        if proprioception.dim() == 1:
            proprioception = proprioception.unsqueeze(0)
        
        # 3. 调用 Action Head
        with torch.no_grad():
            actions = action_head(vlm_features, proprioception)
        
        return actions
    
    return predict


# ============================================================================
# 类型检查辅助
# ============================================================================

def is_cosmos_backbone(obj) -> bool:
    """检查对象是否是 Cosmos Backbone"""
    return isinstance(obj, CosmosReason2BVLBackbone)


def get_backbone_info(backbone: CosmosReason2BVLBackbone) -> dict:
    """获取 backbone 信息"""
    return backbone.get_model_info()

