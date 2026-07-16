"""
S0_1 VLM Module

提供 VLM backbone 和相关工具。

主要功能:
- Qwen3-VL backbone: 提取 hidden states
- 配置系统: YAML + 代码覆盖
- VLM 接口: 与 Action Head 组合

使用示例:
    from VLMs.S0_1 import create_backbone
    
    # 创建 backbone
    backbone = create_backbone(
        model_path="Qwen/Qwen3-VL-2B-Instruct",
        layers=[14],
        device="cuda:0"
    )
    
    # 提取 hidden states
    output = backbone.get_hidden_states(
        images=[agentview_img, wrist_img],
        instruction="pick up the apple"
    )
    
    # 获取 hidden states 列表（用于 Action Head）
    vlm_features = output.hidden_states
"""

# 导入 backbone 模块
from . import backbone

# 便捷导入 - 最常用的类和函数
from .backbone import (
    Qwen3VLBackbone,
    Qwen3VLConfig,
    HiddenStateOutput,
    create_backbone,
    create_backbone_for_model,
)

# 从 qwen3_vl 模块导入更多内容
from .backbone.qwen3_vl import (
    load_config,
    load_yaml_config,
    get_default_config,
    get_config_for_model,
    connect_vlm_to_action_head,
)

__version__ = "0.1.0"

__all__ = [
    # 子模块
    "backbone",
    # 核心类
    "Qwen3VLBackbone",
    "Qwen3VLConfig",
    "HiddenStateOutput",
    # 创建函数
    "create_backbone",
    "create_backbone_for_model",
    # 配置函数
    "load_config",
    "load_yaml_config",
    "get_default_config",
    "get_config_for_model",
    # 组合工具
    "connect_vlm_to_action_head",
]
