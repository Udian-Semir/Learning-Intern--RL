"""
S0_1 Backbone Module

提供各种 VLM backbone 实现和统一的模型选择接口。

可用的 Backbone:
- qwen3_vl: Qwen3-VL 系列模型 (2B, 4B, 7B)
- eagle2_5_vl: Eagle 2.5 VL 模型

使用示例:
    # 1. 使用统一的 model_selector
    from VLMs.S0_1.backbone import create_vlm_backbone
    
    backbone = create_vlm_backbone(
        model_type="qwen3_vl",
        model_path="Qwen/Qwen3-VL-2B-Instruct",
        layers=[14]
    )
    
    # 2. 直接使用特定 backbone
    from VLMs.S0_1.backbone import qwen3_vl
    
    backbone = qwen3_vl.create_backbone(
        model_path="Qwen/Qwen3-VL-2B-Instruct",
        layers=[14]
    )
    
    # 3. 使用 Eagle 2.5 VL
    from VLMs.S0_1.backbone import eagle2_5_vl
    
    backbone = eagle2_5_vl.create_backbone(
        model_path="/path/to/eagle_model",
        layers=[-4]
    )
"""

# 导入子模块
from . import qwen3_vl
from . import eagle2_5_vl

# 导入模型选择器
from .model_selector import (
    create_vlm_backbone,
    create_backbone_from_config,
    get_prompt_config,
    list_available_models,
    list_prompt_templates,
    list_content_orders,
    ModelSelectorConfig,
    PromptConfig,
    ModelType,
    RunMode,
    ContentOrder,
    PROMPT_TEMPLATES,
    CONTENT_ORDER_CONFIGS,
)

# 配置类可以直接导入（无 torch 依赖）
from .qwen3_vl import (
    Qwen3VLConfig,
    load_config as load_qwen_config,
    load_yaml_config as load_qwen_yaml_config,
    get_default_config as get_qwen_default_config,
    get_config_for_model as get_qwen_config_for_model,
)

from .eagle2_5_vl import (
    Eagle25VLConfig,
    load_config as load_eagle_config,
    load_yaml_config as load_eagle_yaml_config,
    get_default_config as get_eagle_default_config,
    get_config_for_model as get_eagle_config_for_model,
)

# 延迟导入的变量
_Qwen3VLBackbone = None
_Eagle25VLBackbone = None
_HiddenStateOutput = None


def _ensure_qwen_loaded():
    """确保 Qwen 组件已加载"""
    global _Qwen3VLBackbone, _HiddenStateOutput
    if _Qwen3VLBackbone is None:
        from .qwen3_vl import Qwen3VLBackbone as _B, HiddenStateOutput as _H
        _Qwen3VLBackbone = _B
        _HiddenStateOutput = _H


def _ensure_eagle_loaded():
    """确保 Eagle 组件已加载"""
    global _Eagle25VLBackbone, _HiddenStateOutput
    if _Eagle25VLBackbone is None:
        from .eagle2_5_vl import Eagle25VLBackbone as _B, HiddenStateOutput as _H
        _Eagle25VLBackbone = _B
        _HiddenStateOutput = _H


def __getattr__(name):
    """延迟导入"""
    if name == "Qwen3VLBackbone":
        _ensure_qwen_loaded()
        return _Qwen3VLBackbone
    elif name == "Eagle25VLBackbone":
        _ensure_eagle_loaded()
        return _Eagle25VLBackbone
    elif name == "HiddenStateOutput":
        _ensure_qwen_loaded()  # HiddenStateOutput 在两者中都有，用任一个
        return _HiddenStateOutput
    elif name == "create_backbone":
        return qwen3_vl.create_backbone
    elif name == "create_backbone_for_model":
        return qwen3_vl.create_backbone_for_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 子模块
    "qwen3_vl",
    "eagle2_5_vl",
    
    # 模型选择器
    "create_vlm_backbone",
    "create_backbone_from_config",
    "get_prompt_config",
    "list_available_models",
    "list_prompt_templates",
    "list_content_orders",
    "ModelSelectorConfig",
    "PromptConfig",
    "ModelType",
    "RunMode",
    "ContentOrder",
    "PROMPT_TEMPLATES",
    "CONTENT_ORDER_CONFIGS",
    
    # Qwen 相关
    "Qwen3VLBackbone",
    "Qwen3VLConfig", 
    "load_qwen_config",
    "load_qwen_yaml_config",
    "get_qwen_default_config",
    "get_qwen_config_for_model",
    
    # Eagle 相关
    "Eagle25VLBackbone",
    "Eagle25VLConfig",
    "load_eagle_config",
    "load_eagle_yaml_config",
    "get_eagle_default_config",
    "get_eagle_config_for_model",
    
    # 通用
    "HiddenStateOutput",
    "create_backbone",
    "create_backbone_for_model",
]
