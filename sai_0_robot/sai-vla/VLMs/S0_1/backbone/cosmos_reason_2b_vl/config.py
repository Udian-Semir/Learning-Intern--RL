"""
Cosmos Reason 2B VL 配置模块

提供配置类定义和配置加载功能。
所有配置值都可以被代码传入的参数覆盖。

使用示例:
    from config import CosmosReason2BVLConfig, load_config
    
    # 加载默认配置
    config = load_config()
    
    # 加载并覆盖部分配置
    config = load_config(
        layers=[-1],
        prompt_template="Custom template: {instruction}"
    )
    
    # 直接创建配置对象
    config = CosmosReason2BVLConfig(
        model_path="/path/to/cosmos_model",
        layers=[-1],
        prompt_template="What action should the robot take to {instruction}?"
    )
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# 配置文件默认路径
DEFAULT_CONFIG_PATH = Path(__file__).parent / "prompt_config.yaml"

# 默认 Eagle3 模型路径 (Eagle-Block2A-2B-v2 目录)
DEFAULT_EAGLE3_MODEL_PATH = Path(__file__).parent / "ori" / "modules" / "nvidia" / "Eagle-Block2A-2B-v2"


@dataclass
class PromptConfig:
    """Prompt 配置"""
    # Prompt 模板，支持 {instruction} 或 {task} 占位符
    template: str = "What action should the robot take to {instruction}?"
    
    # 是否将任务描述转为小写
    lowercase_instruction: bool = True
    
    # 是否添加 generation prompt
    add_generation_prompt: bool = True


@dataclass 
class ContentConfig:
    """消息内容顺序配置"""
    # 内容顺序，每个元素是 {"type": "image"|"text", "key": str}
    order: List[Dict[str, str]] = field(default_factory=lambda: [
        {"type": "image", "key": "agentview"},
        {"type": "image", "key": "wrist"},
        {"type": "text", "key": "prompt"}
    ])
    
    # 预设模板名称（可选，如果指定则使用预设）
    preset: Optional[str] = None


@dataclass
class ImageConfig:
    """图像处理配置"""
    # 最大图像数量
    max_images: int = 2
    
    # 是否翻转图像 (180度旋转)
    flip_images: bool = True
    
    # resize 目标尺寸
    resize_to: Optional[int] = None
    
    # 图像键名
    keys: Dict[str, str] = field(default_factory=lambda: {
        "agentview": "agentview",
        "wrist": "wrist"
    })


@dataclass
class ModelConfig:
    """VLM 模型配置"""
    # 默认提取的隐藏层 (Eagle3 默认使用最后一层)
    default_layers: List[int] = field(default_factory=lambda: [-1])
    
    # 数据类型
    dtype: str = "bfloat16"
    
    # 是否使用 trust_remote_code
    trust_remote_code: bool = True


@dataclass
class CosmosReason2BVLConfig:
    """
    Cosmos Reason 2B VL 完整配置
    
    所有配置项都可以通过构造函数参数覆盖
    """
    # ========== 模型路径 ==========
    model_path: str = ""  # 本地模型路径或 HuggingFace 模型 ID
    processor_path: Optional[str] = None  # None 表示使用 model_path
    
    # 本地 Eagle3 模型配置路径 (Eagle-Block2A-2B-v2 目录)
    eagle3_config_path: Optional[str] = None
    
    # ========== 设备和数据类型 ==========
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    # ========== 隐藏层提取 ==========
    # 要提取的 VLM transformer 层号
    # Eagle3 默认使用最后一层: [-1]
    layers: List[int] = field(default_factory=lambda: [-1])
    
    # ========== Prompt 配置 ==========
    # Prompt 模板
    prompt_template: str = "What action should the robot take to {instruction}?"
    
    # 是否将指令转为小写
    lowercase_instruction: bool = True
    
    # 是否添加 generation prompt
    add_generation_prompt: bool = True
    
    # ========== 内容顺序 ==========
    # 内容顺序列表，每个元素是 {"type": "image"|"text", "key": str}
    content_order: List[Dict[str, str]] = field(default_factory=lambda: [
        {"type": "image", "key": "agentview"},
        {"type": "image", "key": "wrist"},
        {"type": "text", "key": "prompt"}
    ])
    
    # 或使用预设名称: "standard", "text_first", "single_image", "interleaved"
    content_preset: Optional[str] = None
    
    # ========== 图像配置 ==========
    max_images: int = 2
    flip_images: bool = True
    resize_to: Optional[int] = None
    
    # ========== Eagle3 特定配置 ==========
    # 是否使用 pixel shuffle
    use_pixel_shuffle: bool = True
    
    # downsample ratio
    downsample_ratio: float = 0.5
    
    # MLP connector 层数
    mlp_connector_layers: int = 2
    
    # 动态 tile 配置
    min_dynamic_tiles: int = 1
    max_dynamic_tiles: int = 12
    
    # 是否使用 thumbnail
    use_thumbnail: bool = False
    
    # select_layer for vision encoder
    select_layer: int = -1
    
    # ========== 其他配置 ==========
    trust_remote_code: bool = True
    verbose: bool = False
    
    # ========== 模型加载参数 ==========
    # 传递给 from_pretrained 的额外参数
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    processor_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """配置后处理"""
        # 如果指定了预设，加载预设的内容顺序
        if self.content_preset is not None:
            self.content_order = self._get_preset_order(self.content_preset)
        
        # 如果 processor_path 未指定，使用 model_path
        if self.processor_path is None:
            self.processor_path = self.model_path
        
        # 如果 eagle3_config_path 未指定，使用默认路径
        if self.eagle3_config_path is None:
            self.eagle3_config_path = str(DEFAULT_EAGLE3_MODEL_PATH)
    
    def _get_preset_order(self, preset_name: str) -> List[Dict[str, str]]:
        """获取预设的内容顺序"""
        presets = {
            "standard": [
                {"type": "image", "key": "agentview"},
                {"type": "image", "key": "wrist"},
                {"type": "text", "key": "prompt"}
            ],
            "text_first": [
                {"type": "text", "key": "prompt"},
                {"type": "image", "key": "agentview"},
                {"type": "image", "key": "wrist"}
            ],
            "single_image": [
                {"type": "image", "key": "agentview"},
                {"type": "text", "key": "prompt"}
            ],
            "interleaved": [
                {"type": "image", "key": "agentview"},
                {"type": "text", "key": "prompt"},
                {"type": "image", "key": "wrist"}
            ]
        }
        
        if preset_name not in presets:
            raise ValueError(
                f"Unknown content preset: {preset_name}. "
                f"Available presets: {list(presets.keys())}"
            )
        
        return presets[preset_name]
    
    def format_prompt(self, instruction: str) -> str:
        """
        根据模板格式化 prompt
        
        Args:
            instruction: 任务指令
            
        Returns:
            格式化后的 prompt 文本
        """
        if self.lowercase_instruction:
            instruction = instruction.lower()
        
        # 支持 {instruction} 和 {task} 两种占位符
        return self.prompt_template.format(
            instruction=instruction,
            task=instruction
        )
    
    def get_content_order(self) -> List[Dict[str, str]]:
        """获取内容顺序"""
        return self.content_order.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_path": self.model_path,
            "processor_path": self.processor_path,
            "eagle3_config_path": self.eagle3_config_path,
            "device": self.device,
            "dtype": self.dtype,
            "layers": self.layers,
            "prompt_template": self.prompt_template,
            "lowercase_instruction": self.lowercase_instruction,
            "add_generation_prompt": self.add_generation_prompt,
            "content_order": self.content_order,
            "content_preset": self.content_preset,
            "max_images": self.max_images,
            "flip_images": self.flip_images,
            "resize_to": self.resize_to,
            "use_pixel_shuffle": self.use_pixel_shuffle,
            "downsample_ratio": self.downsample_ratio,
            "mlp_connector_layers": self.mlp_connector_layers,
            "min_dynamic_tiles": self.min_dynamic_tiles,
            "max_dynamic_tiles": self.max_dynamic_tiles,
            "use_thumbnail": self.use_thumbnail,
            "select_layer": self.select_layer,
            "trust_remote_code": self.trust_remote_code,
            "verbose": self.verbose,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CosmosReason2BVLConfig":
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


def load_yaml_config(config_path: Union[str, Path] = None) -> Dict[str, Any]:
    """
    加载 YAML 配置文件
    
    Args:
        config_path: 配置文件路径，None 使用默认路径
        
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"[WARN] Config file not found: {config_path}, using defaults")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(
    config_path: Union[str, Path] = None,
    **overrides
) -> CosmosReason2BVLConfig:
    """
    加载配置并应用覆盖
    
    YAML 配置文件中的值会被代码传入的参数覆盖。
    
    Args:
        config_path: YAML 配置文件路径
        **overrides: 要覆盖的配置项
        
    Returns:
        CosmosReason2BVLConfig 配置对象
        
    Example:
        # 使用默认配置
        config = load_config()
        
        # 覆盖部分配置
        config = load_config(
            layers=[-1],
            prompt_template="Custom: {instruction}",
            device="cuda:0"
        )
        
        # 使用自定义配置文件
        config = load_config(
            config_path="./my_config.yaml",
            verbose=True
        )
    """
    # 1. 加载 YAML 配置
    yaml_config = load_yaml_config(config_path)
    
    # 2. 解析 YAML 配置到扁平字典
    config_dict = {}
    
    # Prompt 配置
    if "prompt" in yaml_config:
        prompt_cfg = yaml_config["prompt"]
        config_dict["prompt_template"] = prompt_cfg.get("template")
        config_dict["lowercase_instruction"] = prompt_cfg.get("lowercase_instruction")
        config_dict["add_generation_prompt"] = prompt_cfg.get("add_generation_prompt")
    
    # 内容顺序配置
    if "content" in yaml_config:
        content_cfg = yaml_config["content"]
        config_dict["content_order"] = content_cfg.get("order")
    
    # 图像配置
    if "images" in yaml_config:
        image_cfg = yaml_config["images"]
        config_dict["max_images"] = image_cfg.get("max_images")
        config_dict["flip_images"] = image_cfg.get("flip_images")
        config_dict["resize_to"] = image_cfg.get("resize_to")
    
    # 模型配置
    if "model" in yaml_config:
        model_cfg = yaml_config["model"]
        config_dict["layers"] = model_cfg.get("default_layers")
        config_dict["dtype"] = model_cfg.get("dtype")
        config_dict["trust_remote_code"] = model_cfg.get("trust_remote_code")
    
    # Eagle3 特定配置
    if "eagle3" in yaml_config:
        eagle3_cfg = yaml_config["eagle3"]
        config_dict["use_pixel_shuffle"] = eagle3_cfg.get("use_pixel_shuffle")
        config_dict["downsample_ratio"] = eagle3_cfg.get("downsample_ratio")
        config_dict["mlp_connector_layers"] = eagle3_cfg.get("mlp_connector_layers")
        config_dict["min_dynamic_tiles"] = eagle3_cfg.get("min_dynamic_tiles")
        config_dict["max_dynamic_tiles"] = eagle3_cfg.get("max_dynamic_tiles")
        config_dict["use_thumbnail"] = eagle3_cfg.get("use_thumbnail")
        config_dict["select_layer"] = eagle3_cfg.get("select_layer")
    
    # 3. 移除 None 值
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    
    # 4. 应用代码传入的覆盖（优先级最高）
    config_dict.update(overrides)
    
    # 5. 创建配置对象
    return CosmosReason2BVLConfig(**config_dict)


# ============================================================================
# 便捷函数
# ============================================================================

def get_default_config() -> CosmosReason2BVLConfig:
    """获取默认配置（不加载 YAML 文件）"""
    return CosmosReason2BVLConfig()


def get_config_for_model(model_path: str, **overrides) -> CosmosReason2BVLConfig:
    """
    根据模型路径获取配置
    
    Args:
        model_path: 模型路径 (本地路径或 HuggingFace ID)
        **overrides: 覆盖配置
        
    Returns:
        配置对象
    """
    config_dict = {
        "model_path": model_path,
        "layers": [-1],  # Eagle3 默认使用最后一层
    }
    config_dict.update(overrides)
    
    return CosmosReason2BVLConfig(**config_dict)

