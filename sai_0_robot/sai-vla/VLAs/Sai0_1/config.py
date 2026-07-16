"""
Sai0_1 配置模块

统一管理 VLM backbone 和 Action Head 的配置。

使用示例:
    from VLAs.Sai0_1 import Sai0Config
    
    config = Sai0Config(
        vlm_type="qwen3_vl",
        vlm_model_path="Qwen/Qwen3-VL-2B-Instruct",
        action_head_type="flow_matching",
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


# ============================================================================
# VLM 配置
# ============================================================================

@dataclass
class VLMConfig:
    """
    VLM Backbone 配置
    
    支持的模型类型:
    - qwen3_vl: Qwen3-VL 系列 (2B, 4B, 7B)
    - eagle2_5_vl: Eagle 2.5 VL (GR00T-N1.5-3B)
    """
    
    # 模型类型
    model_type: str = "qwen3_vl"
    """模型类型: qwen3_vl, eagle2_5_vl"""
    
    # 模型路径
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    """模型路径或 HuggingFace ID
    - Qwen: "Qwen/Qwen3-VL-2B-Instruct"
    - Eagle: "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
    """
    
    # 设备
    device: str = "cuda:0"
    """运行设备"""
    
    # 提取的隐藏层
    layers: List[int] = field(default_factory=lambda: [14])
    """要提取的层号
    - Qwen3-VL-2B: [14] (推荐)
    - Eagle 2.5 VL: [-1] (推荐)
    """
    
    # 数据类型
    dtype: str = "bfloat16"
    """数据类型: float32, float16, bfloat16"""
    
    # Prompt 配置
    prompt_template: str = "What action should the robot take to {instruction}?"
    """Prompt 模板"""
    
    content_order: str = "images_first"
    """内容顺序: images_first, text_first, interleaved, single_image"""
    
    lowercase_instruction: bool = True
    """是否将指令转为小写"""
    
    add_generation_prompt: bool = True
    """是否添加 generation prompt (对于聊天模型)"""
    
    # 图像配置
    flip_images: bool = True
    """是否翻转图像 (180度)"""
    
    image_keys: List[str] = field(default_factory=lambda: ["agentview", "wrist"])
    """图像视角键名"""
    
    # 调试
    verbose: bool = False
    """详细输出"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": self.device,
            "layers": self.layers,
            "dtype": self.dtype,
            "prompt_template": self.prompt_template,
            "content_order": self.content_order,
            "lowercase_instruction": self.lowercase_instruction,
            "add_generation_prompt": self.add_generation_prompt,
            "flip_images": self.flip_images,
            "image_keys": self.image_keys,
            "verbose": self.verbose,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VLMConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def for_qwen3_vl_2b(cls, **kwargs) -> "VLMConfig":
        """Qwen3-VL-2B 预设配置"""
        defaults = {
            "model_type": "qwen3_vl",
            "model_path": "Qwen/Qwen3-VL-2B-Instruct",
            "layers": [14],
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_eagle2_5_vl(cls, **kwargs) -> "VLMConfig":
        """Eagle 2.5 VL 预设配置"""
        defaults = {
            "model_type": "eagle2_5_vl",
            "model_path": "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e",
            "layers": [-1],
        }
        defaults.update(kwargs)
        return cls(**defaults)


# ============================================================================
# Action Head 配置
# ============================================================================

@dataclass
class ActionHeadConfig:
    """
    Action Head 配置
    
    支持的类型:
    - flow_matching_0: Flow Matching Action Head (GR00T N1.5 原始架构)
    - flow_matching_1: Flow Matching Action Head (自定义配置，支持多层 VLM)
    - oft_1_0: OFT Action Head (L1 Regression + Transformer)
    
    简写别名:
    - flow_matching → flow_matching_0
    - fm0 → flow_matching_0
    - fm1 → flow_matching_1
    - oft → oft_1_0
    """
    
    # 类型
    head_type: str = "flow_matching_0"
    """Action Head 类型: flow_matching_0, flow_matching_1, oft_1_0"""
    
    # 预训练权重
    pretrained_weights: Optional[str] = None
    """预训练权重路径"""
    
    # 模型维度
    max_action_dim: int = 32
    """最大动作维度 (padding 目标)"""
    
    max_state_dim: int = 64
    """最大状态维度 (padding 目标)"""
    
    # Action Chunking
    num_action_chunks: int = 16
    """动作预测时间步数 (action_horizon)"""
    
    # 归一化
    use_normalization: bool = True
    """是否使用归一化"""
    
    convert_quat_to_axisangle: bool = True
    """是否将四元数转为轴角"""
    
    # Embodiment
    embodiment_id: int = 31
    """Embodiment ID"""
    
    # ========== Flow Matching 1 特有配置 ==========
    vlm_output_dim: int = 2048
    """VLM 输出维度 (Flow Matching 1 用)"""
    
    action_backbone_dim: int = 1536
    """Action backbone 维度 (Flow Matching 1 用)"""
    
    # ========== OFT 特有配置 ==========
    llm_output_dim: int = 4096
    """LLM 输出维度 (OFT 用)"""
    
    num_vlm_hidden_layers: int = 1
    """VLM 隐藏层数量 (OFT 用)"""
    
    action_dim: int = 7
    """实际动作维度 (OFT 用)"""
    
    proprio_dim: int = 8
    """本体感受维度 (OFT 用)"""
    
    use_diffusion: bool = False
    """是否使用 Diffusion (OFT 用，False=L1 Regression)"""
    
    num_transformer_blocks: int = 4
    """Transformer 块数量 (OFT 用)"""
    
    num_attention_heads: int = 8
    """注意力头数量 (OFT 用)"""
    
    dropout: float = 0.1
    """Dropout 比率 (OFT 用)"""
    
    action_head_hidden_dim: int = 4096
    """Action Head 隐藏层维度 (OFT 用)"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionHeadConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def normalize_head_type(cls, head_type: str) -> str:
        """
        标准化 Action Head 类型名称
        
        别名映射:
        - flow_matching, fm0 → flow_matching_0
        - fm1 → flow_matching_1
        - oft → oft_1_0
        """
        aliases = {
            "flow_matching": "flow_matching_0",
            "fm0": "flow_matching_0",
            "fm1": "flow_matching_1",
            "oft": "oft_1_0",
        }
        return aliases.get(head_type.lower(), head_type.lower())
    
    @classmethod
    def for_flow_matching_0(cls, **kwargs) -> "ActionHeadConfig":
        """Flow Matching 0 (GR00T N1.5 原始) 预设配置"""
        defaults = {
            "head_type": "flow_matching_0",
            "max_action_dim": 32,
            "max_state_dim": 64,
            "num_action_chunks": 16,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_flow_matching_1(cls, vlm_output_dim: int = 2048, **kwargs) -> "ActionHeadConfig":
        """Flow Matching 1 (自定义) 预设配置"""
        defaults = {
            "head_type": "flow_matching_1",
            "max_action_dim": 32,
            "max_state_dim": 64,
            "num_action_chunks": 16,
            "vlm_output_dim": vlm_output_dim,
            "action_backbone_dim": 1536,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_oft(cls, llm_output_dim: int = 4096, num_vlm_layers: int = 1, **kwargs) -> "ActionHeadConfig":
        """OFT 1.0 预设配置"""
        defaults = {
            "head_type": "oft_1_0",
            "num_action_chunks": 16,
            "action_dim": 7,
            "proprio_dim": 8,
            "llm_output_dim": llm_output_dim,
            "num_vlm_hidden_layers": num_vlm_layers,
            "use_diffusion": False,
        }
        defaults.update(kwargs)
        return cls(**defaults)


# ============================================================================
# 数据配置
# ============================================================================

@dataclass
class DataConfig:
    """
    数据加载配置
    
    使用 LeRobot 格式数据集
    """
    
    # 数据集路径
    dataset_path: str = ""
    """LeRobot 数据集路径"""
    
    # 批次大小
    batch_size: int = 32
    """训练批次大小"""
    
    # 数据加载
    num_workers: int = 4
    """数据加载 worker 数"""
    
    # Action Chunking
    num_action_chunks: int = 16
    """Action chunk 数量"""
    
    enable_chunking: bool = True
    """是否启用 action chunking"""
    
    # 缓存
    cache_vlm_states: bool = False
    """是否缓存 VLM hidden states"""
    
    max_cached_video_readers: int = 32
    """缓存的视频 reader 数量上限"""
    
    # 图像键名
    image_keys: List[str] = field(default_factory=lambda: ["agentview", "wrist"])
    """图像视角键名 (对应 observation.images.{key})"""
    
    # 验证集
    val_split: float = 0.0
    """验证集比例"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dataset_path": self.dataset_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "num_action_chunks": self.num_action_chunks,
            "enable_chunking": self.enable_chunking,
            "cache_vlm_states": self.cache_vlm_states,
            "max_cached_video_readers": self.max_cached_video_readers,
            "image_keys": self.image_keys,
            "val_split": self.val_split,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DataConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 训练参数
    epochs: int = 100000
    """训练轮数"""
    
    steps: int = 20000
    """最大训练步数"""
    
    lr: float = 1e-4
    """学习率"""
    
    weight_decay: float = 1e-5
    """权重衰减"""
    
    warmup_ratio: float = 0.05
    """预热比例"""
    
    adam_beta1: float = 0.95
    """AdamW beta1"""
    
    adam_beta2: float = 0.999
    """AdamW beta2"""
    
    gradient_accumulation_steps: int = 1
    """梯度累积步数"""
    
    # 混合精度
    use_amp: bool = True
    """是否使用混合精度训练"""
    
    amp_dtype: str = "float16"
    """混合精度数据类型: float16, bfloat16"""
    
    # 保存
    out_dir: str = "./experiments/sai0_1/checkpoints"
    """Checkpoint 输出目录"""
    
    log_dir: str = "./experiments/sai0_1/logs"
    """日志输出目录"""
    
    save_every: int = 1
    """每 N 个 epoch 保存一次"""
    
    save_every_steps: int = 0
    """每 N 个 step 保存一次 (0=禁用)"""
    
    # W&B
    use_wandb: bool = True
    """是否使用 W&B"""
    
    wandb_project: str = "sai0_1_training"
    """W&B 项目名"""
    
    wandb_run_name: str = ""
    """W&B 运行名"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ============================================================================
# 主配置类
# ============================================================================

@dataclass
class Sai0Config:
    """
    Sai0_1 统一配置
    
    整合 VLM、Action Head、数据和训练配置
    
    使用示例:
        config = Sai0Config(
            vlm=VLMConfig.for_qwen3_vl_2b(),
            action_head=ActionHeadConfig(pretrained_weights="./pretrained.pt"),
            data=DataConfig(dataset_path="/path/to/dataset"),
        )
        
        # 或使用预设
        config = Sai0Config.for_qwen3_libero()
    """
    
    vlm: VLMConfig = field(default_factory=VLMConfig)
    """VLM 配置"""
    
    action_head: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    """Action Head 配置"""
    
    data: DataConfig = field(default_factory=DataConfig)
    """数据配置"""
    
    training: TrainingConfig = field(default_factory=TrainingConfig)
    """训练配置"""
    
    # 模式
    mode: str = "train"
    """运行模式: train, eval, inference"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "vlm": self.vlm.to_dict(),
            "action_head": self.action_head.to_dict(),
            "data": self.data.to_dict(),
            "training": self.training.to_dict(),
            "mode": self.mode,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sai0Config":
        """从字典创建"""
        return cls(
            vlm=VLMConfig.from_dict(d.get("vlm", {})),
            action_head=ActionHeadConfig.from_dict(d.get("action_head", {})),
            data=DataConfig.from_dict(d.get("data", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            mode=d.get("mode", "train"),
        )
    
    def save(self, path: str):
        """保存配置到 JSON 文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Sai0Config":
        """从 JSON 文件加载配置"""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    # ========== 预设配置 ==========
    
    @classmethod
    def for_qwen3_libero(
        cls,
        dataset_path: str = "",
        pretrained_weights: str = None,
        **kwargs
    ) -> "Sai0Config":
        """
        Qwen3-VL-2B + LIBERO 数据集预设配置
        """
        return cls(
            vlm=VLMConfig.for_qwen3_vl_2b(),
            action_head=ActionHeadConfig(
                pretrained_weights=pretrained_weights,
                num_action_chunks=16,
            ),
            data=DataConfig(
                dataset_path=dataset_path,
                num_action_chunks=16,
                image_keys=["agentview", "wrist"],
            ),
            **kwargs
        )
    
    @classmethod
    def for_eagle_libero(
        cls,
        dataset_path: str = "",
        pretrained_weights: str = None,
        **kwargs
    ) -> "Sai0Config":
        """
        Eagle 2.5 VL + LIBERO 数据集预设配置
        """
        return cls(
            vlm=VLMConfig.for_eagle2_5_vl(),
            action_head=ActionHeadConfig(
                pretrained_weights=pretrained_weights,
                num_action_chunks=16,
            ),
            data=DataConfig(
                dataset_path=dataset_path,
                num_action_chunks=16,
                image_keys=["agentview", "wrist"],
            ),
            **kwargs
        )


# ============================================================================
# 工具函数
# ============================================================================

def get_vlm_hidden_dim(vlm_type: str, model_path: str = None) -> int:
    """
    获取 VLM 的 hidden dimension
    
    Args:
        vlm_type: 模型类型 (qwen3_vl, eagle2_5_vl)
        model_path: 模型路径 (用于确定具体变体)
    
    Returns:
        hidden dimension
    """
    hidden_dims = {
        "qwen3_vl": {
            "2B": 1536,
            "4B": 2560,
            "7B": 3584,
        },
        "eagle2_5_vl": {
            "default": 2048,
        },
    }
    
    if vlm_type == "qwen3_vl":
        if model_path:
            if "2B" in model_path:
                return hidden_dims["qwen3_vl"]["2B"]
            elif "4B" in model_path:
                return hidden_dims["qwen3_vl"]["4B"]
            elif "7B" in model_path:
                return hidden_dims["qwen3_vl"]["7B"]
        return hidden_dims["qwen3_vl"]["2B"]  # 默认 2B
    
    elif vlm_type == "eagle2_5_vl":
        return hidden_dims["eagle2_5_vl"]["default"]
    
    raise ValueError(f"Unknown VLM type: {vlm_type}")

