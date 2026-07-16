"""
Cosmos Reason 2B VL Backbone 实现

提供 VLM hidden states 提取接口，可与 Action Head 组合使用。

设计理念:
- VLM 只负责输出 hidden states，不关心下游如何使用
- 配置可通过 YAML 文件定义，也可通过代码参数覆盖
- 提供清晰的接口，方便与各种 Action Head 组合

使用示例:
    from backbone import CosmosReason2BVLBackbone
    
    # 基本使用
    backbone = CosmosReason2BVLBackbone(model_path="/path/to/cosmos_model")
    hidden_states = backbone.get_hidden_states(images, instruction)
    
    # 自定义配置
    backbone = CosmosReason2BVLBackbone(
        model_path="/path/to/cosmos_model",
        layers=[-1],
        prompt_template="Robot action for: {instruction}",
        device="cuda:0"
    )
    
    # 与 Action Head 组合
    vlm_features = backbone.get_hidden_states(images, instruction)
    actions = action_head(vlm_features, proprioception)
"""

from __future__ import annotations

import sys
import os
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# 导入配置
from .config import CosmosReason2BVLConfig, load_config


# 类型定义
ImageInput = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]


@dataclass
class HiddenStateOutput:
    """
    Hidden States 输出结构
    
    这个数据类封装了 VLM 输出的所有信息，方便与 Action Head 对接。
    """
    # 核心输出: hidden states 列表
    # 每个元素形状: (batch_size, seq_len, hidden_dim)
    hidden_states: List[torch.Tensor]
    
    # 提取的层号
    layer_indices: List[int]
    
    # 元数据
    metadata: Dict[str, Any]
    
    @property
    def num_layers(self) -> int:
        """返回层数"""
        return len(self.hidden_states)
    
    @property
    def hidden_dim(self) -> int:
        """返回 hidden dimension"""
        if self.hidden_states:
            return self.hidden_states[0].shape[-1]
        return 0
    
    @property
    def seq_len(self) -> int:
        """返回序列长度"""
        if self.hidden_states:
            return self.hidden_states[0].shape[1]
        return 0
    
    @property
    def batch_size(self) -> int:
        """返回 batch size"""
        if self.hidden_states:
            return self.hidden_states[0].shape[0]
        return 0
    
    def to_stacked_tensor(self) -> torch.Tensor:
        """
        将多层 hidden states 堆叠为单个 tensor
        
        Returns:
            shape: (num_layers, batch_size, seq_len, hidden_dim)
        """
        return torch.stack(self.hidden_states, dim=0)
    
    def to_concatenated_tensor(self, dim: int = 1) -> torch.Tensor:
        """
        将多层 hidden states 在指定维度拼接
        
        Args:
            dim: 拼接维度 (默认 1，即序列维度)
            
        Returns:
            dim=1 时: shape (batch_size, seq_len * num_layers, hidden_dim)
        """
        return torch.cat(self.hidden_states, dim=dim)
    
    def get_layer(self, layer_idx: int) -> torch.Tensor:
        """获取指定层的 hidden states"""
        if layer_idx < 0:
            layer_idx = len(self.hidden_states) + layer_idx
        return self.hidden_states[layer_idx]
    
    def to_numpy(self) -> List[np.ndarray]:
        """转换为 numpy 格式（用于保存）"""
        return [hs.cpu().float().numpy() for hs in self.hidden_states]
    
    def to_device(self, device: str) -> "HiddenStateOutput":
        """移动到指定设备"""
        return HiddenStateOutput(
            hidden_states=[hs.to(device) for hs in self.hidden_states],
            layer_indices=self.layer_indices,
            metadata=self.metadata
        )


class CosmosReason2BVLBackbone:
    """
    Cosmos Reason 2B VL Backbone
    
    负责加载 VLM 模型并提取 hidden states。
    这是一个独立的接口，可以与任何 Action Head 组合使用。
    
    基于 Eagle3_VL (nvidia/Eagle-Block2A-2B-v2) 架构。
    
    接口设计:
    - get_hidden_states(): 核心方法，输入图像和指令，输出 hidden states
    - 支持单图、双图（agentview + wrist）或多图输入
    - 支持自定义 prompt 模板和内容顺序
    
    使用示例:
        >>> backbone = CosmosReason2BVLBackbone(
        ...     model_path="/path/to/cosmos_model",
        ...     layers=[-1],
        ...     device="cuda:0"
        ... )
        >>> 
        >>> # 提取 hidden states
        >>> output = backbone.get_hidden_states(
        ...     images=[agentview_img, wrist_img],
        ...     instruction="pick up the apple"
        ... )
        >>> 
        >>> # 输出可直接用于 Action Head
        >>> vlm_features = output.hidden_states  # List[Tensor]
    """
    
    def __init__(
        self,
        model_path: str = None,
        config: CosmosReason2BVLConfig = None,
        config_path: str = None,
        **config_overrides
    ):
        """
        初始化 Cosmos Reason 2B VL Backbone
        
        配置优先级（从高到低）:
        1. config_overrides (代码直接传入的参数)
        2. config (配置对象)
        3. config_path (YAML 配置文件)
        4. 默认配置
        
        Args:
            model_path: 模型路径，覆盖配置中的 model_path
            config: 配置对象
            config_path: YAML 配置文件路径
            **config_overrides: 其他配置覆盖项
        """
        # 1. 加载配置
        if config is not None:
            self.config = config
            # 应用覆盖
            for key, value in config_overrides.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            # 从文件或默认加载
            self.config = load_config(config_path, **config_overrides)
        
        # 如果直接传入 model_path，覆盖配置
        if model_path is not None:
            self.config.model_path = model_path
            self.config.processor_path = model_path
        
        # 2. 解析设备
        self.device = self._resolve_device(self.config.device)
        
        # 3. 加载模型和处理器
        self._load_model_and_processor()
        
        # 5. 打印初始化信息
        if self.config.verbose:
            self._print_init_info()
    
    def _resolve_device(self, device: str) -> str:
        """解析设备字符串"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        """解析数据类型"""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.bfloat16,
        }
        return dtype_map.get(dtype, torch.bfloat16)
    
    def _load_model_and_processor(self):
        """加载模型和处理器"""
        print(f"[INFO] Loading Cosmos Reason 2B VL (Eagle3) from: {self.config.eagle3_config_path}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Layers to extract: {self.config.layers}")
        
        # 解析数据类型
        dtype = self._resolve_dtype(self.config.dtype)
        
        # 设置路径，确保可以导入自定义模型 (使用本地修复版本)
        eagle3_config_path = Path(self.config.eagle3_config_path)
        modules_dir = eagle3_config_path.parent  # nvidia
        ori_modules_dir = modules_dir.parent  # modules
        ori_dir = ori_modules_dir.parent  # ori
        
        # 添加必要的路径到 sys.path (优先级最高)
        for path in [str(eagle3_config_path), str(modules_dir), str(ori_modules_dir), str(ori_dir)]:
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        
        # 使用本地修复的 Eagle3 模块
        
        # 直接从本地导入已修复的模块 (不依赖 HuggingFace 缓存)
        # 这样换服务器后不需要重新修改 HuggingFace 缓存
        import importlib.util
        import types
        
        def load_module_from_file(module_name: str, file_path: Path, parent_package: str = None):
            """加载本地 Python 模块"""
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            # 设置模块名和包名，使相对导入能工作
            module.__name__ = module_name
            if parent_package:
                module.__package__ = parent_package
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        
        # 创建一个虚拟的父包用于相对导入
        package_name = "eagle3_vl_local"
        if package_name not in sys.modules:
            package_module = types.ModuleType(package_name)
            package_module.__path__ = [str(eagle3_config_path)]
            package_module.__package__ = package_name
            sys.modules[package_name] = package_module
        
        # 按依赖顺序加载模块:
        # 1. 首先加载 modeling_siglip2 (被其他模块依赖)
        siglip2_module = load_module_from_file(
            f"{package_name}.modeling_siglip2",
            eagle3_config_path / "modeling_siglip2.py",
            package_name
        )
        # 也注册不带包名的版本，用于相对导入
        sys.modules["modeling_siglip2"] = siglip2_module
        
        # 2. 加载配置类
        config_module = load_module_from_file(
            f"{package_name}.configuration_eagle3_vl",
            eagle3_config_path / "configuration_eagle3_vl.py",
            package_name
        )
        sys.modules["configuration_eagle3_vl"] = config_module
        Eagle3_VLConfig = config_module.Eagle3_VLConfig
        
        # 3. 加载模型类
        model_module = load_module_from_file(
            f"{package_name}.modeling_eagle3_vl",
            eagle3_config_path / "modeling_eagle3_vl.py",
            package_name
        )
        sys.modules["modeling_eagle3_vl"] = model_module
        Eagle3_VLForConditionalGeneration = model_module.Eagle3_VLForConditionalGeneration
        
        # 4. 加载处理器类
        processor_module = load_module_from_file(
            f"{package_name}.processing_eagle3_vl",
            eagle3_config_path / "processing_eagle3_vl.py",
            package_name
        )
        sys.modules["processing_eagle3_vl"] = processor_module
        Eagle3_VLProcessor = processor_module.Eagle3_VLProcessor
        
        # 加载 Eagle3 配置 (从 config.json)
        import json
        config_json_path = eagle3_config_path / "config.json"
        with open(config_json_path, 'r') as f:
            config_dict = json.load(f)
        
        eagle3_config = Eagle3_VLConfig(**config_dict)
        
        # ========== 关键：检测权重文件的实际 LLM 层数 ==========
        # 如果提供了 model_path，从权重文件中检测实际层数
        # 避免创建比权重更多的层（多余的层会是随机初始化的垃圾数据）
        actual_num_layers = None
        if self.config.model_path:
            actual_num_layers = self._detect_llm_layers_from_weights()
        
        if actual_num_layers is not None:
            original_layers = eagle3_config.text_config.num_hidden_layers
            if actual_num_layers != original_layers:
                eagle3_config.text_config.num_hidden_layers = actual_num_layers
                print(f"[INFO] Modified Eagle3 config: {original_layers} -> {actual_num_layers} LLM layers (based on weight files)")
        
        # 尝试设置 flash_attention_2 实现 (如果可用)
        try:
            from transformers.utils import is_flash_attn_2_available
            if is_flash_attn_2_available():
                if hasattr(eagle3_config.text_config, '_attn_implementation'):
                    eagle3_config.text_config._attn_implementation = 'flash_attention_2'
                print("[INFO] Flash Attention 2 is available and enabled")
            else:
                print("[WARN] Flash Attention 2 is not available, using default attention")
        except ImportError:
            print("[WARN] Could not check flash attention availability")
        
        print(f"[INFO] Eagle3 config loaded: {eagle3_config.model_type}")
        print(f"[INFO] Text config: {eagle3_config.text_config.architectures}")
        print(f"[INFO] Text config num_hidden_layers: {eagle3_config.text_config.num_hidden_layers}")
        print(f"[INFO] Vision config: {eagle3_config.vision_config.model_type}")
        
        # 创建模型 (使用本地的模型类，层数已根据权重调整)
        self.model = Eagle3_VLForConditionalGeneration(eagle3_config)
        
        # 如果提供了 model_path，从中加载权重
        if self.config.model_path:
            self._load_model_weights(dtype)
        
        # 移动到目标设备并设置数据类型
        self.model = self.model.to(device=self.device, dtype=dtype)
        self.model.eval()
        
        # 加载处理器 (使用本地的处理器类)
        
        # 使用 transformers 的 AutoTokenizer 加载 tokenizer (通常不需要修改)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.eagle3_config_path,
            trust_remote_code=True
        )
        
        # 加载 image processor
        image_processor = None
        
        # 方案1: 尝试从本地加载 Eagle3_VLImageProcessorFast (需要新版 transformers)
        try:
            image_processor_module = load_module_from_file(
                f"{package_name}.image_processing_eagle3_vl_fast",
                eagle3_config_path / "image_processing_eagle3_vl_fast.py",
                package_name
            )
            sys.modules["image_processing_eagle3_vl_fast"] = image_processor_module
            Eagle3_VLImageProcessorFast = image_processor_module.Eagle3_VLImageProcessorFast
            
            preprocessor_config_path = eagle3_config_path / "preprocessor_config.json"
            if preprocessor_config_path.exists():
                with open(preprocessor_config_path, 'r') as f:
                    preprocessor_config = json.load(f)
                preprocessor_config.pop('image_processor_type', None)
                preprocessor_config.pop('processor_class', None)
                preprocessor_config.pop('auto_map', None)
                image_processor = Eagle3_VLImageProcessorFast(**preprocessor_config)
            else:
                image_processor = Eagle3_VLImageProcessorFast()
            print("[INFO] Loaded local Eagle3_VLImageProcessorFast")
        except Exception as e:
            print(f"[WARN] Fast image processor not available: {e}")
        
        # 方案2: 尝试使用 SiglipImageProcessor (兼容旧版 transformers)
        if image_processor is None:
            try:
                from transformers import SiglipImageProcessor
                preprocessor_config_path = eagle3_config_path / "preprocessor_config.json"
                if preprocessor_config_path.exists():
                    with open(preprocessor_config_path, 'r') as f:
                        preprocessor_config = json.load(f)
                    # 只保留 SiglipImageProcessor 支持的参数
                    # 强制 do_resize=True 以确保图像尺寸正确
                    # 使用 448x448 (能被 patch_size=14 整除，Eagle3 官方推荐)
                    target_size = preprocessor_config.get('size', {'height': 448, 'width': 448})
                    # 确保尺寸能被 14 整除
                    h = target_size.get('height', 448)
                    w = target_size.get('width', 448)
                    # 调整到最近的 14 的倍数
                    h = max(14, (h // 14) * 14)
                    w = max(14, (w // 14) * 14)
                    siglip_params = {
                        'do_resize': True,  # 强制开启 resize
                        'size': {'height': h, 'width': w},
                        'do_rescale': preprocessor_config.get('do_rescale', True),
                        'rescale_factor': preprocessor_config.get('rescale_factor', 1/255),
                        'do_normalize': preprocessor_config.get('do_normalize', True),
                        'image_mean': preprocessor_config.get('image_mean', [0.5, 0.5, 0.5]),
                        'image_std': preprocessor_config.get('image_std', [0.5, 0.5, 0.5]),
                    }
                    image_processor = SiglipImageProcessor(**siglip_params)
                    print(f"[INFO] Using SiglipImageProcessor with size {h}x{w}")
                else:
                    # 默认使用 448x448 (能被 14 整除，Eagle3 官方推荐)
                    image_processor = SiglipImageProcessor(
                        do_resize=True,
                        size={'height': 448, 'width': 448}
                    )
                print("[INFO] Using SiglipImageProcessor as fallback")
            except Exception as e2:
                print(f"[WARN] SiglipImageProcessor not available: {e2}")
        
        # 方案3: 使用通用的 AutoImageProcessor
        if image_processor is None:
            try:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(
                    "google/siglip-base-patch16-224",  # 使用标准 siglip 模型
                    trust_remote_code=False,
                    do_resize=True,
                    size={'height': 448, 'width': 448}  # 能被 14 整除，Eagle3 官方推荐
                )
                print("[INFO] Using generic SigLIP image processor with size 448x448")
            except Exception as e3:
                print(f"[ERROR] All image processor options failed: {e3}")
                raise RuntimeError(f"Cannot load any image processor")
        
        # 加载 chat template
        chat_template = None
        chat_template_path = eagle3_config_path / "chat_template.json"
        if chat_template_path.exists():
            with open(chat_template_path, 'r') as f:
                chat_template_data = json.load(f)
                chat_template = chat_template_data.get('chat_template')
            print("[INFO] Loaded chat template from chat_template.json")
        
        # 如果 tokenizer 已有 chat_template，优先使用
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            chat_template = tokenizer.chat_template
            print("[INFO] Using chat template from tokenizer")
        
        # 创建处理器
        self.processor = Eagle3_VLProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            chat_template=chat_template
        )
        
        # 获取隐藏层维度
        if hasattr(self.model, 'hidden_size'):
            self.hidden_dim = self.model.hidden_size
        elif hasattr(self.model.config, 'text_config'):
            self.hidden_dim = self.model.config.text_config.hidden_size
        else:
            self.hidden_dim = getattr(self.model.config, 'hidden_size', 2048)
        
        print(f"[INFO] Cosmos Reason 2B VL loaded successfully!")
        print(f"[INFO] Hidden dim: {self.hidden_dim}")
    
    def _detect_llm_layers_from_weights(self) -> Optional[int]:
        """
        从权重文件中检测 LLM 的实际层数
        
        通过分析权重键名来确定模型有多少层 LLM layers。
        例如: backbone.model.language_model.model.layers.15.* 表示有 16 层 (0-15)
        
        Returns:
            检测到的层数，如果无法检测则返回 None
        """
        from safetensors import safe_open
        import json
        import re
        
        model_path = Path(self.config.model_path)
        
        # 查找权重文件
        index_file = model_path / "model.safetensors.index.json"
        single_file = model_path / "model.safetensors"
        
        weight_keys = []
        
        if index_file.exists():
            # 多分片权重文件 - 从索引中读取所有键名
            with open(index_file, 'r') as f:
                weight_index = json.load(f)
            weight_keys = list(weight_index.get("weight_map", {}).keys())
        elif single_file.exists():
            # 单个 safetensors 文件 - 读取键名
            with safe_open(str(single_file), framework="pt", device="cpu") as f:
                weight_keys = list(f.keys())
        else:
            print(f"[WARN] No weight files found in {model_path}, cannot detect layer count")
            return None
        
        # 查找 LLM layers 的最大索引
        # 匹配模式如: backbone.model.language_model.model.layers.15.xxx
        # 或: model.language_model.model.layers.15.xxx
        layer_pattern = re.compile(r'(?:backbone\.)?model\.language_model\.model\.layers\.(\d+)\.')
        
        max_layer_idx = -1
        for key in weight_keys:
            match = layer_pattern.search(key)
            if match:
                layer_idx = int(match.group(1))
                max_layer_idx = max(max_layer_idx, layer_idx)
        
        if max_layer_idx >= 0:
            num_layers = max_layer_idx + 1  # 索引从 0 开始，所以层数 = 最大索引 + 1
            print(f"[INFO] Detected {num_layers} LLM layers from weight files (layer indices 0-{max_layer_idx})")
            return num_layers
        else:
            print(f"[WARN] Could not detect LLM layer count from weight keys")
            return None
    
    def _remap_key(self, key: str) -> str:
        """
        重映射权重键名（在读取时调用）
        
        GR00T-N1.6-3B 权重文件中的键名格式: backbone.model.xxx
        Eagle3 模型期望的键名格式: xxx
        
        移除 'backbone.model.' 前缀
        """
        if key.startswith("backbone.model."):
            return key.replace("backbone.model.", "", 1)
        return key
    
    def _load_model_weights(self, dtype: torch.dtype):
        """
        从模型路径加载权重（采用读取时映射策略，和 Eagle 2.5 相同）
        
        只加载 backbone.model.* 权重（VLM 部分），跳过 action_head.* 等其他权重
        """
        from safetensors import safe_open
        import json
        
        model_path = Path(self.config.model_path)
        
        # 检查是否有权重索引文件
        index_file = model_path / "model.safetensors.index.json"
        single_file = model_path / "model.safetensors"
        pytorch_file = model_path / "pytorch_model.bin"
        
        state_dict = {}
        keys_remapped = 0
        
        if index_file.exists():
            # 多分片权重文件（GR00T 使用这种格式）
            with open(index_file, 'r') as f:
                weight_index = json.load(f)
            
            weight_map = weight_index.get("weight_map", {})
            
            # 1. 先扫描 weight_map，找出 backbone.model.* 的键并建立映射
            # 类似 Eagle 2.5 的做法：只加载我们需要的权重
            backbone_weights = {}  # orig_key -> (new_key, file_name)
            files_to_load = set()
            
            for key, file_name in weight_map.items():
                if key.startswith("backbone.model."):
                    # 重映射键名：backbone.model.xxx -> xxx
                    new_key = self._remap_key(key)
                    backbone_weights[key] = (new_key, file_name)
                    files_to_load.add(file_name)
            
            print(f"[INFO] Found {len(backbone_weights)} backbone weights in {len(files_to_load)} files")
            
            # 2. 加载权重，读取时直接使用映射后的键名
            for file_name in files_to_load:
                file_path = model_path / file_name
                if file_path.exists():
                    print(f"[INFO] Loading weights from: {file_name}")
                    with safe_open(str(file_path), framework="pt", device="cpu") as f:
                        for orig_key, (new_key, fn) in backbone_weights.items():
                            if fn == file_name and orig_key in f.keys():
                                tensor = f.get_tensor(orig_key)
                                if tensor.dtype != dtype and tensor.is_floating_point():
                                    tensor = tensor.to(dtype)
                                state_dict[new_key] = tensor  # 直接用映射后的键名存储
                                keys_remapped += 1
            
            print(f"[INFO] Remapped {keys_remapped} keys (removed 'backbone.model.' prefix)")
            
        elif single_file.exists():
            # 单个 safetensors 文件
            print(f"[INFO] Loading weights from: {single_file}")
            with safe_open(str(single_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    # 只加载 backbone.model.* 权重
                    if key.startswith("backbone.model."):
                        tensor = f.get_tensor(key)
                        if tensor.dtype != dtype and tensor.is_floating_point():
                            tensor = tensor.to(dtype)
                        new_key = self._remap_key(key)
                        state_dict[new_key] = tensor
                        keys_remapped += 1
            
            print(f"[INFO] Remapped {keys_remapped} keys (removed 'backbone.model.' prefix)")
            
        elif pytorch_file.exists():
            # PyTorch 权重文件
            print(f"[INFO] Loading weights from: {pytorch_file}")
            raw_state_dict = torch.load(pytorch_file, map_location="cpu")
            
            for key, tensor in raw_state_dict.items():
                # 只加载 backbone.model.* 权重
                if key.startswith("backbone.model."):
                    if tensor.is_floating_point():
                        tensor = tensor.to(dtype)
                    new_key = self._remap_key(key)
                    state_dict[new_key] = tensor
                    keys_remapped += 1
            
            print(f"[INFO] Remapped {keys_remapped} keys (removed 'backbone.model.' prefix)")
        else:
            print(f"[WARNING] No weight files found in {model_path}, using random initialization")
            return
        
        # 3. 加载权重到模型（一次性加载，无需二次修复）
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"[WARNING] Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        
        print(f"[INFO] Loaded {len(state_dict)} backbone weights successfully")
    
    def _print_init_info(self):
        """打印初始化信息"""
        print("\n" + "=" * 60)
        print("Cosmos Reason 2B VL Backbone Configuration")
        print("=" * 60)
        print(f"Model: {self.config.model_path}")
        print(f"Device: {self.device}")
        print(f"Layers: {self.config.layers}")
        print(f"Prompt Template: {self.config.prompt_template}")
        print(f"Content Order: {self.config.content_order}")
        print(f"Flip Images: {self.config.flip_images}")
        print(f"Hidden Dim: {self.hidden_dim}")
        print("=" * 60 + "\n")
    
    # ========================================================================
    # 图像处理
    # ========================================================================
    
    def _process_image(self, image: ImageInput) -> Image.Image:
        """
        处理单张图像，统一转换为 PIL Image
        
        Args:
            image: 输入图像（多种格式支持）
            
        Returns:
            PIL Image 对象
        """
        # 1. 如果是路径，加载图像
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # 2. 如果是 torch.Tensor，转换为 numpy
        elif isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            image = Image.fromarray(arr)
        
        # 3. 如果是 numpy array，转换为 PIL
        elif isinstance(image, np.ndarray):
            arr = image
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.shape[-1] not in (1, 3, 4):
                if arr.shape[0] in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
            image = Image.fromarray(arr)
        
        # 4. 如果已经是 PIL Image，确保是 RGB
        elif isinstance(image, Image.Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
        
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # 5. 翻转图像（如果需要）
        if self.config.flip_images:
            arr = np.array(image)
            arr = arr[::-1, ::-1, :].copy()
            image = Image.fromarray(arr)
        
        # 6. Resize（如果需要）
        if self.config.resize_to is not None:
            image = image.resize(
                (self.config.resize_to, self.config.resize_to),
                Image.Resampling.LANCZOS
            )
        
        return image
    
    def _prepare_images(
        self, 
        images: Union[ImageInput, List[ImageInput], Dict[str, ImageInput]]
    ) -> Dict[str, Image.Image]:
        """
        准备图像字典
        
        Args:
            images: 输入图像，支持多种格式:
                - 单张图像: 作为 image_0
                - 列表 [img1, img2, ...]: 任意数量图像，使用 image_0, image_1, ... 作为 key
                - 字典 {"key1": img1, "key2": img2}: 明确指定 key
        
        Returns:
            图像字典 {"image_0": PIL.Image, "image_1": PIL.Image, ...}
        """
        if isinstance(images, dict):
            # 字典格式，直接处理
            return {k: self._process_image(v) for k, v in images.items()}
        
        elif isinstance(images, (list, tuple)):
            # 列表格式 - 支持任意数量的图像
            result = {}
            for i, img in enumerate(images):
                result[f"image_{i}"] = self._process_image(img)
            return result
        
        else:
            # 单张图像
            return {"image_0": self._process_image(images)}
    
    # ========================================================================
    # Prompt 构建
    # ========================================================================
    
    def _build_messages(
        self,
        images: Dict[str, Image.Image],
        instruction: str,
        prompt_template: Optional[str] = None,
        content_order: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        构建消息列表
        
        Args:
            images: 图像字典
            instruction: 任务指令
            prompt_template: 自定义 prompt 模板（覆盖配置）
            content_order: 自定义内容顺序（覆盖配置）
            
        Returns:
            Eagle3 chat 格式的消息列表
        """
        # 获取 prompt 模板
        template = prompt_template or self.config.prompt_template
        
        # 格式化 prompt
        text_prompt = template.format(
            instruction=instruction.lower() if self.config.lowercase_instruction else instruction,
            task=instruction.lower() if self.config.lowercase_instruction else instruction
        )
        
        # 获取内容顺序
        order = content_order or self.config.content_order
        
        # 构建内容列表
        content = []
        images_added = set()  # 跟踪已添加的图像
        
        for item in order:
            item_type = item["type"]
            item_key = item["key"]
            
            if item_type == "image":
                # 首先尝试精确匹配 key
                if item_key in images:
                    content.append({
                        "type": "image",
                        "image": images[item_key]
                    })
                    images_added.add(item_key)
                else:
                    # 如果没有精确匹配，尝试按顺序使用 image_N 格式的图像
                    for i in range(len(images)):
                        img_key = f"image_{i}"
                        if img_key in images and img_key not in images_added:
                            content.append({
                                "type": "image",
                                "image": images[img_key]
                            })
                            images_added.add(img_key)
                            break
            elif item_type == "text":
                content.append({
                    "type": "text",
                    "text": text_prompt
                })
        
        # 添加剩余未使用的图像（支持超过 content_order 定义数量的图像）
        for i in range(len(images)):
            img_key = f"image_{i}"
            if img_key in images and img_key not in images_added:
                # 在文本之前插入剩余图像
                text_idx = next((idx for idx, c in enumerate(content) if c.get("type") == "text"), len(content))
                content.insert(text_idx, {
                    "type": "image",
                    "image": images[img_key]
                })
                images_added.add(img_key)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def _build_text_with_placeholders(
        self,
        num_images: int,
        instruction: str,
        prompt_template: Optional[str] = None,
        content_order: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        构建带图像占位符的文本
        
        Args:
            num_images: 图像数量
            instruction: 任务指令
            prompt_template: Prompt 模板
            content_order: 内容顺序
            
        Returns:
            带占位符的格式化文本
        """
        # 获取 prompt 模板
        template = prompt_template or self.config.prompt_template
        
        # 格式化 prompt
        text_prompt = template.format(
            instruction=instruction.lower() if self.config.lowercase_instruction else instruction,
            task=instruction.lower() if self.config.lowercase_instruction else instruction
        )
        
        # 获取内容顺序
        order = content_order or self.config.content_order
        
        # 构建文本（按顺序添加图像占位符和文本）
        parts = []
        image_idx = 0
        for item in order:
            item_type = item["type"]
            
            if item_type == "image":
                if image_idx < num_images:
                    image_idx += 1
                    # 添加图像占位符 (Eagle3 格式: <image-N>)
                    parts.append(f"<image-{image_idx}>")
            elif item_type == "text":
                parts.append(text_prompt)
        
        # 确保所有图像都有占位符
        while image_idx < num_images:
            image_idx += 1
            parts.insert(0, f"<image-{image_idx}>")
        
        # 注意：使用空格连接而非换行符，避免 </img> 后直接换行
        user_content = " ".join(parts) if parts else text_prompt
        
        # 构建完整文本 (Eagle3/Qwen 格式)
        if self.config.add_generation_prompt:
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
        else:
            text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n"
        
        return text
    
    # ========================================================================
    # 核心接口: 提取 Hidden States
    # ========================================================================
    
    @torch.no_grad()
    def get_hidden_states(
        self,
        images: Union[ImageInput, List[ImageInput], Dict[str, ImageInput]],
        instruction: str,
        layers: Optional[List[int]] = None,
        prompt_template: Optional[str] = None,
        content_order: Optional[List[Dict[str, str]]] = None,
        return_dict: bool = True
    ) -> Union[HiddenStateOutput, List[torch.Tensor]]:
        """
        提取 VLM Hidden States
        
        这是核心接口，输入图像和指令，输出 hidden states。
        
        Args:
            images: 输入图像，支持:
                - 单张图像 (numpy/PIL/tensor/path)
                - 列表 [agentview, wrist]
                - 字典 {"agentview": img1, "wrist": img2}
            instruction: 任务指令/描述
            layers: 要提取的层号（覆盖配置），Eagle3 支持负数索引
            prompt_template: 自定义 prompt 模板（覆盖配置）
            content_order: 自定义内容顺序（覆盖配置）
            return_dict: 是否返回 HiddenStateOutput 对象
            
        Returns:
            如果 return_dict=True:
                HiddenStateOutput 对象，包含 hidden_states 列表和元数据
            否则:
                List[torch.Tensor]，每个元素形状 (1, seq_len, hidden_dim)
                
        Example:
            >>> output = backbone.get_hidden_states(
            ...     images=[agentview_img, wrist_img],
            ...     instruction="pick up the red apple"
            ... )
            >>> 
            >>> # 获取 hidden states（用于 Action Head）
            >>> vlm_features = output.hidden_states
            >>> 
            >>> # 获取元数据
            >>> seq_len = output.seq_len
            >>> hidden_dim = output.hidden_dim
        """
        # 1. 确定要提取的层
        layer_indices = layers or self.config.layers
        
        # 2. 准备图像
        image_dict = self._prepare_images(images)
        
        # 3. 准备图像列表（按 content_order 顺序）
        order = content_order or self.config.content_order
        image_inputs = []
        for item in order:
            if item["type"] == "image" and item["key"] in image_dict:
                image_inputs.append(image_dict[item["key"]])
        if len(image_inputs) == 0:
            # 如果没有按顺序找到，使用所有图像
            image_inputs = list(image_dict.values())
        
        # 4. 构建带占位符的文本（直接使用 Eagle3 格式，不依赖 chat template）
        text = self._build_text_with_placeholders(
            num_images=len(image_inputs),
            instruction=instruction,
            prompt_template=prompt_template,
            content_order=content_order
        )
        
        # 5. 处理输入
        inputs = self.processor(
            text=[text],
            images=image_inputs if image_inputs else None,
            videos=None,
            return_tensors="pt",
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # 处理 pixel_values（Eagle3 期望列表格式）
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            if isinstance(pixel_values, list):
                inputs["pixel_values"] = [pv.to(self.device) for pv in pixel_values]
            elif torch.is_tensor(pixel_values):
                # 将 tensor 转换为列表（Eagle3 处理方式）
                inputs["pixel_values"] = [pixel_values.to(self.device)]
        
        # Verbose 模式：打印 token 信息
        if self.config.verbose:
            self._print_token_info(inputs)
        
        # 6. 前向传播
        # Eagle3_VLForConditionalGeneration.forward() 支持的参数
        supported_keys = {
            'pixel_values', 'input_ids', 'attention_mask', 'position_ids', 'image_flags',
            'past_key_values', 'labels', 'use_cache', 'output_attentions', 'output_hidden_states',
            'return_dict'
        }
        filtered_inputs = {k: v for k, v in inputs.items() if k in supported_keys}
        
        
        # 前向传播
        with torch.inference_mode():
            outputs = self.model(
                **filtered_inputs, 
                output_hidden_states=True, 
                return_dict=True
            )
        
        # 7. 提取指定层的 hidden states
        # Eagle3 的 hidden_states 在 language_model 的输出中
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            all_hidden_states = outputs.hidden_states
        else:
            raise ValueError("Model did not return hidden_states. Make sure output_hidden_states=True")
        
        hidden_states_list = []
        num_layers = len(all_hidden_states)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",num_layers) # 17
        for layer_idx in layer_indices:
            # 处理负数索引
            if layer_idx < 0:
                actual_idx = num_layers + layer_idx
            else:
                actual_idx = layer_idx
            
            if actual_idx < 0 or actual_idx >= num_layers:
                raise ValueError(
                    f"Layer index {layer_idx} out of range. "
                    f"Model has {num_layers} layers (0 to {num_layers-1})"
                )
            
            hs = all_hidden_states[actual_idx]  # (batch, seq_len, hidden_dim)
            hidden_states_list.append(hs.float())
        
        # 8. 构建输出
        if return_dict:
            metadata = {
                "seq_len": hidden_states_list[0].shape[1],
                "hidden_dim": hidden_states_list[0].shape[-1],
                "num_layers": len(layer_indices),
                "layer_indices": layer_indices,
                "instruction": instruction,
                "input_ids": inputs.get("input_ids"),
                "attention_mask": inputs.get("attention_mask"),
            }
            
            return HiddenStateOutput(
                hidden_states=hidden_states_list,
                layer_indices=layer_indices,
                metadata=metadata
            )
        else:
            return hidden_states_list
    
    def _print_token_info(self, inputs: Dict[str, Any]):
        """打印 token 信息（verbose 模式）"""
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return
        
        print("\n" + "=" * 60)
        print("[Token Information]")
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Total tokens: {input_ids.shape[1]}")
        
        # 打印其他输入键
        print(f"  Input keys: {list(inputs.keys())}")
        if "pixel_values" in inputs:
            pv = inputs["pixel_values"]
            if isinstance(pv, list):
                print(f"  Pixel values: {len(pv)} items")
                for i, p in enumerate(pv):
                    print(f"    [{i}] shape: {p.shape if hasattr(p, 'shape') else type(p)}")
            else:
                print(f"  Pixel values shape: {pv.shape if hasattr(pv, 'shape') else type(pv)}")
        if "image_sizes" in inputs:
            print(f"  Image sizes: {inputs['image_sizes']}")
        if "attention_mask" in inputs:
            am = inputs["attention_mask"]
            print(f"  Attention mask shape: {am.shape if hasattr(am, 'shape') else type(am)}")
        
        # 尝试解码 - 完整打印
        try:
            decoded = self.processor.decode(input_ids[0], skip_special_tokens=False)
            print(f"\n  Decoded tokens (full, {len(decoded)} chars):")
            print("-" * 60)
            print(decoded)
            print("-" * 60)
        except Exception as e:
            print(f"  Could not decode: {e}")
        
        print("=" * 60 + "\n")
    
    # ========================================================================
    # 便捷方法
    # ========================================================================
    
    def extract_and_save(
        self,
        images: Union[ImageInput, List[ImageInput], Dict[str, ImageInput]],
        instruction: str,
        output_path: Union[str, Path],
        **kwargs
    ) -> np.ndarray:
        """
        提取 hidden states 并保存为 numpy 文件
        
        Args:
            images: 输入图像
            instruction: 任务指令
            output_path: 输出文件路径 (.npy)
            **kwargs: 传递给 get_hidden_states 的其他参数
            
        Returns:
            保存的数组，形状 (num_layers, seq_len, hidden_dim)
        """
        output = self.get_hidden_states(images, instruction, **kwargs)
        
        # 堆叠并转换为 numpy
        stacked = output.to_stacked_tensor()  # (num_layers, batch, seq_len, dim)
        arr = stacked[:, 0, :, :].cpu().float().numpy()  # 取 batch=0
        
        # 保存
        np.save(output_path, arr)
        
        return arr
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_path": self.config.model_path,
            "device": self.device,
            "layers": self.config.layers,
            "dtype": self.config.dtype,
            "prompt_template": self.config.prompt_template,
            "content_order": self.config.content_order,
            "hidden_dim": self.hidden_dim,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }


# ============================================================================
# 接口类: 用于与 Action Head 组合
# ============================================================================

class VLMInterface:
    """
    VLM 接口抽象类
    
    定义了 VLM 与 Action Head 交互的标准接口。
    任何 VLM backbone 都应该实现这个接口。
    """
    
    def get_hidden_states(
        self,
        images: Any,
        instruction: str,
        **kwargs
    ) -> HiddenStateOutput:
        """
        提取 hidden states
        
        Args:
            images: 输入图像
            instruction: 任务指令
            
        Returns:
            HiddenStateOutput 对象
        """
        raise NotImplementedError
    
    @property
    def hidden_dim(self) -> int:
        """返回 hidden dimension"""
        raise NotImplementedError
    
    @property
    def num_layers(self) -> int:
        """返回提取的层数"""
        raise NotImplementedError


# CosmosReason2BVLBackbone 已经实现了 VLMInterface 的方法
# 可以直接用作接口

