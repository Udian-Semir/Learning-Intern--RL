"""
Qwen3-VL Backbone 实现

提供 VLM hidden states 提取接口，可与 Action Head 组合使用。

设计理念:
- VLM 只负责输出 hidden states，不关心下游如何使用
- 配置可通过 YAML 文件定义，也可通过代码参数覆盖
- 提供清晰的接口，方便与各种 Action Head 组合

使用示例:
    from backbone import Qwen3VLBackbone
    
    # 基本使用
    backbone = Qwen3VLBackbone(model_path="Qwen/Qwen3-VL-2B-Instruct")
    hidden_states = backbone.get_hidden_states(images, instruction)
    
    # 自定义配置
    backbone = Qwen3VLBackbone(
        model_path="Qwen/Qwen3-VL-4B-Instruct",
        layers=[16, 17, 18],
        prompt_template="Robot action for: {instruction}",
        device="cuda:0"
    )
    
    # 与 Action Head 组合
    vlm_features = backbone.get_hidden_states(images, instruction)
    actions = action_head(vlm_features, proprioception)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# 导入配置
from .config import Qwen3VLConfig, load_config


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


class Qwen3VLBackbone:
    """
    Qwen3-VL Backbone
    
    负责加载 VLM 模型并提取 hidden states。
    这是一个独立的接口，可以与任何 Action Head 组合使用。
    
    接口设计:
    - get_hidden_states(): 核心方法，输入图像和指令，输出 hidden states
    - 支持单图、双图（agentview + wrist）或多图输入
    - 支持自定义 prompt 模板和内容顺序
    
    使用示例:
        >>> backbone = Qwen3VLBackbone(
        ...     model_path="Qwen/Qwen3-VL-2B-Instruct",
        ...     layers=[14],
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
        config: Qwen3VLConfig = None,
        config_path: str = None,
        **config_overrides
    ):
        """
        初始化 Qwen3-VL Backbone
        
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
        
        # 4. 打印初始化信息
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
        from transformers import AutoModelForImageTextToText, AutoProcessor
        
        print(f"[INFO] Loading Qwen3-VL from: {self.config.model_path}")
        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Layers to extract: {self.config.layers}")
        
        # 解析数据类型
        dtype = self._resolve_dtype(self.config.dtype)
        
        # 加载模型
        model_kwargs = dict(self.config.model_kwargs)
        model_kwargs.setdefault("torch_dtype", dtype)
        model_kwargs.setdefault("trust_remote_code", self.config.trust_remote_code)
        
        # 使用 device_map 来指定设备
        if self.device != "cpu":
            model_kwargs.setdefault("device_map", self.device)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path,
            **model_kwargs
        )
        self.model.eval()
        
        # 确定实际设备
        if hasattr(self.model, "device"):
            self.device = str(self.model.device)
        
        # 加载处理器
        processor_kwargs = dict(self.config.processor_kwargs)
        processor_kwargs.setdefault("trust_remote_code", self.config.trust_remote_code)
        
        self.processor = AutoProcessor.from_pretrained(
            self.config.processor_path,
            **processor_kwargs
        )
        
        # 加载 tokenizer（用于 verbose 模式）
        self.tokenizer = None
        if self.config.verbose:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model_path,
                    trust_remote_code=self.config.trust_remote_code
                )
            except Exception as e:
                print(f"[WARN] Could not load tokenizer: {e}")
        
        print(f"[INFO] Qwen3-VL loaded successfully!")
    
    def _print_init_info(self):
        """打印初始化信息"""
        print("\n" + "=" * 60)
        print("Qwen3-VL Backbone Configuration")
        print("=" * 60)
        print(f"Model: {self.config.model_path}")
        print(f"Device: {self.device}")
        print(f"Layers: {self.config.layers}")
        print(f"Prompt Template: {self.config.prompt_template}")
        print(f"Content Order: {self.config.content_order}")
        print(f"Flip Images: {self.config.flip_images}")
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
            Qwen chat 格式的消息列表
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
                    # 查找下一个未使用的 image_N
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
                # 找到 text 的位置
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
            layers: 要提取的层号（覆盖配置）
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
        
        # 3. 构建消息
        messages = self._build_messages(
            images=image_dict,
            instruction=instruction,
            prompt_template=prompt_template,
            content_order=content_order
        )
        
        # 4. 处理输入
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=self.config.add_generation_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Verbose 模式：打印 token 信息
        if self.config.verbose and self.tokenizer is not None:
            self._print_token_info(inputs)
        
        # 5. 前向传播
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        
        # 6. 提取指定层的 hidden states
        hidden_states_list = []
        for layer_idx in layer_indices:
            # hidden_states[0] 是 embedding，[1:] 是 transformer 层
            hs = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)
            hidden_states_list.append(hs.float())
        
        # 7. 构建输出
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
        
        print("\n" + "=" * 70)
        print("🔍 [Verbose] Token Information")
        print("=" * 70)
        print(f"  Input IDs shape: {input_ids.shape}")
        print(f"  Total tokens: {input_ids.shape[1]}")
        
        # 打印 input_ids 的完整内容
        print(f"\n  Input IDs (tensor):")
        print(f"    {input_ids}")
        
        if self.tokenizer is not None:
            decoded = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
            print(f"\n  Decoded content (complete, {len(decoded)} chars):")
            print("-" * 70)
            print(decoded)
            print("-" * 70)
        
        print("=" * 70 + "\n")
    
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


# Qwen3VLBackbone 已经实现了 VLMInterface 的方法
# 可以直接用作接口

