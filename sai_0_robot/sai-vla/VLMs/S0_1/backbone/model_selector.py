#!/usr/bin/env python3
"""
VLM 模型选择器 (Model Selector)

提供统一的接口选择和配置不同的 VLM backbone。
支持在 shell 脚本中通过参数配置模型、prompt 和运行模式。

================================================================================
                              可用参数一览表
================================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                           基础配置参数                                        │
├─────────────────────┬──────────────────────────┬────────────────────────────┤
│ 参数                 │ 默认值                    │ 说明                        │
├─────────────────────┼──────────────────────────┼────────────────────────────┤
│ --model_type        │ qwen3_vl                 │ 模型类型                    │
│                     │                          │ 可选: qwen3_vl, eagle2_5_vl│
│                     │                          │       cosmos_reason_2b_vl  │
│ --model_path        │ Qwen/Qwen3-VL-2B-Instruct│ 模型路径或 HuggingFace ID   │
│                     │ (Eagle 见下方默认路径)    │                            │
│ --device            │ cuda:0                   │ 设备 (cuda:0, cuda:1, cpu) │
│ --layers            │ "14" (Qwen) / "-1" (Eagle)│ 提取层号，逗号分隔多层       │
│ --mode              │ pipeline                 │ 运行模式                    │
│                     │                          │ 可选: pipeline, hidden_state│
│ --dtype             │ bfloat16                 │ 数据类型                    │
│                     │                          │ 可选: float32, float16,    │
│                     │                          │       bfloat16             │
│ --verbose           │ False                    │ 详细输出 (打印 token 信息)  │
└─────────────────────┴──────────────────────────┴────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           Prompt 配置参数                                    │
├─────────────────────┬──────────────────────────┬────────────────────────────┤
│ 参数                 │ 默认值                    │ 说明                        │
├─────────────────────┼──────────────────────────┼────────────────────────────┤
│ --prompt_template   │ action                   │ Prompt 模板 (预设名或自定义)│
│ --content_order     │ images_first             │ 内容顺序                    │
│ --lowercase_instruction│ True                  │ 仅将 {instruction} 转小写   │
│ --no_lowercase_instruction│ (关闭上述)          │ 不转换指令为小写            │
│ --add_generation_prompt│ True                  │ 添加 generation prompt      │
│ --no_generation_prompt│ (关闭上述)              │ 不添加 generation prompt    │
└─────────────────────┴──────────────────────────┴────────────────────────────┘

--lowercase_instruction 详细说明:
    仅将 {instruction} 占位符的内容转为小写，模板其他部分保持原样。
    
    示例: 
      模板: "What action should the robot take to {instruction}?"
      指令: "Pick Up The Apple"
      
      lowercase_instruction=True  → "What action should the robot take to pick up the apple?"
      lowercase_instruction=False → "What action should the robot take to Pick Up The Apple?"
    
    注意: "What" 始终保持大写，只有 {instruction} 部分受影响。

--prompt_template 预设模板:
┌─────────────┬────────────────────────────────────────────────────────────────┐
│ 预设名称     │ 模板内容                                                        │
├─────────────┼────────────────────────────────────────────────────────────────┤
│ action      │ "What action should the robot take to {instruction}?"          │
│ simple      │ "{instruction}"                                                │
│ detailed    │ "Given the robot's current view, what action should be taken  │
│             │  to accomplish: {instruction}"                                 │
│ step_by_step│ "Based on the images, describe the next action step to        │
│             │  {instruction}"                                                │
└─────────────┴────────────────────────────────────────────────────────────────┘
用法: 
  --prompt_template action                                    # 使用预设名称
  --prompt_template "Robot needs to: {instruction}"           # 自定义模板
  --prompt_template "Task: {task}. What action?"              # {task} 等同于 {instruction}
  --prompt_template "执行: {instruction}，下一步动作是什么？"     # 支持中文

--content_order 可选值 (图像按 --image_keys 顺序):
┌───────────────┬──────────────────────────────────────────────────────────────┐
│ 名称           │ 顺序                                                          │
├───────────────┼──────────────────────────────────────────────────────────────┤
│ images_first  │ [图像1] → [图像2] → ... → [文本 prompt]                       │
│ text_first    │ [文本 prompt] → [图像1] → [图像2] → ...                       │
│ interleaved   │ [图像1] → [文本 prompt] → [图像2] → ...                       │
│ single_image  │ [图像1] → [文本 prompt] (仅使用第一张图像)                     │
└───────────────┴──────────────────────────────────────────────────────────────┘
示例: --image_keys "front,side" --content_order images_first
      → [front 图] → [side 图] → [文本 prompt]

┌─────────────────────────────────────────────────────────────────────────────┐
│                           图像配置参数                                        │
├─────────────────────┬──────────────────────────┬────────────────────────────┤
│ 参数                 │ 默认值                    │ 说明                        │
├─────────────────────┼──────────────────────────┼────────────────────────────┤
│ --flip_images       │ True                     │ 翻转图像 (180度)            │
│ --no_flip_images    │ (关闭上述)                │ 不翻转图像                  │
└─────────────────────┴──────────────────────────┴────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    Hidden State 提取模式专用参数                              │
├─────────────────────┬──────────────────────────┬────────────────────────────┤
│ 参数                 │ 默认值                    │ 说明                        │
├─────────────────────┼──────────────────────────┼────────────────────────────┤
│ --dataset_path      │ None (必填)              │ LeRobot 数据集路径          │
│ --output_dir        │ {dataset_path}/          │ 输出目录                    │
│                     │ vlm_hidden_states        │                            │
│ --image_keys        │ agentview,wrist          │ 图像视角键名 (逗号分隔)     │
│ --start_idx         │ None (从头开始)          │ 起始帧索引 (断点续传)       │
│ --end_idx           │ None (到结尾)            │ 结束帧索引                  │
│ --num_workers       │ 4                        │ 数据预加载 worker 数        │
│ --prefetch_size     │ 8                        │ 预加载队列大小              │
│ --save_hidden_states│ True                     │ 保存 hidden states 到文件   │
│ --no_save_hidden_states│ (关闭上述)             │ 不保存 (仅用于测试)         │
└─────────────────────┴──────────────────────────┴────────────────────────────┘

--image_keys 说明:
  - 指定数据集中的图像视角名称，逗号分隔
  - 对应数据集中的路径: videos/chunk-XXX/observation.images.{key}/
  - 示例:
    --image_keys agentview,wrist           # 默认 (两个视角)
    --image_keys left_camera,right_camera  # 自定义视角名
    --image_keys front,side,top            # 三个视角
    --image_keys agentview                 # 单个视角

┌─────────────────────────────────────────────────────────────────────────────┐
│                           其他参数                                           │
├─────────────────────┬──────────────────────────┬────────────────────────────┤
│ 参数                 │ 默认值                    │ 说明                        │
├─────────────────────┼──────────────────────────┼────────────────────────────┤
│ --list_models       │ False                    │ 列出所有可用模型            │
│ --list_prompts      │ False                    │ 列出所有 prompt 模板        │
└─────────────────────┴──────────────────────────┴────────────────────────────┘

================================================================================
                         运行模式说明 (重要!)
================================================================================

本工具支持两种运行模式，它们对 {instruction} 的处理方式不同：

┌─────────────────────────────────────────────────────────────────────────────┐
│ 模式            │ 用途                │ {instruction} 来源                  │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ pipeline        │ 实时推理            │ 用户代码传入:                        │
│                 │ (评估/部署)         │ backbone.get_hidden_states(         │
│                 │                     │     images, instruction="..."       │
│                 │                     │ )                                   │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ hidden_state    │ 批量提取数据集      │ 自动从数据集读取:                    │
│                 │ (预处理)            │ {dataset_path}/meta/tasks.jsonl     │
│                 │                     │ 每帧根据 task_index 获取对应任务描述 │
└─────────────────┴─────────────────────┴─────────────────────────────────────┘

{instruction} 占位符说明:
    - 在 --prompt_template 中使用 {instruction} 作为占位符
    - 运行时会被实际的任务描述替换
    - 例如: "What action should the robot take to {instruction}?"
           → "What action should the robot take to pick up the apple?"

Hidden State 模式的数据集要求:
    {dataset_path}/
    ├── meta/
    │   ├── info.json          # 数据集信息
    │   └── tasks.jsonl        # 任务描述 (包含 task_index 和 task 字段)
    ├── data/
    │   └── chunk-XXX/         # parquet 文件
    └── videos/
        └── chunk-XXX/         # 视频文件
            ├── observation.images.{image_key_1}/  # 第一个视角
            └── observation.images.{image_key_2}/  # 第二个视角 (可选)

    图像路径格式: videos/chunk-XXX/observation.images.{key}/episode_XXXXXX.mp4
    --image_keys 参数对应 {key} 部分，默认: agentview,wrist

================================================================================
                              使用示例
================================================================================

Pipeline 模式 (实时推理):
    python model_selector.py \\
        --model_type qwen3_vl \\
        --model_path Qwen/Qwen3-VL-2B-Instruct \\
        --mode pipeline \\
        --layers 14 \\
        --verbose

Hidden State 提取模式:
    python model_selector.py \\
        --model_type qwen3_vl \\
        --model_path Qwen/Qwen3-VL-2B-Instruct \\
        --mode hidden_state \\
        --dataset_path /path/to/lerobot_dataset \\
        --output_dir /path/to/output \\
        --layers 14,15,16 \\
        --num_workers 4

Eagle 模型:
    # Eagle 2.5 VL (GR00T-N1.5-3B) 默认路径:
    # /home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e
    python model_selector.py \\
        --model_type eagle2_5_vl \\
        --model_path /home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \\
        --mode hidden_state \\
        --dataset_path /path/to/dataset \\
        --layers -1

断点续传:
    python model_selector.py \\
        --model_type qwen3_vl \\
        --model_path Qwen/Qwen3-VL-2B-Instruct \\
        --mode hidden_state \\
        --dataset_path /path/to/dataset \\
        --start_idx 1000 \\
        --end_idx 2000

================================================================================
                              配置优先级
================================================================================
    函数参数/命令行参数 > prompt_config.yaml > 默认值

================================================================================
                              环境变量
================================================================================
    VLM_MODEL_TYPE      : 模型类型 (qwen3_vl, eagle2_5_vl, cosmos_reason_2b_vl)
    VLM_MODEL_PATH      : 模型路径
    VLM_DEVICE          : 设备 (cuda:0, cuda:1, cpu)
    VLM_LAYERS          : 提取的层号 (逗号分隔)
    VLM_MODE            : 运行模式 (hidden_state, pipeline)
    VLM_PROMPT_TEMPLATE : Prompt 模板
    VLM_CONTENT_ORDER   : 内容顺序 (images_first, text_first, interleaved)

================================================================================
                              Python API
================================================================================

Pipeline 模式 (在代码中使用):
    from model_selector import create_vlm_backbone
    
    # 1. 创建 backbone
    backbone = create_vlm_backbone(
        model_type="qwen3_vl",
        model_path="Qwen/Qwen3-VL-2B-Instruct",
        layers=[14]
    )
    
    # 2. 提取 hidden states (用户传入 instruction)
    output = backbone.get_hidden_states(
        images=[agentview_img, wrist_img],  # PIL.Image 或 numpy array
        instruction="pick up the red apple"  # ← 任务描述
    )
    
    # 3. 获取结果
    vlm_features = output.hidden_states  # List[Tensor], 每层一个
    seq_len = output.seq_len             # 序列长度
    hidden_dim = output.hidden_dim       # 隐藏维度

Hidden State 模式 (命令行批量提取):
    # 直接运行命令行，instruction 自动从数据集 tasks.jsonl 读取
    python model_selector.py \\
        --mode hidden_state \\
        --model_type qwen3_vl \\
        --model_path Qwen/Qwen3-VL-2B-Instruct \\
        --dataset_path /path/to/lerobot_dataset
"""

from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading

# 延迟导入的模块 (在需要时导入)
# import cv2
# import numpy as np
# import pandas as pd
# from PIL import Image
# from tqdm import tqdm


# ============================================================================
# 枚举定义
# ============================================================================

class ModelType(Enum):
    """支持的模型类型"""
    QWEN3_VL = "qwen3_vl"
    EAGLE2_5_VL = "eagle2_5_vl"
    COSMOS_REASON_2B_VL = "cosmos_reason_2b_vl"


class RunMode(Enum):
    """运行模式"""
    HIDDEN_STATE = "hidden_state"  # 提取 hidden states 保存
    PIPELINE = "pipeline"  # 实时推理


class ContentOrder(Enum):
    """内容顺序预设"""
    IMAGES_FIRST = "images_first"  # 图像在前
    TEXT_FIRST = "text_first"  # 文本在前
    INTERLEAVED = "interleaved"  # 交错
    SINGLE_IMAGE = "single_image"  # 单图模式


# ============================================================================
# Prompt 配置 (这些配置会覆盖 prompt_config.yaml 中的对应设置)
# ============================================================================

# 预定义的 Prompt 模板
# 覆盖: prompt_config.yaml → prompt.template
# 使用: --prompt_template <预设名称或自定义模板>
PROMPT_TEMPLATES = {
    "action": "What action should the robot take to {instruction}?",
    "simple": "{instruction}",
    "detailed": "What action should the robot take using its right hand to complete {instruction}",
    "step_by_step": "Based on the images, describe the next action step to {instruction}",
}

# 内容顺序配置
# 覆盖: prompt_config.yaml → content.presets
# 使用: --content_order <预设名称>
# 注意: 这里的 "agentview", "wrist" 会被 --image_keys 参数覆盖
CONTENT_ORDER_CONFIGS = {
    "images_first": [
        {"type": "image", "key": "agentview"},   # 第1张图 (对应 --image_keys 第1个)
        {"type": "image", "key": "wrist"},       # 第2张图 (对应 --image_keys 第2个)
        {"type": "text", "key": "prompt"}
    ],
    "text_first": [
        {"type": "text", "key": "prompt"},
        {"type": "image", "key": "agentview"},   # 第1张图
        {"type": "image", "key": "wrist"}        # 第2张图
    ],
    "interleaved": [
        {"type": "image", "key": "agentview"},   # 第1张图
        {"type": "text", "key": "prompt"},
        {"type": "image", "key": "wrist"}        # 第2张图
    ],
    "single_image": [
        {"type": "image", "key": "agentview"},   # 仅第1张图
        {"type": "text", "key": "prompt"}
    ],
}


@dataclass
class PromptConfig:
    """
    Prompt 配置类
    
    用于配置 VLM 的输入格式，包括 prompt 模板和内容顺序。
    """
    # Prompt 模板
    template: str = PROMPT_TEMPLATES["action"]
    
    # 内容顺序
    content_order: List[Dict[str, str]] = field(
        default_factory=lambda: CONTENT_ORDER_CONFIGS["images_first"]
    )
    
    # 是否将 {instruction} 部分转为小写
    # 注意: 仅影响 {instruction} 占位符内容，模板其他部分保持原样
    # 例如: "What action...to pick up the apple?" 中 "What" 保持大写
    lowercase_instruction: bool = True
    
    # 是否添加 generation prompt
    add_generation_prompt: bool = True
    
    @classmethod
    def from_preset(
        cls,
        template_name: str = "action",
        order_name: str = "images_first",
        **kwargs
    ) -> "PromptConfig":
        """
        从预设创建配置
        
        Args:
            template_name: 模板名称 (action, simple, detailed, step_by_step)
            order_name: 顺序名称 (images_first, text_first, interleaved, single_image)
            **kwargs: 其他配置
        """
        template = PROMPT_TEMPLATES.get(template_name, template_name)
        order = CONTENT_ORDER_CONFIGS.get(order_name, CONTENT_ORDER_CONFIGS["images_first"])
        
        return cls(
            template=template,
            content_order=order,
            **kwargs
        )


# ============================================================================
# 模型选择配置
# ============================================================================

@dataclass
class ModelSelectorConfig:
    """
    模型选择器配置
    
    统一配置所有 VLM 相关参数
    """
    # 基本配置
    model_type: str = "qwen3_vl"
    model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "cuda:0"
    
    # 隐藏层配置
    layers: List[int] = field(default_factory=lambda: [14])
    
    # 运行模式
    mode: str = "pipeline"
    
    # Prompt 配置
    prompt_template: str = PROMPT_TEMPLATES["action"]
    content_order: str = "images_first"
    lowercase_instruction: bool = True
    add_generation_prompt: bool = True
    
    # 图像配置
    flip_images: bool = True
    resize_to: Optional[int] = None
    
    # 数据类型
    dtype: str = "bfloat16"
    
    # 详细输出
    verbose: bool = False
    
    @classmethod
    def from_env(cls) -> "ModelSelectorConfig":
        """
        从环境变量创建配置
        
        支持的环境变量:
        - VLM_MODEL_TYPE
        - VLM_MODEL_PATH
        - VLM_DEVICE
        - VLM_LAYERS (逗号分隔)
        - VLM_MODE
        - VLM_PROMPT_TEMPLATE
        - VLM_CONTENT_ORDER
        """
        layers_str = os.environ.get("VLM_LAYERS", "14")
        layers = [int(x.strip()) for x in layers_str.split(",")]
        
        return cls(
            model_type=os.environ.get("VLM_MODEL_TYPE", "qwen3_vl"),
            model_path=os.environ.get("VLM_MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct"),
            device=os.environ.get("VLM_DEVICE", "cuda:0"),
            layers=layers,
            mode=os.environ.get("VLM_MODE", "pipeline"),
            prompt_template=os.environ.get("VLM_PROMPT_TEMPLATE", PROMPT_TEMPLATES["action"]),
            content_order=os.environ.get("VLM_CONTENT_ORDER", "images_first"),
        )
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ModelSelectorConfig":
        """从命令行参数创建配置"""
        layers = [int(x.strip()) for x in args.layers.split(",")]
        
        return cls(
            model_type=args.model_type,
            model_path=args.model_path,
            device=args.device,
            layers=layers,
            mode=args.mode,
            prompt_template=args.prompt_template,
            content_order=args.content_order,
            lowercase_instruction=args.lowercase_instruction,
            add_generation_prompt=args.add_generation_prompt,
            flip_images=args.flip_images,
            dtype=args.dtype,
            verbose=args.verbose,
        )
    
    def get_prompt_config(self) -> PromptConfig:
        """获取 Prompt 配置"""
        order = CONTENT_ORDER_CONFIGS.get(
            self.content_order, 
            CONTENT_ORDER_CONFIGS["images_first"]
        )
        
        return PromptConfig(
            template=self.prompt_template,
            content_order=order,
            lowercase_instruction=self.lowercase_instruction,
            add_generation_prompt=self.add_generation_prompt,
        )


# ============================================================================
# 工厂函数
# ============================================================================

def create_vlm_backbone(
    model_type: str = "qwen3_vl",
    model_path: str = None,
    device: str = "cuda:0",
    layers: List[int] = None,
    prompt_template: str = None,
    content_order: str = None,
    flip_images: bool = None,
    dtype: str = None,
    verbose: bool = False,
    use_yaml_config: bool = True,
    **kwargs
):
    """
    创建 VLM Backbone
    
    统一的工厂函数，根据 model_type 创建对应的 backbone。
    
    配置优先级: 函数参数 > prompt_config.yaml > 默认值
    
    Args:
        model_type: 模型类型 ("qwen3_vl", "eagle2_5_vl")
        model_path: 模型路径 (本地路径或 HuggingFace ID)
        device: 设备
        layers: 要提取的层号 (None 表示使用 yaml 配置或模型默认值)
        prompt_template: Prompt 模板 (None 表示使用 yaml 配置)
        content_order: 内容顺序 (None 表示使用 yaml 配置)
        flip_images: 是否翻转图像 (None 表示使用 yaml 配置)
        dtype: 数据类型 (None 表示使用 yaml 配置)
        verbose: 详细输出
        use_yaml_config: 是否从 prompt_config.yaml 加载配置 (默认 True)
        **kwargs: 其他配置 (会覆盖 yaml 配置)
        
    Returns:
        VLM Backbone 实例
        
    Example:
        >>> # 使用 yaml 默认配置
        >>> backbone = create_vlm_backbone(
        ...     model_type="qwen3_vl",
        ...     model_path="Qwen/Qwen3-VL-2B-Instruct"
        ... )
        
        >>> # 覆盖部分配置
        >>> backbone = create_vlm_backbone(
        ...     model_type="eagle2_5_vl",
        ...     model_path="/path/to/eagle",
        ...     layers=[-1],  # 覆盖 yaml 中的 default_layers
        ...     prompt_template="Custom: {instruction}"
        ... )
        
        >>> output = backbone.get_hidden_states(images, instruction)
    """
    model_type = model_type.lower()
    
    # 构建覆盖配置字典 (只包含非 None 的值)
    overrides = {"verbose": verbose}
    if model_path is not None:
        overrides["model_path"] = model_path
    if device is not None:
        overrides["device"] = device
    if layers is not None:
        overrides["layers"] = layers
    if dtype is not None:
        overrides["dtype"] = dtype
    if flip_images is not None:
        overrides["flip_images"] = flip_images
    
    # 处理 prompt_template
    if prompt_template is not None:
        # 如果是预设名称，转换为实际模板
        if prompt_template in PROMPT_TEMPLATES:
            overrides["prompt_template"] = PROMPT_TEMPLATES[prompt_template]
        else:
            overrides["prompt_template"] = prompt_template
    
    # 处理 content_order
    if content_order is not None:
        order = CONTENT_ORDER_CONFIGS.get(content_order, CONTENT_ORDER_CONFIGS["images_first"])
        overrides["content_order"] = order
    
    # 合并 kwargs
    overrides.update(kwargs)
    
    if model_type == "qwen3_vl":
        from .qwen3_vl import Qwen3VLBackbone, load_config as load_qwen_config
        
        # 设置 Qwen 特有的默认值
        if "model_path" not in overrides:
            overrides["model_path"] = "Qwen/Qwen3-VL-2B-Instruct"
        
        if use_yaml_config:
            # 从 yaml 加载配置，再用 overrides 覆盖
            config = load_qwen_config(**overrides)
        else:
            # 直接使用参数创建配置
            from .qwen3_vl import Qwen3VLConfig
            config = Qwen3VLConfig(**overrides)
        
        return Qwen3VLBackbone(config=config)
    
    elif model_type == "eagle2_5_vl":
        from .eagle2_5_vl import Eagle25VLBackbone, load_config as load_eagle_config
        
        # Eagle 默认模型路径 (GR00T-N1.5-3B)
        if "model_path" not in overrides:
            overrides["model_path"] = "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
        
        if use_yaml_config:
            # 从 yaml 加载配置，再用 overrides 覆盖
            # Eagle 的模型特有配置 (mlp_connector_layers, use_thumbnail 等) 会从 yaml 加载
            config = load_eagle_config(**overrides)
        else:
            # 直接使用参数创建配置
            from .eagle2_5_vl import Eagle25VLConfig
            config = Eagle25VLConfig(**overrides)
        
        return Eagle25VLBackbone(config=config)
    
    elif model_type == "cosmos_reason_2b_vl":
        from .cosmos_reason_2b_vl import CosmosReason2BVLBackbone, load_config as load_cosmos_config
        
        # Cosmos 默认使用本地 Eagle-Block2A-2B-v2 模型
        if "model_path" not in overrides:
            # 使用相对于 backbone 目录的路径
            import os
            backbone_dir = os.path.dirname(os.path.abspath(__file__))
            overrides["model_path"] = os.path.join(
                backbone_dir, "cosmos_reason_2b_vl", "ori", "modules", "nvidia", "Eagle-Block2A-2B-v2"
            )
        
        if use_yaml_config:
            # 从 yaml 加载配置，再用 overrides 覆盖
            config = load_cosmos_config(**overrides)
        else:
            # 直接使用参数创建配置
            from .cosmos_reason_2b_vl import CosmosReason2BVLConfig
            config = CosmosReason2BVLConfig(**overrides)
        
        return CosmosReason2BVLBackbone(config=config)
    
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Supported types: qwen3_vl, eagle2_5_vl, cosmos_reason_2b_vl"
        )


def create_backbone_from_config(config: ModelSelectorConfig):
    """
    从配置对象创建 backbone
    
    Args:
        config: ModelSelectorConfig 配置对象
        
    Returns:
        VLM Backbone 实例
    """
    return create_vlm_backbone(
        model_type=config.model_type,
        model_path=config.model_path,
        device=config.device,
        layers=config.layers,
        prompt_template=config.prompt_template,
        content_order=config.content_order,
        flip_images=config.flip_images,
        dtype=config.dtype,
        verbose=config.verbose,
    )


def get_prompt_config(
    template_name: str = "action",
    order_name: str = "images_first",
    **kwargs
) -> PromptConfig:
    """
    获取 Prompt 配置
    
    Args:
        template_name: 模板名称 或 自定义模板字符串
        order_name: 顺序名称
        **kwargs: 其他配置
        
    Returns:
        PromptConfig 实例
    """
    return PromptConfig.from_preset(template_name, order_name, **kwargs)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """
    列出所有可用的模型
    
    Returns:
        模型信息字典
    """
    return {
        "qwen3_vl": {
            "description": "Qwen3-VL 系列模型",
            "variants": ["2B", "4B", "7B"],
            "default_model_path": "Qwen/Qwen3-VL-2B-Instruct",
            "default_layers": [14],
            "hidden_dim": {"2B": 1536, "4B": 2560, "7B": 3584},
        },
        "eagle2_5_vl": {
            "description": "Eagle 2.5 VL 模型 (GR00T-N1.5-3B)",
            "variants": ["base"],
            "default_model_path": "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e",
            "default_layers": [-1],
            "hidden_dim": {"base": 2048},
        },
        "cosmos_reason_2b_vl": {
            "description": "Cosmos Reason 2B VL 模型 (Eagle-Block2A-2B-v2)",
            "variants": ["base"],
            "default_model_path": "VLMs/S0_1/backbone/cosmos_reason_2b_vl/ori/modules/nvidia/Eagle-Block2A-2B-v2",
            "default_layers": [-1],
            "hidden_dim": {"base": 2048},
        },
    }


def list_prompt_templates() -> Dict[str, str]:
    """列出所有可用的 Prompt 模板"""
    return PROMPT_TEMPLATES.copy()


def list_content_orders() -> Dict[str, List[Dict[str, str]]]:
    """列出所有可用的内容顺序配置"""
    return CONTENT_ORDER_CONFIGS.copy()


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="VLM Model Selector - 选择和配置 VLM backbone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用 Qwen3-VL-2B
    python model_selector.py --model_type qwen3_vl --model_path Qwen/Qwen3-VL-2B-Instruct
    
    # 使用 Eagle 2.5 VL
    python model_selector.py --model_type eagle2_5_vl --model_path /path/to/eagle_model
    
    # 自定义 prompt
    python model_selector.py --model_type qwen3_vl --prompt_template "Robot should: {instruction}"
    
    # 修改内容顺序
    python model_selector.py --model_type qwen3_vl --content_order text_first
    
环境变量:
    VLM_MODEL_TYPE: 模型类型
    VLM_MODEL_PATH: 模型路径
    VLM_DEVICE: 设备
    VLM_LAYERS: 提取的层号
    VLM_MODE: 运行模式
    VLM_PROMPT_TEMPLATE: Prompt 模板
    VLM_CONTENT_ORDER: 内容顺序
        """
    )
    
    # 基本配置
    parser.add_argument(
        "--model_type", type=str, default="qwen3_vl",
        choices=["qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"],
        help="模型类型"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="模型路径 (本地路径或 HuggingFace ID)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="设备"
    )
    parser.add_argument(
        "--layers", type=str, default="14",
        help="要提取的层号 (逗号分隔)"
    )
    
    # 运行模式
    parser.add_argument(
        "--mode", type=str, default="pipeline",
        choices=["hidden_state", "pipeline"],
        help="运行模式: hidden_state (预处理), pipeline (实时推理)"
    )
    
    # Prompt 配置
    parser.add_argument(
        "--prompt_template", type=str, 
        default=PROMPT_TEMPLATES["action"],
        help="Prompt 模板，可以是预设名称 (action, simple, detailed) 或自定义模板"
    )
    parser.add_argument(
        "--content_order", type=str, default="images_first",
        choices=["images_first", "text_first", "interleaved", "single_image"],
        help="内容顺序"
    )
    parser.add_argument(
        "--lowercase_instruction", action="store_true", default=True,
        help="仅将 {instruction} 部分转为小写，模板其他部分不变 (如 'What action...' 保持大写)"
    )
    parser.add_argument(
        "--no_lowercase_instruction", action="store_false", dest="lowercase_instruction",
        help="不转换指令为小写，保持原始大小写"
    )
    parser.add_argument(
        "--add_generation_prompt", action="store_true", default=True,
        help="添加 generation prompt"
    )
    parser.add_argument(
        "--no_generation_prompt", action="store_false", dest="add_generation_prompt",
        help="不添加 generation prompt"
    )
    
    # 图像配置
    parser.add_argument(
        "--flip_images", action="store_true", default=True,
        help="翻转图像 (180度)"
    )
    parser.add_argument(
        "--no_flip_images", action="store_false", dest="flip_images",
        help="不翻转图像"
    )
    
    # 其他
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="数据类型"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--list_models", action="store_true",
        help="列出所有可用模型"
    )
    parser.add_argument(
        "--list_prompts", action="store_true",
        help="列出所有 prompt 模板"
    )
    
    # ========== Hidden State 提取模式专用参数 ==========
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="[hidden_state 模式] LeRobot 数据集路径"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="[hidden_state 模式] Hidden states 输出目录 (默认: {dataset_path}/vlm_hidden_states)"
    )
    parser.add_argument(
        "--start_idx", type=int, default=None,
        help="[hidden_state 模式] 起始帧索引 (用于断点续传)"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None,
        help="[hidden_state 模式] 结束帧索引 (用于断点续传)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="[hidden_state 模式] 数据预加载的 worker 数量"
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=8,
        help="[hidden_state 模式] 预加载队列大小"
    )
    parser.add_argument(
        "--save_hidden_states", action="store_true", default=True,
        help="[hidden_state 模式] 是否保存 hidden states 到文件 (默认 True)"
    )
    parser.add_argument(
        "--no_save_hidden_states", action="store_false", dest="save_hidden_states",
        help="[hidden_state 模式] 不保存 hidden states (仅用于测试)"
    )
    parser.add_argument(
        "--image_keys", type=str, default="agentview,wrist",
        help="[hidden_state 模式] 图像视角键名，逗号分隔 (默认: agentview,wrist)"
    )
    
    return parser.parse_args()


# ============================================================================
# Hidden State 提取功能
# ============================================================================

def load_dataset_info(dataset_path: str):
    """加载数据集元信息"""
    meta_path = Path(dataset_path) / "meta"
    
    with open(meta_path / "info.json", "r") as f:
        info = json.load(f)
    
    tasks = {}
    with open(meta_path / "tasks.jsonl", "r") as f:
        for line in f:
            task = json.loads(line)
            tasks[task["task_index"]] = task["task"]
    
    return info, tasks


def get_all_frames_info(dataset_path: str, info: dict):
    """获取所有帧的信息列表"""
    import pandas as pd
    
    frames_info = []
    data_path = Path(dataset_path) / "data"
    chunks_size = info["chunks_size"]
    total_episodes = info["total_episodes"]
    
    for episode_idx in range(total_episodes):
        chunk_idx = episode_idx // chunks_size
        parquet_path = data_path / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
        
        if not parquet_path.exists():
            continue
        
        df = pd.read_parquet(parquet_path)
        min_index = df["index"].min()
        
        for _, row in df.iterrows():
            frames_info.append({
                "episode_index": row["episode_index"],
                "frame_index": row["index"] - min_index,
                "global_index": row["vlm_hidden_state_index"] if "vlm_hidden_state_index" in row else row["index"],
                "task_index": row["task_index"],
                "chunk_index": chunk_idx,
            })
    
    return frames_info


def get_video_frame(video_path: str, frame_idx: int):
    """从视频中提取指定帧"""
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取视频帧: {video_path}, frame {frame_idx}")
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_images_from_video(
    dataset_path: str, 
    episode_idx: int, 
    frame_idx: int, 
    chunk_idx: int, 
    image_keys: List[str] = None,
    flip: bool = False
):
    """
    从视频中获取指定视角的图像
    
    Args:
        dataset_path: 数据集路径
        episode_idx: episode 索引
        frame_idx: 帧索引
        chunk_idx: chunk 索引
        image_keys: 图像视角键名列表，如 ["agentview", "wrist"]
        flip: 是否翻转图像 (180度)
    
    Returns:
        图像列表 [img1, img2, ...]
    """
    from PIL import Image
    import numpy as np
    
    if image_keys is None:
        image_keys = ["agentview", "wrist"]
    
    videos_path = Path(dataset_path) / "videos"
    images = []
    
    for key in image_keys:
        video_path = videos_path / f"chunk-{chunk_idx:03d}" / f"observation.images.{key}" / f"episode_{episode_idx:06d}.mp4"
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        img = get_video_frame(str(video_path), frame_idx)
        
        if flip:
            img = img[::-1, ::-1, :].copy()
        
        images.append(Image.fromarray(img))
    
    return images


class HiddenStateExtractor:
    """
    Hidden State 提取器
    
    使用 VLM backbone 从数据集提取 hidden states 并保存
    
    支持:
    - 单层或多层 hidden states 提取
    - 可选是否保存到文件
    - 多线程数据预加载
    - 断点续传
    - 自定义图像视角键名
    """
    
    def __init__(
        self,
        backbone,
        dataset_path: str,
        output_dir: str,
        tasks: dict,
        flip_images: bool = True,
        num_workers: int = 4,
        prefetch_size: int = 8,
        save_to_file: bool = True,
        image_keys: List[str] = None,
        save_dtype: str = "float32",  # 保存 hidden states 的数据类型
    ):
        self.backbone = backbone
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.tasks = tasks
        self.flip_images = flip_images
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        self.save_to_file = save_to_file
        self.image_keys = image_keys or ["agentview", "wrist"]
        self.save_dtype = save_dtype  # 保存 hidden states 的数据类型
        
        self.queue = Queue(maxsize=prefetch_size)
        self.stop_event = threading.Event()
        
        # 获取提取层信息
        self.num_layers = len(backbone.config.layers)
        self.layers = backbone.config.layers
    
    def _load_single_frame(self, frame_info: dict) -> Optional[dict]:
        """加载单帧数据 (per-frame 模式)"""
        global_idx = frame_info["global_index"]
        output_path = os.path.join(self.output_dir, f"hidden_state_{global_idx:06d}.npy")
        
        # 如果已存在，跳过
        if os.path.exists(output_path):
            return None
        
        try:
            task_description = self.tasks[frame_info["task_index"]]
            images = get_images_from_video(
                self.dataset_path,
                frame_info["episode_index"],
                frame_info["frame_index"],
                frame_info["chunk_index"],
                image_keys=self.image_keys,
                flip=False
            )
            return {
                "global_idx": global_idx,
                "output_path": output_path,
                "images": images,
                "instruction": task_description,
            }
        except Exception as e:
            return {"error": str(e), "global_idx": global_idx}

    def _load_single_frame_for_episode(self, frame_info: dict) -> Optional[dict]:
        """加载单帧数据 (per-episode 模式，不检查单帧文件是否存在)"""
        try:
            task_description = self.tasks[frame_info["task_index"]]
            images = get_images_from_video(
                self.dataset_path,
                frame_info["episode_index"],
                frame_info["frame_index"],
                frame_info["chunk_index"],
                image_keys=self.image_keys,
                flip=False
            )
            return {
                "frame_index": frame_info["frame_index"],
                "images": images,
                "instruction": task_description,
            }
        except Exception as e:
            return {"error": str(e), "frame_index": frame_info["frame_index"]}
    
    def _loader_worker(self, frames_info: list):
        """后台加载线程"""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            frame_iter = iter(frames_info)
            
            for _ in range(min(self.num_workers * 2, len(frames_info))):
                try:
                    frame_info = next(frame_iter)
                    futures.append(executor.submit(self._load_single_frame, frame_info))
                except StopIteration:
                    break
            
            while futures and not self.stop_event.is_set():
                future = futures.pop(0)
                result = future.result()
                
                if result is not None:
                    self.queue.put(result)
                
                try:
                    frame_info = next(frame_iter)
                    futures.append(executor.submit(self._load_single_frame, frame_info))
                except StopIteration:
                    pass
            
            self.queue.put(None)  # 结束信号
    
    def extract(self, frames_info: list):
        """
        提取 hidden states
        
        Args:
            frames_info: 帧信息列表
        
        Returns:
            (processed_count, error_count)
        """
        import numpy as np
        from tqdm import tqdm
        
        # 打印提取信息
        print(f"\n📊 提取配置:")
        print(f"  - 提取层: {self.layers} (共 {self.num_layers} 层)")
        print(f"  - 保存到文件: {self.save_to_file}")
        print(f"  - 保存数据类型: {self.save_dtype}")
        if self.save_to_file:
            print(f"  - 输出格式: (num_layers={self.num_layers}, seq_len, hidden_dim)")
        
        if self.save_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 计算需要处理的总数（跳过已存在的文件）
            total_to_process = sum(
                1 for f in frames_info 
                if not os.path.exists(os.path.join(self.output_dir, f"hidden_state_{f['global_index']:06d}.npy"))
            )
            
            if total_to_process == 0:
                print("\n✓ 所有帧已处理完成，无需重复提取")
                return 0, 0
        else:
            total_to_process = len(frames_info)
        
        # 启动加载线程
        loader_thread = threading.Thread(target=self._loader_worker, args=(frames_info,), daemon=True)
        loader_thread.start()
        
        processed_count = 0
        error_count = 0
        first_output_shape = None
        
        with tqdm(total=total_to_process, desc="提取 Hidden States") as pbar:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                
                if "error" in item:
                    print(f"\n⚠️ 帧 {item['global_idx']} 出错: {item['error']}")
                    error_count += 1
                    continue
                
                try:
                    # 使用 backbone 提取 hidden states
                    output = self.backbone.get_hidden_states(
                        images=item["images"],
                        instruction=item["instruction"]
                    )
                    
                    # 堆叠: (num_layers, batch, seq_len, dim)
                    stacked = output.to_stacked_tensor()
                    # 取 batch=0: (num_layers, seq_len, dim)
                    # 根据 save_dtype 转换数据类型
                    if self.save_dtype == "float16":
                        arr = stacked[:, 0, :, :].cpu().half().numpy()
                    else:  # float32 (默认)
                        arr = stacked[:, 0, :, :].cpu().float().numpy()
                    
                    # 记录第一次的输出形状
                    if first_output_shape is None:
                        first_output_shape = arr.shape
                        print(f"\n  首个输出形状: {first_output_shape}")
                        print(f"    - num_layers: {arr.shape[0]}")
                        print(f"    - seq_len: {arr.shape[1]}")
                        print(f"    - hidden_dim: {arr.shape[2]}")
                    
                    # 保存到文件
                    if self.save_to_file:
                        np.save(item["output_path"], arr)
                    
                    processed_count += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\n⚠️ 帧 {item['global_idx']} 提取出错: {e}")
                    error_count += 1
        
        self.stop_event.set()
        loader_thread.join(timeout=1)
        
        return processed_count, error_count

    def extract_per_episode(self, frames_info: list, chunks_size: int = 1000):
        """
        按 episode 分组提取 hidden states，按 chunk 打包为 .npz 文件。

        输出文件: chunk-{chunk_idx:03d}.npz
        npz 内部每个 key: "episode_{ep_idx:06d}" -> (num_frames, num_layers, seq_len, hidden_dim)

        兼容: 同时检查旧格式 episode_XXXXXX.npy 和新格式 chunk-XXX.npz 做断点续传。

        Args:
            frames_info: 帧信息列表 (按 episode 顺序排列)
            chunks_size: 每个 chunk 包含的 episode 数 (默认 1000)

        Returns:
            (processed_count, error_count)
        """
        import numpy as np
        from collections import OrderedDict
        from tqdm import tqdm

        print(f"\n📊 提取配置 (per-episode chunk-npz 模式):")
        print(f"  - 提取层: {self.layers} (共 {self.num_layers} 层)")
        print(f"  - 保存到文件: {self.save_to_file}")
        print(f"  - 保存数据类型: {self.save_dtype}")
        print(f"  - chunk 大小: {chunks_size} episodes/chunk")
        if self.save_to_file:
            print(f"  - 输出格式: chunk-XXX.npz, 内含 episode_XXXXXX keys")

        episodes: OrderedDict[int, list] = OrderedDict()
        for fi in frames_info:
            ep_idx = fi["episode_index"]
            episodes.setdefault(ep_idx, []).append(fi)

        def _episode_already_done(ep_idx: int) -> bool:
            """检查 episode 是否已在某个 chunk npz 中（或旧格式 .npy 中）。"""
            chunk_idx = ep_idx // chunks_size
            npz_path = os.path.join(self.output_dir, f"chunk-{chunk_idx:03d}.npz")
            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path, allow_pickle=False) as data:
                        if f"episode_{ep_idx:06d}" in data:
                            return True
                except Exception:
                    pass
            old_npy = os.path.join(self.output_dir, f"episode_{ep_idx:06d}.npy")
            return os.path.exists(old_npy)

        if self.save_to_file:
            os.makedirs(self.output_dir, exist_ok=True)
            episodes_to_process = OrderedDict()
            for ep_idx, ep_frames in episodes.items():
                if not _episode_already_done(ep_idx):
                    episodes_to_process[ep_idx] = ep_frames

            if not episodes_to_process:
                print("\n✓ 所有 episode 已处理完成，无需重复提取")
                return 0, 0

            total_frames = sum(len(fs) for fs in episodes_to_process.values())
            print(f"  - 待处理 episode 数: {len(episodes_to_process)}")
            print(f"  - 待处理帧数: {total_frames}")
        else:
            episodes_to_process = episodes
            total_frames = sum(len(fs) for fs in episodes_to_process.values())

        processed_count = 0
        error_count = 0
        first_output_shape = None

        chunk_buffer: dict[int, np.ndarray] = {}
        current_chunk_idx: int | None = None

        def _flush_chunk(chunk_idx: int):
            """将攒好的 chunk_buffer 保存为 .npz 文件。"""
            if not chunk_buffer or not self.save_to_file:
                return
            npz_path = os.path.join(self.output_dir, f"chunk-{chunk_idx:03d}.npz")
            existing = {}
            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path, allow_pickle=False) as old:
                        existing = {k: old[k] for k in old.files}
                except Exception:
                    pass
            existing.update(chunk_buffer)
            np.savez(npz_path, **existing)
            print(f"\n  💾 保存 chunk-{chunk_idx:03d}.npz ({len(existing)} episodes)")

        with tqdm(total=total_frames, desc="提取 Hidden States (per-episode)") as pbar:
            for ep_idx, ep_frames in episodes_to_process.items():
                ep_chunk_idx = ep_idx // chunks_size

                if current_chunk_idx is not None and ep_chunk_idx != current_chunk_idx:
                    _flush_chunk(current_chunk_idx)
                    chunk_buffer.clear()
                current_chunk_idx = ep_chunk_idx

                ep_frames_sorted = sorted(ep_frames, key=lambda x: x["frame_index"])
                ep_results = []
                ep_error = False

                for frame_info in ep_frames_sorted:
                    loaded = self._load_single_frame_for_episode(frame_info)
                    if loaded is None:
                        continue

                    if "error" in loaded:
                        print(f"\n⚠️ Episode {ep_idx} 帧 {loaded['frame_index']} 加载出错: {loaded['error']}")
                        error_count += 1
                        ep_error = True
                        pbar.update(1)
                        continue

                    try:
                        output = self.backbone.get_hidden_states(
                            images=loaded["images"],
                            instruction=loaded["instruction"],
                        )
                        stacked = output.to_stacked_tensor()
                        if self.save_dtype == "float16":
                            arr = stacked[:, 0, :, :].cpu().half().numpy()
                        else:
                            arr = stacked[:, 0, :, :].cpu().float().numpy()

                        if first_output_shape is None:
                            first_output_shape = arr.shape
                            print(f"\n  首个输出形状: {first_output_shape}")
                            print(f"    - num_layers: {arr.shape[0]}")
                            print(f"    - seq_len: {arr.shape[1]}")
                            print(f"    - hidden_dim: {arr.shape[2]}")

                        ep_results.append(arr)
                        processed_count += 1
                        pbar.update(1)

                    except Exception as e:
                        print(f"\n⚠️ Episode {ep_idx} 帧 {loaded['frame_index']} 提取出错: {e}")
                        error_count += 1
                        ep_error = True
                        pbar.update(1)

                if ep_results:
                    ep_arr = np.stack(ep_results, axis=0)
                    chunk_buffer[f"episode_{ep_idx:06d}"] = ep_arr
                    if ep_error:
                        print(f"  ⚠️ Episode {ep_idx} 有帧出错，仍保存已成功的 {len(ep_results)} 帧")

        if chunk_buffer and current_chunk_idx is not None:
            _flush_chunk(current_chunk_idx)

        return processed_count, error_count


def run_hidden_state_extraction(args, backbone):
    """
    运行 hidden state 提取
    
    Args:
        args: 命令行参数
        backbone: VLM backbone
    """
    if args.dataset_path is None:
        raise ValueError("hidden_state 模式需要指定 --dataset_path")
    
    # 解析图像键名
    image_keys = [k.strip() for k in args.image_keys.split(",")]
    
    # 获取层信息
    layers = backbone.config.layers
    num_layers = len(layers)
    
    print("\n" + "=" * 70)
    print("Hidden State 提取模式")
    print("=" * 70)
    print(f"模型类型: {args.model_type}")
    print(f"提取层: {layers} (共 {num_layers} 层)")
    print(f"图像视角: {image_keys} (共 {len(image_keys)} 个)")
    print(f"保存到文件: {args.save_hidden_states}")
    
    # 加载数据集信息
    print(f"\n📂 加载数据集: {args.dataset_path}")
    info, tasks = load_dataset_info(args.dataset_path)
    print(f"  总 episodes: {info['total_episodes']}")
    print(f"  总帧数: {info['total_frames']}")
    print(f"  任务数: {len(tasks)}")
    
    # 获取所有帧信息
    print("\n📋 收集帧信息...")
    frames_info = get_all_frames_info(args.dataset_path, info)
    print(f"  收集到 {len(frames_info)} 帧")
    
    # 确定处理范围
    start_idx = args.start_idx if args.start_idx is not None else 0
    end_idx = args.end_idx if args.end_idx is not None else len(frames_info)
    frames_to_process = frames_info[start_idx:end_idx]
    print(f"  处理范围: [{start_idx}, {end_idx}), 共 {len(frames_to_process)} 帧")
    
    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else os.path.join(args.dataset_path, "vlm_hidden_states")
    if args.save_hidden_states:
        print(f"\n💾 输出目录: {output_dir}")
    
    # 创建提取器
    save_dtype = getattr(args, 'save_dtype', 'float32')  # 默认 float32
    extractor = HiddenStateExtractor(
        backbone=backbone,
        dataset_path=args.dataset_path,
        output_dir=output_dir,
        tasks=tasks,
        flip_images=args.flip_images,
        num_workers=args.num_workers,
        prefetch_size=args.prefetch_size,
        save_to_file=args.save_hidden_states,
        image_keys=image_keys,
        save_dtype=save_dtype,
    )
    
    # 提取
    save_per_episode = getattr(args, 'save_per_episode', True)
    chunks_size = info.get("chunks_size", 1000)
    print(f"\n🚀 开始提取... (模式: {'per-episode' if save_per_episode else 'per-frame'})")
    if save_per_episode:
        processed, errors = extractor.extract_per_episode(frames_to_process, chunks_size=chunks_size)
    else:
        processed, errors = extractor.extract(frames_to_process)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("提取完成!")
    print("=" * 70)
    print(f"  成功处理: {processed} 帧")
    print(f"  错误: {errors} 帧")
    print(f"  提取层数: {num_layers} 层 ({layers})")
    if args.save_hidden_states:
        print(f"  输出目录: {output_dir}")
        if save_per_episode:
            print(f"  文件格式: episode_XXXXXX.npy, (num_frames, num_layers={num_layers}, seq_len, hidden_dim)")
        else:
            print(f"  文件格式: hidden_state_XXXXXX.npy, (num_layers={num_layers}, seq_len, hidden_dim)")
    print("=" * 70)


def main():
    """主函数"""
    args = parse_args()
    
    # 列出模型
    if args.list_models:
        print("\n可用的模型:")
        print("-" * 60)
        for name, info in list_available_models().items():
            print(f"\n{name}:")
            print(f"  描述: {info['description']}")
            print(f"  变体: {info['variants']}")
            print(f"  默认路径: {info['default_model_path']}")
            print(f"  默认层: {info['default_layers']}")
        return
    
    # 列出 prompt 模板
    if args.list_prompts:
        print("\n可用的 Prompt 模板:")
        print("-" * 60)
        for name, template in list_prompt_templates().items():
            print(f"\n{name}:")
            print(f"  {template}")
        
        print("\n可用的内容顺序:")
        print("-" * 60)
        for name, order in list_content_orders().items():
            print(f"\n{name}:")
            for item in order:
                print(f"  - {item['type']}: {item['key']}")
        return
    
    # 创建配置
    config = ModelSelectorConfig.from_args(args)
    
    # 打印配置信息
    print("\n" + "=" * 60)
    print("VLM Model Selector 配置")
    print("=" * 60)
    print(f"模型类型: {config.model_type}")
    print(f"模型路径: {config.model_path}")
    print(f"设备: {config.device}")
    print(f"提取层: {config.layers}")
    print(f"运行模式: {config.mode}")
    print(f"Prompt 模板: {config.prompt_template}")
    print(f"内容顺序: {config.content_order}")
    print(f"翻转图像: {config.flip_images}")
    print(f"数据类型: {config.dtype}")
    print("=" * 60)
    
    # 创建 backbone (仅在非列表模式时)
    if args.model_path is not None or args.model_type == "qwen3_vl":
        print("\n正在创建 backbone...")
        backbone = create_backbone_from_config(config)
        print(f"\n✓ Backbone 创建成功!")
        print(f"  模型信息: {backbone.get_model_info()}")
        
        # 根据运行模式执行
        if config.mode == "hidden_state":
            run_hidden_state_extraction(args, backbone)
        else:
            print("\n✅ Pipeline 模式: backbone 已准备好，可用于推理")
            print("   使用方法:")
            print("     output = backbone.get_hidden_states(images, instruction)")
    else:
        print("\n⚠️ 需要指定 --model_path 才能创建 backbone")


if __name__ == "__main__":
    main()

