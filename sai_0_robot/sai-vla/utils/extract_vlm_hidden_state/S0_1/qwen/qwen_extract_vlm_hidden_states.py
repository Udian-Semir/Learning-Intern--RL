#!/usr/bin/env python3
"""
Qwen3-VL Hidden States 提取脚本

从 LeRobot 格式数据集中使用 Qwen3-VL 模型提取 hidden states。

================================================================================
                              使用说明
================================================================================

基本用法:
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/lerobot_dataset \\
        --model_path Qwen/Qwen3-VL-2B-Instruct

使用默认 Qwen3-VL-2B 模型:
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/lerobot_dataset

自定义提取层:
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/lerobot_dataset \\
        --layers "14,15,16"

断点续传:
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/lerobot_dataset \\
        --start_idx 1000 \\
        --end_idx 2000

自定义图像视角:
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/lerobot_dataset \\
        --image_keys "top,left_wrist"

================================================================================
                              参数说明
================================================================================

必需参数:
    --dataset_path      : LeRobot 数据集路径

可选参数:
    --model_path        : Qwen3-VL 模型路径 (默认: Qwen/Qwen3-VL-2B-Instruct)
    --output_dir        : 输出目录 (默认: {dataset_path}/vlm_hidden_states)
    --layers            : 提取的层号，逗号分隔 (默认: "14")
                          Qwen3-VL-2B 共有 28 层 transformer (层号 1-28)
    --image_keys        : 图像视角键名，逗号分隔 (默认: "agentview,wrist")
    --device            : 设备 (默认: "cuda:0")
    --num_workers       : 数据预加载 worker 数 (默认: 4)
    --prefetch_size     : 预加载队列大小 (默认: 8)
    --start_idx         : 起始帧索引 (断点续传)
    --end_idx           : 结束帧索引 (断点续传)
    --prompt_template   : Prompt 模板 (默认: "action")
    --content_order     : 内容顺序 (默认: "images_first")
    --flip_images       : 翻转图像 (默认: True)
    --no_flip_images    : 不翻转图像
    --verbose           : 详细输出

================================================================================
                              数据集要求
================================================================================

数据集结构:
    {dataset_path}/
    ├── meta/
    │   ├── info.json           # 数据集信息
    │   └── tasks.jsonl         # 任务描述
    ├── data/
    │   └── chunk-XXX/          # parquet 文件
    └── videos/
        └── chunk-XXX/
            ├── observation.images.{key1}/  # 第一个视角
            └── observation.images.{key2}/  # 第二个视角

输出格式:
    {output_dir}/
    ├── chunk-XXX.npz            # 默认 (per-episode / chunk 打包)
    │                            # 每个 key: "episode_XXXXXX" -> (num_frames, num_layers, seq_len, hidden_dim)
    └── hidden_state_XXXXXX.npy  # 旧格式 (--no_save_per_episode), shape: (num_layers, seq_len, hidden_dim)

================================================================================
                              模型信息
================================================================================

Qwen3-VL 系列模型:
    - Qwen3-VL-2B-Instruct: 28 层 transformer, hidden_dim=1536
    - Qwen3-VL-4B-Instruct: 28 层 transformer, hidden_dim=2560
    - Qwen3-VL-7B-Instruct: 28 层 transformer, hidden_dim=3584

层号说明:
    - 层号范围: 1-28 (Qwen3-VL-2B 为例)
    - hidden_states[0]: embedding 层输出
    - hidden_states[1-28]: transformer 第 1-28 层的输出
    - 推荐层: 中间层 (如 14) 或后几层

================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

# 添加项目根目录到 Python 路径
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]  # utils/extract_vlm_hidden_state/S0_1/qwen -> project root
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 默认配置
# ============================================================================

# Qwen3-VL 默认模型路径
DEFAULT_QWEN_MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"

# 默认提取层 (Qwen3-VL-2B 的中间层)
DEFAULT_LAYERS = [14]

# 默认图像键
DEFAULT_IMAGE_KEYS = ["agentview", "wrist"]

# 模型信息 (用于验证)
QWEN_MODEL_INFO = {
    "2b": {
        "model_path": "Qwen/Qwen3-VL-2B-Instruct",
        "num_layers": 28,
        "hidden_dim": 1536,
    },
    "4b": {
        "model_path": "Qwen/Qwen3-VL-4B-Instruct",
        "num_layers": 28,
        "hidden_dim": 2560,
    },
    "7b": {
        "model_path": "Qwen/Qwen3-VL-7B-Instruct",
        "num_layers": 28,
        "hidden_dim": 3584,
    },
}


# ============================================================================
# 参数解析
# ============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-VL Hidden States 提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法
    python qwen_extract_vlm_hidden_states.py --dataset_path /path/to/dataset
    
    # 指定模型和层
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/dataset \\
        --model_path Qwen/Qwen3-VL-2B-Instruct \\
        --layers "14,15,16"
    
    # 断点续传
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/dataset \\
        --start_idx 1000 --end_idx 2000
    
    # 使用 4B 模型
    python qwen_extract_vlm_hidden_states.py \\
        --dataset_path /path/to/dataset \\
        --model_path Qwen/Qwen3-VL-4B-Instruct \\
        --layers "16,17,18"
        """
    )
    
    # 数据集配置
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="LeRobot 数据集路径 (必填)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录 (默认: {dataset_path}/vlm_hidden_states)"
    )
    
    # 模型配置
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_QWEN_MODEL_PATH,
        help=f"Qwen3-VL 模型路径 (默认: {DEFAULT_QWEN_MODEL_PATH})"
    )
    parser.add_argument(
        "--layers", type=str, default="14",
        help="提取的层号，逗号分隔 (默认: 14)。Qwen3-VL-2B 有 28 层 (1-28)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="设备 (默认: cuda:0)"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="数据类型 (默认: bfloat16)"
    )
    
    # 图像配置
    parser.add_argument(
        "--image_keys", type=str, default=None,
        help="图像视角键名，逗号分隔；不传时默认自动检测"
    )
    parser.add_argument(
        "--auto_image_keys", action="store_true", default=True,
        help="自动从数据集检测图像视角键名 (默认: True)"
    )
    parser.add_argument(
        "--no_auto_image_keys", action="store_false", dest="auto_image_keys",
        help="关闭自动检测，未指定 --image_keys 时回退到默认值"
    )
    parser.add_argument(
        "--flip_images", action="store_true", default=True,
        help="翻转图像 180 度 (默认: True)"
    )
    parser.add_argument(
        "--no_flip_images", action="store_false", dest="flip_images",
        help="不翻转图像"
    )
    
    # Prompt 配置
    parser.add_argument(
        "--prompt_template", type=str, default="action",
        help="Prompt 模板名称或自定义模板 (默认: action)"
    )
    parser.add_argument(
        "--content_order", type=str, default="images_first",
        choices=["images_first", "text_first", "interleaved", "single_image"],
        help="内容顺序 (默认: images_first)"
    )
    parser.add_argument(
        "--lowercase_instruction", action="store_true", default=True,
        help="将指令转为小写 (默认: True)"
    )
    parser.add_argument(
        "--no_lowercase_instruction", action="store_false", dest="lowercase_instruction",
        help="不转换指令为小写"
    )
    parser.add_argument(
        "--add_generation_prompt", action="store_true", default=True,
        help="添加 generation prompt (默认: True)"
    )
    parser.add_argument(
        "--no_generation_prompt", action="store_false", dest="add_generation_prompt",
        help="不添加 generation prompt"
    )
    
    # 处理配置
    parser.add_argument(
        "--start_idx", type=int, default=None,
        help="起始帧索引 (用于断点续传)"
    )
    parser.add_argument(
        "--end_idx", type=int, default=None,
        help="结束帧索引 (用于断点续传)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="数据预加载 worker 数量 (默认: 4)"
    )
    parser.add_argument(
        "--prefetch_size", type=int, default=8,
        help="预加载队列大小 (默认: 8)"
    )
    
    # 输出配置
    parser.add_argument(
        "--save_hidden_states", action="store_true", default=True,
        help="保存 hidden states 到文件 (默认: True)"
    )
    parser.add_argument(
        "--no_save_hidden_states", action="store_false", dest="save_hidden_states",
        help="不保存 hidden states (仅用于测试)"
    )
    parser.add_argument(
        "--save_dtype", type=str, default="float32", choices=["float32", "float16"],
        help="保存 hidden states 的数据类型 (默认: float32, float16 可减半文件大小)"
    )
    parser.add_argument(
        "--save_per_episode", action="store_true", default=True,
        help="按 episode 分组并打包为 chunk-XXX.npz (默认: True, 推荐, 减少 inode 占用)"
    )
    parser.add_argument(
        "--no_save_per_episode", action="store_false", dest="save_per_episode",
        help="关闭 chunk-npz 模式, 保存为 per-frame hidden_state_XXXXXX.npy (旧格式)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="详细输出 (打印 token 信息)"
    )
    
    return parser.parse_args()


# ============================================================================
# 辅助函数
# ============================================================================

def validate_layers(layers: List[int], model_path: str) -> None:
    """
    验证层号是否有效
    
    Args:
        layers: 层号列表
        model_path: 模型路径
    """
    # 检测模型类型
    max_layers = 28  # 默认值 (Qwen3-VL-2B)
    
    model_path_lower = model_path.lower()
    if "7b" in model_path_lower:
        max_layers = 28
    elif "4b" in model_path_lower:
        max_layers = 36
    elif "2b" in model_path_lower:
        max_layers = 28
    
    for layer_idx in layers:
        if layer_idx < 1 or layer_idx > max_layers:
            raise ValueError(
                f"层号必须在 [1, {max_layers}] 范围内，当前: {layer_idx}。"
                f"Qwen3-VL 的 hidden_states[0] 是 embedding 层，"
                f"hidden_states[1-{max_layers}] 是 transformer 第 1-{max_layers} 层。"
            )


def discover_dataset_image_keys(dataset_path: str) -> List[str]:
    """
    自动发现 LeRobot 数据集里的图像键名。

    优先从 `meta/info.json` 的 `features` 中读取 `observation.images.*`，
    如果没有再回退到 `videos/` 目录扫描。
    """
    dataset_path = Path(dataset_path)
    info_path = dataset_path / "meta" / "info.json"
    image_keys: List[str] = []

    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)

        features = info.get("features", {})
        for full_key, spec in features.items():
            if not full_key.startswith("observation.images."):
                continue

            if isinstance(spec, dict) and spec.get("dtype") not in (None, "video"):
                continue

            image_keys.append(full_key.replace("observation.images.", "", 1))

    if image_keys:
        return image_keys

    videos_path = dataset_path / "videos"
    if not videos_path.exists():
        return []

    chunk_dirs = sorted(
        [p for p in videos_path.iterdir() if p.is_dir() and p.name.startswith("chunk-")]
    )
    if chunk_dirs:
        candidate_dirs = [p for p in chunk_dirs[0].iterdir() if p.is_dir()]
    else:
        candidate_dirs = [p for p in videos_path.iterdir() if p.is_dir()]

    for path in sorted(candidate_dirs):
        if path.name.startswith("observation.images."):
            image_keys.append(path.name.replace("observation.images.", "", 1))

    return image_keys


def resolve_image_keys(
    dataset_path: str,
    image_keys_arg: Optional[str] = None,
    auto_image_keys: bool = True,
) -> List[str]:
    """解析最终使用的图像键名，支持手动覆盖或自动检测。"""
    if image_keys_arg:
        image_keys = [k.strip() for k in image_keys_arg.split(",") if k.strip()]
        if not image_keys:
            raise ValueError("--image_keys 不能为空字符串")
        return image_keys

    if auto_image_keys:
        image_keys = discover_dataset_image_keys(dataset_path)
        if image_keys:
            return image_keys
        raise ValueError(
            f"未能从数据集自动检测到 image keys: {dataset_path}。"
            "请手动通过 --image_keys 指定。"
        )

    return DEFAULT_IMAGE_KEYS


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    args = parse_args()
    
    # 打印 Banner
    print("\n" + "=" * 70)
    print("🔮 Qwen3-VL Hidden States 提取工具")
    print("=" * 70)
    
    # 验证数据集路径
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"\n❌ 错误: 数据集路径不存在: {dataset_path}")
        sys.exit(1)
    
    # 验证必要的文件
    info_path = dataset_path / "meta" / "info.json"
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if not info_path.exists():
        print(f"\n❌ 错误: 缺少 meta/info.json: {info_path}")
        sys.exit(1)
    if not tasks_path.exists():
        print(f"\n❌ 错误: 缺少 meta/tasks.jsonl: {tasks_path}")
        sys.exit(1)
    
    # 解析层号
    layers = [int(x.strip()) for x in args.layers.split(",")]
    
    # 验证层号
    try:
        validate_layers(layers, args.model_path)
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
    
    # 解析图像键
    try:
        image_keys = resolve_image_keys(
            dataset_path=str(dataset_path),
            image_keys_arg=args.image_keys,
            auto_image_keys=args.auto_image_keys,
        )
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
    
    # 设置输出目录
    output_dir = args.output_dir or str(dataset_path / "vlm_hidden_states")
    
    # 打印配置信息
    print(f"\n📋 配置信息:")
    print(f"  模型类型: Qwen3-VL")
    print(f"  模型路径: {args.model_path}")
    print(f"  数据集路径: {args.dataset_path}")
    print(f"  输出目录: {output_dir}")
    print(f"  提取层: {layers} (共 {len(layers)} 层)")
    image_keys_source = "手动指定" if args.image_keys else (
        "自动检测" if args.auto_image_keys else "默认值"
    )
    print(f"  图像视角: {image_keys} (共 {len(image_keys)} 个, 来源: {image_keys_source})")
    print(f"  设备: {args.device}")
    print(f"  数据类型: {args.dtype}")
    print(f"  翻转图像: {args.flip_images}")
    print(f"  Prompt 模板: {args.prompt_template}")
    print(f"  内容顺序: {args.content_order}")
    if args.start_idx is not None or args.end_idx is not None:
        print(f"  处理范围: [{args.start_idx or 0}, {args.end_idx or 'end'})")
    print(f"  Workers: {args.num_workers}")
    print(f"  预加载队列: {args.prefetch_size}")
    print(f"  保存到文件: {args.save_hidden_states}")
    print(f"  保存模式: {'per-episode chunk-npz' if args.save_per_episode else 'per-frame npy'}")
    
    # 导入 model_selector
    print("\n🔧 导入 model_selector...")
    try:
        from VLMs.S0_1.backbone.model_selector import (
            create_vlm_backbone,
            run_hidden_state_extraction,
            PROMPT_TEMPLATES,
        )
    except ImportError as e:
        print(f"\n❌ 导入失败: {e}")
        print("请确保在项目根目录下运行，或正确设置 PYTHONPATH")
        sys.exit(1)
    
    # 创建 Qwen3-VL backbone
    print("\n🚀 创建 Qwen3-VL Backbone...")
    
    # 处理 prompt_template
    prompt_template = args.prompt_template
    if prompt_template in PROMPT_TEMPLATES:
        prompt_template = PROMPT_TEMPLATES[prompt_template]
    
    try:
        backbone = create_vlm_backbone(
            model_type="qwen3_vl",
            model_path=args.model_path,
            device=args.device,
            layers=layers,
            prompt_template=prompt_template,
            content_order=args.content_order,
            flip_images=args.flip_images,
            dtype=args.dtype,
            verbose=args.verbose,
            lowercase_instruction=args.lowercase_instruction,
            add_generation_prompt=args.add_generation_prompt,
        )
        print(f"✓ Backbone 创建成功!")
        print(f"  模型信息: {backbone.get_model_info()}")
    except Exception as e:
        print(f"\n❌ 创建 Backbone 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 创建一个伪 args 对象传递给 run_hidden_state_extraction
    class ExtractArgs:
        pass
    
    extract_args = ExtractArgs()
    extract_args.model_type = "qwen3_vl"
    extract_args.dataset_path = args.dataset_path
    extract_args.output_dir = output_dir
    extract_args.image_keys = ",".join(image_keys)
    extract_args.flip_images = args.flip_images
    extract_args.start_idx = args.start_idx
    extract_args.end_idx = args.end_idx
    extract_args.num_workers = args.num_workers
    extract_args.prefetch_size = args.prefetch_size
    extract_args.save_hidden_states = args.save_hidden_states
    extract_args.save_dtype = args.save_dtype
    extract_args.save_per_episode = args.save_per_episode
    
    # 运行提取
    try:
        run_hidden_state_extraction(extract_args, backbone)
    except Exception as e:
        print(f"\n❌ 提取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n✅ 提取完成!")


# ============================================================================
# Python API
# ============================================================================

def extract_qwen_hidden_states(
    dataset_path: str,
    model_path: str = None,
    output_dir: str = None,
    layers: List[int] = None,
    image_keys: List[str] = None,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    flip_images: bool = True,
    prompt_template: str = "action",
    content_order: str = "images_first",
    lowercase_instruction: bool = True,
    add_generation_prompt: bool = True,
    start_idx: int = None,
    end_idx: int = None,
    num_workers: int = 4,
    prefetch_size: int = 8,
    save_to_file: bool = True,
    verbose: bool = False,
) -> tuple:
    """
    Python API: 提取 Qwen3-VL hidden states
    
    Args:
        dataset_path: LeRobot 数据集路径
        model_path: Qwen3-VL 模型路径 (默认: Qwen/Qwen3-VL-2B-Instruct)
        output_dir: 输出目录 (默认: {dataset_path}/vlm_hidden_states)
        layers: 提取的层号列表 (默认: [14])
        image_keys: 图像视角键名列表。为 None 时自动从数据集检测
        device: 设备
        dtype: 数据类型
        flip_images: 是否翻转图像
        prompt_template: Prompt 模板名称或自定义模板
        content_order: 内容顺序
        lowercase_instruction: 是否将指令转为小写
        add_generation_prompt: 是否添加 generation prompt
        start_idx: 起始帧索引
        end_idx: 结束帧索引
        num_workers: 数据预加载 worker 数量
        prefetch_size: 预加载队列大小
        save_to_file: 是否保存到文件
        verbose: 详细输出
        
    Returns:
        (processed_count, error_count): 成功处理数和错误数
        
    Example:
        >>> processed, errors = extract_qwen_hidden_states(
        ...     dataset_path="/path/to/dataset",
        ...     layers=[14, 15, 16],
        ...     image_keys=["top", "left_wrist"]
        ... )
        >>> print(f"成功: {processed}, 错误: {errors}")
    """
    from VLMs.S0_1.backbone.model_selector import (
        create_vlm_backbone,
        load_dataset_info,
        get_all_frames_info,
        HiddenStateExtractor,
        PROMPT_TEMPLATES,
    )
    
    # 设置默认值
    if model_path is None:
        model_path = DEFAULT_QWEN_MODEL_PATH
    if layers is None:
        layers = DEFAULT_LAYERS
    if image_keys is None:
        image_keys = resolve_image_keys(
            dataset_path=dataset_path,
            image_keys_arg=None,
            auto_image_keys=True,
        )
    if output_dir is None:
        output_dir = os.path.join(dataset_path, "vlm_hidden_states")
    
    # 验证层号
    validate_layers(layers, model_path)
    
    # 处理 prompt_template
    if prompt_template in PROMPT_TEMPLATES:
        prompt_template = PROMPT_TEMPLATES[prompt_template]
    
    # 创建 backbone
    backbone = create_vlm_backbone(
        model_type="qwen3_vl",
        model_path=model_path,
        device=device,
        layers=layers,
        prompt_template=prompt_template,
        content_order=content_order,
        flip_images=flip_images,
        dtype=dtype,
        verbose=verbose,
        lowercase_instruction=lowercase_instruction,
        add_generation_prompt=add_generation_prompt,
    )
    
    # 加载数据集信息
    info, tasks = load_dataset_info(dataset_path)
    
    # 获取帧信息
    frames_info = get_all_frames_info(dataset_path, info)
    
    # 确定处理范围
    start = start_idx if start_idx is not None else 0
    end = end_idx if end_idx is not None else len(frames_info)
    frames_to_process = frames_info[start:end]
    
    # 创建提取器
    extractor = HiddenStateExtractor(
        backbone=backbone,
        dataset_path=dataset_path,
        output_dir=output_dir,
        tasks=tasks,
        flip_images=flip_images,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
        save_to_file=save_to_file,
        image_keys=image_keys,
    )
    
    # 提取
    processed, errors = extractor.extract(frames_to_process)
    
    return processed, errors


# ============================================================================
# 入口点
# ============================================================================

if __name__ == "__main__":
    main()

