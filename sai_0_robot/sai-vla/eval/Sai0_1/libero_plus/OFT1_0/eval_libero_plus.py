#!/usr/bin/env python3
"""
LIBERO-plus 评估脚本 - 基于 Sai0_1 OFT1_0 模块化架构
====================================================

使用 LIBERO-plus 官方 benchmark 评估 OFT Action Head, 输出与 LIBERO-plus
leaderboard 一致的"按 7 类扰动 + 难度等级"分组成功率。

⚠️ 与原版 ``eval/Sai0_1/libero/OFT1_0/eval_Sai0_1.py`` 的关键差异:

1. ``state`` 格式与 ``libero_plus_lerobot`` 数据集对齐:
       [eef_pos_x/y/z, eef_euler_r/p/y, gripper_qpos_left, gripper_qpos_right]   8维, 用 ``scipy.Rotation`` 把 ``robot0_eef_quat`` 转成欧拉 RPY,
   不再是原版的 ``quat→axis-angle``

2. 每个任务默认只跑 ``num_trials_per_task=1`` (LIBERO-plus 官方约定):
   每个 task 本身就是一个独立的扰动 variant，10K+ 个 variant 已经覆盖
   足够多的随机性，因此官方建议把 trial=1

3. 评测结果按 LIBERO-plus 论文的"7 类扰动 + 难度等级"分组聚合,
   完全对齐官方 task_classification.json

4. 支持 ``--resume``: 中途中断后再跑可以接着已完成的 task 继续

5. 自动处理 LIBERO-plus 的环境引导:
   - 在 ``/data_disk1/hwl/LIBERO-plus/libero/`` touch 出 ``__init__.py``
   - 把 conda env 中的 LIBERO assets 软链到 LIBERO-plus 目录
   - 写一份独立的 ``~/.libero_plus_sai0/config.yaml``
   - 自动 monkey-patch ``torch.load(weights_only=False)`` 以兼容 PyTorch 2.6+

依赖 (qwen_eagle_hwl):
    - LIBERO-plus 的 ``extra_requirements.txt``: ``Wand``, ``scikit-image``
    - ``gym==0.25.2`` (LIBERO-plus venv 模块依赖, robosuite 不需要但 LIBERO-plus
      的 ``libero/libero/envs/venv.py`` 顶层 import 了 gym)
    > pip install Wand scikit-image "gym==0.25.2"

基本用法:
    # 默认评估 libero_spatial 全部 2402 个任务
    CUDA_VISIBLE_DEVICES=0 python -m eval.Sai0_1.libero_plus.OFT1_0.eval_libero_plus \\
        --checkpoint_path /path/to/action_head.pt \\
        --vlm_model_path /path/to/qwen_or_eagle \\
        --vlm_type qwen3_vl \\
        --vlm_layers 14 \\
        --vlm_output_dim 2048 \\
        --dataset_path /data_disk2/hwl/datasets/libero_plus_lerobot \\
        --task_suite_name libero_spatial

    # 只评估某一类扰动
    ... --categories "Camera Viewpoints,Robot Initial States"

    # 只评估前 100 个任务 (快速调试)
    ... --max_tasks 100

    # 中断后接着跑
    ... --resume

参数说明:
    --checkpoint_path  : 训练好的 action_head.pt
    --vlm_model_path   : VLM backbone (Qwen3-VL / Eagle / Cosmos)
    --vlm_type         : qwen3_vl / eagle2_5_vl / cosmos_reason_2b_vl
    --dataset_path     : 训练时使用的 lerobot 数据集路径 (用于读 stats.json
                         做 state/action 反归一化)
    --task_suite_name  : libero_spatial / libero_object / libero_goal / libero_10
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# ============================================================================
# 路径设置 - 必须放在所有 libero / robosuite 相关导入之前
# ============================================================================
EVAL_DIR = Path(__file__).resolve().parent
SAI0_ROOT = EVAL_DIR.parents[3]  # eval/Sai0_1/libero_plus/OFT1_0 -> sai0-vla
if str(SAI0_ROOT) not in sys.path:
    sys.path.insert(0, str(SAI0_ROOT))


DEFAULT_LIBERO_PLUS_ROOT = "/data_disk1/hwl/LIBERO-plus"
DEFAULT_CONDA_LIBERO_ROOT = (
    "/home/dev/miniconda3/envs/qwen_eagle_hwl/lib/python3.10/site-packages/libero"
)
DEFAULT_LIBERO_PLUS_CONFIG_DIR = os.path.expanduser("~/.libero_plus_sai0")


# ============================================================================
# LIBERO-plus 环境引导工具 - 第一阶段, 不能依赖 libero / robosuite
# ============================================================================

def _bootstrap_libero_plus(
    libero_plus_root: str = DEFAULT_LIBERO_PLUS_ROOT,
    conda_libero_root: str = DEFAULT_CONDA_LIBERO_ROOT,
    libero_plus_config_dir: str = DEFAULT_LIBERO_PLUS_CONFIG_DIR,
) -> None:
    """让 ``import libero`` 解析到 LIBERO-plus 的副本，且使用独立的 config.yaml。"""
    lp_root = Path(libero_plus_root).expanduser()
    if not lp_root.is_dir():
        raise FileNotFoundError(
            f"LIBERO-plus 仓库不存在: {lp_root}\n"
            f"  请先 git clone 到 {libero_plus_root}, 或使用 --libero_plus_root 指向正确路径"
        )

    # 1) touch __init__.py 让 libero/ 成为合法 package, 否则 sys.path[0] 注入无效
    pkg_init = lp_root / "libero" / "__init__.py"
    if not pkg_init.exists():
        try:
            pkg_init.touch()
        except PermissionError:
            print(
                f"⚠️ 无法 touch {pkg_init}, 请手动创建该空文件后重跑。",
                file=sys.stderr,
            )
            raise

    # 2) sys.path 优先级最高
    if str(lp_root) not in sys.path:
        sys.path.insert(0, str(lp_root))
    else:
        sys.path.remove(str(lp_root))
        sys.path.insert(0, str(lp_root))

    # 3) 准备独立的 config.yaml, 不污染原版 LIBERO 的 ~/.libero/config.yaml
    config_dir = Path(libero_plus_config_dir).expanduser()
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / "config.yaml"
    benchmark_root = lp_root / "libero" / "libero"
    desired_cfg = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    if cfg_path.exists():
        try:
            with cfg_path.open("r") as f:
                existing_cfg = yaml.safe_load(f) or {}
        except Exception:  # noqa: BLE001
            existing_cfg = {}
        if existing_cfg != desired_cfg:
            with cfg_path.open("w") as f:
                yaml.safe_dump(desired_cfg, f, default_flow_style=False)
    else:
        with cfg_path.open("w") as f:
            yaml.safe_dump(desired_cfg, f, default_flow_style=False)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    # 4) 把原版 LIBERO 的 assets 软链过来, 避免重新下载
    src_assets = Path(conda_libero_root) / "libero" / "assets"
    dst_assets = benchmark_root / "assets"
    if src_assets.exists():
        dst_assets.mkdir(parents=True, exist_ok=True)
        for item in src_assets.iterdir():
            target = dst_assets / item.name
            if not target.exists() and not target.is_symlink():
                try:
                    target.symlink_to(item)
                except OSError:
                    pass


def _fix_robosuite_log_permission() -> None:
    """robosuite 默认写 ``/tmp/robosuite.log``, 多用户机器上经常没权限。"""
    target_log = "/tmp/robosuite.log"
    try:
        with open(target_log, "a"):
            pass
        return
    except (PermissionError, OSError):
        pass

    user_log = os.path.expanduser("~/.robosuite/robosuite.log")
    os.makedirs(os.path.dirname(user_log), exist_ok=True)

    _orig_filehandler = logging.FileHandler

    class _PatchedFileHandler(logging.FileHandler):  # noqa: WPS431
        def __init__(self, filename, mode="a", encoding=None, delay=False):  # noqa: D401
            if filename == target_log:
                filename = user_log
            super().__init__(filename, mode, encoding, delay)

    logging.FileHandler = _PatchedFileHandler


def _patch_torch_load_for_legacy_pickles() -> None:
    """LIBERO-plus 的 ``.pruned_init`` 是用 numpy reconstruct 的旧 pickle,
    PyTorch 2.6+ 默认 ``weights_only=True`` 会拒绝加载。"""
    import torch  # noqa: WPS433

    if getattr(torch.load, "_libero_plus_patched", False):
        return

    _orig_load = torch.load

    def _patched_load(*args, **kwargs):  # noqa: D401
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    _patched_load._libero_plus_patched = True  # type: ignore[attr-defined]
    torch.load = _patched_load  # type: ignore[assignment]


def _patch_numpy_legacy_aliases() -> None:
    """LIBERO-plus 的 ``envs/env_wrapper.py`` 实现 fog/snow/spatter 等
    Sensor Noise 时使用了 NumPy 1.x 已弃用 / NumPy 2.0 移除的别名 (主要是
    ``np.float_``, 见 ``plasma_fractal``). 用 NumPy 2.x 跑这些 task 会触发
    ``AttributeError: np.float_ was removed in the NumPy 2.0 release``,
    导致 worker 在第一个 Sensor Noise task 上整体崩掉.

    这里把已知会被引用的旧别名补回 numpy 命名空间, 等价于 ``np.float_``
    ``np.bool_`` ``np.complex_`` 等常见 1.x 写法. 必须在 import
    ``libero`` / ``robosuite`` 之前执行.
    """
    import numpy as _np  # noqa: WPS433

    _legacy_aliases = {
        "float_": _np.float64,
        "complex_": _np.complex128,
        "object_": _np.object_ if hasattr(_np, "object_") else object,
        "str_": _np.str_ if hasattr(_np, "str_") else str,
        "bytes_": _np.bytes_ if hasattr(_np, "bytes_") else bytes,
        "int_": _np.int64,
        "long": _np.int64,
        "bool_": _np.bool_ if hasattr(_np, "bool_") else bool,
    }
    for _name, _alias in _legacy_aliases.items():
        if not hasattr(_np, _name):
            setattr(_np, _name, _alias)


def _set_default_offline_hf_envs() -> None:
    """让 transformers / huggingface_hub 进入离线模式 (除非用户显式设了在线)。

    第一次 ``import`` 触发的元数据探测在没网或 HF 抽风时会卡 1-3 分钟,
    评估时模型权重早就已经下载到本地缓存,完全可以走离线。
    """
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ============================================================================
# 第一阶段引导 (在导入 libero / robosuite / Sai0_1 之前)
# ============================================================================
_bootstrap_libero_plus(
    libero_plus_root=os.environ.get("LIBERO_PLUS_ROOT", DEFAULT_LIBERO_PLUS_ROOT),
    conda_libero_root=os.environ.get("LIBERO_CONDA_ROOT", DEFAULT_CONDA_LIBERO_ROOT),
    libero_plus_config_dir=os.environ.get(
        "LIBERO_PLUS_CONFIG_DIR", DEFAULT_LIBERO_PLUS_CONFIG_DIR
    ),
)
_fix_robosuite_log_permission()
_patch_torch_load_for_legacy_pickles()
_patch_numpy_legacy_aliases()
_set_default_offline_hf_envs()


# ============================================================================
# 第二阶段 - 现在可以安全 import 重型依赖
# ============================================================================
import cv2  # noqa: E402
import imageio  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial.transform import Rotation as ScipyRotation  # noqa: E402

# Sai0_1 模块
from VLAs.Sai0_1 import (  # noqa: E402
    load_normalization_stats,
)
from VLMs.S0_1.backbone import create_vlm_backbone  # noqa: E402

# LIBERO-plus 延迟导入, 避免 robosuite 在 GPU 设置完成前提前初始化 CUDA
LIBERO_AVAILABLE = False
benchmark_module = None
get_libero_path = None
OffScreenRenderEnv = None


def _init_libero() -> bool:
    global LIBERO_AVAILABLE, benchmark_module, get_libero_path, OffScreenRenderEnv
    if LIBERO_AVAILABLE:
        return True
    try:
        from libero.libero import benchmark as _benchmark, get_libero_path as _get_path  # noqa: WPS433
        from libero.libero.envs import OffScreenRenderEnv as _OSEnv  # noqa: WPS433
        benchmark_module = _benchmark
        get_libero_path = _get_path
        OffScreenRenderEnv = _OSEnv
        LIBERO_AVAILABLE = True
        return True
    except ImportError as e:  # noqa: BLE001
        print(
            f"[ERROR] 无法导入 LIBERO-plus: {e}\n"
            "  请确认: \n"
            f"    1) {DEFAULT_LIBERO_PLUS_ROOT}/libero/__init__.py 存在 (空文件即可)\n"
            "    2) 已经 pip install Wand scikit-image 'gym==0.25.2'",
            file=sys.stderr,
        )
        return False


# ============================================================================
# 常量
# ============================================================================
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# LIBERO-plus 论文定义的 7 类扰动 (用于排序与汇总)
LIBERO_PLUS_CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


# ============================================================================
# 工具函数
# ============================================================================

def quat_xyzw_to_euler_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """robosuite 的 ``robot0_eef_quat`` 是 (qx, qy, qz, qw) 顺序。

    libero_plus_lerobot 数据集中的 ``observation.state[3:6]`` 用的是
    XYZ 内旋顺序的 RPY 欧拉角 (与 robosuite/MuJoCo 一致)。
    """
    return ScipyRotation.from_quat(quat_xyzw).as_euler("xyz")


def get_libero_dummy_action() -> List[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def get_libero_image(
    obs: Dict[str, np.ndarray],
    resize_to: Optional[int] = 128,
    flip: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 LIBERO obs 提取 agentview 和 wrist 图像。"""
    img = obs["agentview_image"].copy()
    wrist_img = obs["robot0_eye_in_hand_image"].copy()

    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if wrist_img.dtype != np.uint8:
        wrist_img = (
            (wrist_img * 255).astype(np.uint8)
            if wrist_img.max() <= 1.0
            else wrist_img.astype(np.uint8)
        )

    if resize_to is not None:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        wrist_img = cv2.resize(wrist_img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)

    if flip:
        img = np.ascontiguousarray(img[::-1, ::-1, :])
        wrist_img = np.ascontiguousarray(wrist_img[::-1, ::-1, :])

    return img, wrist_img


def save_rollout_video(
    top_view: List[np.ndarray],
    wrist_view: List[np.ndarray],
    idx: int,
    success: bool,
    task_description: str,
    video_dir: str,
    fps: int = 10,
) -> Optional[str]:
    rollout_dir = f"{video_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    safe_desc = (
        task_description.lower()
        .replace(" ", "_")
        .replace("\n", "_")
        .replace(".", "_")[:50]
    )
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={safe_desc}.mp4"
    )
    if not top_view or not wrist_view:
        return None
    try:
        writer = imageio.get_writer(
            mp4_path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        for img1, img2 in zip(top_view, wrist_view):
            if img1.dtype != np.uint8:
                img1 = img1.astype(np.uint8)
            if img2.dtype != np.uint8:
                img2 = img2.astype(np.uint8)
            writer.append_data(np.hstack((img1, img2)))
        writer.close()
        return mp4_path
    except Exception as e:  # noqa: BLE001
        print(f"❌ 视频保存失败: {e}")
        return None


# ============================================================================
# LIBERO-plus 任务分类信息
# ============================================================================

class TaskClassificationInfo:
    """从 task_classification.json 读取每个 task 的 category 和 difficulty_level。

    LIBERO-plus 官方 leaderboard 是按 (suite x category) 二元组聚合的。
    """

    def __init__(self, libero_plus_root: str = DEFAULT_LIBERO_PLUS_ROOT):
        self.path = (
            Path(libero_plus_root)
            / "libero"
            / "libero"
            / "benchmark"
            / "task_classification.json"
        )
        if not self.path.exists():
            raise FileNotFoundError(f"task_classification.json 不存在: {self.path}")
        with self.path.open("r") as f:
            self.raw = json.load(f)

        # suite -> task_name -> {category, difficulty_level}
        self.lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for suite, tasks in self.raw.items():
            self.lookup[suite] = {}
            for t in tasks:
                self.lookup[suite][t["name"]] = {
                    "id": t.get("id"),
                    "category": t.get("category", "Unknown"),
                    "difficulty_level": t.get("difficulty_level"),
                }

    def get(self, suite: str, task_name: str) -> Dict[str, Any]:
        return self.lookup.get(suite, {}).get(
            task_name,
            {"id": None, "category": "Unknown", "difficulty_level": None},
        )


# ============================================================================
# Sai0Policy - 与原版基本一致, state 重组为 [xyz, euler, gripper]
# ============================================================================

class Sai0Policy:
    """Sai0_1 OFT Action Head 推理封装 (libero_plus 版)。"""

    def __init__(
        self,
        checkpoint_path: str,
        vlm_model_path: str,
        vlm_type: str = "qwen3_vl",
        vlm_layers: Optional[List[int]] = None,
        vlm_output_dim: int = 2048,
        dataset_path: Optional[str] = None,
        device: str = "cuda:0",
        num_transformer_blocks: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        action_head_hidden_dim: int = 4096,
        action_chunk_size: int = 16,
        num_action_chunks: int = 16,
        action_dim: int = 7,
        flip_images: bool = True,
        content_order: str = "images_first",
        lowercase_instruction: bool = True,
        add_generation_prompt: bool = True,
        add_action_prompt: bool = False,
        verbose: bool = False,
        binarize_gripper: bool = True,
    ):
        self.device = device
        self.vlm_layers = vlm_layers or [-1]
        self.vlm_output_dim = vlm_output_dim
        self.action_chunk_size = action_chunk_size
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim
        self.flip_images = flip_images
        self.content_order = content_order
        self.lowercase_instruction = lowercase_instruction
        self.add_generation_prompt = add_generation_prompt
        self.add_action_prompt = add_action_prompt
        self.verbose = verbose
        self.binarize_gripper = binarize_gripper

        self.action_queue: List[np.ndarray] = []

        # 1) VLM Backbone
        print(f"\n📦 加载 VLM Backbone: {vlm_model_path}")
        print(f"  - 类型: {vlm_type}")
        print(f"  - 提取层: {self.vlm_layers}")
        print(f"  - 内容顺序: {content_order}")
        print(f"  - Action Prompt: {add_action_prompt}")
        self.vlm_backbone = create_vlm_backbone(
            model_type=vlm_type,
            model_path=vlm_model_path,
            device=device,
            layers=self.vlm_layers,
            flip_images=flip_images,
            content_order=content_order,
            prompt_template="simple",
            lowercase_instruction=False,
            verbose=verbose,
        )
        print("✓ VLM Backbone 加载成功")

        # 2) OFT Action Head
        print(f"\n📦 加载 OFT Action Head: {checkpoint_path}")

        # ── ① 先加载 checkpoint, 自动检测 action 输出形状 ──
        #     目的: ``Action_Heads.OFT1_0.constants`` 在 import 时会从环境变量
        #     读取 ``ACTION_DIM`` / ``NUM_ACTIONS_CHUNK`` 并把 ``L1RegressionActionHead``
        #     的 fc2 输出维度固定为 ``ACTION_DIM * NUM_ACTIONS_CHUNK``。
        #     训练脚本 (train_qwen_datasets_libero_plus_action7_chunk16_no_state.sh)
        #     里 export 了 ACTION_DIM=7 NUM_ACTIONS_CHUNK=16, 但 eval 脚本默认没有,
        #     所以这里在 import 之前根据 checkpoint 自动设置一次。
        state_dict = torch.load(checkpoint_path, map_location=device)
        ckpt_fc2_out = None
        if "action_head.model.fc2.weight" in state_dict:
            ckpt_fc2_out = state_dict["action_head.model.fc2.weight"].shape[0]
            expected = num_action_chunks * action_dim
            if ckpt_fc2_out != expected:
                # 优先信任 checkpoint
                print(
                    f"  ⚠️ checkpoint fc2 输出维度={ckpt_fc2_out}, "
                    f"与 (num_action_chunks={num_action_chunks}) * (action_dim={action_dim}) "
                    f"={expected} 不一致, 以 checkpoint 为准"
                )
                if ckpt_fc2_out % action_dim == 0:
                    num_action_chunks = ckpt_fc2_out // action_dim
                    self.num_action_chunks = num_action_chunks
                elif ckpt_fc2_out % num_action_chunks == 0:
                    action_dim = ckpt_fc2_out // num_action_chunks
                    self.action_dim = action_dim
            print(
                f"  - 推断: NUM_ACTIONS_CHUNK={num_action_chunks}, "
                f"ACTION_DIM={action_dim} (fc2 out={ckpt_fc2_out})"
            )

        os.environ["NUM_ACTIONS_CHUNK"] = str(num_action_chunks)
        os.environ["ACTION_DIM"] = str(action_dim)
        if "proprio_projector.layer_norm.weight" in state_dict:
            detected_dim = state_dict["proprio_projector.layer_norm.weight"].shape[0]
            if detected_dim != vlm_output_dim:
                print(f"  ⚠️ 自动从 checkpoint 检测到 vlm_output_dim={detected_dim}")
                vlm_output_dim = detected_dim
        # 同步 PROPRIO_DIM 到环境变量 (默认 8 是 libero/libero_plus 的 state 维度,
        # 与 ProprioProjector 的输入对齐, libero_plus_lerobot 也是 8 维)
        os.environ.setdefault("PROPRIO_DIM", "8")

        # ── ② import 并构造 pipeline ──
        from Action_Heads.OFT1_0 import create_vlm2oft_pipeline  # noqa: WPS433

        self.model = create_vlm2oft_pipeline(
            num_transformer_blocks=num_transformer_blocks,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            action_head_hidden_dim=action_head_hidden_dim,
            num_vlm_layers=len(self.vlm_layers),
            vlm_output_dim=vlm_output_dim,
        ).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✓ OFT Action Head 加载成功")

        # 3) 归一化统计 (libero_plus_lerobot 已经是 8 维 euler, 不需要 quat→axisangle 转换)
        self.normalizers = None
        if dataset_path:
            self.normalizers = load_normalization_stats(
                dataset_path,
                convert_quat_to_axisangle=False,
            )
            if self.normalizers:
                print("✓ 归一化统计量加载成功 (convert_quat_to_axisangle=False)")
            else:
                print("⚠️ 归一化统计量未找到, 将不进行 normalize/denormalize")

    def reset_action_queue(self) -> None:
        self.action_queue = []

    def get_action(self, obs: Dict[str, np.ndarray], lang: str) -> np.ndarray:
        if self.action_queue:
            return self.action_queue.pop(0)

        # === libero_plus_lerobot state 格式: [xyz(3), euler_rpy(3), gripper_qpos(2)] ===
        xyz = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        euler = quat_xyzw_to_euler_rpy(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        state = np.concatenate([xyz, euler, gripper], axis=0).astype(np.float32)
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 8)

        # 图像 (flip 由 VLM backbone 内部处理)
        img, wrist_img = get_libero_image(obs, flip=False)
        images = [Image.fromarray(img), Image.fromarray(wrist_img)]

        # 指令
        task_desc = lang.lower() if self.lowercase_instruction else lang
        instruction = (
            f"What action should the robot take to {task_desc}?"
            if self.add_action_prompt
            else task_desc
        )

        # VLM hidden states
        vlm_output = self.vlm_backbone.get_hidden_states(images=images, instruction=instruction)
        vlm_hidden_states = vlm_output.hidden_states  # List[Tensor]

        # state 归一化
        if self.normalizers and "state" in self.normalizers:
            state_tensor = self.normalizers["state"].normalize(state_tensor)

        # 推理
        with torch.no_grad():
            vlm_states_dev = [v.to(self.device) for v in vlm_hidden_states]
            pred_actions = self.model(vlm_states_dev, state_tensor)
            # pred_actions: (1, 1, num_action_chunks * action_dim)

        pred_actions = pred_actions[0, 0].cpu().numpy()
        pred_actions = pred_actions.reshape(self.num_action_chunks, self.action_dim)

        # action 反归一化
        if self.normalizers and "action" in self.normalizers:
            pred_t = torch.from_numpy(pred_actions)
            pred_actions = self.normalizers["action"].denormalize(pred_t).numpy()

        # 入队
        n_take = min(self.action_chunk_size, self.num_action_chunks)
        for i in range(n_take):
            action = pred_actions[i].astype(np.float32)
            if self.binarize_gripper:
                action[-1] = float(np.sign(action[-1]))
            self.action_queue.append(action)

        return self.action_queue.pop(0)


# ============================================================================
# 评估配置
# ============================================================================

@dataclass
class EvalConfig:
    # 模型
    checkpoint_path: str = ""
    vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_type: str = "qwen3_vl"
    dataset_path: str = ""

    # VLM
    vlm_layers: List[int] = field(default_factory=lambda: [-1])
    vlm_output_dim: int = 2048

    # Prompt
    content_order: str = "images_first"
    lowercase_instruction: bool = True
    add_generation_prompt: bool = True
    add_action_prompt: bool = False

    # OFT
    num_transformer_blocks: int = 4
    num_attention_heads: int = 8
    dropout: float = 0.1
    action_head_hidden_dim: int = 4096
    num_action_chunks: int = 16
    action_dim: int = 7
    binarize_gripper: bool = True

    # LIBERO-plus
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 1
    task_ids: Optional[List[int]] = None
    max_tasks: int = -1
    num_steps_wait: int = 10
    max_steps: int = 600
    env_seed: Optional[int] = None
    categories: Optional[List[str]] = None
    difficulty_levels: Optional[List[int]] = None

    # 推理
    action_chunk_size: int = 16
    execute_all_chunks: bool = True

    # 系统
    device: str = "cuda:0"
    # flip_images: 是否把 LIBERO env 直出的图像做 180° 翻转后再喂给 VLM backbone.
    # LIBERO env 给的 obs (agentview_image) 是 OpenGL 渲染原方向, 整张图是颠倒的,
    # 必须翻一次才是"人/模型预期看到"的正向. 训练时 LeRobot 数据集已经是正向方向,
    # 所以评估时 backbone 也必须做这个翻转 (与训练对齐) => 默认 True.
    flip_images: bool = True
    # video_flip 仅控制保存到 mp4 的展示方向. 跟 flip_images 解耦, 这样如果以后
    # 想让模型看 raw 方向但视频仍展示翻转方向 (或反过来) 都好调.
    video_flip: bool = True
    video_dir: str = "./eval_rollouts"
    log_dir: str = ""
    save_videos: bool = False
    save_video_every: int = 1
    verbose: bool = False
    resume: bool = True
    # 实时打印
    print_per_task: bool = True   # 每完成一个 task 在 tqdm 之上 print 一行(不被覆盖)
    summary_every: int = 100       # 每 N 个 episode 打印一次完整 per-category 进度小结, 0=关掉

    # 多卡 shard (data parallel)
    shard_index: int = 0     # 当前 worker 在所有 shard 中的索引 [0, num_shards)
    num_shards: int = 1      # 总 shard 数 (= 使用的 GPU 数). 默认 1 表示单卡


# ============================================================================
# 评估主循环
# ============================================================================

def _filter_task_ids(
    task_suite,
    task_info: TaskClassificationInfo,
    suite_name: str,
    cfg: EvalConfig,
) -> List[int]:
    """根据 cfg.task_ids / cfg.max_tasks / cfg.categories / cfg.difficulty_levels 过滤。"""
    n_tasks = task_suite.n_tasks
    base_ids = list(range(n_tasks)) if cfg.task_ids is None else list(cfg.task_ids)

    if cfg.categories is not None:
        cat_set = set(cfg.categories)
        base_ids = [
            i
            for i in base_ids
            if task_info.get(suite_name, task_suite.tasks[i].name)["category"] in cat_set
        ]
    if cfg.difficulty_levels is not None:
        diff_set = set(cfg.difficulty_levels)
        base_ids = [
            i
            for i in base_ids
            if task_info.get(suite_name, task_suite.tasks[i].name).get("difficulty_level")
            in diff_set
        ]

    if cfg.max_tasks > 0:
        base_ids = base_ids[: cfg.max_tasks]

    # 多卡 shard 切分: 用 step-stride (每张卡都拿到散落在整个 task 列表里的 task)
    # 这样每张卡的 category / difficulty 分布大致相同, 跑完时间也大致相同,
    # 避免某张卡都是难 task 而另一张卡都是简单 task.
    if cfg.num_shards > 1:
        if not (0 <= cfg.shard_index < cfg.num_shards):
            raise ValueError(
                f"shard_index={cfg.shard_index} 超出范围 [0, {cfg.num_shards})"
            )
        base_ids = base_ids[cfg.shard_index :: cfg.num_shards]

    return base_ids


def _save_results(results_path: Path, payload: Dict[str, Any]) -> None:
    tmp = results_path.with_suffix(".tmp.json")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    tmp.replace(results_path)


def _load_existing_results(results_path: Path) -> Dict[str, Any]:
    if not results_path.exists():
        return {}
    try:
        with results_path.open("r") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _aggregate_metrics(
    task_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """按 category 和 difficulty_level 聚合, 并计算 overall。"""
    overall_eps = 0
    overall_succ = 0
    cat_buckets: Dict[str, Dict[str, int]] = {c: {"ep": 0, "succ": 0} for c in LIBERO_PLUS_CATEGORIES}
    cat_buckets["Unknown"] = {"ep": 0, "succ": 0}
    diff_buckets: Dict[str, Dict[str, int]] = {}

    for _, r in task_results.items():
        cat = r.get("category", "Unknown")
        diff = r.get("difficulty_level")
        ep = r.get("episodes", 0)
        succ = r.get("successes", 0)
        overall_eps += ep
        overall_succ += succ

        cat_buckets.setdefault(cat, {"ep": 0, "succ": 0})
        cat_buckets[cat]["ep"] += ep
        cat_buckets[cat]["succ"] += succ

        diff_key = "None" if diff is None else f"L{diff}"
        diff_buckets.setdefault(diff_key, {"ep": 0, "succ": 0})
        diff_buckets[diff_key]["ep"] += ep
        diff_buckets[diff_key]["succ"] += succ

    def _rate(b: Dict[str, int]) -> float:
        return b["succ"] / b["ep"] if b["ep"] else 0.0

    def _norm(b: Dict[str, int]) -> Dict[str, Any]:
        return {
            "episodes": b["ep"],
            "successes": b["succ"],
            "success_rate": _rate(b),
        }

    return {
        "overall": {
            "episodes": overall_eps,
            "successes": overall_succ,
            "success_rate": (overall_succ / overall_eps) if overall_eps else 0.0,
        },
        "by_category": {
            c: _norm(v) for c, v in cat_buckets.items() if v["ep"]
        },
        "by_difficulty": {
            k: _norm(v) for k, v in diff_buckets.items() if v["ep"]
        },
    }


def _print_summary(metrics: Dict[str, Any], suite_name: str) -> None:
    print("\n" + "=" * 70)
    print(f"📊 LIBERO-plus 评估结果汇总 - {suite_name}")
    print("=" * 70)
    o = metrics["overall"]
    print(f"  总 episodes: {o['episodes']}, 成功: {o['successes']}, "
          f"Overall SR: {o['success_rate'] * 100:.2f}%")
    print("-" * 70)
    print(f"  {'Category':<25}{'Episodes':>10}{'Succ':>8}{'SR':>10}")
    for cat in list(LIBERO_PLUS_CATEGORIES) + ["Unknown"]:
        if cat in metrics["by_category"]:
            v = metrics["by_category"][cat]
            print(
                f"  {cat:<25}{v['episodes']:>10}{v['successes']:>8}"
                f"{v['success_rate'] * 100:>9.2f}%"
            )
    print("-" * 70)
    print(f"  {'Difficulty':<25}{'Episodes':>10}{'Succ':>8}{'SR':>10}")
    for k in sorted(metrics["by_difficulty"].keys()):
        v = metrics["by_difficulty"][k]
        print(
            f"  {k:<25}{v['episodes']:>10}{v['successes']:>8}"
            f"{v['success_rate'] * 100:>9.2f}%"
        )
    print("=" * 70)


def eval_libero_plus(cfg: EvalConfig) -> Dict[str, Any]:
    if not _init_libero():
        raise RuntimeError("LIBERO-plus 初始化失败")

    log_dir = cfg.log_dir if cfg.log_dir else cfg.video_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(cfg.video_dir, exist_ok=True)

    # 加载 task 分类信息
    libero_plus_root = os.environ.get("LIBERO_PLUS_ROOT", DEFAULT_LIBERO_PLUS_ROOT)
    task_info = TaskClassificationInfo(libero_plus_root=libero_plus_root)

    # 创建 Policy
    policy = Sai0Policy(
        checkpoint_path=cfg.checkpoint_path,
        vlm_model_path=cfg.vlm_model_path,
        vlm_type=cfg.vlm_type,
        vlm_layers=cfg.vlm_layers,
        vlm_output_dim=cfg.vlm_output_dim,
        dataset_path=cfg.dataset_path,
        device=cfg.device,
        num_transformer_blocks=cfg.num_transformer_blocks,
        num_attention_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
        action_head_hidden_dim=cfg.action_head_hidden_dim,
        action_chunk_size=cfg.num_action_chunks if cfg.execute_all_chunks else cfg.action_chunk_size,
        num_action_chunks=cfg.num_action_chunks,
        action_dim=cfg.action_dim,
        flip_images=cfg.flip_images,
        content_order=cfg.content_order,
        lowercase_instruction=cfg.lowercase_instruction,
        add_generation_prompt=cfg.add_generation_prompt,
        add_action_prompt=cfg.add_action_prompt,
        verbose=cfg.verbose,
        binarize_gripper=cfg.binarize_gripper,
    )

    # benchmark
    bench_dict = benchmark_module.get_benchmark_dict()
    if cfg.task_suite_name not in bench_dict:
        raise ValueError(
            f"未知 task_suite_name={cfg.task_suite_name}, 可选: {list(bench_dict.keys())}"
        )
    task_suite = bench_dict[cfg.task_suite_name]()

    target_task_ids = _filter_task_ids(task_suite, task_info, cfg.task_suite_name, cfg)
    print(f"\n{'=' * 70}")
    print(f"LIBERO-plus 评估 (Sai0_1 OFT1_0)")
    print(f"{'=' * 70}")
    print(f"  Task Suite       : {cfg.task_suite_name}")
    print(f"  Total tasks      : {task_suite.n_tasks}")
    print(f"  Filtered tasks   : {len(target_task_ids)}")
    if cfg.categories:
        print(f"  Categories       : {cfg.categories}")
    if cfg.difficulty_levels:
        print(f"  Difficulty levels: {cfg.difficulty_levels}")
    if cfg.num_shards > 1:
        print(
            f"  Shard            : {cfg.shard_index}/{cfg.num_shards} "
            f"(stride 切分: shard_index={cfg.shard_index}, num_shards={cfg.num_shards})"
        )
    print(f"  Trials per task  : {cfg.num_trials_per_task}")
    print(f"  Max steps        : {cfg.max_steps}")
    print(f"{'=' * 70}\n")

    # resume — 多卡时每张卡用独立结果文件 (eval_results_<suite>_shardK_of_N.json)
    if cfg.num_shards > 1:
        results_filename = (
            f"eval_results_{cfg.task_suite_name}"
            f"_shard{cfg.shard_index}_of_{cfg.num_shards}.json"
        )
    else:
        results_filename = f"eval_results_{cfg.task_suite_name}.json"
    results_path = Path(cfg.video_dir) / results_filename
    existing = _load_existing_results(results_path) if cfg.resume else {}
    task_results: Dict[str, Dict[str, Any]] = existing.get("task_results", {}) if existing else {}
    if task_results:
        print(f"🔁 检测到已有结果, 从 {len(task_results)} 个已完成 task 继续 (resume=True)")

    # 日志文件
    log_suffix = (
        f"_shard{cfg.shard_index}_of_{cfg.num_shards}" if cfg.num_shards > 1 else ""
    )
    log_path = Path(log_dir) / f"eval_{cfg.task_suite_name}_{DATE_TIME}{log_suffix}.log"
    log_file = log_path.open("w")
    log_file.write(f"task_suite     : {cfg.task_suite_name}\n")
    log_file.write(f"checkpoint     : {cfg.checkpoint_path}\n")
    log_file.write(f"vlm_model_path : {cfg.vlm_model_path}\n")
    log_file.write(f"vlm_type       : {cfg.vlm_type}\n")
    log_file.write(f"target_tasks   : {len(target_task_ids)}\n\n")
    log_file.flush()

    save_every = max(1, cfg.save_video_every)
    # 用 shard 内"实际跑过的 task 计数"判断是否保存视频, 而不是 task_id % save_every,
    # 否则在多 GPU stride 切分下, 部分 shard 永远不会命中 (例如 stride=8/save_every=50
    # 时只有 shard 0/2/4/6 能命中, shard 1/3/5/7 一个视频都没).
    local_run_idx = 0

    pbar = tqdm.tqdm(target_task_ids, desc=f"{cfg.task_suite_name}")
    for task_id in pbar:
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_meta = task_info.get(cfg.task_suite_name, task_name)
        category = task_meta["category"]
        difficulty = task_meta["difficulty_level"]
        task_description = task.language

        # resume 跳过
        key = str(task_id)
        if key in task_results and task_results[key].get("episodes", 0) >= cfg.num_trials_per_task:
            continue

        # 准备 env
        try:
            bddl_path = task_suite.get_task_bddl_file_path(task_id)
            env_kwargs = {
                "bddl_file_name": bddl_path,
                "camera_heights": 256,
                "camera_widths": 256,
            }
            env = OffScreenRenderEnv(**env_kwargs)
            if cfg.env_seed is not None:
                env.seed(cfg.env_seed)
            else:
                env.seed(int(np.random.randint(0, 2**31 - 1)))
        except FileNotFoundError as e:
            msg = f"⚠️ Task {task_id} ({task_name}) 资产/bddl 缺失, 跳过: {e}"
            print(msg)
            log_file.write(msg + "\n")
            task_results[key] = {
                "task_id": task_id,
                "task_name": task_name,
                "category": category,
                "difficulty_level": difficulty,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
                "skipped": True,
                "error": str(e),
            }
            _save_results(
                results_path,
                {
                    "suite": cfg.task_suite_name,
                    "vlm_type": cfg.vlm_type,
                    "vlm_model_path": cfg.vlm_model_path,
                    "checkpoint_path": cfg.checkpoint_path,
                    "task_results": task_results,
                    "metrics": _aggregate_metrics(task_results),
                },
            )
            continue
        except Exception as e:  # noqa: BLE001
            msg = f"⚠️ Task {task_id} ({task_name}) 初始化失败: {type(e).__name__}: {e}"
            print(msg)
            log_file.write(msg + "\n")
            task_results[key] = {
                "task_id": task_id,
                "task_name": task_name,
                "category": category,
                "difficulty_level": difficulty,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
                "skipped": True,
                "error": f"{type(e).__name__}: {e}",
            }
            _save_results(
                results_path,
                {
                    "suite": cfg.task_suite_name,
                    "vlm_type": cfg.vlm_type,
                    "vlm_model_path": cfg.vlm_model_path,
                    "checkpoint_path": cfg.checkpoint_path,
                    "task_results": task_results,
                    "metrics": _aggregate_metrics(task_results),
                },
            )
            continue

        # 初始状态
        try:
            initial_states = task_suite.get_task_init_states(task_id)
        except Exception as e:  # noqa: BLE001
            msg = f"⚠️ Task {task_id} init_states 加载失败: {e}"
            print(msg)
            log_file.write(msg + "\n")
            with contextlib.suppress(Exception):
                env.close()
            task_results[key] = {
                "task_id": task_id,
                "task_name": task_name,
                "category": category,
                "difficulty_level": difficulty,
                "episodes": 0,
                "successes": 0,
                "success_rate": 0.0,
                "skipped": True,
                "error": f"init_states: {e}",
            }
            continue

        task_succ = 0
        trial_records: List[Dict[str, Any]] = []
        # 进入 trial 循环代表本 task 真的会运行, 此时再决定是否保存视频
        should_save_video = bool(cfg.save_videos) and (local_run_idx % save_every == 0)
        local_run_idx += 1
        for trial in range(cfg.num_trials_per_task):
            trial_t0 = time.time()
            env.reset()
            state_idx = trial % initial_states.shape[0]
            obs = env.set_init_state(initial_states[state_idx])

            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action())

            policy.reset_action_queue()

            done = False
            success = False
            top_view: List[np.ndarray] = []
            wrist_view: List[np.ndarray] = []
            steps_used = 0
            for step in range(cfg.max_steps):
                steps_used = step
                if done:
                    break
                if should_save_video:
                    img, wrist_img = get_libero_image(obs, flip=cfg.video_flip)
                    top_view.append(img)
                    wrist_view.append(wrist_img)

                try:
                    action = policy.get_action(obs, task_description)
                except Exception as e:  # noqa: BLE001
                    print(f"❌ Task {task_id} 推理失败: {type(e).__name__}: {e}")
                    log_file.write(
                        f"  Task {task_id} trial {trial}: 推理失败 {type(e).__name__}: {e}\n"
                    )
                    break

                obs, reward, done, info = env.step(action.tolist())
                if done and reward > 0:
                    success = True

            if success:
                task_succ += 1
            trial_records.append(
                {
                    "trial": trial,
                    "success": bool(success),
                    "steps": int(steps_used + 1),
                    "wall_time_sec": float(time.time() - trial_t0),
                }
            )

            if should_save_video and top_view:
                mp4_path = save_rollout_video(
                    top_view,
                    wrist_view,
                    idx=trial,
                    success=success,
                    task_description=task_description,
                    video_dir=f"{cfg.video_dir}/task_{task_id:05d}",
                )
                if mp4_path:
                    print(f"  💾 视频已保存: {mp4_path}")
                    log_file.write(f"  video: {mp4_path}\n")

        with contextlib.suppress(Exception):
            env.close()

        rate = task_succ / cfg.num_trials_per_task if cfg.num_trials_per_task else 0.0
        task_results[key] = {
            "task_id": task_id,
            "task_name": task_name,
            "category": category,
            "difficulty_level": difficulty,
            "episodes": cfg.num_trials_per_task,
            "successes": task_succ,
            "success_rate": rate,
            "task_description": task_description,
            "trials": trial_records,
        }

        # 实时聚合 + 保存
        metrics = _aggregate_metrics(task_results)
        _save_results(
            results_path,
            {
                "suite": cfg.task_suite_name,
                "shard_index": cfg.shard_index,
                "num_shards": cfg.num_shards,
                "vlm_type": cfg.vlm_type,
                "vlm_model_path": cfg.vlm_model_path,
                "checkpoint_path": cfg.checkpoint_path,
                "dataset_path": cfg.dataset_path,
                "config": {
                    "vlm_layers": cfg.vlm_layers,
                    "vlm_output_dim": cfg.vlm_output_dim,
                    "content_order": cfg.content_order,
                    "lowercase_instruction": cfg.lowercase_instruction,
                    "add_action_prompt": cfg.add_action_prompt,
                    "num_action_chunks": cfg.num_action_chunks,
                    "action_dim": cfg.action_dim,
                    "execute_all_chunks": cfg.execute_all_chunks,
                    "max_steps": cfg.max_steps,
                    "num_trials_per_task": cfg.num_trials_per_task,
                    "categories": cfg.categories,
                    "difficulty_levels": cfg.difficulty_levels,
                },
                "task_results": task_results,
                "metrics": metrics,
            },
        )

        log_file.write(
            f"task_id={task_id} cat={category} diff={difficulty} "
            f"name={task_name} success={task_succ}/{cfg.num_trials_per_task}\n"
        )
        log_file.flush()

        # 当前 task 单条状态 (在 tqdm 之上单独一行打印, 不会被覆盖)
        last_mark = "✓" if task_succ == cfg.num_trials_per_task else (
            "·" if task_succ > 0 else "✗"
        )
        cur_cat_metrics = metrics["by_category"].get(category, {})
        cur_diff_key = f"L{difficulty}" if difficulty is not None else "None"
        cur_diff_metrics = metrics["by_difficulty"].get(cur_diff_key, {})
        if cfg.print_per_task:
            tqdm.tqdm.write(
                f"[{last_mark}] task#{task_id:>5} | {category:<22} L{difficulty} | "
                f"trial_succ={task_succ}/{cfg.num_trials_per_task} | "
                f"overall={metrics['overall']['successes']}/{metrics['overall']['episodes']} "
                f"({metrics['overall']['success_rate'] * 100:.2f}%) | "
                f"{category[:8]}={cur_cat_metrics.get('successes', 0)}/{cur_cat_metrics.get('episodes', 0)} "
                f"({cur_cat_metrics.get('success_rate', 0.0) * 100:.1f}%) | "
                f"L{difficulty}={cur_diff_metrics.get('successes', 0)}/{cur_diff_metrics.get('episodes', 0)} "
                f"({cur_diff_metrics.get('success_rate', 0.0) * 100:.1f}%)"
            )

        # tqdm postfix 多字段实时刷新
        pbar.set_postfix_str(
            f"{last_mark} cur={category[:6]}/L{difficulty} "
            f"overall={metrics['overall']['success_rate'] * 100:.2f}% "
            f"({metrics['overall']['successes']}/{metrics['overall']['episodes']}) "
            f"{category[:6]}={cur_cat_metrics.get('success_rate', 0.0) * 100:.1f}% "
            f"L{difficulty}={cur_diff_metrics.get('success_rate', 0.0) * 100:.1f}%"
        )

        # 每 N 个 task 完整 dump 一次 per-category / per-difficulty 表格
        if (
            cfg.summary_every > 0
            and metrics["overall"]["episodes"] > 0
            and metrics["overall"]["episodes"] % cfg.summary_every == 0
        ):
            tqdm.tqdm.write(
                f"\n────── 进度小结 (已评 {metrics['overall']['episodes']} ep) ──────"
            )
            tqdm.tqdm.write(
                f"  Overall: {metrics['overall']['successes']}/{metrics['overall']['episodes']} "
                f"= {metrics['overall']['success_rate'] * 100:.2f}%"
            )
            for cat_name in list(LIBERO_PLUS_CATEGORIES) + ["Unknown"]:
                if cat_name in metrics["by_category"]:
                    v = metrics["by_category"][cat_name]
                    tqdm.tqdm.write(
                        f"    {cat_name:<22} {v['successes']:>4}/{v['episodes']:<4} "
                        f"= {v['success_rate'] * 100:>6.2f}%"
                    )
            for diff_key in sorted(metrics["by_difficulty"].keys()):
                v = metrics["by_difficulty"][diff_key]
                tqdm.tqdm.write(
                    f"    {diff_key:<22} {v['successes']:>4}/{v['episodes']:<4} "
                    f"= {v['success_rate'] * 100:>6.2f}%"
                )
            tqdm.tqdm.write("─" * 50)

    log_file.close()

    # 终态汇总
    final_metrics = _aggregate_metrics(task_results)
    _print_summary(final_metrics, cfg.task_suite_name)

    summary_payload = {
        "suite": cfg.task_suite_name,
        "shard_index": cfg.shard_index,
        "num_shards": cfg.num_shards,
        "vlm_type": cfg.vlm_type,
        "vlm_model_path": cfg.vlm_model_path,
        "checkpoint_path": cfg.checkpoint_path,
        "dataset_path": cfg.dataset_path,
        "task_results": task_results,
        "metrics": final_metrics,
    }
    _save_results(results_path, summary_payload)
    print(f"\n💾 结果已保存到: {results_path}")
    return summary_payload


# ============================================================================
# 入口
# ============================================================================

def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIBERO-plus 评估 (Sai0_1 OFT1_0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 模型
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--vlm_model_path", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument(
        "--vlm_type", type=str, default="qwen3_vl",
        choices=["qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"],
    )
    parser.add_argument("--dataset_path", type=str, default="")

    # VLM
    parser.add_argument("--vlm_layers", type=str, default="14")
    parser.add_argument("--vlm_output_dim", type=int, default=2048)

    # Prompt
    parser.add_argument(
        "--content_order", type=str, default="images_first",
        choices=["images_first", "text_first", "interleaved", "single_image"],
    )
    parser.add_argument("--lowercase_instruction", action="store_true", default=True)
    parser.add_argument(
        "--no_lowercase_instruction", action="store_false", dest="lowercase_instruction",
    )
    parser.add_argument("--add_generation_prompt", action="store_true", default=True)
    parser.add_argument(
        "--no_generation_prompt", action="store_false", dest="add_generation_prompt",
    )
    parser.add_argument("--add_action_prompt", action="store_true", default=False)
    parser.add_argument(
        "--no_action_prompt", action="store_false", dest="add_action_prompt",
    )

    # OFT
    parser.add_argument("--num_transformer_blocks", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--action_head_hidden_dim", type=int, default=4096)
    parser.add_argument("--num_action_chunks", type=int, default=16)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--no_binarize_gripper", action="store_false", dest="binarize_gripper")
    parser.add_argument("--binarize_gripper", action="store_true", default=True)

    # LIBERO-plus
    parser.add_argument(
        "--task_suite_name", type=str, default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    parser.add_argument("--num_trials_per_task", type=int, default=1,
                        help="LIBERO-plus 官方约定为 1, 因为每个 task 已经是一个独立扰动 variant")
    parser.add_argument("--task_ids", type=str, default=None)
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--env_seed", type=int, default=None)
    parser.add_argument(
        "--categories", type=str, default=None,
        help='逗号分隔, 如 "Camera Viewpoints,Robot Initial States". '
             "可选: " + ", ".join(LIBERO_PLUS_CATEGORIES),
    )
    parser.add_argument(
        "--difficulty_levels", type=str, default=None,
        help='逗号分隔, 如 "1,2,3"',
    )

    # 推理
    parser.add_argument("--action_chunk_size", type=int, default=16)
    parser.add_argument("--execute_all_chunks", action="store_true", default=True)
    parser.add_argument(
        "--no_execute_all_chunks", action="store_false", dest="execute_all_chunks",
    )

    # 系统
    parser.add_argument("--device", type=str, default="cuda:0")
    # LIBERO env 直出的图像是 OpenGL 颠倒方向, 必须翻 180° 才是正向 (与训练对齐).
    parser.add_argument("--flip_images", action="store_true", default=True,
                        help="翻转喂给 VLM backbone 的图像 (默认 True, 把 OpenGL 颠倒图翻正)")
    parser.add_argument("--no_flip_images", action="store_false", dest="flip_images")
    # video_flip 跟 flip_images 解耦, 但默认值一致, 保持二者展示和模型输入方向一致
    parser.add_argument("--video_flip", action="store_true", default=True,
                        help="保存到 mp4 时翻转图像 (仅影响视频, 不影响给模型的输入)")
    parser.add_argument("--no_video_flip", action="store_false", dest="video_flip")
    parser.add_argument("--video_dir", type=str, default="./eval_rollouts")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--save_videos", action="store_true",
                        help="保存 rollout 视频 (10K+ tasks 默认关闭以省磁盘)")
    parser.add_argument("--save_video_every", type=int, default=1,
                        help="每隔多少个 task 才保存一次视频 (默认 1=每个 task 都存; 仅当 --save_videos 时生效)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no_resume", action="store_false", dest="resume",
                        help="不复用已有 eval_results JSON")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument(
        "--no_print_per_task", action="store_false", dest="print_per_task",
        help="关闭 'tqdm 每完成 1 个 task 单独 print 一行' 的实时输出 (默认开启)",
    )
    parser.add_argument(
        "--print_per_task", action="store_true", default=True,
        help="开启 (默认) 每完成 1 个 task 单独 print 一行带 ✓/✗ 标记的成功率",
    )
    parser.add_argument(
        "--summary_every", type=int, default=100,
        help="每 N 个 episode 在 stdout 打印一次完整 per-category / per-difficulty 进度表 "
             "(默认 100, 设为 0 关闭)",
    )
    parser.add_argument(
        "--shard_index", type=int, default=0,
        help="多卡 data parallel 时本 worker 的 shard 索引 [0, num_shards). 单卡时设 0.",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="多卡 data parallel 时总 shard 数 (= 使用的 GPU 数). 单卡时设 1.",
    )

    args = parser.parse_args()

    cfg = EvalConfig(
        checkpoint_path=args.checkpoint_path,
        vlm_model_path=args.vlm_model_path,
        vlm_type=args.vlm_type,
        dataset_path=args.dataset_path,
        vlm_layers=_parse_int_list(args.vlm_layers) or [-1],
        vlm_output_dim=args.vlm_output_dim,
        content_order=args.content_order,
        lowercase_instruction=args.lowercase_instruction,
        add_generation_prompt=args.add_generation_prompt,
        add_action_prompt=args.add_action_prompt,
        num_transformer_blocks=args.num_transformer_blocks,
        num_attention_heads=args.num_attention_heads,
        dropout=args.dropout,
        action_head_hidden_dim=args.action_head_hidden_dim,
        num_action_chunks=args.num_action_chunks,
        action_dim=args.action_dim,
        binarize_gripper=args.binarize_gripper,
        task_suite_name=args.task_suite_name,
        num_trials_per_task=args.num_trials_per_task,
        task_ids=_parse_int_list(args.task_ids),
        max_tasks=args.max_tasks,
        num_steps_wait=args.num_steps_wait,
        max_steps=args.max_steps,
        env_seed=args.env_seed,
        categories=_parse_str_list(args.categories),
        difficulty_levels=_parse_int_list(args.difficulty_levels),
        action_chunk_size=args.action_chunk_size,
        execute_all_chunks=args.execute_all_chunks,
        device=args.device,
        flip_images=args.flip_images,
        video_flip=args.video_flip,
        video_dir=args.video_dir,
        log_dir=args.log_dir,
        save_videos=args.save_videos,
        save_video_every=args.save_video_every,
        verbose=args.verbose,
        resume=args.resume,
        print_per_task=args.print_per_task,
        summary_every=args.summary_every,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    eval_libero_plus(cfg)


if __name__ == "__main__":
    main()
