"""
Dataset196 Open-Loop Evaluation
================================

针对 OFT checkpoint:
  /data_disk2/hwl/checkpoints/unitree_v2_pretrain/action7_chunk16/
   p6000_bsz500*1*8_tb4_400000steps_layer14_20260511_wp...USE_AMP_true_MD_true/checkpoints/step_300000

数据集 (默认 = velocity-filter 版, 与 _then_tele 训练脚本一致):
  /data_disk2/hwl/datasets/dataset196/
    lerobot_dataset196_rightarm_state_joint_action_eefdelta_filter_by_velocity

VLM hidden state 来源:
  默认 (--use_pre_extracted_vlm=True):
      <dataset_path>/vlm_hidden_states/hidden_state_XXXXXX.npy
      文件名编号 = parquet 列 'vlm_hidden_state_index' (全局 frame index)
      shape = (num_layers, seq_len, hidden_dim)  或  (seq_len, hidden_dim)
      与 utils/extract_vlm_hidden_state/.../run_qwen_*_extraction.sh 输出一致.
      训练脚本 train_qwen_pretrain_unitree_v2_multi_dataset_then_tele.sh
      也是吃同一份 npy.
  当 --no_use_pre_extracted_vlm:
      脚本退回 "在线" 模式: 进程内加载 Qwen3-VL-2B 做 VLM 推理 (老逻辑).

流程:
  1. (按需) 加载 Qwen3-VL-2B / 加载 OFT Pipeline
  2. 从 73 episode 里随机选 5 条 (可指定 --seed / --episodes)
  3. 对每条 episode 做完整 open-loop:
        for start in range(0, num_steps, chunk_size):
            读 state[start]
            读 hidden_state_{vlm_index}.npy   (或 在线 VLM 推理两路图像)
            -> OFT pipeline -> 16 步 delta (归一化空间)
            把全部 16 步采纳, 拼到预测序列
     直到拼到原 episode 长度
  4. 反归一化 (使用本数据集 stats.json)
  5. 画图: 每维 GT vs Pred + 累积曲线 + xyz 累积轨迹 + 首帧

VLM 提取参数 (与训练时 extraction 完全一致):
    model_path        = Qwen/Qwen3-VL-2B-Instruct
    layers            = [14]
    flip_images       = False
    prompt_template   = "simple"   (即 "{instruction}")
    content_order     = "images_first"
    lowercase_instruction  = True
    add_generation_prompt  = True
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import av
from PIL import Image

# ---- 项目路径 ----
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _read_oft_config(ckpt_dir: Path) -> dict:
    with open(ckpt_dir / "config.json", "r") as f:
        return json.load(f)


def _setup_constants_env(ckpt_dir: Path):
    """
    Action_Heads/OFT1_0/constants.py 在 module import 时读取
    ACTION_DIM / PROPRIO_DIM / NUM_ACTIONS_CHUNK 环境变量,
    所以必须在 import vlm2oft_pipeline 之前设置.
    """
    cfg = _read_oft_config(ckpt_dir)
    os.environ["ACTION_DIM"] = str(cfg["action_dim"])
    os.environ["PROPRIO_DIM"] = str(cfg["proprio_dim"])
    os.environ["NUM_ACTIONS_CHUNK"] = str(cfg["num_actions_chunk"])
    print(
        f"[constants.py env] ACTION_DIM={os.environ['ACTION_DIM']} "
        f"PROPRIO_DIM={os.environ['PROPRIO_DIM']} "
        f"NUM_ACTIONS_CHUNK={os.environ['NUM_ACTIONS_CHUNK']}"
    )


# ============================================================================
# Dataset helpers
# ============================================================================

def load_info(dataset_path: Path) -> dict:
    with open(dataset_path / "meta" / "info.json", "r") as f:
        return json.load(f)


def load_stats(dataset_path: Path) -> dict:
    with open(dataset_path / "meta" / "stats.json", "r") as f:
        return json.load(f)


def load_tasks(dataset_path: Path) -> Dict[int, str]:
    p = dataset_path / "meta" / "tasks.jsonl"
    if not p.exists():
        return {}
    out: Dict[int, str] = {}
    with open(p, "r") as f:
        for line in f:
            t = json.loads(line.strip())
            out[int(t["task_index"])] = t["task"]
    return out


def list_episode_indices(dataset_path: Path) -> List[int]:
    """从 episodes.jsonl 读全部 episode_index。"""
    p = dataset_path / "meta" / "episodes.jsonl"
    eps = []
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            eps.append(int(entry["episode_index"]))
    return sorted(eps)


def _read_all_frames(video_path: Path, expected_n: Optional[int]) -> List[np.ndarray]:
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if expected_n is not None and len(frames) >= expected_n:
            break
    container.close()
    return frames


def _read_first_frame(video_path: Path) -> Optional[np.ndarray]:
    """只解码第一帧 (用于 first_frame.png 可视化, 避免整段视频解码)."""
    try:
        container = av.open(str(video_path))
    except Exception as e:
        print(f"[WARN] open video failed: {video_path} ({e})")
        return None
    try:
        for frame in container.decode(video=0):
            return frame.to_ndarray(format="rgb24")
    finally:
        container.close()
    return None


def load_episode(
    dataset_path: Path,
    episode_index: int,
    info: dict,
    tasks: Dict[int, str],
    load_full_video: bool = True,
) -> dict:
    """
    Args:
        load_full_video:
            True  -> 解码整段 mp4 (用于在线 VLM 推理, 每帧都要拿图片)
            False -> 只解码第 0 帧 (用于 pre-extracted-VLM 模式, 只为画 first_frame.png)
    """
    chunks_size = info["chunks_size"]
    chunk_idx = episode_index // chunks_size
    pq = (
        dataset_path
        / "data"
        / f"chunk-{chunk_idx:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )
    df = pd.read_parquet(pq)
    n = len(df)

    states = np.array(df["observation.state"].tolist(), dtype=np.float32)
    actions = np.array(df["action"].tolist(), dtype=np.float32)

    if "vlm_hidden_state_index" in df.columns:
        vlm_indices = df["vlm_hidden_state_index"].to_numpy().astype(np.int64)
    else:
        vlm_indices = None

    image_keys = sorted(
        [k for k in info["features"] if k.startswith("observation.images.")]
    )
    if not image_keys:
        raise ValueError("No image keys in dataset info.")

    image_streams: Dict[str, List[np.ndarray]] = {}
    first_frames: Dict[str, np.ndarray] = {}
    for key in image_keys:
        vid = (
            dataset_path
            / "videos"
            / f"chunk-{chunk_idx:03d}"
            / key
            / f"episode_{episode_index:06d}.mp4"
        )
        if not vid.exists():
            print(f"[WARN] video missing: {vid}, skip key={key}")
            continue
        if load_full_video:
            frames = _read_all_frames(vid, n)
            if len(frames) != n:
                print(f"[WARN] {key} got {len(frames)} frames, expected {n}")
            image_streams[key] = frames
            first_frames[key] = frames[0] if frames else None
        else:
            f0 = _read_first_frame(vid)
            if f0 is not None:
                first_frames[key] = f0

    # 任务指令
    instruction = ""
    if "task_index" in df.columns:
        instruction = tasks.get(int(df.iloc[0]["task_index"]), "")

    return {
        "episode_index": episode_index,
        "num_steps": n,
        "states": states,
        "actions": actions,
        "vlm_indices": vlm_indices,
        "image_streams": image_streams,
        "first_frames": first_frames,
        "image_keys": image_keys,
        "instruction": instruction,
    }


# ============================================================================
# Normalizer
# ============================================================================

class MinMaxNormalizer:
    def __init__(self, vmin: np.ndarray, vmax: np.ndarray):
        self.vmin = vmin.astype(np.float32)
        self.vmax = vmax.astype(np.float32)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        rng = self.vmax - self.vmin
        out = np.zeros_like(x, dtype=np.float32)
        mask = rng != 0
        if mask.any():
            out[..., mask] = 2.0 * (x[..., mask] - self.vmin[mask]) / rng[mask] - 1.0
        return out

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x + 1.0) / 2.0 * (self.vmax - self.vmin) + self.vmin


def build_normalizer(stats: dict, key: str) -> Optional[MinMaxNormalizer]:
    if key not in stats:
        return None
    s = stats[key]
    if "min" not in s or "max" not in s:
        return None
    return MinMaxNormalizer(
        np.asarray(s["min"], dtype=np.float32),
        np.asarray(s["max"], dtype=np.float32),
    )


# ============================================================================
# Open-loop predictor
# ============================================================================

class OpenLoopPredictor:
    """Wraps (可选) Qwen3-VL backbone + OFT pipeline 做 step-wise 推理.

    use_pre_extracted_vlm=True 时, 跳过 VLM backbone 加载,
    直接吃 npy 形式的 hidden states (与训练 dataloader 一致).
    """

    def __init__(
        self,
        ckpt_path: Path,
        device: str = "cuda:0",
        use_pre_extracted_vlm: bool = True,
        vlm_model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        layers: Tuple[int, ...] = (14,),
        prompt_template: str = "simple",
        content_order: str = "images_first",
        flip_images: bool = False,
        lowercase_instruction: bool = True,
        add_generation_prompt: bool = True,
    ):
        self.device = device
        self.use_pre_extracted_vlm = use_pre_extracted_vlm

        # ---- 读 OFT config.json ----
        ckpt_dir = ckpt_path.parent
        with open(ckpt_dir / "config.json", "r") as f:
            cfg = json.load(f)

        self.chunk_size: int = cfg["num_actions_chunk"]
        self.action_dim: int = cfg["action_dim"]
        self.proprio_dim: int = cfg["proprio_dim"]
        self.num_blocks: int = cfg["num_blocks"]
        self.num_attention_heads: int = cfg["num_attention_heads"]
        self.action_head_hidden_dim: int = cfg["action_head_hidden_dim"]
        self.vlm_output_dim: int = cfg["llm_output_dim"]
        self.num_vlm_layers: int = cfg["num_vlm_hidden_layers"]
        self.dropout: float = cfg.get("dropout", 0.1)

        print(
            f"[OFT cfg] chunk={self.chunk_size} action_dim={self.action_dim} "
            f"proprio={self.proprio_dim} blocks={self.num_blocks} heads={self.num_attention_heads} "
            f"hidden={self.action_head_hidden_dim} vlm_dim={self.vlm_output_dim} "
            f"vlm_layers={self.num_vlm_layers}"
        )

        # ---- 加载 VLM backbone (按需) ----
        self.vlm = None
        if self.use_pre_extracted_vlm:
            print("[VLM] using pre-extracted hidden states from .npy "
                  "(skip loading Qwen3-VL backbone)")
        else:
            print(f"[VLM] loading {vlm_model_path} on {device} layers={list(layers)}")
            self.vlm = create_vlm_backbone(
                model_type="qwen3_vl",
                model_path=vlm_model_path,
                device=device,
                layers=list(layers),
                flip_images=flip_images,
                content_order=content_order,
                prompt_template=prompt_template,
                lowercase_instruction=lowercase_instruction,
                add_generation_prompt=add_generation_prompt,
            )
            print("[VLM] ready.")

        # ---- 加载 OFT pipeline ----
        print(f"[OFT] building pipeline + loading {ckpt_path}")
        self.oft = create_vlm2oft_pipeline(
            num_transformer_blocks=self.num_blocks,
            num_attention_heads=self.num_attention_heads,
            num_vlm_layers=self.num_vlm_layers,
            vlm_output_dim=self.vlm_output_dim,
            action_head_hidden_dim=self.action_head_hidden_dim,
            dropout=self.dropout,
        ).to(device)
        sd = torch.load(str(ckpt_path), map_location=device)
        self.oft.load_state_dict(sd)
        self.oft.eval()
        print("[OFT] ready.")

    # ----------------------------------------------------------------
    # numpy hidden state -> List[Tensor], 喂给 OFT pipeline 用
    # ----------------------------------------------------------------
    def _np_to_hidden_states_list(self, arr: np.ndarray) -> List[torch.Tensor]:
        """
        npy 可能是:
          - (seq_len, hidden_dim)            ->  视为 1 层
          - (num_layers, seq_len, hidden_dim)
          - (1, seq_len, hidden_dim)         ->  num_layers=1
        返回 List[Tensor], 每个 Tensor shape = (1, seq, hidden_dim), float32 on device
        """
        if arr.ndim == 2:
            layers = [arr]
        elif arr.ndim == 3:
            layers = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError(f"unexpected vlm hidden state shape: {arr.shape}")

        if self.vlm_output_dim and layers[0].shape[-1] != self.vlm_output_dim:
            raise ValueError(
                f"vlm hidden_dim={layers[0].shape[-1]} != model expects "
                f"{self.vlm_output_dim}, 检查 npy 来源是否匹配 checkpoint."
            )
        if len(layers) != self.num_vlm_layers:
            raise ValueError(
                f"npy has {len(layers)} layers but model expects "
                f"{self.num_vlm_layers}. (检查训练 NUM_VLM_LAYERS / 提取 LAYERS 是否一致)"
            )

        return [
            torch.from_numpy(np.ascontiguousarray(l)).unsqueeze(0).to(self.device).float()
            for l in layers
        ]

    @torch.inference_mode()
    def predict_chunk_from_hidden(
        self,
        hidden_state_np: np.ndarray,
        proprio: np.ndarray,
    ) -> np.ndarray:
        """直接吃预提取 npy 的版本 (与训练 dataloader 一致)."""
        hs = self._np_to_hidden_states_list(hidden_state_np)
        proprio_t = torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(self.device)

        action_pred = self.oft(hs, proprio_t)  # (1, 1, chunk*action_dim)
        actions = action_pred[0, 0].detach().float().cpu().numpy()
        actions = actions.reshape(self.chunk_size, self.action_dim)
        return actions

    @torch.inference_mode()
    def predict_chunk(
        self,
        images: List[np.ndarray],
        proprio: np.ndarray,
        instruction: str,
    ) -> np.ndarray:
        """
        在线 VLM 版本, 仅当 use_pre_extracted_vlm=False 时使用.
        Args:
            images:    list of HxWx3 uint8 RGB (与训练时图像数一致, 通常 2 张)
            proprio:   shape (proprio_dim,) float32, 已经归一化到 [-1,1]
            instruction: str
        Returns:
            np.ndarray, shape (chunk_size, action_dim), 在 [-1,1] 归一化空间
        """
        if self.vlm is None:
            raise RuntimeError(
                "predict_chunk() called but VLM backbone not loaded. "
                "请用 predict_chunk_from_hidden(), 或构造时 use_pre_extracted_vlm=False."
            )
        vlm_out = self.vlm.get_hidden_states(
            images=images, instruction=instruction
        )
        hs = [v.to(self.device) for v in vlm_out.hidden_states]

        proprio_t = torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(self.device)

        action_pred = self.oft(hs, proprio_t)  # (1, 1, chunk*action_dim)
        actions = action_pred[0, 0].detach().float().cpu().numpy()
        actions = actions.reshape(self.chunk_size, self.action_dim)
        return actions


# ============================================================================
# Plotting
# ============================================================================

ACTION_NAMES = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
STATE_NAMES = [f"state[{i}]" for i in range(8)]


def plot_action_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    chunk_size: int,
    instruction: str,
    ep_idx: int,
    out_path: Path,
):
    n = min(len(gt), len(pred))
    dim = gt.shape[-1]
    t = np.arange(n)

    nrows = (dim + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(18, 3.4 * nrows))
    axes = np.array(axes).reshape(nrows, 2).flatten()

    fig.suptitle(
        f"Episode {ep_idx} | {n} steps | chunk={chunk_size} | open-loop\n"
        f"\"{instruction}\"",
        fontsize=13, fontweight="bold", y=1.0,
    )

    for i in range(dim):
        ax = axes[i]
        g, p = gt[:n, i], pred[:n, i]
        ax.plot(t, g, "b-", label="GT", lw=1.4, alpha=0.85)
        ax.plot(t, p, "r--", label="Pred", lw=1.4, alpha=0.85)
        for b in range(chunk_size, n, chunk_size):
            ax.axvline(b, color="gray", ls=":", alpha=0.4)
        rmse = float(np.sqrt(np.mean((g - p) ** 2)))
        mae = float(np.mean(np.abs(g - p)))
        nm = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"dim{i}"
        ax.set_title(f"{nm}  RMSE={rmse:.4f}  MAE={mae:.4f}", fontsize=11)
        ax.set_xlabel("Step")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(dim, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {out_path}")


def plot_action_cumsum_comparison(
    gt: np.ndarray,
    pred: np.ndarray,
    chunk_size: int,
    instruction: str,
    ep_idx: int,
    out_path: Path,
    gripper_col: Optional[int] = None,
):
    """
    每维 action 的累加曲线对比 (GT cumsum vs Pred cumsum).

    - 前 N-1 维 (delta xyz / delta rpy) 直接 cumsum, 表示从 0 出发的累计位移/旋转.
    - gripper 列 (binary 0/1) 也做 cumsum, 但语义是"累计闭合次数",
      在标题里特别标注一下.
    """
    n = min(len(gt), len(pred))
    dim = gt.shape[-1]
    t = np.arange(n)

    gt_cs = np.cumsum(gt[:n], axis=0)
    pr_cs = np.cumsum(pred[:n], axis=0)

    nrows = (dim + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(18, 3.4 * nrows))
    axes = np.array(axes).reshape(nrows, 2).flatten()

    fig.suptitle(
        f"Episode {ep_idx} | {n} steps | chunk={chunk_size} | open-loop"
        f" | per-dim CUMSUM\n\"{instruction}\"",
        fontsize=13, fontweight="bold", y=1.0,
    )

    for i in range(dim):
        ax = axes[i]
        g, p = gt_cs[:, i], pr_cs[:, i]
        ax.plot(t, g, "b-", label="GT cumsum", lw=1.5, alpha=0.9)
        ax.plot(t, p, "r--", label="Pred cumsum", lw=1.5, alpha=0.9)
        for b in range(chunk_size, n, chunk_size):
            ax.axvline(b, color="gray", ls=":", alpha=0.35)
        end_diff = float(p[-1] - g[-1])
        end_abs = float(abs(end_diff))
        nm = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"dim{i}"
        tag = ""
        if gripper_col is not None and i == gripper_col:
            tag = " [binary -> cum-close-count]"
        ax.set_title(
            f"{nm}{tag}  end GT={g[-1]:+.3f} Pred={p[-1]:+.3f} "
            f"diff={end_diff:+.3f}  |diff|={end_abs:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("cumsum")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(dim, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {out_path}")


def plot_xyz_trajectory(
    gt: np.ndarray,
    pred: np.ndarray,
    instruction: str,
    ep_idx: int,
    out_path: Path,
    init_xyz: Optional[np.ndarray] = None,
):
    """Cumulative xyz trajectory comparison from delta actions."""
    if gt.shape[-1] < 3 or pred.shape[-1] < 3:
        return
    n = min(len(gt), len(pred))
    init = init_xyz.astype(np.float32) if init_xyz is not None else np.zeros(3, dtype=np.float32)
    gt_traj = init + np.cumsum(gt[:n, :3], axis=0)
    pr_traj = init + np.cumsum(pred[:n, :3], axis=0)

    fig = plt.figure(figsize=(15, 5))

    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    ax3d.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], "b-", lw=1.4, label="GT")
    ax3d.plot(pr_traj[:, 0], pr_traj[:, 1], pr_traj[:, 2], "r--", lw=1.4, label="Pred")
    ax3d.scatter(*gt_traj[0], color="green", s=80, marker="o", label="start")
    ax3d.scatter(*gt_traj[-1], color="blue", s=80, marker="x", label="GT end")
    ax3d.scatter(*pr_traj[-1], color="red", s=80, marker="^", label="Pred end")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("XYZ cumsum trajectory")
    ax3d.legend(fontsize=8)

    # 1D 累计轨迹分轴
    ax1 = fig.add_subplot(1, 3, 2)
    for i, lab in enumerate(["x", "y", "z"]):
        ax1.plot(gt_traj[:, i], label=f"GT {lab}", alpha=0.85)
        ax1.plot(pr_traj[:, i], "--", label=f"Pred {lab}", alpha=0.85)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cumulative position")
    ax1.set_title("Cumulative xyz")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(1, 3, 3)
    err = np.linalg.norm(gt_traj - pr_traj, axis=1)
    ax2.plot(err, "k-", lw=1.4)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("|Pred - GT|")
    ax2.set_title(f"Cumulative pos err  end={err[-1]:.3f}")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Episode {ep_idx} cumulative xyz trajectory | \"{instruction}\"",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {out_path}")


def save_first_frame(img: np.ndarray, instruction: str, path: Path):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    print(f"  saved first frame -> {path} (\"{instruction}\")")


# ============================================================================
# Open-loop run for one episode
# ============================================================================

def run_open_loop(
    predictor: OpenLoopPredictor,
    ep: dict,
    state_normalizer: Optional[MinMaxNormalizer],
    action_normalizer: Optional[MinMaxNormalizer],
    binarize_gripper_col: Optional[int],
    gripper_threshold: float,
    out_dir: Path,
    vlm_hidden_states_dir: Optional[Path] = None,
    print_every: int = 5,
):
    ep_idx = ep["episode_index"]
    n = ep["num_steps"]
    chunk_size = predictor.chunk_size
    action_dim = predictor.action_dim
    proprio_dim = predictor.proprio_dim
    instruction = ep["instruction"]

    gt_actions = ep["actions"]
    states = ep["states"]
    streams = ep["image_streams"]
    first_frames = ep.get("first_frames", {})
    keys = ep["image_keys"]
    vlm_indices = ep.get("vlm_indices", None)

    # ---- 准备 proprio (可选) 归一化 ----
    if state_normalizer is not None:
        proprio_states = state_normalizer.normalize(states.copy())
    else:
        proprio_states = states.copy()

    use_pre = predictor.use_pre_extracted_vlm
    print(
        f"\n>>> Episode {ep_idx}: {n} steps, chunk={chunk_size}, "
        f"action_dim={action_dim}, proprio_dim={proprio_dim}, "
        f"image_keys={keys}, vlm_mode={'npy' if use_pre else 'online'}"
    )
    print(f'    instruction="{instruction}"')

    if use_pre:
        if vlm_hidden_states_dir is None:
            raise RuntimeError("use_pre_extracted_vlm=True 但没传 vlm_hidden_states_dir")
        if vlm_indices is None:
            raise RuntimeError(
                "parquet 缺少 'vlm_hidden_state_index' 列, 无法定位 npy 文件"
            )

    pred_norm_chunks = []
    starts = list(range(0, n, chunk_size))
    t0 = time.time()
    for i, start in enumerate(starts):
        remaining = n - start
        take = min(chunk_size, remaining)

        proprio = proprio_states[start, :proprio_dim]

        t_call = time.time()
        if use_pre:
            vlm_idx = int(vlm_indices[start])
            npy_path = vlm_hidden_states_dir / f"hidden_state_{vlm_idx:06d}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(
                    f"missing VLM npy: {npy_path}  "
                    f"(ep={ep_idx} step={start} vlm_idx={vlm_idx}); "
                    "请先把这部分 episode 的 vlm hidden states 跑出来"
                )
            hidden_np = np.load(npy_path)
            chunk_norm = predictor.predict_chunk_from_hidden(
                hidden_state_np=hidden_np, proprio=proprio
            )
        else:
            frames = [streams[k][start] for k in keys]
            chunk_norm = predictor.predict_chunk(
                images=frames, proprio=proprio, instruction=instruction
            )
        dt = time.time() - t_call

        pred_norm_chunks.append(chunk_norm[:take])

        if i < 3 or i == len(starts) - 1 or (i + 1) % print_every == 0:
            extra = (
                f"vlm_idx={int(vlm_indices[start])} " if use_pre and vlm_indices is not None else ""
            )
            print(
                f"    [{i + 1}/{len(starts)}] start={start} {extra}"
                f"take={take}/{chunk_norm.shape[0]} call={dt:.2f}s"
            )

    pred_norm = np.concatenate(pred_norm_chunks, axis=0)  # (N, action_dim)
    total = time.time() - t0
    print(
        f"<<< Episode {ep_idx} done: total={total:.1f}s, "
        f"avg per chunk={total / max(1, len(starts)):.2f}s"
    )

    # ---- 反归一化 + gripper 二值化 ----
    if action_normalizer is not None:
        pred_phys = action_normalizer.denormalize(pred_norm)
    else:
        pred_phys = pred_norm.copy()

    if binarize_gripper_col is not None and 0 <= binarize_gripper_col < pred_phys.shape[-1]:
        pred_phys[:, binarize_gripper_col] = (
            pred_phys[:, binarize_gripper_col] > gripper_threshold
        ).astype(np.float32)

    # ---- 指标 ----
    mn = min(len(gt_actions), len(pred_phys))
    g = gt_actions[:mn]
    p = pred_phys[:mn]
    cdim = min(g.shape[-1], p.shape[-1])
    g = g[:, :cdim]
    p = p[:, :cdim]

    mse = float(np.mean((g - p) ** 2))
    mae = float(np.mean(np.abs(g - p)))
    per_dim_rmse = np.sqrt(np.mean((g - p) ** 2, axis=0))

    print(f"    physical-space metrics: MSE={mse:.5f}  MAE={mae:.5f}")
    for i, r in enumerate(per_dim_rmse):
        nm = ACTION_NAMES[i] if i < len(ACTION_NAMES) else f"dim{i}"
        print(f"      {nm}: RMSE={r:.5f}")

    # ---- 绘图 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    first_img = None
    if first_frames and keys[0] in first_frames:
        first_img = first_frames[keys[0]]
    elif streams and keys[0] in streams and streams[keys[0]]:
        first_img = streams[keys[0]][0]
    if first_img is not None:
        save_first_frame(
            first_img,
            instruction,
            out_dir / f"ep{ep_idx:06d}_first_frame.png",
        )
    else:
        print("  [WARN] no first frame available, skip first_frame.png")
    plot_action_comparison(
        gt=g,
        pred=p,
        chunk_size=chunk_size,
        instruction=instruction,
        ep_idx=ep_idx,
        out_path=out_dir / f"ep{ep_idx:06d}_action_open_loop.png",
    )
    plot_action_cumsum_comparison(
        gt=g,
        pred=p,
        chunk_size=chunk_size,
        instruction=instruction,
        ep_idx=ep_idx,
        out_path=out_dir / f"ep{ep_idx:06d}_action_cumsum.png",
        gripper_col=binarize_gripper_col,
    )
    plot_xyz_trajectory(
        gt=g,
        pred=p,
        instruction=instruction,
        ep_idx=ep_idx,
        out_path=out_dir / f"ep{ep_idx:06d}_xyz_traj.png",
    )

    # ---- npz dump ----
    np.savez_compressed(
        out_dir / f"ep{ep_idx:06d}_results.npz",
        gt_actions=g,
        pred_actions=p,
        pred_actions_norm=pred_norm[:mn, :cdim],
        chunk_size=chunk_size,
        instruction=instruction,
    )

    return {
        "episode_index": ep_idx,
        "num_steps": n,
        "mse": mse,
        "mae": mae,
        "per_dim_rmse": per_dim_rmse.tolist(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "/data_disk2/hwl/checkpoints/unitree_v2_pretrain/action7_chunk16/"
            "p6000_bsz500*1*8_tb4_400000steps_layer14_20260511_wpunitree_v2_pretrain_"
            "unitree_v2_pretrain_CBC_true_dtypefloat32_USE_AMP_true_MD_true/"
            "checkpoints/step_300000"
        ),
        help="checkpoint 目录, 内含 action_head.pt + config.json",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=(
            "/data_disk2/hwl/datasets/dataset196/"
            "lerobot_dataset196_rightarm_state_joint_action_eefdelta_filter_by_velocity"
        ),
    )
    parser.add_argument(
        "--vlm_hidden_states_dir",
        type=str,
        default=None,
        help=(
            "预提取的 VLM hidden states 目录 (per-frame .npy 格式), "
            "为空则默认 <dataset_path>/vlm_hidden_states"
        ),
    )
    parser.add_argument(
        "--use_pre_extracted_vlm",
        action="store_true",
        default=True,
        help="(默认) 从 vlm_hidden_states/*.npy 读 hidden states, 跳过 VLM backbone",
    )
    parser.add_argument(
        "--no_use_pre_extracted_vlm",
        dest="use_pre_extracted_vlm",
        action="store_false",
        help="退回在线模式: 进程内加载 Qwen3-VL backbone 实时跑 VLM",
    )
    parser.add_argument(
        "--vlm_model_path",
        type=str,
        default=(
            "/home/dev/.cache/huggingface/hub/"
            "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
            "89644892e4d85e24eaac8bacfd4f463576704203"
        ),
        help="Qwen3-VL-2B-Instruct 本地快照路径 (仅在线模式用)",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
        help="若提供则覆盖 --num_episodes / --seed, 直接用这些 episode 序号",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SCRIPT_DIR.parent / "outputs" / "openloop_eval"),
    )

    # ---- VLM 推理选项 (默认与训练时 extraction 一致) ----
    parser.add_argument("--prompt_template", type=str, default="simple")
    parser.add_argument("--content_order", type=str, default="images_first")
    parser.add_argument("--no_flip_images", action="store_true", default=True)
    parser.add_argument(
        "--lowercase_instruction", action="store_true", default=True
    )
    parser.add_argument(
        "--add_generation_prompt", action="store_true", default=True
    )
    parser.add_argument("--layers", type=int, nargs="+", default=[14])

    # ---- proprio / action 归一化开关 ----
    parser.add_argument(
        "--use_state_normalize",
        action="store_true",
        default=True,
        help="使用 dataset196 stats.json 把 state 归一化为 [-1,1]",
    )
    parser.add_argument(
        "--no_state_normalize",
        dest="use_state_normalize",
        action="store_false",
    )
    parser.add_argument(
        "--use_action_denormalize",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_action_denormalize",
        dest="use_action_denormalize",
        action="store_false",
    )

    # ---- gripper 二值化 ----
    parser.add_argument(
        "--gripper_col",
        type=int,
        default=6,
        help="action 中 gripper 的列, -1 表示不做二值化",
    )
    parser.add_argument(
        "--gripper_threshold",
        type=float,
        default=0.5,
        help="反归一化后阈值, gripper>thr 置 1",
    )

    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint)
    ckpt_path = ckpt_dir / "action_head.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"action_head.pt not found in {ckpt_dir}")
    if not (ckpt_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in {ckpt_dir}")

    # 在 import VLM/OFT 之前先设置 constants.py 期望的 env vars
    _setup_constants_env(ckpt_dir)

    # 延迟 import: 让 constants.py 拿到正确的 ACTION_DIM 等
    # VLM backbone 仅在线模式才需要, 默认 (use_pre_extracted_vlm=True) 跳过.
    global create_vlm_backbone, create_vlm2oft_pipeline
    create_vlm_backbone = None
    if not args.use_pre_extracted_vlm:
        from VLMs.S0_1.backbone import create_vlm_backbone as _create_vlm_backbone
        create_vlm_backbone = _create_vlm_backbone
    from Action_Heads.OFT1_0.vlm2oft_pipeline import (
        create_vlm2oft_pipeline as _create_vlm2oft_pipeline,
    )
    create_vlm2oft_pipeline = _create_vlm2oft_pipeline

    dataset_path = Path(args.dataset_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析 vlm_hidden_states_dir
    if args.use_pre_extracted_vlm:
        vlm_dir = (
            Path(args.vlm_hidden_states_dir)
            if args.vlm_hidden_states_dir
            else dataset_path / "vlm_hidden_states"
        )
        if not vlm_dir.exists():
            raise FileNotFoundError(
                f"vlm_hidden_states_dir 不存在: {vlm_dir}\n"
                f"  请先用 utils/extract_vlm_hidden_state/.../run_qwen_*_extraction.sh "
                f"把 hidden states 抽出来, 或者加 --no_use_pre_extracted_vlm 走在线模式."
            )
        print(f"[VLM] pre-extracted dir = {vlm_dir}")
    else:
        vlm_dir = None
        print("[VLM] online mode (will load Qwen3-VL backbone)")

    info = load_info(dataset_path)
    stats = load_stats(dataset_path)
    tasks = load_tasks(dataset_path)

    # episode 选择
    all_eps = list_episode_indices(dataset_path)
    print(f"[Dataset] {dataset_path}\n          episodes={len(all_eps)} fps={info.get('fps')}")
    if args.episodes:
        chosen = list(args.episodes)
        unknown = [e for e in chosen if e not in set(all_eps)]
        if unknown:
            raise ValueError(f"未知 episode_index: {unknown}, 数据集只有 {all_eps[:5]}...")
    else:
        rng = random.Random(args.seed)
        chosen = sorted(rng.sample(all_eps, k=min(args.num_episodes, len(all_eps))))
    print(f"[Episodes] chosen = {chosen}")

    # normalizer
    state_normalizer = build_normalizer(stats, "observation.state") if args.use_state_normalize else None
    action_normalizer = build_normalizer(stats, "action") if args.use_action_denormalize else None
    print(
        f"[Norm] state_normalize={state_normalizer is not None}, "
        f"action_denormalize={action_normalizer is not None}, "
        f"gripper_col={args.gripper_col}, gripper_threshold={args.gripper_threshold}"
    )

    # predictor
    predictor = OpenLoopPredictor(
        ckpt_path=ckpt_path,
        device=args.device,
        use_pre_extracted_vlm=args.use_pre_extracted_vlm,
        vlm_model_path=args.vlm_model_path,
        layers=tuple(args.layers),
        prompt_template=args.prompt_template,
        content_order=args.content_order,
        flip_images=False,
        lowercase_instruction=bool(args.lowercase_instruction),
        add_generation_prompt=bool(args.add_generation_prompt),
    )

    # 跑每个 episode
    all_metrics = []
    for ep_idx in chosen:
        ep = load_episode(
            dataset_path,
            ep_idx,
            info,
            tasks,
            load_full_video=not args.use_pre_extracted_vlm,
        )
        m = run_open_loop(
            predictor=predictor,
            ep=ep,
            state_normalizer=state_normalizer,
            action_normalizer=action_normalizer,
            binarize_gripper_col=args.gripper_col if args.gripper_col >= 0 else None,
            gripper_threshold=args.gripper_threshold,
            out_dir=out_dir,
            vlm_hidden_states_dir=vlm_dir,
        )
        all_metrics.append(m)

    # 汇总
    summary = {
        "checkpoint": str(ckpt_dir),
        "dataset_path": str(dataset_path),
        "chunk_size": predictor.chunk_size,
        "action_dim": predictor.action_dim,
        "proprio_dim": predictor.proprio_dim,
        "use_pre_extracted_vlm": bool(args.use_pre_extracted_vlm),
        "vlm_hidden_states_dir": str(vlm_dir) if vlm_dir is not None else None,
        "vlm_model_path": args.vlm_model_path,
        "vlm_layers": list(args.layers),
        "use_state_normalize": state_normalizer is not None,
        "use_action_denormalize": action_normalizer is not None,
        "gripper_col": args.gripper_col,
        "gripper_threshold": args.gripper_threshold,
        "metrics": all_metrics,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Summary] -> {out_dir / 'summary.json'}")
    if all_metrics:
        avg_mse = np.mean([m["mse"] for m in all_metrics])
        avg_mae = np.mean([m["mae"] for m in all_metrics])
        print(f"  avg MSE = {avg_mse:.5f}, avg MAE = {avg_mae:.5f} over {len(all_metrics)} episodes")


if __name__ == "__main__":
    main()
