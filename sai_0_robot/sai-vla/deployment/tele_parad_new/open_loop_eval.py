"""
open_loop_eval.py -- LeRobot open-loop evaluation with OFT server

Load an episode from a LeRobot dataset (parquet + video), predict with the
OFT server at every execute_horizon steps using GT image + GT state,
concatenate the first execute_horizon actions from each chunk, then compare
with the ground-truth actions.

When execute_horizon == chunk_size (default) it is pure open-loop evaluation.
When execute_horizon < chunk_size it is receding-horizon (MPC-style).

Usage:
    # 1) start OFT server
    python server.py --config config.yaml --offline

    # 2) pure open-loop
    python open_loop_eval.py \
        --dataset_path /path/to/lerobot_dataset \
        --config config.yaml \
        --server_url http://localhost:8000

    # 3) receding-horizon
    python open_loop_eval.py \
        --dataset_path /path/to/lerobot_dataset \
        --config config.yaml \
        --execute_horizon 10

    # 4) specific episode
    python open_loop_eval.py \
        --dataset_path /path/to/lerobot_dataset \
        --config config.yaml \
        --episode_index 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml

try:
    import av
except ImportError:
    print("ERROR: PyAV is required.  pip install av")
    sys.exit(1)


# ==================== LeRobot data helpers ====================

def load_metadata(dataset_path):
    with open(Path(dataset_path) / "meta" / "info.json", "r") as f:
        return json.load(f)


def load_tasks(dataset_path):
    p = Path(dataset_path) / "meta" / "tasks.jsonl"
    if not p.exists():
        return {}
    out = {}
    with open(p, "r") as f:
        for line in f:
            t = json.loads(line.strip())
            out[t["task_index"]] = t["task"]
    return out


def load_episode(dataset_path, episode_index, info):
    dataset_path = Path(dataset_path)
    chunks_size = info["chunks_size"]
    chunk_idx = episode_index // chunks_size
    pq = dataset_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"Parquet not found: {pq}")

    df = pd.read_parquet(pq)
    n = len(df)

    states = np.array(df["observation.state"].tolist(), dtype=np.float32)
    actions = np.array(df["action"].tolist(), dtype=np.float32)

    image_keys = sorted([k for k in info["features"] if k.startswith("observation.images.")])
    if not image_keys:
        raise ValueError("No image keys in dataset")

    all_image_streams: Dict[str, List] = {}
    for key in image_keys:
        vid = dataset_path / "videos" / f"chunk-{chunk_idx:03d}" / key / f"episode_{episode_index:06d}.mp4"
        if vid.exists():
            all_image_streams[key] = _read_all_frames(vid, n)
        else:
            print(f"  [WARN] Video not found for key '{key}': {vid}")

    if not all_image_streams:
        raise FileNotFoundError("No video files found for any image key")

    main_key = image_keys[0]
    images = all_image_streams[main_key]

    task_map = load_tasks(dataset_path)
    instruction = ""
    if "task_index" in df.columns:
        instruction = task_map.get(int(df.iloc[0]["task_index"]), "")
    elif "annotation.human.action.task_description" in df.columns:
        instruction = task_map.get(int(df.iloc[0]["annotation.human.action.task_description"]), "")

    print(f"\n=== Episode {episode_index} ===")
    print(f"  Steps       : {n}")
    print(f"  Instruction : {instruction}")
    print(f"  State shape : {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Image keys  : {list(all_image_streams.keys())}")
    for key, frames in all_image_streams.items():
        print(f"    {key}: {len(frames)} frames, shape={frames[0].shape}")

    return states, actions, all_image_streams, instruction, image_keys


def _read_all_frames(video_path, n):
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= n:
            break
    container.close()
    if len(frames) < n:
        print(f"  [WARN] video has {len(frames)} frames, expected {n}")
    return frames


# ==================== Action stats / post-processing ====================

def load_action_stats(stats_file, stats_key="action"):
    if not stats_file or not os.path.exists(stats_file):
        return None, None
    with open(stats_file, "r") as f:
        data = json.load(f)
    if stats_key not in data:
        return None, None
    s = data[stats_key]
    return np.array(s["min"], dtype=np.float32), np.array(s["max"], dtype=np.float32)


def denormalize_actions(a, a_min, a_max, skip_cols=None):
    out = a.copy()
    d = min(a.shape[-1], len(a_min))
    cols = [c for c in range(d) if skip_cols is None or c not in skip_cols]
    ca = np.array(cols)
    out[..., ca] = (out[..., ca] + 1.0) / 2.0 * (a_max[ca] - a_min[ca]) + a_min[ca]
    return out


def normalize_actions(a, a_min, a_max):
    """物理值 → [-1, 1]，与训练时 MinMaxNormalizer.normalize 一致"""
    out = a.copy()
    d = min(a.shape[-1], len(a_min))
    ca = np.arange(d)
    rng = a_max[ca] - a_min[ca]
    mask = rng != 0
    out[..., ca[mask]] = 2.0 * (out[..., ca[mask]] - a_min[ca[mask]]) / rng[mask] - 1.0
    return out


def binarize_gripper(a, columns, threshold, bce_mode):
    out = a.copy()
    for c in columns:
        if c < out.shape[-1]:
            if bce_mode:
                prob = 1.0 / (1.0 + np.exp(-out[..., c]))
                out[..., c] = np.where(prob > threshold, 1.0, 0.0)
            else:
                out[..., c] = np.where(out[..., c] > threshold, 1.0, 0.0)
    return out


def strip_padding(actions, prepend, insert_before_last):
    s = actions[:, prepend:]
    if insert_before_last > 0:
        d = s.shape[-1]
        sp = d - insert_before_last - 1
        s = np.concatenate([s[:, :sp], s[:, sp + insert_before_last:]], axis=-1)
    return s


# ==================== Server ====================

def get_server_health(url):
    r = requests.get(f"{url}/health", timeout=10)
    r.raise_for_status()
    cfg = r.json()
    print("\n=== Server Health ===")
    print(json.dumps(cfg, indent=2))
    return cfg


def send_predict(url, image_list, state_list, instruction=""):
    """image_list: list of numpy arrays (one per image stream)"""
    payload = {"image_arrays": [img.tolist() for img in image_list], "state": state_list}
    if instruction:
        payload["instruction"] = instruction
    r = requests.post(f"{url}/predict", json=payload, timeout=120)
    r.raise_for_status()
    res = r.json()
    if "error" in res:
        print(f"  Server error: {res['error']}")
        sys.exit(1)
    return np.array(res["actions"], dtype=np.float32)


def predict_full_episode(url, all_image_streams, states, instruction, chunk_size, exec_h=None):
    """all_image_streams: dict {key: [frame0, frame1, ...]}"""
    stream_keys = sorted(all_image_streams.keys())
    n = len(list(all_image_streams.values())[0])
    step = exec_h or chunk_size
    mode = "receding-horizon" if step < chunk_size else "full-chunk open-loop"
    total_calls = (n + step - 1) // step
    print(f"\n  mode={mode}, chunk={chunk_size}, exec_h={step}, ~{total_calls} calls")
    print(f"  image streams: {stream_keys} ({len(stream_keys)} images per request)")

    parts = []
    start = 0
    idx = 0
    while start < n:
        remaining = n - start
        t0 = time.time()
        frame_list = [all_image_streams[k][start] for k in stream_keys]
        padded = send_predict(url, frame_list, states[start].tolist(), instruction)
        dt = time.time() - t0
        take = min(step, remaining, padded.shape[0])
        parts.append(padded[:take])
        idx += 1
        print(f"  [{idx}/{total_calls}] frame={start}, took {take}/{padded.shape[0]}, {dt:.2f}s")
        start += take

    return np.concatenate(parts, axis=0)


# ==================== Trajectory ====================

def compute_state_trajectory(init, deltas, joint_dims=7):
    n = len(deltas)
    dim = len(init)
    traj = np.zeros((n + 1, dim), dtype=np.float32)
    traj[0] = init
    jd = min(joint_dims, dim, deltas.shape[-1])
    for t in range(n):
        traj[t + 1] = traj[t].copy()
        traj[t + 1, :jd] += deltas[t, :jd]
        if deltas.shape[-1] > jd and dim > jd:
            traj[t + 1, jd] = deltas[t, jd]
    return traj


# ==================== Plotting ====================

def plot_action_comparison(gt, pred, chunk_size, instruction, ep_idx, path,
                           exec_h=None, names=None):
    n = min(len(gt), len(pred))
    dim = gt.shape[-1]
    t = np.arange(n)
    step = exec_h or chunk_size
    if names is None:
        names = [f"Dim {i}" for i in range(dim)]

    nrows = max(1, (dim + 1) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(18, 4 * nrows))
    if nrows == 1:
        axes = np.array(axes).reshape(1, 2)
    axes = axes.flatten()

    tag = f"chunk={chunk_size}, exec_h={step}" if step < chunk_size else f"chunk={chunk_size}"
    fig.suptitle(
        f"Action Comparison - Episode {ep_idx} | {n} steps | {tag}\n\"{instruction}\"",
        fontsize=13, fontweight="bold", y=0.99)

    for i in range(dim):
        ax = axes[i]
        g, p = gt[:n, i], pred[:n, i]
        ax.plot(t, g, "b-", label="GT", lw=1.5, alpha=0.8)
        ax.plot(t, p, "r--", label="Pred", lw=1.5, alpha=0.8)
        for b in range(step, n, step):
            ax.axvline(x=b, color="gray", ls=":", alpha=0.4)
        rmse = np.sqrt(np.mean((g - p) ** 2))
        mae = np.mean(np.abs(g - p))
        nm = names[i] if i < len(names) else f"Dim {i}"
        ax.set_title(f"{nm}  (RMSE={rmse:.4f}, MAE={mae:.4f})", fontsize=11)
        ax.set_xlabel("Step")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(dim, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Action plot -> {path}")


def plot_state_comparison(gt, pred, chunk_size, instruction, ep_idx, path,
                          exec_h=None, names=None):
    n = min(len(gt), len(pred))
    dim = gt.shape[-1]
    t = np.arange(n)
    step = exec_h or chunk_size
    if names is None:
        names = [f"Dim {i}" for i in range(dim)]

    nrows = max(1, (dim + 1) // 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(18, 4 * nrows))
    if nrows == 1:
        axes = np.array(axes).reshape(1, 2)
    axes = axes.flatten()

    fig.suptitle(
        f"State Trajectory - Episode {ep_idx} | {n} steps\n\"{instruction}\"",
        fontsize=13, fontweight="bold", y=0.99)

    for i in range(dim):
        ax = axes[i]
        g, p = gt[:n, i], pred[:n, i]
        ax.plot(t, g, "b-", label="GT", lw=1.5, alpha=0.8)
        ax.plot(t, p, "r--", label="Pred", lw=1.5, alpha=0.8)
        for b in range(step, n, step):
            ax.axvline(x=b, color="gray", ls=":", alpha=0.4)
        rmse = np.sqrt(np.mean((g - p) ** 2))
        nm = names[i] if i < len(names) else f"Dim {i}"
        ax.set_title(f"{nm}  (RMSE={rmse:.4f})", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("deg" if i < dim - 1 else "binary")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(dim, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  State plot -> {path}")


def save_first_frame(img, instruction, path):
    """保存首帧为 PNG。不用 matplotlib 放大小图，避免 imshow+savefig 插值导致的块状/锯齿感。"""
    from PIL import Image

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.size and float(arr.max()) <= 1.0:
            arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        Image.fromarray(arr).save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
    print(f"  First frame -> {path}")
    if instruction:
        print(f"    instruction: \"{instruction}\"")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Open-loop eval on LeRobot dataset with OFT server",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="LeRobot dataset root directory")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Server config.yaml (for post-processing params)")
    parser.add_argument("--server_url", type=str, default="http://localhost:8000")
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--execute_horizon", type=int, default=None,
                        help="Actions to keep per chunk (None=chunk_size)")
    parser.add_argument("--output_dir", type=str,
                        default="/tmp/open_loop_eval_lerobot")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load server config ----
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    action_dim = cfg.get("action_dim", 7)
    chunk_size = cfg.get("chunk_size", 16)

    ap = cfg.get("action_postprocess", {})
    denorm_cfg = ap.get("action_denormalize", {})
    grip_cfg = ap.get("gripper_binarize", {})
    pad_cfg = ap.get("padding_zeros", {})

    pad_enabled = pad_cfg.get("enabled", False)
    pad_prepend = pad_cfg.get("prepend_count", 0) if pad_enabled else 0
    pad_insert = pad_cfg.get("insert_before_last", 0) if pad_enabled else 0

    denorm_enabled = denorm_cfg.get("enabled", False)
    a_min, a_max = None, None
    if denorm_enabled:
        a_min, a_max = load_action_stats(
            denorm_cfg.get("stats_file", ""),
            denorm_cfg.get("stats_key", "action"))
        if a_min is None:
            print("[WARN] action stats load failed, GT not denormalized")
            denorm_enabled = False

    grip_enabled = grip_cfg.get("enabled", False)
    grip_cols = grip_cfg.get("columns", [])
    grip_thresh = grip_cfg.get("threshold", 0.5)
    grip_bce = grip_cfg.get("bce_mode", False)
    bce_skip = set(grip_cols) if grip_bce and grip_enabled else set()

    # ---- server health ----
    srv = get_server_health(args.server_url)
    srv_chunk = srv.get("chunk_size", chunk_size)
    if srv_chunk != chunk_size:
        print(f"[WARN] server chunk={srv_chunk} != config chunk={chunk_size},"
              f" using server value")
        chunk_size = srv_chunk

    # exec_h 优先级: 命令行 > server 返回 > config 文件 > chunk_size
    if args.execute_horizon is not None:
        exec_h = args.execute_horizon
    else:
        srv_exec_h = srv.get("execute_horizon", None)
        cfg_exec_h = cfg.get("execute_horizon", None)
        exec_h = srv_exec_h or cfg_exec_h or chunk_size
        if srv_exec_h is not None:
            print(f"  [INFO] 从 server 获取 execute_horizon={srv_exec_h}")
        elif cfg_exec_h is not None:
            print(f"  [INFO] 从 config 获取 execute_horizon={cfg_exec_h}")

    if exec_h > chunk_size:
        print(f"[WARN] exec_h={exec_h} > chunk={chunk_size}, clamping")
        exec_h = chunk_size

    print("=== Config ===")
    print(f"  action_dim={action_dim}, chunk={chunk_size}, exec_h={exec_h}")
    print(f"  denorm={denorm_enabled}, grip={grip_enabled}"
          f"(cols={grip_cols},bce={grip_bce})")
    print(f"  pad: prepend={pad_prepend}, insert_before_last={pad_insert}")

    # ---- load episode ----
    info = load_metadata(dataset_path)
    all_states, all_actions, all_image_streams, instruction, image_keys = load_episode(
        dataset_path, args.episode_index, info)
    num_steps = len(all_states)

    main_key = image_keys[0]
    save_first_frame(
        all_image_streams[main_key][0], instruction,
        str(output_dir / f"ep{args.episode_index}_first_frame.png"))

    # ---- GT post-processing (align with server output) ----
    # 数据集中的 action 已经是物理值 (非归一化)，不需要 denormalize。
    # Server 输出也是反归一化后的物理值，所以直接比较即可。
    # 只对 gripper 做二值化以匹配 server 的输出。
    gt_actions = all_actions.copy()
    if grip_enabled and grip_cols:
        gt_actions = binarize_gripper(
            gt_actions, grip_cols, grip_thresh, grip_bce)

    # ---- predict ----
    print(f"\nPredicting {num_steps} steps, chunk={chunk_size}, exec_h={exec_h}")
    print(f"  ~{(num_steps + exec_h - 1) // exec_h} server calls")

    padded_pred = predict_full_episode(
        args.server_url, all_image_streams, all_states, instruction,
        chunk_size, exec_h)

    pred_actions = strip_padding(padded_pred, pad_prepend, pad_insert)

    # ---- align dims ----
    mdim = pred_actions.shape[-1]
    gdim = gt_actions.shape[-1]
    cdim = min(mdim, gdim)
    nc = min(len(gt_actions), len(pred_actions))
    gt_cmp = gt_actions[:nc, :cdim]
    pred_cmp = pred_actions[:nc, :cdim]

    # ---- metrics (physical space) ----
    mse = np.mean((gt_cmp - pred_cmp) ** 2)
    mae = np.mean(np.abs(gt_cmp - pred_cmp))
    per_dim = np.sqrt(np.mean((gt_cmp - pred_cmp) ** 2, axis=0))

    joint_dims = [i for i in range(cdim) if i not in set(grip_cols)]
    grip_dims = [i for i in range(cdim) if i in set(grip_cols)]

    print(f"\n=== Metrics - Physical Space (ep {args.episode_index},"
          f" {nc} steps, {cdim} dims) ===")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    if joint_dims:
        j_mse = np.mean((gt_cmp[:, joint_dims] - pred_cmp[:, joint_dims]) ** 2)
        j_mae = np.mean(np.abs(gt_cmp[:, joint_dims] - pred_cmp[:, joint_dims]))
        print(f"  Joints-only MSE: {j_mse:.6f}  MAE: {j_mae:.6f}")
    for i, r in enumerate(per_dim):
        tag = " (gripper)" if i in set(grip_cols) else ""
        print(f"  Dim {i} RMSE: {r:.6f}{tag}")

    if grip_dims:
        grip_err = gt_cmp[:, grip_dims] != pred_cmp[:, grip_dims]
        grip_error_rate = grip_err.mean()
        print(f"  Gripper error rate: {grip_error_rate*100:.1f}%"
              f" ({int(grip_err.sum())}/{nc} steps)")

    # ---- metrics (normalized [-1,1] space, comparable to training loss) ----
    if a_min is not None and a_max is not None:
        gt_norm = normalize_actions(gt_cmp, a_min[:cdim], a_max[:cdim])
        pred_norm = normalize_actions(pred_cmp, a_min[:cdim], a_max[:cdim])
        norm_mae = np.mean(np.abs(gt_norm - pred_norm))
        norm_mse = np.mean((gt_norm - pred_norm) ** 2)
        norm_per_dim = np.sqrt(np.mean((gt_norm - pred_norm) ** 2, axis=0))

        print(f"\n=== Metrics - Normalized [-1,1] Space (compare with training loss) ===")
        print(f"  Normalized L1 (MAE): {norm_mae:.6f}  (training loss ≈ L1)")
        print(f"  Normalized MSE: {norm_mse:.6f}")
        if joint_dims:
            jn_mae = np.mean(np.abs(gt_norm[:, joint_dims] - pred_norm[:, joint_dims]))
            jn_mse = np.mean((gt_norm[:, joint_dims] - pred_norm[:, joint_dims]) ** 2)
            print(f"  Joints-only normalized L1: {jn_mae:.6f}  MSE: {jn_mse:.6f}")
        for i, r in enumerate(norm_per_dim):
            tag = " (gripper)" if i in set(grip_cols) else ""
            print(f"  Dim {i} norm RMSE: {r:.6f}{tag}")

    jnames = ([f"Joint {i+1}" for i in range(cdim - 1)] + ["Gripper"]
              if cdim > 1 else [f"Dim 0"])

    # ---- action comparison plot ----
    plot_action_comparison(
        gt_cmp, pred_cmp, chunk_size, instruction, args.episode_index,
        str(output_dir / f"ep{args.episode_index}_action_comparison.png"),
        exec_h=exec_h, names=jnames)

    # ---- state trajectory comparison ----
    sel = cfg.get("state_select_indices", None)
    gt_sel = all_states[:, sel] if sel is not None else all_states
    sdim = gt_sel.shape[-1]
    jd = min(sdim - 1, cdim - 1) if sdim > 1 else sdim

    pred_traj = compute_state_trajectory(gt_sel[0], pred_cmp, joint_dims=jd)
    pred_traj = pred_traj[:len(gt_sel)]
    sd = min(gt_sel.shape[-1], pred_traj.shape[-1])
    snames = ([f"Joint {i+1}" for i in range(sd - 1)] + ["Gripper"]
              if sd > 1 else [f"Dim 0"])

    plot_state_comparison(
        gt_sel[:, :sd], pred_traj[:, :sd],
        chunk_size, instruction, args.episode_index,
        str(output_dir / f"ep{args.episode_index}_state_comparison.png"),
        exec_h=exec_h, names=snames)

    print(f"\n=== Done ===")
    print(f"  Outputs: {output_dir}")


if __name__ == "__main__":
    main()
