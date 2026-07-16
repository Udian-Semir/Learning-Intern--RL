#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_lerobot_datasets.py
把多个 LeRobot v2.0 数据集合并成一个 LeRobot 数据集（方案 A：物理合并）。

目标使用场景
------------
OFT action head 的 pretrain。 Action_Heads/OFT1_0/train_multigpu.py 只支持单个
--data_path。为了同时用多个已经提好 VLM hidden states 的 LeRobot 数据集做 pretrain，
先把它们全部合并到一个 LeRobot 目录下。

做的事情
--------
1. 全局重编号 episode_index / index / vlm_hidden_state_index
2. 合并 meta/tasks.jsonl（按 task 文本去重，建立 src_task_idx -> new_task_idx 映射）
3. 合并 meta/episodes.jsonl（重编号 episode_index + 重映射 tasks[] 里的 task_index）
4. 复制并改写 data/chunk-XXX/episode_XXXXXX.parquet 中：
   - episode_index
   - index
   - vlm_hidden_state_index
   - task_index
   - annotation.human.*.task_description（如果有）
5. 按新 episode 号重新打包 vlm_hidden_states/chunk-XXX.npz
   （流式写入，内存占用只等于单个 episode 的 hidden state）
6. 重建 meta/info.json, meta/stats.json, meta/modality.json
7. 可选复制 videos/（默认跳过，因为训练用 SKIP_IMAGES=true 不需要）

使用方式
--------
    python -m utils.mix_dataset.merge_lerobot_datasets \
        --datasets_from_file utils/mix_dataset/datasets_pretrain10.txt \
        --output /data_disk1/hwl/pretrain10_merged \
        --overwrite

之后把 train_qwen_datasets_pretrain10_22.sh 中的 DATA_PATH 改为 /data_disk1/hwl/pretrain10_merged 即可。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib import format as npy_fmt
from tqdm import tqdm


# =====================================================================
# 元信息读取
# =====================================================================
def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_tasks(meta_dir: Path) -> Dict[int, str]:
    """meta/tasks.jsonl -> {task_index: task_text}"""
    tasks: Dict[int, str] = {}
    for entry in load_jsonl(meta_dir / "tasks.jsonl"):
        tasks[int(entry["task_index"])] = entry["task"]
    return tasks


# =====================================================================
# VLM 流式 npz 写入（单 episode 内存占用）
# =====================================================================
class StreamNpzWriter:
    """流式写 .npz。np.savez 内部用 zipfile ZIP_STORED，这里手写等价逻辑，
    避免一次性把一个 chunk（最多 chunks_size 个 episode）全部驻留内存。"""

    def __init__(self, path: Path):
        self.path = path
        # allowZip64 保证单文件 >4GB 也能写
        self._zf = zipfile.ZipFile(
            path, mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        )
        self._count = 0

    def write(self, key: str, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr)
        # numpy.savez 约定的命名：key + ".npy"
        with self._zf.open(f"{key}.npy", mode="w", force_zip64=True) as f:
            npy_fmt.write_array(f, arr, allow_pickle=False)
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None

    def __enter__(self) -> "StreamNpzWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# =====================================================================
# 在线统计（min/max/mean/std）
# =====================================================================
class OnlineStats:
    """沿样本维 (N) 聚合 min/max/mean/std。用在 observation.state / action 上。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)
        self.count = 0

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x64 = x.astype(np.float64, copy=False)
        self.min = np.minimum(self.min, x64.min(axis=0))
        self.max = np.maximum(self.max, x64.max(axis=0))
        self.sum += x64.sum(axis=0)
        self.sumsq += (x64 ** 2).sum(axis=0)
        self.count += x.shape[0]

    def finalize(self) -> dict:
        n = max(self.count, 1)
        mean = self.sum / n
        var = np.maximum(self.sumsq / n - mean ** 2, 0.0)
        std = np.sqrt(var)
        return {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "count": int(self.count),
        }


# =====================================================================
# 源数据集描述
# =====================================================================
@dataclass
class SourceDataset:
    path: Path
    info: dict
    tasks: Dict[int, str]
    episodes: List[dict]
    ep_offset: int = 0      # 新 episode_index 起点
    idx_offset: int = 0     # 新 frame-level index 起点
    vlm_offset: int = 0     # 新 vlm_hidden_state_index 起点
    task_remap: Dict[int, int] = field(default_factory=dict)

    @property
    def total_episodes(self) -> int:
        return int(self.info["total_episodes"])

    @property
    def total_frames(self) -> int:
        return int(self.info["total_frames"])

    @property
    def chunks_size(self) -> int:
        return int(self.info["chunks_size"])

    @classmethod
    def load(cls, path: Path) -> "SourceDataset":
        meta = path / "meta"
        info = load_json(meta / "info.json")
        tasks = load_tasks(meta)
        episodes = load_jsonl(meta / "episodes.jsonl")
        return cls(path=path, info=info, tasks=tasks, episodes=episodes)


# =====================================================================
# 主合并流程
# =====================================================================
def collect_datasets(args) -> List[Path]:
    paths: List[Path] = []
    if args.datasets:
        paths.extend(Path(p).expanduser().resolve() for p in args.datasets)
    if args.datasets_from_file:
        with open(args.datasets_from_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(Path(line).expanduser().resolve())
    if not paths:
        raise SystemExit("❌ 必须通过 --datasets 或 --datasets_from_file 至少指定一个源数据集")
    seen = set()
    unique = []
    for p in paths:
        key = str(p)
        if key in seen:
            print(f"⚠️  重复路径，已去重: {p}")
            continue
        seen.add(key)
        unique.append(p)
    return unique


def check_compatibility(sources: List[SourceDataset]) -> Tuple[int, int, int]:
    """检查所有数据集的 action/state 维度与 fps，返回 (state_dim, action_dim, fps)."""
    info0 = sources[0].info
    s_dim = int(info0["features"]["observation.state"]["shape"][0])
    a_dim = int(info0["features"]["action"]["shape"][0])
    fps0 = int(info0.get("fps", 0))

    for src in sources[1:]:
        sd = int(src.info["features"]["observation.state"]["shape"][0])
        ad = int(src.info["features"]["action"]["shape"][0])
        if sd != s_dim:
            raise SystemExit(
                f"❌ observation.state 维度不一致: {sources[0].path.name}={s_dim}, "
                f"{src.path.name}={sd}\n"
                f"   若确认要混合，请手动对齐维度（例如补零或截断）。"
            )
        if ad != a_dim:
            raise SystemExit(
                f"❌ action 维度不一致: {sources[0].path.name}={a_dim}, "
                f"{src.path.name}={ad}"
            )
        fpsi = int(src.info.get("fps", 0))
        if fps0 and fpsi and fpsi != fps0:
            print(f"⚠️  fps 不一致: {sources[0].path.name}={fps0}, "
                  f"{src.path.name}={fpsi} (合并后 info.json 取第一个)")
    return s_dim, a_dim, fps0


def compute_global_offsets(sources: List[SourceDataset]) -> Tuple[int, int]:
    """为每个源数据集设置 ep/idx/vlm offset，返回 (total_episodes, total_frames)."""
    ep_acc = 0
    idx_acc = 0
    for src in sources:
        src.ep_offset = ep_acc
        src.idx_offset = idx_acc
        # 当前所有源的 vlm_hidden_state_index 都等同于 frame 的全局 index，
        # 所以 vlm_offset == idx_offset
        src.vlm_offset = idx_acc
        ep_acc += src.total_episodes
        idx_acc += src.total_frames
    return ep_acc, idx_acc


def build_task_table(sources: List[SourceDataset]) -> List[str]:
    """按 task 文本去重合并 tasks，返回新 task 表 (list of task_text)。"""
    task_to_new_idx: "OrderedDict[str, int]" = OrderedDict()
    for src in sources:
        local_remap: Dict[int, int] = {}
        for src_idx in sorted(src.tasks.keys()):
            text = src.tasks[src_idx]
            if text not in task_to_new_idx:
                task_to_new_idx[text] = len(task_to_new_idx)
            local_remap[src_idx] = task_to_new_idx[text]
        src.task_remap = local_remap
    return list(task_to_new_idx.keys())


# ---------------------------------------------------------------------
# parquet 重写
# ---------------------------------------------------------------------
_ANNOTATION_TASK_DESC_COLS = (
    "annotation.human.action.task_description",
    "annotation.human.validity",
)


def rewrite_parquets(
    sources: List[SourceDataset],
    output: Path,
    out_chunks_size: int,
    state_stats: OnlineStats,
    action_stats: OnlineStats,
    merged_episodes: List[dict],
    dry_run: bool,
) -> None:
    print("\n🔄 [1/2] 重写 parquet + 聚合 episodes.jsonl + 统计 state/action ...")
    for src in sources:
        base_ep = src.ep_offset
        base_idx = src.idx_offset
        base_vlm = src.vlm_offset
        src_chunks_size = src.chunks_size
        src_ep_by_idx = {int(e["episode_index"]): e for e in src.episodes}

        iterator = range(src.total_episodes)
        pbar = tqdm(iterator, desc=f"parquet:{src.path.name}", unit="ep")
        for src_ep in pbar:
            src_chunk = src_ep // src_chunks_size
            src_parquet = (
                src.path / "data" / f"chunk-{src_chunk:03d}" / f"episode_{src_ep:06d}.parquet"
            )
            if not src_parquet.exists():
                raise FileNotFoundError(f"缺少源 parquet: {src_parquet}")

            df = pd.read_parquet(src_parquet)

            new_ep = base_ep + src_ep
            df["episode_index"] = np.int64(new_ep)
            if "index" in df.columns:
                df["index"] = df["index"].astype(np.int64) + np.int64(base_idx)
            if "vlm_hidden_state_index" in df.columns:
                df["vlm_hidden_state_index"] = (
                    df["vlm_hidden_state_index"].astype(np.int64) + np.int64(base_vlm)
                )

            # task_index / annotation.human.*.task_description 做重映射
            remap = src.task_remap
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].astype(np.int64).map(remap).astype(np.int64)
            for col in _ANNOTATION_TASK_DESC_COLS:
                if col in df.columns and df[col].dtype.kind in ("i", "u"):
                    df[col] = df[col].astype(np.int64).map(remap).astype(np.int64)

            # 聚合 state / action 统计
            if "observation.state" in df.columns:
                state_arr = np.stack(df["observation.state"].to_numpy())
                state_stats.update(state_arr)
            if "action" in df.columns:
                action_arr = np.stack(df["action"].to_numpy())
                action_stats.update(action_arr)

            # episodes.jsonl 条目：复制原条目并重编号，remap tasks[]
            src_entry = src_ep_by_idx.get(src_ep)
            if src_entry is None:
                src_entry = {
                    "episode_index": src_ep,
                    "tasks": [],
                    "length": len(df),
                }
            new_entry = dict(src_entry)
            new_entry["episode_index"] = new_ep
            if isinstance(new_entry.get("tasks"), list):
                new_entry["tasks"] = [
                    int(remap.get(int(t), int(t))) for t in new_entry["tasks"]
                ]
            merged_episodes.append(new_entry)

            # 写到新路径
            new_chunk = new_ep // out_chunks_size
            out_dir = output / "data" / f"chunk-{new_chunk:03d}"
            out_path = out_dir / f"episode_{new_ep:06d}.parquet"
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_parquet(out_path, index=False)


# ---------------------------------------------------------------------
# VLM chunk-npz 流式合并
# ---------------------------------------------------------------------
def merge_vlm_hidden_states(
    sources: List[SourceDataset],
    output: Path,
    out_chunks_size: int,
    dry_run: bool,
) -> None:
    print("\n🧠 [2/2] 合并 VLM hidden states (chunk-npz) ...")

    current_new_chunk: Optional[int] = None
    current_writer: Optional[StreamNpzWriter] = None
    vlm_dir_out = output / "vlm_hidden_states"
    if not dry_run:
        vlm_dir_out.mkdir(parents=True, exist_ok=True)

    try:
        for src in sources:
            vlm_dir = src.path / "vlm_hidden_states"
            src_chunks = sorted(vlm_dir.glob("chunk-*.npz"))
            if not src_chunks:
                raise FileNotFoundError(
                    f"❌ {src.path.name} 未找到 vlm_hidden_states/chunk-*.npz，"
                    f"请先跑 run_qwen_multi_dataset_multi_gpu_save_per_episode.sh"
                )

            pbar = tqdm(total=src.total_episodes, desc=f"vlm:{src.path.name}", unit="ep")
            seen_src_eps: set = set()
            for chunk_path in src_chunks:
                with np.load(chunk_path, allow_pickle=False) as data:
                    keys = sorted(data.files, key=lambda k: int(k.split("_")[-1]))
                    for k in keys:
                        src_ep = int(k.split("_")[-1])
                        if src_ep >= src.total_episodes:
                            # 有些 npz 可能包含越界 key，跳过
                            continue
                        new_ep = src.ep_offset + src_ep
                        new_chunk = new_ep // out_chunks_size

                        if new_chunk != current_new_chunk:
                            if current_writer is not None:
                                current_writer.close()
                                current_writer = None
                            current_new_chunk = new_chunk
                            if not dry_run:
                                out_npz = vlm_dir_out / f"chunk-{new_chunk:03d}.npz"
                                current_writer = StreamNpzWriter(out_npz)

                        arr = np.asarray(data[k])
                        if current_writer is not None:
                            current_writer.write(f"episode_{new_ep:06d}", arr)
                        seen_src_eps.add(src_ep)
                        pbar.update(1)
                        del arr
            pbar.close()

            missing = set(range(src.total_episodes)) - seen_src_eps
            if missing:
                miss_list = sorted(missing)[:10]
                raise RuntimeError(
                    f"❌ {src.path.name}: 有 {len(missing)} 个 episode 没有在 chunk-npz 中找到 "
                    f"(前 10 个缺失: {miss_list})。请确认 VLM 抽取已完成。"
                )
    finally:
        if current_writer is not None:
            current_writer.close()


# ---------------------------------------------------------------------
# 写 meta: info.json / stats.json / tasks.jsonl / episodes.jsonl / modality.json
# ---------------------------------------------------------------------
def write_meta(
    sources: List[SourceDataset],
    output: Path,
    out_chunks_size: int,
    total_episodes: int,
    total_frames: int,
    state_stats: OnlineStats,
    action_stats: OnlineStats,
    merged_episodes: List[dict],
    new_tasks: List[str],
    s_dim: int,
    a_dim: int,
    fps: int,
    dry_run: bool,
) -> dict:
    meta_out = output / "meta"
    if not dry_run:
        meta_out.mkdir(parents=True, exist_ok=True)

    # tasks.jsonl
    tasks_lines = [
        json.dumps({"task_index": i, "task": t}, ensure_ascii=False)
        for i, t in enumerate(new_tasks)
    ]

    # episodes.jsonl（按 episode_index 排序）
    merged_episodes_sorted = sorted(merged_episodes, key=lambda e: int(e["episode_index"]))
    episodes_lines = [json.dumps(e, ensure_ascii=False) for e in merged_episodes_sorted]

    # info.json
    info0 = sources[0].info
    features = dict(info0.get("features", {}))
    if "observation.state" in features:
        features["observation.state"] = dict(features["observation.state"])
        features["observation.state"]["shape"] = [s_dim]
    if "action" in features:
        features["action"] = dict(features["action"])
        features["action"]["shape"] = [a_dim]

    total_chunks = (total_episodes + out_chunks_size - 1) // out_chunks_size
    info_out = {
        "codebase_version": info0.get("codebase_version", "v2.0"),
        "robot_type": "mixed",
        "source_datasets": [src.path.name for src in sources],
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(new_tasks),
        "total_videos": 0,
        "total_chunks": total_chunks,
        "chunks_size": out_chunks_size,
        "fps": fps or info0.get("fps", 10),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": info0.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        ),
        "vlm_hidden_state_path": "vlm_hidden_states/chunk-{episode_chunk:03d}.npz",
        "features": features,
    }

    # stats.json
    stats_out = {
        "observation.state": state_stats.finalize(),
        "action": action_stats.finalize(),
    }

    if dry_run:
        print("\n[dry-run] 不写 meta/ 文件，这里是预览：")
        print(f"  tasks.jsonl        : {len(tasks_lines)} 行")
        print(f"  episodes.jsonl     : {len(episodes_lines)} 行")
        print(f"  info.total_episodes: {total_episodes}")
        print(f"  info.total_frames  : {total_frames}")
        print(f"  info.chunks_size   : {out_chunks_size}")
        print(f"  info.total_chunks  : {total_chunks}")
        return info_out

    (meta_out / "tasks.jsonl").write_text("\n".join(tasks_lines) + "\n", encoding="utf-8")
    (meta_out / "episodes.jsonl").write_text("\n".join(episodes_lines) + "\n", encoding="utf-8")
    with open(meta_out / "info.json", "w", encoding="utf-8") as f:
        json.dump(info_out, f, indent=2, ensure_ascii=False)
    with open(meta_out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)

    # modality.json: 优先继承第一个源
    src_modality = sources[0].path / "meta" / "modality.json"
    if src_modality.exists():
        shutil.copy2(src_modality, meta_out / "modality.json")

    return info_out


# ---------------------------------------------------------------------
# 可选：拷贝/软链 videos
# ---------------------------------------------------------------------
def link_videos(
    sources: List[SourceDataset],
    output: Path,
    out_chunks_size: int,
    dry_run: bool,
    use_hardlink: bool = True,
) -> None:
    print("\n🎬 合并 videos/ (symlink/hardlink, 不复制字节) ...")
    videos_out = output / "videos"
    if not dry_run:
        videos_out.mkdir(parents=True, exist_ok=True)

    for src in sources:
        src_videos = src.path / "videos"
        if not src_videos.exists():
            print(f"  跳过 {src.path.name}: 没有 videos/ 目录")
            continue
        src_chunks_size = src.chunks_size

        for src_ep in tqdm(range(src.total_episodes), desc=f"videos:{src.path.name}"):
            src_chunk = src_ep // src_chunks_size
            chunk_dir = src_videos / f"chunk-{src_chunk:03d}"
            if not chunk_dir.exists():
                continue
            for video_key_dir in chunk_dir.iterdir():
                if not video_key_dir.is_dir():
                    continue
                src_mp4 = video_key_dir / f"episode_{src_ep:06d}.mp4"
                if not src_mp4.exists():
                    continue
                new_ep = src.ep_offset + src_ep
                new_chunk = new_ep // out_chunks_size
                dst_dir = videos_out / f"chunk-{new_chunk:03d}" / video_key_dir.name
                dst_mp4 = dst_dir / f"episode_{new_ep:06d}.mp4"
                if dry_run:
                    continue
                dst_dir.mkdir(parents=True, exist_ok=True)
                if dst_mp4.exists():
                    dst_mp4.unlink()
                try:
                    if use_hardlink:
                        os.link(src_mp4, dst_mp4)
                    else:
                        dst_mp4.symlink_to(src_mp4.resolve())
                except OSError:
                    # 跨设备/权限问题时回退成 symlink
                    dst_mp4.symlink_to(src_mp4.resolve())


# =====================================================================
# main
# =====================================================================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--datasets", nargs="+", default=None,
        help="源数据集路径列表（空格分隔）",
    )
    ap.add_argument(
        "--datasets_from_file", default=None,
        help="文本文件，一行一个源数据集路径，# 开头视为注释",
    )
    ap.add_argument(
        "--output", required=True,
        help="输出合并数据集目录",
    )
    ap.add_argument(
        "--chunks_size", type=int, default=1000,
        help="合并后每个 data-chunk / vlm-chunk 容纳多少个 episode（默认 1000）",
    )
    ap.add_argument(
        "--include_videos", action="store_true",
        help="同时合并 videos/（默认跳过，训练用 SKIP_IMAGES=true 时不需要）",
    )
    ap.add_argument(
        "--link_mode", choices=["hardlink", "symlink"], default="hardlink",
        help="合并 videos 时用的方式（--include_videos 才生效）",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="输出目录已存在时先清空",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="只打印计划，不写任何文件",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    ds_paths = collect_datasets(args)

    print("📂 源数据集列表:")
    for i, p in enumerate(ds_paths):
        print(f"  [{i:2d}] {p}")
    print()

    # 加载每个源数据集的 meta
    sources: List[SourceDataset] = []
    for p in ds_paths:
        assert (p / "meta" / "info.json").exists(), f"不是有效的 LeRobot 数据集: {p}"
        sources.append(SourceDataset.load(p))

    s_dim, a_dim, fps = check_compatibility(sources)
    total_episodes, total_frames = compute_global_offsets(sources)
    new_tasks = build_task_table(sources)

    print("📊 合并统计:")
    print(f"  state_dim       = {s_dim}")
    print(f"  action_dim      = {a_dim}")
    print(f"  fps (第一个源)  = {fps}")
    print(f"  total_episodes  = {total_episodes}")
    print(f"  total_frames    = {total_frames}")
    print(f"  unique tasks    = {len(new_tasks)}")
    print(f"  chunks_size     = {args.chunks_size}")
    print(f"  total_chunks    = {(total_episodes + args.chunks_size - 1) // args.chunks_size}")
    print()
    for src in sources:
        print(f"  · {src.path.name}: "
              f"ep [{src.ep_offset}, {src.ep_offset + src.total_episodes}), "
              f"idx [{src.idx_offset}, {src.idx_offset + src.total_frames}), "
              f"tasks_remap={dict(list(src.task_remap.items())[:3])}...")
    print()

    # 输出目录
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(
                f"❌ 输出目录已存在: {output}\n   加上 --overwrite 强制清空后重建"
            )
        if not args.dry_run:
            print(f"🧹 清空已存在的输出目录: {output}")
            shutil.rmtree(output)

    if not args.dry_run:
        (output / "data").mkdir(parents=True, exist_ok=True)
        (output / "meta").mkdir(parents=True, exist_ok=True)
        (output / "vlm_hidden_states").mkdir(parents=True, exist_ok=True)

    # === 步骤 1: 重写 parquet + 聚合 episodes + 统计量 ===
    state_stats = OnlineStats(s_dim)
    action_stats = OnlineStats(a_dim)
    merged_episodes: List[dict] = []
    rewrite_parquets(
        sources=sources,
        output=output,
        out_chunks_size=args.chunks_size,
        state_stats=state_stats,
        action_stats=action_stats,
        merged_episodes=merged_episodes,
        dry_run=args.dry_run,
    )

    # === 步骤 2: 合并 VLM chunk-npz ===
    merge_vlm_hidden_states(
        sources=sources,
        output=output,
        out_chunks_size=args.chunks_size,
        dry_run=args.dry_run,
    )

    # === 步骤 3: 写 meta ===
    info_out = write_meta(
        sources=sources,
        output=output,
        out_chunks_size=args.chunks_size,
        total_episodes=total_episodes,
        total_frames=total_frames,
        state_stats=state_stats,
        action_stats=action_stats,
        merged_episodes=merged_episodes,
        new_tasks=new_tasks,
        s_dim=s_dim,
        a_dim=a_dim,
        fps=fps,
        dry_run=args.dry_run,
    )

    # === 可选: 合并 videos ===
    if args.include_videos:
        link_videos(
            sources=sources,
            output=output,
            out_chunks_size=args.chunks_size,
            dry_run=args.dry_run,
            use_hardlink=(args.link_mode == "hardlink"),
        )

    # === 写合并清单，便于复现 ===
    if not args.dry_run:
        manifest = {
            "output": str(output),
            "chunks_size": args.chunks_size,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(new_tasks),
            "datasets": [
                {
                    "name": src.path.name,
                    "path": str(src.path),
                    "ep_offset": src.ep_offset,
                    "idx_offset": src.idx_offset,
                    "vlm_offset": src.vlm_offset,
                    "total_episodes": src.total_episodes,
                    "total_frames": src.total_frames,
                    "task_remap": {int(k): int(v) for k, v in src.task_remap.items()},
                }
                for src in sources
            ],
        }
        with open(output / "merge_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n✅ 合并完成")
    print(f"   输出路径: {output}")
    print(f"   total_episodes = {total_episodes}")
    print(f"   total_frames   = {total_frames}")
    print(f"   total_tasks    = {len(new_tasks)}")
    print()
    print("👉 训练时:")
    print(f"   把 train_qwen_datasets_pretrain10_22.sh 里的 DATA_PATH 改成:")
    print(f"     DATA_PATH=\"{output}\"")
    print(f"   并保持 SKIP_IMAGES=\"true\"（因为只有 VLM hidden states，没复制视频）")


if __name__ == "__main__":
    main()
