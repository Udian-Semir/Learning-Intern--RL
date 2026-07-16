#!/usr/bin/env python3
"""
将 HuggingFace Datasets 格式的 libero_plus (video+label) 转换为 LeRobot v2 目录格式。

源格式 (/data_disk1/hwl/libero_plus/train):
    - Arrow 文件, 28694 行
    - 每行: video (mp4 bytes) + label (0=front, 1=wrist)
    - 14347 个 episode, 每个有 front + wrist 两个视角

目标格式 (与 austin_buds_dataset_converted_externally_to_rlds 一致):
    dataset_name/
    ├── meta/
    │   ├── info.json
    │   ├── episodes.jsonl
    │   ├── tasks.jsonl
    │   └── stats.json
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    └── videos/
        └── chunk-000/
            ├── observation.images.front/
            │   ├── episode_000000.mp4
            │   └── ...
            └── observation.images.wrist/
                ├── episode_000000.mp4
                └── ...

用法:
    conda run -n qwen_eagle_hwl python utils/convert_libero_plus_to_lerobot_v2.py \
        --src /data_disk1/hwl/libero_plus \
        --dst /data_disk1/hwl/unitree_train_v2_recipe_lerobot/libero_plus
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa


LABEL_TO_CAMERA = {0: "front", 1: "wrist"}
CHUNKS_SIZE = 1000


def get_video_frame_count(video_path: str) -> tuple[int, float]:
    """用 ffprobe 获取视频帧数和 fps。"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    nb_frames = int(stream["nb_frames"])
    r_fps_str = stream["r_frame_rate"]  # e.g. "20/1"
    num, den = map(int, r_fps_str.split("/"))
    fps = num / den
    return nb_frames, fps


def parse_args():
    parser = argparse.ArgumentParser(description="libero_plus -> LeRobot v2")
    parser.add_argument("--src", type=str, default="/data_disk1/hwl/libero_plus",
                        help="HuggingFace Datasets 格式的 libero_plus 路径")
    parser.add_argument("--dst", type=str,
                        default="/data_disk1/hwl/unitree_train_v2_recipe_lerobot/libero_plus",
                        help="输出的 LeRobot v2 目录")
    parser.add_argument("--workers", type=int, default=16,
                        help="写视频文件的并行 worker 数")
    parser.add_argument("--task_description", type=str, default="",
                        help="任务描述文本 (可选)")
    return parser.parse_args()


def main():
    args = parse_args()
    src_path = Path(args.src)
    dst_path = Path(args.dst)

    print(f"源路径: {src_path}")
    print(f"目标路径: {dst_path}")

    # ------------------------------------------------------------------
    # 1. 加载 HuggingFace Dataset (arrow 格式, 不解码视频)
    # ------------------------------------------------------------------
    print("\n[1/6] 加载数据集...")
    from datasets import load_from_disk
    ds = load_from_disk(str(src_path / "train"))
    ds_arrow = ds.with_format("arrow")
    total_rows = len(ds)
    print(f"  总行数: {total_rows}")

    labels_all = ds_arrow["label"].to_pylist()

    # ------------------------------------------------------------------
    # 2. 建立 episode 映射: ep_number -> {front_row_idx, wrist_row_idx}
    # ------------------------------------------------------------------
    print("\n[2/6] 建立 episode 映射...")
    episodes: dict[int, dict] = {}
    for i in range(total_rows):
        row = ds_arrow[i : i + 1]
        vid_meta = row.column("video").chunk(0)[0].as_py()
        lab = labels_all[i]
        path_str: str = vid_meta["path"]  # e.g. "episode_000123.mp4"
        ep_num = int(path_str.replace("episode_", "").replace(".mp4", ""))

        if ep_num not in episodes:
            episodes[ep_num] = {}

        cam = LABEL_TO_CAMERA[lab]
        episodes[ep_num][f"{cam}_row"] = i

        if i % 5000 == 0:
            print(f"  已扫描 {i}/{total_rows}...")

    ep_numbers = sorted(episodes.keys())
    num_episodes = len(ep_numbers)
    ep_num_to_new_idx = {ep_num: idx for idx, ep_num in enumerate(ep_numbers)}
    print(f"  共 {num_episodes} 个 episode (编号 {ep_numbers[0]}~{ep_numbers[-1]})")

    # ------------------------------------------------------------------
    # 3. 提取视频文件到目标目录
    # ------------------------------------------------------------------
    print("\n[3/6] 提取视频文件...")
    for cam in ("front", "wrist"):
        (dst_path / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(
            parents=True, exist_ok=True
        )

    def write_video(ep_num: int, cam: str) -> tuple[int, str, int]:
        """写单个视频并返回 (new_ep_idx, cam, frame_count)."""
        new_idx = ep_num_to_new_idx[ep_num]
        row_idx = episodes[ep_num][f"{cam}_row"]
        chunk_idx = new_idx // CHUNKS_SIZE
        chunk_dir = f"chunk-{chunk_idx:03d}"

        out_dir = dst_path / "videos" / chunk_dir / f"observation.images.{cam}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode_{new_idx:06d}.mp4"

        if out_path.exists():
            nb_frames, _ = get_video_frame_count(str(out_path))
            return new_idx, cam, nb_frames

        row = ds_arrow[row_idx : row_idx + 1]
        video_bytes = row.column("video").chunk(0)[0].as_py()["bytes"]
        out_path.write_bytes(video_bytes)

        nb_frames, _ = get_video_frame_count(str(out_path))
        return new_idx, cam, nb_frames

    ep_frame_counts: dict[int, dict[str, int]] = {}
    tasks_to_submit = []
    for ep_num in ep_numbers:
        for cam in ("front", "wrist"):
            if f"{cam}_row" in episodes[ep_num]:
                tasks_to_submit.append((ep_num, cam))

    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(write_video, ep_num, cam): (ep_num, cam)
            for ep_num, cam in tasks_to_submit
        }
        for fut in as_completed(futures):
            new_idx, cam, nf = fut.result()
            ep_frame_counts.setdefault(new_idx, {})[cam] = nf
            done_count += 1
            if done_count % 2000 == 0:
                print(f"  已写入 {done_count}/{len(tasks_to_submit)} 个视频文件...")

    print(f"  完成, 共写入 {done_count} 个视频文件")

    # ------------------------------------------------------------------
    # 4. 获取 fps 和分辨率 (从第一个视频)
    # ------------------------------------------------------------------
    print("\n[4/6] 读取视频元数据...")
    first_video = dst_path / "videos" / "chunk-000" / "observation.images.front" / "episode_000000.mp4"
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", str(first_video),
    ]
    probe = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    stream = probe["streams"][0]
    fps_num, fps_den = map(int, stream["r_frame_rate"].split("/"))
    fps = fps_num / fps_den
    width = int(stream["width"])
    height = int(stream["height"])
    codec = stream["codec_name"]
    pix_fmt = stream["pix_fmt"]
    print(f"  fps={fps}, size={width}x{height}, codec={codec}, pix_fmt={pix_fmt}")

    # ------------------------------------------------------------------
    # 5. 生成 parquet 数据文件 & episodes.jsonl
    # ------------------------------------------------------------------
    print("\n[5/6] 生成 parquet 数据文件...")
    (dst_path / "data").mkdir(parents=True, exist_ok=True)
    meta_path = dst_path / "meta"
    meta_path.mkdir(parents=True, exist_ok=True)

    episodes_info = []
    global_index = 0

    for new_idx in range(num_episodes):
        chunk_idx = new_idx // CHUNKS_SIZE
        chunk_dir = dst_path / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        fc = ep_frame_counts.get(new_idx, {})
        front_frames = fc.get("front", 0)
        wrist_frames = fc.get("wrist", 0)
        n_frames = max(front_frames, wrist_frames)

        if n_frames == 0:
            print(f"  WARNING: episode {new_idx} 帧数为 0, 跳过")
            continue

        frame_indices = list(range(n_frames))
        timestamps = [i / fps for i in frame_indices]
        indices = list(range(global_index, global_index + n_frames))
        episode_indices = [new_idx] * n_frames
        task_indices = [0] * n_frames
        done_flags = [False] * n_frames
        done_flags[-1] = True

        table = pa.table({
            "timestamp": pa.array(timestamps, type=pa.float64()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(indices, type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
            "next.done": pa.array(done_flags, type=pa.bool_()),
        })

        pq.write_table(table, chunk_dir / f"episode_{new_idx:06d}.parquet")

        episodes_info.append({
            "episode_index": new_idx,
            "tasks": [0],
            "length": n_frames,
        })

        global_index += n_frames

        if (new_idx + 1) % 2000 == 0:
            print(f"  已生成 {new_idx + 1}/{num_episodes} 个 parquet...")

    total_frames = global_index
    num_chunks = (num_episodes - 1) // CHUNKS_SIZE + 1
    print(f"  完成, 共 {num_episodes} episode, {total_frames} 帧, {num_chunks} chunks")

    # ------------------------------------------------------------------
    # 6. 生成 meta 文件
    # ------------------------------------------------------------------
    print("\n[6/6] 生成 meta 文件...")

    # episodes.jsonl
    with open(meta_path / "episodes.jsonl", "w") as f:
        for ep in episodes_info:
            f.write(json.dumps(ep) + "\n")

    # tasks.jsonl
    task_desc = args.task_description or "manipulate objects on the table"
    with open(meta_path / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_desc}) + "\n")

    # info.json
    video_info_block = {
        "video.fps": fps,
        "video.codec": codec,
        "video.pix_fmt": pix_fmt,
        "video.is_depth_map": False,
        "has_audio": False,
    }
    info = {
        "codebase_version": "v2.0",
        "robot_type": "unknown",
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 2,
        "total_chunks": num_chunks,
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{num_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.front": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": video_info_block,
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": video_info_block,
            },
            "timestamp": {"dtype": "float64", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
        },
    }
    with open(meta_path / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # stats.json (空统计, 因为没有 state/action 数据)
    with open(meta_path / "stats.json", "w") as f:
        json.dump({}, f, indent=2)

    # modality.json
    modality = {
        "video": {
            "front": {"original_key": "observation.images.front"},
            "wrist": {"original_key": "observation.images.wrist"},
        },
        "annotation": {
            "human.action.task_description": {},
            "human.validity": {},
        },
    }
    with open(meta_path / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    print("\n" + "=" * 60)
    print("转换完成!")
    print(f"  输出路径:    {dst_path}")
    print(f"  Episodes:    {num_episodes}")
    print(f"  总帧数:      {total_frames}")
    print(f"  Chunks:      {num_chunks}")
    print(f"  FPS:         {fps}")
    print(f"  分辨率:      {width}x{height}")
    print("=" * 60)


if __name__ == "__main__":
    main()
