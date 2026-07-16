#!/usr/bin/env python3
"""
将已有的 per-episode .npy VLM hidden states 打包为 chunk-level .npz 文件。

转换前:
    vlm_hidden_states/
    ├── episode_000000.npy
    ├── episode_000001.npy
    └── ...

转换后:
    vlm_hidden_states/
    ├── chunk-000.npz   (episode 0~999)
    ├── chunk-001.npz   (episode 1000~1999)
    └── ...

用法:
    # 迁移单个数据集
    python utils/migrate_vlm_npy_to_chunk_npz.py \
        --dataset_root /data_disk1/hwl/unitree_train_v2_recipe_lerobot/austin_buds_dataset_converted_externally_to_rlds

    # 迁移整个根目录下所有数据集
    python utils/migrate_vlm_npy_to_chunk_npz.py \
        --dataset_root /data_disk1/hwl/unitree_train_v2_recipe_lerobot \
        --all

    # 迁移后删除旧 .npy 文件 (释放 inode)
    python utils/migrate_vlm_npy_to_chunk_npz.py \
        --dataset_root /data_disk1/hwl/unitree_train_v2_recipe_lerobot \
        --all --delete_old
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


def get_chunks_size(dataset_path: Path) -> int:
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            return json.load(f).get("chunks_size", 1000)
    return 1000


def migrate_single_dataset(dataset_path: Path, delete_old: bool = False):
    vlm_dir = dataset_path / "vlm_hidden_states"
    if not vlm_dir.exists():
        print(f"  跳过 (无 vlm_hidden_states): {dataset_path.name}")
        return

    npy_files = sorted(vlm_dir.glob("episode_*.npy"))
    if not npy_files:
        if any(vlm_dir.glob("chunk-*.npz")):
            print(f"  跳过 (已是 chunk-npz 格式): {dataset_path.name}")
        else:
            print(f"  跳过 (无 episode npy 文件): {dataset_path.name}")
        return

    chunks_size = get_chunks_size(dataset_path)
    print(f"\n  迁移: {dataset_path.name} ({len(npy_files)} episodes, chunks_size={chunks_size})")

    chunk_groups: dict[int, list[tuple[str, Path]]] = defaultdict(list)
    for npy_file in npy_files:
        ep_idx = int(npy_file.stem.replace("episode_", ""))
        chunk_idx = ep_idx // chunks_size
        chunk_groups[chunk_idx].append((f"episode_{ep_idx:06d}", npy_file))

    files_to_delete = []

    for chunk_idx in tqdm(sorted(chunk_groups.keys()), desc=f"  Packing {dataset_path.name}"):
        npz_path = vlm_dir / f"chunk-{chunk_idx:03d}.npz"

        existing = {}
        if npz_path.exists():
            try:
                with np.load(npz_path, allow_pickle=False) as old:
                    existing = {k: old[k] for k in old.files}
            except Exception:
                pass

        for ep_key, npy_file in chunk_groups[chunk_idx]:
            if ep_key not in existing:
                existing[ep_key] = np.load(npy_file)

        np.savez(npz_path, **existing)
        files_to_delete.extend(f for _, f in chunk_groups[chunk_idx])

    num_chunks = len(chunk_groups)
    print(f"  完成: {len(npy_files)} npy -> {num_chunks} npz")

    if delete_old:
        for f in files_to_delete:
            f.unlink()
        print(f"  已删除 {len(files_to_delete)} 个旧 .npy 文件")
    else:
        print(f"  旧 .npy 文件保留 (使用 --delete_old 删除)")


def main():
    parser = argparse.ArgumentParser(description="迁移 per-episode .npy -> chunk .npz")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="单个数据集路径 或 包含多个数据集的根目录")
    parser.add_argument("--all", action="store_true",
                        help="遍历 dataset_root 下所有子数据集")
    parser.add_argument("--delete_old", action="store_true",
                        help="迁移完成后删除旧的 .npy 文件")
    args = parser.parse_args()

    root = Path(args.dataset_root)

    if args.all:
        datasets = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "vlm_hidden_states").exists()
        )
        print(f"找到 {len(datasets)} 个含 vlm_hidden_states 的数据集")
        for ds in datasets:
            migrate_single_dataset(ds, args.delete_old)
    else:
        if (root / "vlm_hidden_states").exists():
            migrate_single_dataset(root, args.delete_old)
        elif (root / "meta" / "info.json").exists():
            migrate_single_dataset(root, args.delete_old)
        else:
            print(f"路径 {root} 不是有效的数据集目录")
            return

    print("\n迁移全部完成!")


if __name__ == "__main__":
    main()
