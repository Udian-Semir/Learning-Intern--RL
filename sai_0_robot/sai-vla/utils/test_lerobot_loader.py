"""
新版 LeRobot Dataset Loader 测试脚本

目标：
1. 验证每个样本是否只加载 Anchor 帧对应的 VLM hidden state
2. 确认 chunk 的最后一帧遇到 next.done=True 后，episode 数据即刻终止
3. 打印实际被加载的样本（当前帧）的 Frame ⇔ VLM index 映射
4. 通过 DataLoader 快速检查张量形状与 batch 内容
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lerobot_dataset_loader import LeRobotDataset, collate_fn


class LoaderValidator:
    """封装多项校验逻辑，便于扩展和复用"""

    def __init__(self, dataset_path: str, num_action_chunks: int, episodes: List[int] | None = None):
        self.dataset = LeRobotDataset(
            dataset_path=dataset_path,
            split="train",
            num_action_chunks=num_action_chunks,
            enable_chunking=True,
            episode_indices=episodes,
            cache_vlm_states=False,
            verbose=False,
        )
        self.dataset_path = Path(dataset_path)
        self.horizon = self.dataset.chunk_horizon

    def _load_episode_df(self, episode_idx: int) -> pd.DataFrame:
        parquet_path = self.dataset._episode_parquet_path(episode_idx)
        return self.dataset._load_parquet(str(parquet_path))

    def check_vlm_alignment(self, num_samples: int = 20) -> None:
        print("=" * 80)
        print(f"检查 1：首 {num_samples} 个样本的 VLM index 对齐情况")
        print("=" * 80)

        for idx in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[idx]
            episode_idx = sample['episode_index']
            frame_idx = sample['frame_index']
            vlm_idx = sample['vlm_index']

            df = self._load_episode_df(episode_idx)
            expected_idx = int(df.iloc[frame_idx]['vlm_hidden_state_index'])
            assert expected_idx == vlm_idx, (
                f"样本 {idx} VLM index 不匹配: got {vlm_idx}, expected {expected_idx}"
            )

            if idx < 5:
                print(f"  样本 {idx}: Episode {episode_idx}, Frame {frame_idx}, VLM {vlm_idx}")

        print("✓ 所有抽样样本的 VLM index 与 parquet 完全一致 (仅加载当前帧)")

    def check_terminal_chunks(self, episodes_to_check: int = 3) -> None:
        print("\n" + "=" * 80)
        print(f"检查 2：前 {episodes_to_check} 个 episode 的 chunk 终止逻辑")
        print("=" * 80)

        for episode_idx in range(min(episodes_to_check, len(self.dataset.episodes))):
            # 收集该 episode 的全部起始帧
            starts = [frame_idx for ep_idx, frame_idx, _ in self.dataset.index_map if ep_idx == episode_idx]
            if not starts:
                print(f"  Episode {episode_idx}: 无可用样本，跳过")
                continue

            last_start = starts[-1]
            df = self._load_episode_df(episode_idx)
            done_idx = None
            if 'next.done' in df.columns:
                done_series = df['next.done'].astype(bool).to_numpy()
                done_positions = np.where(done_series)[0]
                if done_positions.size > 0:
                    done_idx = int(done_positions[-1])

            expected_last = last_start + self.horizon - 1
            if done_idx is not None:
                assert expected_last == done_idx, (
                    f"Episode {episode_idx} 终止帧应为 {done_idx}，但最后一个 chunk 覆盖到 {expected_last}"
                )
                print(f"  Episode {episode_idx}: 终止帧 {done_idx}，chunk 正确停在 next.done=True 位置")
            else:
                assert expected_last == len(df) - 1, (
                    f"Episode {episode_idx} 无 done 标记，应覆盖到末尾 {len(df)-1}，当前 {expected_last}"
                )
                print(f"  Episode {episode_idx}: 无 done 标记，chunk 覆盖完整 episode")

    def inspect_dataloader(self, batch_size: int = 4, num_batches: int = 2) -> None:
        print("\n" + "=" * 80)
        print(f"检查 3：DataLoader batch 形状 (batch_size={batch_size})")
        print("=" * 80)

        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        for batch_idx, batch in enumerate(loader):
            print(f"\nBatch {batch_idx}:")
            print(f"  Episodes: {batch['episode_index'].tolist()}")
            print(f"  Frames:   {batch['frame_index'].tolist()}")
            print(f"  VLM idx:  {batch['vlm_index'].tolist()}")
            print(f"  VLM tensor: {tuple(batch['vlm_hidden_states'].shape)}")
            print(f"  Actions tensor: {tuple(batch['actions'].shape)}")
            if 'vlm_attention_mask' in batch:
                print(f"  VLM mask: {tuple(batch['vlm_attention_mask'].shape)}")

            if batch_idx + 1 >= num_batches:
                break

        print("\n✓ DataLoader 输出张量形状检查完毕")

    def dump_loaded_vlm_pairs(self, episodes_to_dump: int = 2) -> None:
        print("\n" + "=" * 80)
        print(f"附加检查：打印前 {episodes_to_dump} 个 episode 的实际加载样本 (Frame ⇔ VLM Index)")
        print("=" * 80)

        selected = self.dataset.episodes[:episodes_to_dump]
        if not selected:
            print("⚠️ 没有可用的 episode")
            return

        for episode in selected:
            episode_idx = episode['episode_index']
            episode_df = self._load_episode_df(episode_idx)
            loaded_frames = [frame_idx for ep_idx, frame_idx, _ in self.dataset.index_map if ep_idx == episode_idx]

            if not loaded_frames:
                print(f"\nEpisode {episode_idx}: 无加载样本 (可能长度不足以形成 chunk)")
                continue

            print(f"\nEpisode {episode_idx} - 已加载 {len(loaded_frames)} 个样本 (仅包含 anchor 帧)")
            print(f"{'Frame':>8} | {'VLM Index':>10}")
            print('-' * 24)
            for frame_idx in loaded_frames:
                vlm_idx = int(episode_df.iloc[frame_idx]['vlm_hidden_state_index'])
                print(f"{frame_idx:8d} | {vlm_idx:10d}")


def parse_args():
    parser = argparse.ArgumentParser(description="验证 LeRobot Dataset Loader 是否正确加载数据")
    parser.add_argument("--dataset_path", type=str, required=True, help="LeRobot 数据集根目录")
    parser.add_argument("--num_action_chunks", type=int, default=25, help="chunk 长度 (action horizon)")
    parser.add_argument("--episodes_to_check", type=int, default=3, help="需要检查的 episode 数量 (用于终止逻辑)")
    parser.add_argument("--mapping_episodes", type=int, default=2, help="需要打印 Frame ⇔ VLM index 映射的 episode 数量")
    parser.add_argument("--sample_checks", type=int, default=20, help="逐条检查的样本数量")
    parser.add_argument("--batch_size", type=int, default=4, help="DataLoader 检查的 batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    print("LeRobot Dataset Loader 综合校验")
    print("=" * 80)
    print(f"数据集: {dataset_path}")
    print(f"Chunk 长度: {args.num_action_chunks}\n")

    validator = LoaderValidator(str(dataset_path), args.num_action_chunks)
    validator.check_vlm_alignment(num_samples=args.sample_checks)
    validator.check_terminal_chunks(episodes_to_check=args.episodes_to_check)
    validator.inspect_dataloader(batch_size=args.batch_size)
    validator.dump_loaded_vlm_pairs(episodes_to_dump=args.mapping_episodes)

    print("\n" + "=" * 80)
    print("✅ 所有检查通过！")
    print("=" * 80)


if __name__ == "__main__":
    main()

# ./test_lerobot_loader.py --dataset_path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0/libero_lerobot_10 --num_action_chunks 25 --episodes_to_check 2 --sample_checks 5 --batch_size 2 --mapping_episodes 2