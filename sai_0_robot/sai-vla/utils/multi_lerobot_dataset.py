"""
MultiLeRobotDataset —— 把多个独立的 LeRobotDataset 子集"拼"成一个统一 Dataset。

设计要点
========
1. 每个子集底层仍走 utils.lerobot_dataset_loader.LeRobotDataset, 零修改。
2. 多源场景下 state/action 归一化 + right-pad 在 __getitem__ 里就完成
   (per-sample 用对应子集的 normalizer), DataLoader 拿到的 batch 已是
   同形状, collate_fn (oft_collate_fn) 收到 normalizers=None 即可跳过二次归一化。
3. shared_vlm_cache 通过 SubsetSharedVLMCacheView 包装: 多个子集共用同一块 SHM,
   每个子集 LeRobotDataset 仍按"子集内 vlm_index_local"取数据, 内部代码零修改。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from .lerobot_dataset_loader import LeRobotDataset
from .multi_dataset_index import MultiDatasetIndex
from .normalization_stats_merge import MultiDatasetNormalizers


# ============================================================================
# SubsetSharedVLMCacheView
# ============================================================================

class SubsetSharedVLMCacheView:
    """
    给单个子集 LeRobotDataset 看的 shared_vlm_cache 视图。

    内部仍指向 multi_chunk_batch_cache 创建的全局 SHM, 但 vlm_index 的 key 是
    "子集内的 vlm_index_local" (与 parquet 里 vlm_hidden_state_index 列一致),
    与 chunk_batch_cache.RemappedSharedVLMCache 行为完全一致 (鸭子类型)。
    """

    def __init__(
        self,
        base_cache,
        local_remap: Dict[int, int],
        subset_total_frames: int,
    ) -> None:
        self.base_cache = base_cache
        self._remap = local_remap
        self.num_samples = max(int(subset_total_frames), 1)
        self._missing_warned = False

    @property
    def name(self) -> str:
        return self.base_cache.name

    @property
    def seq_lens_name(self) -> str:
        return self.base_cache.seq_lens_name

    @property
    def dtype(self):
        return self.base_cache.dtype

    @property
    def sample_shape(self):
        return self.base_cache.sample_shape

    @property
    def is_creator(self) -> bool:
        return self.base_cache.is_creator

    @property
    def seq_lens(self):
        return self.base_cache.seq_lens

    @property
    def data(self):
        return self.base_cache.data

    def get_sample(self, vlm_index_local: int) -> np.ndarray:
        local_idx = self._remap.get(int(vlm_index_local))
        if local_idx is None:
            if not self._missing_warned:
                print(
                    f"⚠️ SubsetSharedVLMCacheView: vlm_index_local={vlm_index_local} "
                    f"不在当前 chunk 批内 (remap 大小={len(self._remap)})。"
                )
                self._missing_warned = True
            raise KeyError(
                f"vlm_index_local={vlm_index_local} not in current chunk batch"
            )
        return self.base_cache.get_sample(local_idx)

    def close(self) -> None:
        # 不主动 close base_cache, 由 MultiChunkBatchCache 管理生命周期
        pass

    def unlink(self) -> None:
        pass


# ============================================================================
# MultiLeRobotDataset
# ============================================================================

class MultiLeRobotDataset(Dataset):
    """多子集 LeRobot 包装 Dataset。"""

    def __init__(
        self,
        index: MultiDatasetIndex,
        normalizers: MultiDatasetNormalizers,
        target_state_dim: int,
        target_action_dim: int,
        num_action_chunks: int,
        enable_chunking: bool = True,
        skip_images: bool = True,
        vlm_cache_dtype: str = "float32",
        max_cached_video_readers: int = 8,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.index = index
        self.normalizers = normalizers
        self.target_state_dim = int(target_state_dim)
        self.target_action_dim = int(target_action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.enable_chunking = bool(enable_chunking)
        self.skip_images = bool(skip_images)
        self.vlm_cache_dtype = vlm_cache_dtype
        self.max_cached_video_readers = int(max_cached_video_readers)
        self.verbose = bool(verbose)

        for s in index.subsets:
            if s.action_dim > self.target_action_dim:
                raise ValueError(
                    f"[MultiLeRobotDataset] subset {s.name} action_dim={s.action_dim} "
                    f"> target_action_dim={self.target_action_dim}"
                )
            if s.state_dim > self.target_state_dim:
                raise ValueError(
                    f"[MultiLeRobotDataset] subset {s.name} state_dim={s.state_dim} "
                    f"> target_state_dim={self.target_state_dim}"
                )

        self._concat: Optional[ConcatDataset] = None
        self._sub_idx_per_position: List[int] = []
        self._active_subsets: Dict[int, LeRobotDataset] = {}

    def set_active_episodes(
        self,
        ep_uids: Iterable[int],
        subset_shared_caches: Dict[int, SubsetSharedVLMCacheView],
    ) -> None:
        """每批 ChunkBatch 切换时调用, 把当前 ep_uids 设为活动。"""
        grouped: Dict[int, List[int]] = {}
        for ep_uid in ep_uids:
            sub_idx, ep_idx_local = self.index.split_ep_uid(int(ep_uid))
            grouped.setdefault(sub_idx, []).append(ep_idx_local)

        if self.verbose:
            print(
                f"[MultiLeRobotDataset] set_active_episodes: "
                f"{len(grouped)} subsets, total {sum(len(v) for v in grouped.values())} eps"
            )

        self._release_active_subsets()

        per_subset_datasets: List[Tuple[int, LeRobotDataset]] = []
        for sub_idx in sorted(grouped.keys()):
            ep_idx_locals = sorted(grouped[sub_idx])
            sinfo = self.index.get_subset(sub_idx)
            shared_cache = subset_shared_caches.get(sub_idx)
            if shared_cache is None:
                raise KeyError(
                    f"[MultiLeRobotDataset] subset_shared_caches 缺少 sub_idx={sub_idx} "
                    f"({sinfo.name})"
                )
            ds = LeRobotDataset(
                dataset_path=str(sinfo.dataset_path),
                num_action_chunks=self.num_action_chunks,
                enable_chunking=self.enable_chunking,
                episode_indices=ep_idx_locals,
                skip_images=self.skip_images,
                vlm_cache_dtype=self.vlm_cache_dtype,
                max_cached_video_readers=self.max_cached_video_readers,
                shared_vlm_cache=shared_cache,
                verbose=False,
            )
            per_subset_datasets.append((sub_idx, ds))
            self._active_subsets[sub_idx] = ds

        self._concat = ConcatDataset([d for _, d in per_subset_datasets])
        self._sub_idx_per_position = []
        for sub_idx, ds in per_subset_datasets:
            self._sub_idx_per_position.extend([sub_idx] * len(ds))

    def close_video_readers(self) -> None:
        """对外接口: 释放所有底层子集 dataset 的资源 (与 LeRobotDataset 同名)。"""
        self._release_active_subsets()

    def _release_active_subsets(self) -> None:
        for ds in self._active_subsets.values():
            try:
                ds.close_video_readers()
            except Exception:
                pass
        self._active_subsets = {}
        self._concat = None
        self._sub_idx_per_position = []

    def __len__(self) -> int:
        if self._concat is None:
            return 0
        return len(self._concat)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self._concat is None:
            raise RuntimeError(
                "MultiLeRobotDataset 还未调用 set_active_episodes(), 无法取 sample"
            )
        sub_idx = int(self._sub_idx_per_position[idx])
        sample = self._concat[idx]
        sample = self._normalize_and_pad_sample(sub_idx, sample)
        sample["sub_idx"] = sub_idx
        sample["subset_name"] = self.index.get_subset(sub_idx).name
        return sample

    def _normalize_and_pad_sample(
        self, sub_idx: int, sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """state / observation_states_chunk / actions: 归一化 + right-pad 到 target dim。"""
        obs = sample.get("observation_state", None)
        if obs is not None:
            obs_t = _ensure_tensor(obs)
            sample["observation_state"] = (
                self.normalizers.normalize_state(sub_idx, obs_t).cpu().numpy()
            )

        obs_chunk = sample.get("observation_states_chunk", None)
        if obs_chunk is not None:
            obs_chunk_t = _ensure_tensor(obs_chunk)
            sample["observation_states_chunk"] = (
                self.normalizers.normalize_state(sub_idx, obs_chunk_t).cpu().numpy()
            )

        act = sample.get("actions", None)
        if act is not None:
            act_t = _ensure_tensor(act)
            sample["actions"] = (
                self.normalizers.normalize_action(sub_idx, act_t).cpu().numpy()
            )

        return sample


def _ensure_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(torch.float32)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)
