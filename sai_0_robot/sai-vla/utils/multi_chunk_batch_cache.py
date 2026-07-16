"""
MultiChunkBatchCache —— 多子集 LeRobot 的"分批进内存"训练缓存。

与 utils.chunk_batch_cache.ChunkBatchCache 的关系
=================================================
- ChunkBatchCache: 单数据集 → chunk 池 → 贪心打包 → SHM
- MultiChunkBatchCache: N 个子集 → 跨子集合并 chunk 池 → 全局 shuffle 后打包 → 单一全局 SHM
  load_next_batch() 返回的 subset_caches 是按 sub_idx 分桶的 view, 让每个子集
  的 LeRobotDataset 仍然按"子集内 vlm_index_local"取数据。

核心要点
========
1. 全局 chunk 池 = MultiDatasetIndex.global_chunks
2. 不同子集的 (num_layers, hidden_dim, dtype) 必须一致; seq_len 各自不同, 全局
   取 max_seq_len_global, 短的写入时 [..., :s_l, :], 剩余靠 SHM 天然 zero-fill。
3. vlm_index 语义: 子集内"按 episode_index 升序累加" (与 parquet 里
   vlm_hidden_state_index 列一致)。SHM 局部 idx 是全局 running counter。
4. load_next_batch() 返回 (ep_uids, subset_caches)。
"""

from __future__ import annotations

import os
import random
import re
import time
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_EP_KEY_RE = re.compile(r"^episode_(\d+)$")

try:
    from .chunk_batch_cache import (
        _NpzMmapReader,
        _bytes_to_gib,
        _format_size,
        _ram_available_bytes,
        _read_npz_shapes_only,
        _shm_available_bytes,
    )
    from .lerobot_dataset_loader import SharedVLMCache, get_numpy_dtype
    from .multi_dataset_index import GlobalChunk, MultiDatasetIndex
    from .multi_lerobot_dataset import SubsetSharedVLMCacheView
except ImportError:
    from chunk_batch_cache import (  # type: ignore[no-redef]
        _NpzMmapReader,
        _bytes_to_gib,
        _format_size,
        _ram_available_bytes,
        _read_npz_shapes_only,
        _shm_available_bytes,
    )
    from lerobot_dataset_loader import SharedVLMCache, get_numpy_dtype  # type: ignore[no-redef]
    from multi_dataset_index import GlobalChunk, MultiDatasetIndex  # type: ignore[no-redef]
    from multi_lerobot_dataset import SubsetSharedVLMCacheView  # type: ignore[no-redef]


class MultiChunkBatchCache:
    """跨子集的 ChunkBatchCache。"""

    def __init__(
        self,
        index: MultiDatasetIndex,
        cache_dtype: str = "float32",
        safety_ratio: float = 0.65,
        min_chunks_per_batch: int = 1,
        max_chunks_per_batch: Optional[int] = None,
        manual_chunks_per_batch: Optional[int] = None,
        ram_inflation_factor: float = 1.05,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
        seed: int = 42,
    ) -> None:
        self.index = index
        self.cache_dtype_str = cache_dtype
        self.cache_np_dtype = get_numpy_dtype(cache_dtype)
        self.safety_ratio = float(safety_ratio)
        self.min_chunks_per_batch = max(1, int(min_chunks_per_batch))
        self.max_chunks_per_batch = max_chunks_per_batch
        self.manual_chunks_per_batch = manual_chunks_per_batch
        self.ram_inflation_factor = float(ram_inflation_factor)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.verbose = bool(verbose) and self.rank == 0
        self.base_seed = int(seed)

        if not index.global_chunks:
            raise ValueError("MultiChunkBatchCache: index.global_chunks 为空")

        self.global_chunks: List[GlobalChunk] = list(index.global_chunks)
        self.num_chunks_total = len(self.global_chunks)

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(
                f"📏 [MultiChunkBatchCache] 探测全局 VLM 形状 "
                f"({self.num_chunks_total} chunks) ..."
            )

        # 全局形状探测 (rank0 一次, 顺带把损坏的 npz 标记/剔除)
        # NOTE 1: 必须扫描每个 chunk 内**所有 episode** 的 shape, 因为同一个数据集 (例如
        #   utaustin_mutex) 不同 episode 的 seq_len 可能不一样 (208 vs 234), 只看
        #   episode_000000 会低估 max_seq_len_global, 后续 SHM 分配按低估值, 装载到
        #   大 seq_len episode 时就会触发 "seq_len X > probed max_seq_len_global Y"。
        # NOTE 2: 顺便把 npz 内**实际存在的 episode keys** 拿出来, 与 index 端按
        #   `e // chunks_size == cidx` 推出的预期 episodes 取交集; 如果数据生成阶段
        #   有残缺 (例如 furniture_bench/chunk-000.npz 只有 episode_000909~999),
        #   就把 gc.episodes / episode_lengths / total_frames 修剪到实际能取到的
        #   子集, 而不是装载时报 KeyError 整段崩溃。
        # _read_npz_shapes_only 只读 .npy header, 不加载 array data, 几乎零开销。
        num_layers_all: set = set()
        hidden_dim_all: set = set()
        dtype_all: set = set()
        max_seq_len_global = 0
        broken: List[Tuple[GlobalChunk, str]] = []
        good_chunks: List[GlobalChunk] = []
        partial_chunks: List[Tuple[GlobalChunk, int, int, int]] = []
        total_missing_eps = 0
        total_missing_frames = 0
        for gc in self.global_chunks:
            try:
                shapes = _read_npz_shapes_only(gc.npz_path)
                if not shapes:
                    raise ValueError("npz has no entries")
                actual_eps: set = set()
                for ep_key, (shape, dtype) in shapes.items():
                    if len(shape) == 4:
                        _, n_l, s_l, h_d = shape
                    elif len(shape) == 3:
                        _, s_l, h_d = shape
                        n_l = 1
                    else:
                        raise ValueError(
                            f"Unexpected ndim={len(shape)} for {ep_key} "
                            f"in {gc.npz_path.name}"
                        )
                    num_layers_all.add(int(n_l))
                    hidden_dim_all.add(int(h_d))
                    dtype_all.add(dtype)
                    if int(s_l) > max_seq_len_global:
                        max_seq_len_global = int(s_l)
                    m = _EP_KEY_RE.match(ep_key)
                    if m is not None:
                        actual_eps.add(int(m.group(1)))

                expected_eps = set(gc.episodes)
                missing = expected_eps - actual_eps
                if missing:
                    keep_eps = sorted(expected_eps & actual_eps)
                    if not keep_eps:
                        # 整个 chunk 没有任何可用 episode -> 直接当损坏跳过
                        raise ValueError(
                            f"npz contains no expected episode keys "
                            f"(expected {len(expected_eps)}, "
                            f"actual {len(actual_eps)})"
                        )
                    missing_frames = sum(
                        gc.episode_lengths[e] for e in missing
                    )
                    new_total_frames = sum(
                        gc.episode_lengths[e] for e in keep_eps
                    )
                    new_lengths = {e: gc.episode_lengths[e] for e in keep_eps}
                    gc.episodes = keep_eps
                    gc.episode_lengths = new_lengths
                    gc.total_frames = int(new_total_frames)
                    partial_chunks.append(
                        (gc, len(expected_eps), len(keep_eps), missing_frames)
                    )
                    total_missing_eps += len(missing)
                    total_missing_frames += missing_frames

                good_chunks.append(gc)
            except (zipfile.BadZipFile, EOFError, OSError, ValueError) as exc:
                # 损坏 / 截断 / 完全空文件 / 不规则 shape 都跳过
                broken.append((gc, f"{type(exc).__name__}: {exc}"))
                if self.verbose:
                    sub_name = self.index.get_subset(gc.sub_idx).name
                    print(
                        f"   ⚠️ 损坏 chunk 已跳过: [{sub_name}] "
                        f"chunk-{gc.chunk_idx:03d}.npz  原因: "
                        f"{type(exc).__name__}: {str(exc)[:80]}",
                        flush=True,
                    )

        if broken:
            print(
                f"   ⚠️ 共发现 {len(broken)} 个损坏 npz (已剔除), "
                f"剩余有效 chunks: {len(good_chunks)}/{self.num_chunks_total}",
                flush=True,
            )
        if partial_chunks and self.verbose:
            print(
                f"   ⚠️ 发现 {len(partial_chunks)} 个 chunk 内 episode 不完整 "
                f"(npz 实际 keys 少于 episodes.jsonl 预期), 已自动修剪:",
                flush=True,
            )
            for gc, exp_n, keep_n, miss_frames in partial_chunks:
                sub_name = self.index.get_subset(gc.sub_idx).name
                print(
                    f"      [{sub_name}] chunk-{gc.chunk_idx:03d}.npz  "
                    f"expected {exp_n} eps -> kept {keep_n} eps "
                    f"(missing {exp_n - keep_n} eps / {miss_frames} frames)",
                    flush=True,
                )
            print(
                f"   ⚠️ 总计丢弃 {total_missing_eps} 个 episode / "
                f"{total_missing_frames} 帧, 训练继续",
                flush=True,
            )
        self.global_chunks = good_chunks
        self.num_chunks_total = len(self.global_chunks)
        self._broken_chunks = broken
        self._partial_chunks = partial_chunks

        if not self.global_chunks:
            raise RuntimeError(
                "MultiChunkBatchCache: 所有 chunk 都损坏了, 无法继续训练。"
                " 请重新生成损坏的 vlm_hidden_states/chunk-XXX.npz。"
            )

        if len(num_layers_all) != 1:
            raise ValueError(f"多子集 num_layers 不一致: {sorted(num_layers_all)}")
        if len(hidden_dim_all) != 1:
            raise ValueError(f"多子集 hidden_dim 不一致: {sorted(hidden_dim_all)}")
        if len(dtype_all) != 1:
            raise ValueError(f"多子集 on-disk dtype 不一致: {dtype_all}")

        self.num_layers: int = int(next(iter(num_layers_all)))
        self.hidden_dim: int = int(next(iter(hidden_dim_all)))
        self._on_disk_dtype = next(iter(dtype_all))
        self.max_seq_len_global: int = int(max_seq_len_global)

        bytes_per_frame_in_cache = (
            self.num_layers
            * self.max_seq_len_global
            * self.hidden_dim
            * np.dtype(self.cache_np_dtype).itemsize
        )
        self._chunk_est_ram: Dict[Tuple[int, int], int] = {}
        for gc in self.global_chunks:
            ram = int(gc.total_frames * bytes_per_frame_in_cache * self.ram_inflation_factor)
            self._chunk_est_ram[gc.global_id] = ram

        if manual_chunks_per_batch is not None:
            self.chunks_per_batch = max(1, int(manual_chunks_per_batch))
            self.budget_bytes = -1
            if self.verbose:
                print(f"  ↪ 手动指定每批 chunk 数: {self.chunks_per_batch}")
        else:
            ram_avail = _ram_available_bytes()
            shm_avail = _shm_available_bytes()
            base_budget = min(ram_avail, shm_avail) if shm_avail > 0 else ram_avail
            self.budget_bytes = max(int(base_budget * self.safety_ratio), 1)
            avg_ram = sum(self._chunk_est_ram.values()) // max(1, len(self._chunk_est_ram))
            est_chunks = max(1, self.budget_bytes // max(1, avg_ram))
            if self.max_chunks_per_batch is not None:
                est_chunks = min(est_chunks, int(self.max_chunks_per_batch))
            est_chunks = max(est_chunks, self.min_chunks_per_batch)
            est_chunks = min(est_chunks, self.num_chunks_total)
            self.chunks_per_batch = int(est_chunks)

        # ----------------------------------------------------------------
        # 跨 rank 同步 (max_seq_len_global, chunks_per_batch)
        # ----------------------------------------------------------------
        # auto 模式下 chunks_per_batch 取决于 _ram_available_bytes() / _shm_available_bytes()
        # 这两个读出的是**当前进程时刻的系统可用量**, 不同 rank 进程在不同时刻调用,
        # 看到的值会有差异 (其他进程占用、OS cache 状态、cgroups 等), 算出来的
        # chunks_per_batch 在 rank 间可能不一致 -> _shuffle_and_pack 切出不同的
        # batches -> num_frames 不一致 -> rank 0 创建的 SHM size 与其他 rank attach
        # 时算出来的 numpy buffer 大小不匹配 -> "TypeError: buffer is too small"。
        #
        # max_seq_len_global 理论上也应该所有 rank 一致 (都基于同样 npz 文件),
        # 但在 NFS / 网络 FS / 文件系统 cache 不一致等极端情况下也可能漂移,
        # 一并 broadcast 兜底。
        try:
            import torch
            import torch.distributed as dist
            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                if torch.cuda.is_available():
                    local_rank = int(os.environ.get(
                        "LOCAL_RANK",
                        self.rank % max(1, torch.cuda.device_count()),
                    ))
                    sync_device = torch.device(f"cuda:{local_rank}")
                else:
                    sync_device = torch.device("cpu")
                sync_tensor = torch.tensor(
                    [self.max_seq_len_global, self.chunks_per_batch],
                    dtype=torch.int64,
                    device=sync_device,
                )
                dist.broadcast(sync_tensor, src=0)
                synced = sync_tensor.cpu().tolist()
                new_max_seq_len = int(synced[0])
                new_cpb = int(synced[1])
                if (
                    new_max_seq_len != self.max_seq_len_global
                    or new_cpb != self.chunks_per_batch
                ):
                    if self.rank != 0:
                        # 仅在被改动时打印 (不污染 rank0 stdout)
                        print(
                            f"   🔁 [rank {self.rank}] sync (max_seq_len, "
                            f"chunks_per_batch) "
                            f"({self.max_seq_len_global}, "
                            f"{self.chunks_per_batch}) -> "
                            f"({new_max_seq_len}, {new_cpb})",
                            flush=True,
                        )
                    self.max_seq_len_global = new_max_seq_len
                    self.chunks_per_batch = new_cpb
        except ImportError:
            pass

        self.current_epoch = -1
        self.shuffled_chunks: List[GlobalChunk] = list(self.global_chunks)
        self.batches: List[List[GlobalChunk]] = []
        self._shuffle_and_pack(self.base_seed)

        self.current_batch_idx: int = 0
        self.current_base_cache: Optional[SharedVLMCache] = None
        self.current_subset_caches: Dict[int, SubsetSharedVLMCacheView] = {}
        self.current_ep_uids: List[int] = []
        self.current_num_frames: int = 0
        self._frozen_packing: bool = False
        self._shm_prefix_base = f"mcbc_{os.getpid()}_{int(time.time())}"
        self._first_load_done = False

        if self.verbose:
            self._print_init_summary()

    @property
    def num_batches(self) -> int:
        return len(self.batches)

    @property
    def total_chunks(self) -> int:
        return self.num_chunks_total

    @property
    def total_frames(self) -> int:
        return sum(gc.total_frames for gc in self.global_chunks)

    @property
    def num_segments(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        if self._frozen_packing:
            self.current_epoch = int(epoch)
            return
        if epoch == self.current_epoch:
            self.current_batch_idx = 0
            return
        self._shuffle_and_pack(self.base_seed + int(epoch))
        self.current_epoch = int(epoch)
        self.current_batch_idx = 0

    def freeze_packing(self, seed: Optional[int] = None) -> None:
        pack_seed = self.base_seed if seed is None else int(seed)
        self._shuffle_and_pack(pack_seed)
        self._frozen_packing = True
        self.current_epoch = 0
        self.current_batch_idx = 0
        if self.verbose:
            print(
                f"🔒 [MultiChunkBatchCache] 已冻结 chunk 打包 (seed={pack_seed}): "
                f"{self.num_batches} segments, {self.chunks_per_batch} chunks/segment"
            )

    def load_segment(
        self, segment_idx: int
    ) -> Tuple[List[int], Dict[int, int], int, Optional[SubsetSharedVLMCacheView]]:
        if not (0 <= int(segment_idx) < len(self.batches)):
            raise IndexError(
                f"segment_idx={segment_idx} out of range [0, {len(self.batches)})"
            )
        self.current_batch_idx = int(segment_idx)
        return self.load_next_batch()

    def load_next_batch(
        self,
    ) -> Tuple[List[int], Dict[int, int], int, Optional[SubsetSharedVLMCacheView]]:
        """
        与 ChunkBatchCache.load_next_batch 同样的 4-tuple 签名 (鸭子类型),
        让 train_multigpu.py 单源/多源代码路径可以统一 unpack。

        Returns:
            ep_uids: List[int] 全局 ep_uid 列表 (多源场景下下游需用 split_ep_uid 还原)
            _placeholder_remap: 永远是 {}, 多源 remap 已分桶到 self.current_subset_caches
            num_frames: int
            None: 多源场景下 train_dataset 用 MultiLeRobotDataset.set_active_episodes 直接拿
                  self.current_subset_caches, 不再共享单一 RemappedSharedVLMCache
        """
        try:
            import torch
            import torch.distributed as dist
        except ImportError as e:
            raise RuntimeError("MultiChunkBatchCache.load_next_batch requires torch") from e

        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        batch_idx = self.current_batch_idx
        if batch_idx >= len(self.batches):
            raise IndexError(
                f"current_batch_idx={batch_idx} but num_batches={len(self.batches)}"
            )
        chunks_this_batch = self.batches[batch_idx]

        self._release_current_cache(verbose=self.verbose)
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        # 计算本批 ep_uids + total frames + per-subset remap
        ep_uids: List[int] = []
        subset_remap: Dict[int, Dict[int, int]] = {}
        local_running = 0
        for gc in chunks_this_batch:
            sinfo = self.index.get_subset(gc.sub_idx)
            for ep_idx_local in sorted(gc.episodes):
                ep_uid = self.index.episode_uid(gc.sub_idx, ep_idx_local)
                ep_uids.append(ep_uid)
                ep_len = gc.episode_lengths[ep_idx_local]
                vlm_start = sinfo.ep_local_vlm_start[ep_idx_local]
                remap = subset_remap.setdefault(gc.sub_idx, {})
                for fi in range(ep_len):
                    remap[vlm_start + fi] = local_running + fi
                local_running += ep_len
        num_frames = local_running

        sample_shape = (self.num_layers, self.max_seq_len_global, self.hidden_dim)
        shm_prefix = f"{self._shm_prefix_base}_b{batch_idx}_e{max(0, self.current_epoch)}"

        if self.verbose:
            sub_names_in_batch = sorted(
                {self.index.get_subset(gc.sub_idx).name for gc in chunks_this_batch}
            )
            print(f"\n{'-' * 60}")
            print(
                f"📂 [MultiChunkBatchCache] 加载批 {batch_idx + 1}/{self.num_batches}  "
                f"(epoch={self.current_epoch})"
            )
            print(
                f"   chunks: {len(chunks_this_batch)} (来自 "
                f"{len(sub_names_in_batch)} 个子集) {sub_names_in_batch[:5]}"
                f"{' ...' if len(sub_names_in_batch) > 5 else ''}"
            )
            print(
                f"   episodes={len(ep_uids)}  frames={num_frames}  "
                f"sample_shape={sample_shape}  cache_dtype={self.cache_dtype_str}"
            )
            est = sum(self._chunk_est_ram[gc.global_id] for gc in chunks_this_batch)
            print(f"   预计 SHM 占用 ≈ {_format_size(est)}")

        base_cache: Optional[SharedVLMCache] = None
        cache_name = ""
        seq_lens_name = ""

        if self.rank == 0:
            t0 = time.time()
            base_cache = SharedVLMCache.create(
                num_samples=num_frames,
                sample_shape=sample_shape,
                dtype=self.cache_np_dtype,
                name_prefix=shm_prefix,
            )
            shm_data = base_cache.data
            shm_seq_lens = base_cache.seq_lens
            local_idx = 0
            num_layers_g = self.num_layers
            hidden_dim_g = self.hidden_dim
            max_seq_len = self.max_seq_len_global
            total_frames_planned = num_frames

            for chunk_pos, gc in enumerate(chunks_this_batch):
                sinfo = self.index.get_subset(gc.sub_idx)
                t_chunk = time.time()
                ep_count = 0
                if self.verbose:
                    print(
                        f"     ▶ [{sinfo.name}] chunk-{gc.chunk_idx:03d} "
                        f"({chunk_pos+1}/{len(chunks_this_batch)}) 开始装载: "
                        f"{gc.total_frames} frames / {len(gc.episodes)} eps  "
                        f"(disk: {_format_size(gc.npz_size_bytes)})",
                        flush=True,
                    )
                try:
                    data_ctx = _NpzMmapReader(gc.npz_path)
                    use_mmap_path = True
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"     ⚠️ _NpzMmapReader 失败 ({exc.__class__.__name__}: {exc}), "
                            f"回退 np.load",
                            flush=True,
                        )
                    data_ctx = np.load(gc.npz_path, allow_pickle=False)
                    use_mmap_path = False

                with data_ctx as data:
                    avail_keys = set(data.files)
                    sorted_eps = sorted(gc.episodes)
                    for ep_idx_local in sorted_eps:
                        ep_key = f"episode_{ep_idx_local:06d}"
                        if ep_key not in avail_keys:
                            raise KeyError(f"Episode key {ep_key} not in {gc.npz_path.name}")
                        ep_arr = data[ep_key]
                        if ep_arr.ndim == 3:
                            ep_arr = ep_arr[:, None, :, :]
                        elif ep_arr.ndim != 4:
                            raise ValueError(
                                f"Unexpected ndim={ep_arr.ndim} for {ep_key} "
                                f"in {gc.npz_path.name}"
                            )
                        n_f, n_l, s_l, h_d = ep_arr.shape
                        if (n_l, h_d) != (num_layers_g, hidden_dim_g):
                            raise ValueError(
                                f"VLM shape mismatch in {ep_key} of {gc.npz_path.name}"
                            )
                        if s_l > max_seq_len:
                            raise ValueError(
                                f"seq_len {s_l} > probed max_seq_len_global {max_seq_len}"
                            )
                        end_idx = local_idx + n_f
                        if end_idx > total_frames_planned:
                            raise RuntimeError(
                                f"Frame overflow in {ep_key}: end_idx={end_idx} "
                                f"> planned={total_frames_planned}"
                            )
                        ep_arr_cast = ep_arr.astype(self.cache_np_dtype, copy=False)
                        if s_l == max_seq_len:
                            shm_data[local_idx:end_idx] = ep_arr_cast
                        else:
                            shm_data[local_idx:end_idx, :, :s_l, :] = ep_arr_cast
                        shm_seq_lens[local_idx:end_idx] = s_l
                        local_idx = end_idx
                        ep_count += 1
                    try:
                        del ep_arr  # type: ignore[possibly-unbound]
                    except NameError:
                        pass
                    try:
                        del ep_arr_cast  # type: ignore[possibly-unbound]
                    except NameError:
                        pass
                if self.verbose:
                    dt = time.time() - t_chunk
                    print(
                        f"     ✓ [{sinfo.name}] chunk-{gc.chunk_idx:03d}  "
                        f"{gc.total_frames} frames ({ep_count} eps)  载入完成 {dt:.1f}s",
                        flush=True,
                    )
            if local_idx != num_frames:
                raise RuntimeError(
                    f"Loaded {local_idx} frames but expected {num_frames}"
                )
            cache_name = base_cache.name
            seq_lens_name = base_cache.seq_lens_name
            if self.verbose:
                dt_total = time.time() - t0
                gibps = _bytes_to_gib(
                    int(num_frames) * num_layers_g * max_seq_len * hidden_dim_g
                    * np.dtype(self.cache_np_dtype).itemsize
                ) / max(dt_total, 1e-3)
                print(
                    f"   ✓ rank 0 装载完成: {dt_total:.1f}s ≈ {gibps:.2f} GiB/s"
                )

        # 广播 SHM 名 → 其他 rank attach
        if self.world_size > 1 and dist.is_initialized():
            try:
                local_rank = int(
                    os.environ.get(
                        "LOCAL_RANK",
                        self.rank % max(1, torch.cuda.device_count()),
                    )
                )
                if torch.cuda.is_available():
                    device = torch.device(f"cuda:{local_rank}")
                else:
                    device = torch.device("cpu")
            except Exception:
                device = torch.device("cpu")

            if self.rank == 0:
                name_buf = list(cache_name.encode("ascii"))
                seqn_buf = list(seq_lens_name.encode("ascii"))
            else:
                name_buf = []
                seqn_buf = []
            name_pad = name_buf + [0] * (256 - len(name_buf))
            seqn_pad = seqn_buf + [0] * (256 - len(seqn_buf))

            name_tensor = torch.tensor(name_pad, dtype=torch.int32, device=device)
            seqn_tensor = torch.tensor(seqn_pad, dtype=torch.int32, device=device)
            dist.broadcast(name_tensor, src=0)
            dist.broadcast(seqn_tensor, src=0)
            dist.barrier()

            if self.rank != 0:
                cache_name = bytes(
                    c for c in name_tensor.cpu().tolist() if c != 0
                ).decode("ascii")
                seq_lens_name = bytes(
                    c for c in seqn_tensor.cpu().tolist() if c != 0
                ).decode("ascii")
                base_cache = SharedVLMCache.attach(
                    name=cache_name,
                    seq_lens_name=seq_lens_name,
                    num_samples=num_frames,
                    sample_shape=sample_shape,
                    dtype=self.cache_np_dtype,
                )

        if base_cache is None:
            raise RuntimeError("Failed to obtain SharedVLMCache after broadcast")

        # 包装 per-subset views
        subset_caches: Dict[int, SubsetSharedVLMCacheView] = {}
        for sub_idx, remap in subset_remap.items():
            sinfo = self.index.get_subset(sub_idx)
            subset_caches[sub_idx] = SubsetSharedVLMCacheView(
                base_cache=base_cache,
                local_remap=remap,
                subset_total_frames=sinfo.total_frames,
            )

        self.current_base_cache = base_cache
        self.current_subset_caches = subset_caches
        self.current_ep_uids = ep_uids
        self.current_num_frames = num_frames
        self.current_batch_idx = batch_idx + 1
        self._first_load_done = True

        # 4-tuple 与 ChunkBatchCache 兼容; 多源 remap 已分桶到 self.current_subset_caches
        return ep_uids, {}, num_frames, None

    def cleanup(self) -> None:
        self._release_current_cache(verbose=self.verbose)

    def _shuffle_and_pack(self, seed: int) -> None:
        rng = random.Random(int(seed))
        chunk_list = list(self.global_chunks)
        rng.shuffle(chunk_list)
        self.shuffled_chunks = chunk_list
        cpb = self.chunks_per_batch if self.chunks_per_batch > 0 else 1
        batches: List[List[GlobalChunk]] = []
        for i in range(0, len(chunk_list), cpb):
            batches.append(chunk_list[i : i + cpb])
        self.batches = batches

    def _release_current_cache(self, verbose: bool = False) -> None:
        if self.current_base_cache is None:
            return
        try:
            if verbose:
                print(f"   ♻️  释放上一批 SHM: {self.current_base_cache.name}")
            if self.rank == 0:
                self.current_base_cache.close()
                self.current_base_cache.unlink()
            else:
                self.current_base_cache.close()
        except Exception as e:
            print(f"⚠️ MultiChunkBatchCache 释放 SHM 时出错 (已忽略): {e}")
        self.current_base_cache = None
        self.current_subset_caches = {}
        self.current_ep_uids = []
        self.current_num_frames = 0

    def _print_init_summary(self) -> None:
        ram_gib = _bytes_to_gib(_ram_available_bytes())
        shm_gib = _bytes_to_gib(_shm_available_bytes())
        budget_gib = _bytes_to_gib(self.budget_bytes) if self.budget_bytes > 0 else -1.0
        avg_ram_gib = _bytes_to_gib(
            sum(self._chunk_est_ram.values()) // max(1, len(self._chunk_est_ram))
        )
        print(f"📦 [MultiChunkBatchCache] 初始化完成")
        print(f"   root_dir:            {self.index.root_dir}")
        print(f"   subsets:             {len(self.index.subsets)}")
        print(f"   global episodes:     {self.index.total_global_episodes}")
        print(f"   global frames:       {self.index.total_global_frames}")
        print(f"   selected chunks:     {self.num_chunks_total}")
        print(
            f"   VLM shape per frame: ({self.num_layers}, "
            f"{self.max_seq_len_global}, {self.hidden_dim})"
        )
        print(f"   on-disk dtype:       {self._on_disk_dtype}")
        print(
            f"   cache dtype:         {self.cache_dtype_str} "
            f"({np.dtype(self.cache_np_dtype).itemsize} B/scalar)"
        )
        print(f"   avg chunk RAM(est):  {avg_ram_gib:.2f} GiB")
        if self.budget_bytes > 0:
            print(f"   RAM available:       {ram_gib:.2f} GiB")
            print(f"   /dev/shm available:  {shm_gib:.2f} GiB")
            print(f"   safety_ratio:        {self.safety_ratio}")
            print(f"   budget per batch:    {budget_gib:.2f} GiB")
        else:
            print(f"   manual chunks/batch: {self.chunks_per_batch}")
        print(f"   chunks per batch:    {self.chunks_per_batch}")
        print(f"   number of batches:   {self.num_batches}")
        print(f"{'=' * 60}")