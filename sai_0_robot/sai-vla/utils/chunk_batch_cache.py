"""
ChunkBatchCache —— 基于 lerobot chunk-XXX.npz 的"分批进内存"训练缓存。

设计目标
========
当 vlm_hidden_states 总量大于本机内存（例如 2.6 TiB 数据 + 2 TiB RAM）时，
USE_SHARED_CACHE 模式会因为 rank 0 一次性加载全部数据而 OOM。

ChunkBatchCache 的做法是把 vlm_hidden_states/chunk-*.npz 当作
WebDataset 里的 tar 分片来用：

  1) 启动时扫描全部 chunk 的字节大小（os.path.getsize，秒级）；
     再读取首个 chunk 的首个 episode 拿到 (num_layers, seq_len, hidden_dim, dtype)
     用作打包预算估计。
  2) 自动用 min(RAM_avail, /dev/shm_avail) × safety_ratio 作为单批内存预算，
     贪心打包出 N 个 batch（每个 batch 是若干完整的 chunk）。
  3) 每个 epoch 用 seed+epoch 重排 chunk 列表，再重新打包，
     保证不同 epoch 的"批组合"不同（跨批 shuffle）。
  4) 切批：rank 0 释放上一批 SHM → 读取本批 chunk 的所有 episode →
     padding 到 max_seq_len → 写入新 SHM → 广播 SHM 名/形状 → 其他 rank attach。
  5) 训练时 LeRobotDataset 用 (episode_indices=本批 ep 列表, shared_vlm_cache=本批 wrap 后的 cache)，
     `vlm_hidden_state_index` 通过 RemappedSharedVLMCache 透明地映射到本批 cache 的 local idx。
  6) 批内 shuffle 由 DistributedSampler.set_epoch(epoch * N + batch_idx) 完成。

接口跟 utils/webdataset_utils.py:ShardBatchCache 对齐，
方便在 train_multigpu.py 复用 `for shard_batch_idx in range(num_shard_batches)` 外层循环。
"""

from __future__ import annotations

import json
import math
import mmap as _mmap_mod
import os
import random
import shutil
import struct
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.format import read_array_header_1_0, read_array_header_2_0, read_magic

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

try:
    from .lerobot_dataset_loader import (
        SharedVLMCache,
        get_numpy_dtype,
    )
except ImportError:  # 直接 python 运行时
    from lerobot_dataset_loader import (  # type: ignore[no-redef]
        SharedVLMCache,
        get_numpy_dtype,
    )


# ============================================================================
# 零拷贝 NPZ 读取 (ZIP_STORED only) —— 走 mmap，避开 numpy 内部的解压/拷贝路径
# ============================================================================
# numpy 的 `np.load(npz)[key]` 内部流程：
#   ZipFile.open(key) → 创建 ZipExtFile (即便 STORED 也是 BufferedReader 包一层)
#     → numpy.lib.format.read_array() 调用 fromfile/read+frombuffer
#     → 返回独立内存的 ndarray (一次完整 memcpy)
# 实测在 NVMe 上仍然只跑得到 ~0.7 GB/s（受限于 Python 单线程的 read()/copy 带宽）。
#
# 我们这里直接：
#   1) 用 zipfile 一次性读出每个 entry 在 .npz 内的 `.npy header offset`
#   2) 解析 .npy header 拿到 dtype/shape，记录 array data 的绝对偏移
#   3) 对整个 .npz 做 mmap(MAP_PRIVATE, READ_ONLY) + posix_fadvise(SEQUENTIAL)
#   4) `__getitem__` 返回 `np.frombuffer(self._mmap, ..., offset=arr_off)` —— 零拷贝 view
# 后续 `shm_dst[...] = view` 是单次 memcpy，可吃满 NVMe 顺序读 + RAM 顺序写带宽。
class _NpzMmapReader:
    """Zero-copy reader for *uncompressed* .npz (ZIP_STORED) archives via mmap.

    用法兼容 `np.load(path)` 的子集：`reader[key]` 返回 ndarray view。
    需要保证 .npz 是 `np.savez` (而不是 `np.savez_compressed`) 产生的。
    """

    _ZIP_LOCAL_HEADER_SIG = 0x04034B50

    def __init__(self, path: os.PathLike) -> None:
        self._path = str(path)
        self._fobj = None
        self._mmap = None
        # key -> (data_offset_bytes, np.dtype, shape_tuple, fortran_order_bool)
        self._index: Dict[str, Tuple[int, np.dtype, Tuple[int, ...], bool]] = {}
        self._open()

    def _open(self) -> None:
        f = open(self._path, "rb")
        self._fobj = f
        try:
            with zipfile.ZipFile(f, mode="r") as zf:
                for zinfo in zf.infolist():
                    if zinfo.compress_type != zipfile.ZIP_STORED:
                        raise ValueError(
                            f"_NpzMmapReader: entry '{zinfo.filename}' in "
                            f"{self._path} is compressed (compress_type="
                            f"{zinfo.compress_type}); 必须用 np.savez (非 _compressed)"
                        )
                    f.seek(zinfo.header_offset)
                    hdr = f.read(30)
                    if len(hdr) < 30:
                        raise ValueError(
                            f"_NpzMmapReader: bad ZIP local header at "
                            f"{zinfo.header_offset} in {self._path}"
                        )
                    sig = struct.unpack("<I", hdr[0:4])[0]
                    if sig != self._ZIP_LOCAL_HEADER_SIG:
                        raise ValueError(
                            f"_NpzMmapReader: bad ZIP signature 0x{sig:08x} "
                            f"in {self._path} for entry {zinfo.filename}"
                        )
                    fname_len = struct.unpack("<H", hdr[26:28])[0]
                    extra_len = struct.unpack("<H", hdr[28:30])[0]
                    npy_off = zinfo.header_offset + 30 + fname_len + extra_len
                    f.seek(npy_off)
                    version = read_magic(f)
                    if version == (1, 0):
                        shape, fortran_order, dtype = read_array_header_1_0(f)
                    elif version == (2, 0):
                        shape, fortran_order, dtype = read_array_header_2_0(f)
                    else:
                        raise ValueError(
                            f"_NpzMmapReader: unsupported .npy version "
                            f"{version} in {self._path}::{zinfo.filename}"
                        )
                    arr_off = f.tell()
                    key = zinfo.filename
                    if key.endswith(".npy"):
                        key = key[:-4]
                    self._index[key] = (arr_off, dtype, tuple(shape), bool(fortran_order))
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
            self._mmap = _mmap_mod.mmap(
                f.fileno(), 0, prot=_mmap_mod.PROT_READ
            )
            try:
                self._mmap.madvise(_mmap_mod.MADV_SEQUENTIAL)
            except (AttributeError, OSError):
                pass
        except Exception:
            try:
                f.close()
            finally:
                self._fobj = None
            raise

    @property
    def files(self) -> List[str]:
        return list(self._index.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in self._index:
            raise KeyError(key)
        if self._mmap is None:
            raise RuntimeError(f"_NpzMmapReader already closed: {self._path}")
        offset, dtype, shape, fortran_order = self._index[key]
        nelem = 1
        for d in shape:
            nelem *= int(d)
        arr = np.frombuffer(
            self._mmap, dtype=dtype, count=nelem, offset=offset
        )
        if fortran_order:
            return arr.reshape(shape, order="F")
        return arr.reshape(shape)

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                # 仍有 ndarray view 引用 mmap 时 close 会抛 BufferError；
                # 此时把 mmap 引用置 None，等 ndarray 被 GC 后 OS 自动 munmap。
                pass
            self._mmap = None
        if self._fobj is not None:
            try:
                self._fobj.close()
            except OSError:
                pass
            self._fobj = None

    def __enter__(self) -> "_NpzMmapReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# ============================================================================
# 仅读 .npy header 的 shape/dtype 探测器 (不加载任何 array 数据)
# ============================================================================
# np.load(npz)[key] 会把整个 array 解压/拷贝进内存; 而我们在 init 探测全局
# (num_layers, max_seq_len, hidden_dim) 时只需要 shape/dtype, 不需要 array data。
#
# 性能关键: 对 ZIP_STORED entry, 直接走底层文件 `f.seek(header_offset)` + `f.read(30)`
# 跳过 ZIP local header, 再用 numpy.lib.format.read_array_header 读 .npy header,
# 全程没有 ZipExtFile 包装 (zf.open(...) 那条路径每次都要构造 BufferedReader,
# 在几十万次调用下会慢到难以接受)。仅当 entry 是压缩存储时回退到 zf.open。
_ZIP_LOCAL_HEADER_SIG = 0x04034B50


def _read_npy_header_at(
    f, npy_offset: int, source_desc: str
) -> Tuple[Tuple[int, ...], np.dtype]:
    """读 .npy header, 返回 (shape, dtype). 文件指针停在 array data 起点。"""
    f.seek(npy_offset)
    version = read_magic(f)
    if version == (1, 0):
        shape, _fortran, dtype = read_array_header_1_0(f)
    elif version == (2, 0):
        shape, _fortran, dtype = read_array_header_2_0(f)
    else:
        raise ValueError(
            f"_read_npz_shapes_only: unsupported .npy version "
            f"{version} in {source_desc}"
        )
    return tuple(int(d) for d in shape), np.dtype(dtype)


def _read_npz_shapes_only(
    path: os.PathLike,
) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
    """读取 .npz 内**所有** entry 的 (shape, dtype), 不加载 array 数据。

    适用于压缩 / 非压缩 npz; 仅用于初始化阶段探测最大 seq_len。
    对常规 (np.savez) 非压缩 npz 走 mmap-friendly 的 seek/read 快路径,
    单文件几十万 entry 也只是数十 ms 级别。

    截断校验
    --------
    对 ZIP_STORED entry, 我们能拿到 array data 的绝对起点 (npy_offset 之后)
    以及 dtype/shape, 因此可以校验
        npy_offset + header_size + nelem * itemsize <= file_size
    若不成立, 说明 npz 写到一半被中断了 (header 完整但 data 截断), 此时
    `_NpzMmapReader.__getitem__` 后续会用 np.frombuffer 触发
    "ValueError: buffer is smaller than requested size"。这里直接把这种
    entry 当作 "不存在", 上层 probe 会自动把它从 gc.episodes 修剪掉。
    对压缩 entry 无法预先估算 raw data 大小, 只能等装载时由 zlib 报错。
    """
    out: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
    truncated: List[Tuple[str, int, int]] = []
    spath = str(path)
    file_size = os.path.getsize(spath)
    with open(spath, "rb") as f:
        with zipfile.ZipFile(f, mode="r") as zf:
            for zinfo in zf.infolist():
                name = zinfo.filename
                if not name.endswith(".npy"):
                    continue
                key = name[:-4]
                if zinfo.compress_type == zipfile.ZIP_STORED:
                    # 快路径: 跳过 ZIP local header, 直接定位到 .npy magic
                    f.seek(zinfo.header_offset)
                    hdr = f.read(30)
                    if len(hdr) < 30:
                        raise ValueError(
                            f"_read_npz_shapes_only: bad ZIP local header at "
                            f"{zinfo.header_offset} in {spath}"
                        )
                    sig = struct.unpack("<I", hdr[0:4])[0]
                    if sig != _ZIP_LOCAL_HEADER_SIG:
                        raise ValueError(
                            f"_read_npz_shapes_only: bad ZIP signature "
                            f"0x{sig:08x} in {spath} for entry {name}"
                        )
                    fname_len = struct.unpack("<H", hdr[26:28])[0]
                    extra_len = struct.unpack("<H", hdr[28:30])[0]
                    npy_off = zinfo.header_offset + 30 + fname_len + extra_len
                    shape, dtype = _read_npy_header_at(
                        f, npy_off, f"{spath}::{name}"
                    )
                    data_off = f.tell()
                    nelem = 1
                    for d in shape:
                        nelem *= int(d)
                    expected_end = data_off + nelem * int(dtype.itemsize)
                    if expected_end > file_size:
                        truncated.append((name, expected_end, file_size))
                        continue
                    out[key] = (shape, dtype)
                else:
                    # 压缩 entry 必须走解压流, 但只读极少字节, 性能影响可接受
                    with zf.open(zinfo) as ef:
                        version = read_magic(ef)
                        if version == (1, 0):
                            shape, _fortran, dtype = read_array_header_1_0(ef)
                        elif version == (2, 0):
                            shape, _fortran, dtype = read_array_header_2_0(ef)
                        else:
                            raise ValueError(
                                f"_read_npz_shapes_only: unsupported .npy "
                                f"version {version} in {spath}::{name}"
                            )
                    out[key] = (
                        tuple(int(d) for d in shape),
                        np.dtype(dtype),
                    )
    if truncated:
        # 仅 print 前几条, 防止单文件巨量截断刷屏
        head = ", ".join(
            f"{n}(need={ne}>file={fs})" for n, ne, fs in truncated[:3]
        )
        more = f" (+{len(truncated) - 3} more)" if len(truncated) > 3 else ""
        print(
            f"   ⚠️ {spath}: 检测到 {len(truncated)} 个 entry data 被截断, "
            f"已剔除: {head}{more}",
            flush=True,
        )
    return out


# ============================================================================
# 工具函数
# ============================================================================

def _bytes_to_gib(n: int) -> float:
    return n / (1024 ** 3)


def _format_size(n_bytes: int) -> str:
    return f"{_bytes_to_gib(n_bytes):.2f} GiB"


def _ram_available_bytes() -> int:
    """返回当前可用物理内存（字节）。psutil 不可用时退回到 /proc/meminfo。"""
    if psutil is not None:
        return int(psutil.virtual_memory().available)

    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kib = int(line.split()[1])
                    return kib * 1024
    except Exception:
        pass

    return 0


def _shm_available_bytes() -> int:
    """返回 /dev/shm 可用字节数。"""
    try:
        return int(shutil.disk_usage("/dev/shm").free)
    except Exception:
        return 0


# ============================================================================
# RemappedSharedVLMCache —— 让 LeRobotDataset 透明地通过 global vlm_index 取本批数据
# ============================================================================

class RemappedSharedVLMCache:
    """
    包装 SharedVLMCache，把全局 vlm_index 透明映射到本批 cache 的 local index。

    - LeRobotDataset 内部按 `cache.get_sample(vlm_index)` 查询；
    - 这里的 vlm_index 是 parquet 里 `vlm_hidden_state_index` 列的全局值；
    - 本批 cache 只装了一部分 episode，所以维护一个 dict: global_vlm_index -> local_idx；
    - num_samples 取 `total_frames`（数据集总帧数），以避免 LeRobotDataset 的 in-range
      检查走到 fallback 分支；不在本批的 vlm_index 会触发 KeyError，但因为我们用
      episode_indices=本批 ep 列表 限制了 Dataset，正常路径下不会越界。
    """

    def __init__(
        self,
        base_cache: SharedVLMCache,
        vlm_index_remap: Dict[int, int],
        total_frames: int,
    ) -> None:
        self.base_cache = base_cache
        self._remap = vlm_index_remap
        self.num_samples = max(int(total_frames), 1)
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

    def get_sample(self, vlm_index: int) -> np.ndarray:
        local_idx = self._remap.get(int(vlm_index))
        if local_idx is None:
            if not self._missing_warned:
                print(
                    f"⚠️ RemappedSharedVLMCache: vlm_index={vlm_index} 不在当前 chunk 批内，"
                    f"LeRobotDataset 将回退到 mmap。请确认 episode_indices 设置正确。"
                )
                self._missing_warned = True
            raise KeyError(f"vlm_index {vlm_index} not in current chunk batch")
        return self.base_cache.get_sample(local_idx)

    def close(self) -> None:
        self.base_cache.close()

    def unlink(self) -> None:
        self.base_cache.unlink()


# ============================================================================
# ChunkBatchCache —— 主类
# ============================================================================

class ChunkBatchCache:
    """
    基于 lerobot chunk-XXX.npz 的分批共享内存缓存。

    用法（仿 ShardBatchCache）::

        cache = ChunkBatchCache(
            dataset_path="/data/.../dataset196",
            episode_indices=train_episode_indices,
            cache_dtype="float32",
            safety_ratio=0.65,
            rank=rank,
            world_size=world_size,
        )

        for epoch in range(args.epochs):
            cache.set_epoch(epoch)
            for batch_idx in range(cache.num_batches):
                episode_indices, vlm_remap, n_frames, shared_cache = cache.load_next_batch()
                # 用 episode_indices + shared_cache 创建 LeRobotDataset
                ...
    """

    def __init__(
        self,
        dataset_path: str,
        episode_indices: Optional[List[int]] = None,
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
        """
        Args:
            dataset_path: lerobot 数据集根目录，需含 meta/info.json、meta/episodes.jsonl、
                         vlm_hidden_states/chunk-*.npz。
            episode_indices: 限定的 episode 索引（用于训练集/验证集划分）。None 表示全部。
            cache_dtype: 共享内存中数据类型 ("float32", "float16", "bfloat16")。
                        bfloat16 在 numpy 里以 float16 存。
            safety_ratio: 内存预算安全系数。预算 = min(RAM_avail, SHM_avail) × safety_ratio。
            min_chunks_per_batch: 每批最少 chunk 数（即使内存允许更少，也至少这么多）。
            max_chunks_per_batch: 每批最多 chunk 数。None 表示不限。
            manual_chunks_per_batch: 直接指定每批 chunk 数（不再做内存预算计算）。
                                     设置后 safety_ratio / min/max_chunks_per_batch 都被忽略。
            ram_inflation_factor: 估算每个 chunk 的 cache 占用时的安全放大。1.05 表示在精确估算
                                 基础上再加 5% 余量。
            rank/world_size: 分布式参数。
            seed: 用于 shuffle 的随机种子（每 epoch shuffle 用 seed+epoch）。
        """
        self.dataset_path = Path(dataset_path)
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

        # ---------- 加载 metadata ----------
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"meta/info.json not found: {info_path}")
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        self.chunks_size: int = int(info.get("chunks_size", 1000))
        self.total_frames_dataset: int = int(info["total_frames"])
        self.total_episodes_dataset: int = int(info["total_episodes"])

        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        if not episodes_path.exists():
            raise FileNotFoundError(f"meta/episodes.jsonl not found: {episodes_path}")

        # 读取每个 episode 的长度（按 episode_index 排序）
        all_episodes: Dict[int, int] = {}
        with open(episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                ep_idx = int(rec["episode_index"])
                ep_len = int(rec["length"])
                all_episodes[ep_idx] = ep_len

        # 计算每个 episode 在 vlm_hidden_state_index 上的全局起点
        # （按 episode_index 升序累加）
        ep_idx_sorted = sorted(all_episodes.keys())
        self._ep_global_vlm_start: Dict[int, int] = {}
        running = 0
        for e in ep_idx_sorted:
            self._ep_global_vlm_start[e] = running
            running += all_episodes[e]

        # episode_indices 过滤
        if episode_indices is None:
            allowed = set(ep_idx_sorted)
        else:
            allowed = set(int(e) for e in episode_indices)

        # 把 episode 按 chunk_idx 分组（仅保留 allowed 内的）
        self._chunks: Dict[int, Dict[str, Any]] = {}
        for e in ep_idx_sorted:
            if e not in allowed:
                continue
            cidx = e // self.chunks_size
            entry = self._chunks.setdefault(
                cidx,
                {
                    "chunk_idx": cidx,
                    "episodes": [],
                    "episode_lengths": {},
                    "npz_path": None,
                    "npz_size": 0,
                    "estimated_ram_bytes": 0,
                    "total_frames": 0,
                },
            )
            entry["episodes"].append(e)
            entry["episode_lengths"][e] = all_episodes[e]
            entry["total_frames"] += all_episodes[e]

        if not self._chunks:
            raise ValueError(
                f"No chunks selected for dataset {self.dataset_path}. "
                f"episode_indices={episode_indices}"
            )

        # 校验 npz 文件存在 + 记录文件大小
        vlm_dir = self.dataset_path / "vlm_hidden_states"
        for cidx, entry in self._chunks.items():
            npz_path = vlm_dir / f"chunk-{cidx:03d}.npz"
            if not npz_path.exists():
                raise FileNotFoundError(
                    f"VLM chunk file missing: {npz_path} "
                    f"(expected for chunk_idx={cidx})"
                )
            entry["npz_path"] = npz_path
            entry["npz_size"] = int(os.path.getsize(npz_path))

        self.all_chunk_indices: List[int] = sorted(self._chunks.keys())
        self.num_chunks_total = len(self.all_chunk_indices)

        # ---------- 探测 VLM 形状（扫描所有 chunk 内**全部** episode 的 .npy header）----
        # NOTE: 早期版本只读 chunk-000 的 episode_000000 抽样, 但实际数据集 (例如
        # utaustin_mutex) 同一 chunk 内不同 episode 的 seq_len 可能不一样
        # (208 vs 226), 抽样会低估 max_seq_len_global, 后续 SHM 按低估值分配,
        # 装载到大 seq_len episode 时会报 "seq_len X > probed max_seq_len_global Y"。
        # _read_npz_shapes_only 只读 .npy header, 不加载 array data, 几乎零开销。
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(
                f"📏 [ChunkBatchCache] 探测 VLM 形状 "
                f"({self.num_chunks_total} chunks, all episodes) ..."
            )
        num_layers_all: set = set()
        hidden_dim_all: set = set()
        dtype_all: set = set()
        max_seq_len_global = 0
        for cidx in self.all_chunk_indices:
            npz_path = self._chunks[cidx]["npz_path"]
            shapes = _read_npz_shapes_only(npz_path)
            if not shapes:
                raise ValueError(f"chunk file is empty: {npz_path}")
            for ep_key, (shape, dtype) in shapes.items():
                if len(shape) == 4:
                    _, n_l, s_l, h_d = shape
                elif len(shape) == 3:
                    _, s_l, h_d = shape
                    n_l = 1
                else:
                    raise ValueError(
                        f"Unexpected VLM array ndim={len(shape)} "
                        f"shape={shape} in {npz_path.name}::{ep_key}"
                    )
                num_layers_all.add(int(n_l))
                hidden_dim_all.add(int(h_d))
                dtype_all.add(dtype)
                if int(s_l) > max_seq_len_global:
                    max_seq_len_global = int(s_l)
        if len(num_layers_all) != 1:
            raise ValueError(f"chunk 之间 num_layers 不一致: {sorted(num_layers_all)}")
        if len(hidden_dim_all) != 1:
            raise ValueError(f"chunk 之间 hidden_dim 不一致: {sorted(hidden_dim_all)}")
        if len(dtype_all) != 1:
            raise ValueError(f"chunk 之间 on-disk dtype 不一致: {dtype_all}")

        self.num_layers = int(next(iter(num_layers_all)))
        self.max_seq_len_global = int(max_seq_len_global)
        self.hidden_dim = int(next(iter(hidden_dim_all)))
        self._on_disk_dtype = next(iter(dtype_all))

        # ---------- 估算每个 chunk 的 RAM 占用 ----------
        # 每帧字节数 = num_layers × max_seq_len × hidden_dim × dtype_size
        bytes_per_frame_in_cache = (
            self.num_layers
            * self.max_seq_len_global
            * self.hidden_dim
            * np.dtype(self.cache_np_dtype).itemsize
        )
        for entry in self._chunks.values():
            ram = int(entry["total_frames"] * bytes_per_frame_in_cache * self.ram_inflation_factor)
            entry["estimated_ram_bytes"] = ram

        # ---------- 计算每批最多塞多少 chunk ----------
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
            avg_ram = sum(e["estimated_ram_bytes"] for e in self._chunks.values()) // max(
                1, len(self._chunks)
            )
            est_chunks = max(1, self.budget_bytes // max(1, avg_ram))
            if self.max_chunks_per_batch is not None:
                est_chunks = min(est_chunks, int(self.max_chunks_per_batch))
            est_chunks = max(est_chunks, self.min_chunks_per_batch)
            est_chunks = min(est_chunks, self.num_chunks_total)
            self.chunks_per_batch = int(est_chunks)

        # ---------- shuffle + 打包成 batch ----------
        self.current_epoch = -1
        self.shuffled_chunk_indices: List[int] = list(self.all_chunk_indices)
        self.batches: List[List[int]] = []
        self._shuffle_and_pack(self.base_seed)  # epoch 0 起始打包

        # ---------- 状态 ----------
        self.current_batch_idx: int = 0
        self.current_shared_cache: Optional[RemappedSharedVLMCache] = None
        self.current_episode_indices: List[int] = []
        self.current_vlm_remap: Dict[int, int] = {}
        self.current_num_frames: int = 0
        # step-segments 模式下整训冻结 chunk 顺序，set_epoch 不再 reshuffle。
        # 由 freeze_packing() 切到 True；默认 False 兼容旧逻辑。
        self._frozen_packing: bool = False
        # 共享内存命名前缀（保证多进程不冲突）
        self._shm_prefix_base = f"cbc_{os.getpid()}_{int(time.time())}"
        # 一些日志状态
        self._first_load_done = False

        if self.verbose:
            self._print_init_summary()

    # --------------------------------------------------------------------- #
    # public properties
    # --------------------------------------------------------------------- #

    @property
    def num_batches(self) -> int:
        return len(self.batches)

    @property
    def total_chunks(self) -> int:
        return self.num_chunks_total

    @property
    def total_frames(self) -> int:
        return sum(e["total_frames"] for e in self._chunks.values())

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def set_epoch(self, epoch: int) -> None:
        """每个 epoch 开始时调用。重新 shuffle chunk 顺序并打包。

        如果调用过 freeze_packing()（step-segments 模式），则只更新 epoch 编号，
        chunk 顺序与打包保持冻结，不再重新 shuffle。
        """
        if self._frozen_packing:
            # 冻结模式：保留 chunk 顺序与 batches；不动 current_batch_idx，
            # 调用方应使用 load_segment(idx) 显式定位。
            self.current_epoch = int(epoch)
            return
        if epoch == self.current_epoch:
            self.current_batch_idx = 0
            return
        self._shuffle_and_pack(self.base_seed + int(epoch))
        self.current_epoch = int(epoch)
        self.current_batch_idx = 0

    def freeze_packing(self, seed: Optional[int] = None) -> None:
        """冻结 chunk 顺序与打包（用于 step-segments 模式）。

        调用后整训过程 segment→ChunkBatch 的映射保持不变，
        让 resume 时可以稳定回到对应 segment。

        Args:
            seed: 用于一次性 shuffle 的种子。None 时使用 base_seed（即 chunk_batch_seed）。
        """
        pack_seed = self.base_seed if seed is None else int(seed)
        self._shuffle_and_pack(pack_seed)
        self._frozen_packing = True
        # 进入冻结模式后 epoch 不再驱动 reshuffle，把状态归位为 0
        self.current_epoch = 0
        self.current_batch_idx = 0
        if self.verbose:
            print(
                f"🔒 [ChunkBatchCache] 已冻结 chunk 打包 (seed={pack_seed}): "
                f"{self.num_batches} segments, {self.chunks_per_batch} chunks/segment"
            )

    @property
    def num_segments(self) -> int:
        """step-segments 模式下的段数。等价于 num_batches，但语义更清晰。"""
        return self.num_batches

    def load_segment(
        self,
        segment_idx: int,
    ) -> Tuple[List[int], Dict[int, int], int, "RemappedSharedVLMCache"]:
        """直接加载指定 segment（== 内部 batch_idx）的数据，不依赖遍历顺序。

        与 load_next_batch 的差别：
        - load_next_batch 按 self.current_batch_idx 顺序前进；
        - load_segment 任意跳转到 segment_idx，便于 step-segments 模式按 step 边界切换。

        建议先调用 freeze_packing() 冻结 chunk 顺序，否则不同 epoch 的 segment_idx
        对应的 chunk 集合会变。
        """
        if not (0 <= int(segment_idx) < len(self.batches)):
            raise IndexError(
                f"segment_idx={segment_idx} out of range "
                f"[0, {len(self.batches)})"
            )
        self.current_batch_idx = int(segment_idx)
        return self.load_next_batch()

    def load_next_batch(
        self,
    ) -> Tuple[List[int], Dict[int, int], int, RemappedSharedVLMCache]:
        """
        加载当前批 chunk 的数据到共享内存。

        Returns:
            episode_indices: 当前批包含的 episode_idx 列表（按读取顺序）。
            vlm_index_remap: dict{global_vlm_index -> local_idx in cache}。
            num_frames: 本批总帧数。
            shared_cache: RemappedSharedVLMCache，可直接传给 LeRobotDataset(shared_vlm_cache=...)。
        """
        try:
            import torch
            import torch.distributed as dist
        except ImportError as e:
            raise RuntimeError("ChunkBatchCache.load_next_batch requires torch") from e

        # 切批前所有 rank 同步
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        batch_idx = self.current_batch_idx
        if batch_idx >= len(self.batches):
            raise IndexError(
                f"current_batch_idx={batch_idx} but num_batches={len(self.batches)}"
            )
        chunk_indices_this_batch = self.batches[batch_idx]

        # 释放上一批，并再 barrier 一次，确保上一批 SHM 真的被 OS 回收，
        # 否则紧接着的大块 create 可能因 /dev/shm 来不及释放而 ENOSPC。
        self._release_current_cache(verbose=self.verbose)
        if self.world_size > 1 and dist.is_initialized():
            dist.barrier()

        # 计算本批 episode 列表 + frame 总数 + remap
        episode_indices: List[int] = []
        episode_lengths: List[int] = []
        for cidx in chunk_indices_this_batch:
            entry = self._chunks[cidx]
            for ep in sorted(entry["episodes"]):
                episode_indices.append(int(ep))
                episode_lengths.append(int(entry["episode_lengths"][ep]))
        num_frames = int(sum(episode_lengths))

        # 全局 vlm_index 起点 → 本批 local_idx
        vlm_index_remap: Dict[int, int] = {}
        local_running = 0
        for ep_idx, ep_len in zip(episode_indices, episode_lengths):
            global_start = self._ep_global_vlm_start[ep_idx]
            for fi in range(ep_len):
                vlm_index_remap[global_start + fi] = local_running + fi
            local_running += ep_len

        # 形状（all chunks 共享 num_layers / max_seq_len / hidden_dim；逐 chunk 加载时再次验证）
        sample_shape = (self.num_layers, self.max_seq_len_global, self.hidden_dim)

        # 跨 rank 共享 SHM 名（rank 0 创建 → 广播）
        shm_prefix = f"{self._shm_prefix_base}_b{batch_idx}_e{max(0, self.current_epoch)}"

        if self.verbose:
            print(f"\n{'-' * 60}")
            print(
                f"📂 [ChunkBatchCache] 加载批 {batch_idx + 1}/{self.num_batches}  "
                f"(epoch={self.current_epoch})"
            )
            print(f"   chunks: {chunk_indices_this_batch}  共 {len(chunk_indices_this_batch)} 个")
            print(
                f"   episodes={len(episode_indices)}  frames={num_frames}  "
                f"sample_shape={sample_shape}  cache_dtype={self.cache_dtype_str}"
            )
            est = sum(self._chunks[c]["estimated_ram_bytes"] for c in chunk_indices_this_batch)
            print(f"   预计 SHM 占用 ≈ {_format_size(est)}")

        # ---------- rank 0 创建 SHM 并加载数据 ----------
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
            local_idx = 0
            shm_data = base_cache.data            # numpy view on shared memory (vectorized writes)
            shm_seq_lens = base_cache.seq_lens
            max_seq_len = self.max_seq_len_global
            num_layers_g = self.num_layers
            hidden_dim_g = self.hidden_dim
            total_frames_planned = num_frames
            bytes_per_frame_in_cache = (
                num_layers_g * max_seq_len * hidden_dim_g
                * np.dtype(self.cache_np_dtype).itemsize
            )
            # 每 N 个 episode 打印一次进度（避免完全沉默 + 不刷屏）
            progress_every_eps = 100
            for chunk_pos, cidx in enumerate(chunk_indices_this_batch):
                entry = self._chunks[cidx]
                npz_path = entry["npz_path"]
                t_chunk = time.time()
                ep_count = 0
                chunk_start_idx = local_idx
                if self.verbose:
                    print(
                        f"     ▶ chunk-{cidx:03d} ({chunk_pos+1}/{len(chunk_indices_this_batch)}) "
                        f"开始装载: {entry['total_frames']} frames / "
                        f"{len(entry['episodes'])} eps"
                        f"  (size on disk: {_format_size(entry['npz_size'])})",
                        flush=True,
                    )
                # 使用 mmap 零拷贝读 npz；fallback 走 np.load（兼容 compressed npz）
                try:
                    data_ctx = _NpzMmapReader(npz_path)
                    use_mmap_path = True
                except Exception as exc:
                    if self.verbose:
                        print(
                            f"     ⚠️ _NpzMmapReader 失败 ({exc.__class__.__name__}: "
                            f"{exc})，回退 np.load",
                            flush=True,
                        )
                    data_ctx = np.load(npz_path, allow_pickle=False)
                    use_mmap_path = False
                with data_ctx as data:
                    avail_keys = set(data.files)
                    sorted_eps = sorted(entry["episodes"])
                    total_eps_in_chunk = len(sorted_eps)
                    for ep_idx in sorted_eps:
                        ep_key = f"episode_{ep_idx:06d}"
                        if ep_key not in avail_keys:
                            raise KeyError(
                                f"Episode key {ep_key} not in {npz_path.name}"
                            )
                        # mmap 路径下 ep_arr 是 read-only view（零拷贝）；
                        # np.load 路径下 ep_arr 是独立内存（一次解压拷贝）。
                        ep_arr = data[ep_key]
                        if ep_arr.ndim == 3:
                            # (frames, seq_len, hidden_dim) → (frames, 1, seq_len, hidden_dim)
                            ep_arr = ep_arr[:, None, :, :]
                        elif ep_arr.ndim != 4:
                            raise ValueError(
                                f"Unexpected ndim={ep_arr.ndim} for {ep_key} in "
                                f"{npz_path.name}"
                            )
                        n_f, n_l, s_l, h_d = ep_arr.shape
                        if (n_l, h_d) != (num_layers_g, hidden_dim_g):
                            raise ValueError(
                                f"VLM shape mismatch in {ep_key} of {npz_path.name}: "
                                f"got ({n_l},{s_l},{h_d}) expected "
                                f"({num_layers_g},*,{hidden_dim_g})"
                            )
                        if s_l > max_seq_len:
                            raise ValueError(
                                f"seq_len {s_l} > probed max_seq_len_global "
                                f"{max_seq_len} in {ep_key} of "
                                f"{npz_path.name}; 请用更大的 sample 探测形状或重做扫描"
                            )
                        # ---------- vectorized batch write (whole episode at once) ----------
                        end_idx = local_idx + n_f
                        if end_idx > total_frames_planned:
                            raise RuntimeError(
                                f"Frame overflow in {ep_key}: end_idx={end_idx} "
                                f"> total_frames_planned={total_frames_planned}"
                            )
                        ep_arr_cast = ep_arr.astype(self.cache_np_dtype, copy=False)
                        if s_l == max_seq_len:
                            # 全长，一次性 memcpy 整个 episode
                            shm_data[local_idx:end_idx] = ep_arr_cast
                        else:
                            # 只写有效部分；padded 区域 [..., s_l:, :] 保持 SHM 默认 0。
                            # SharedVLMCache.create 走 shm_open+ftruncate，POSIX 保证新页 zero-filled，
                            # 且下游 get_sample 只读 [:, :actual_seq_len, :]，从不访问 padded 区。
                            # 去掉冗余 zero-fill 后单帧 RAM 写入量从 max_seq_len 降到 s_l (~2x 提速)。
                            shm_data[local_idx:end_idx, :, :s_l, :] = ep_arr_cast
                        # 一次性写入这一段所有 seq_len（避免循环）
                        shm_seq_lens[local_idx:end_idx] = s_l
                        local_idx = end_idx
                        ep_count += 1

                        # 实时进度（避免长时间无输出让用户以为卡死）
                        if self.verbose and ep_count % progress_every_eps == 0:
                            dt_sofar = time.time() - t_chunk
                            done_bytes_chunk = (local_idx - chunk_start_idx) * bytes_per_frame_in_cache
                            gibps = done_bytes_chunk / 1024**3 / max(dt_sofar, 1e-3)
                            pct = 100.0 * ep_count / total_eps_in_chunk
                            print(
                                f"        chunk-{cidx:03d}  {ep_count}/{total_eps_in_chunk} eps "
                                f"({pct:.1f}%)  {dt_sofar:.0f}s  ≈ {gibps:.2f} GiB/s"
                                + ("  [mmap]" if use_mmap_path else "  [np.load]"),
                                flush=True,
                            )
                    # 清掉对 mmap 的强引用，让 with 退出时 mmap.close() 不抛 BufferError
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
                    gibps = (entry["total_frames"]
                             * num_layers_g * max_seq_len * hidden_dim_g
                             * np.dtype(self.cache_np_dtype).itemsize) / 1024**3 / max(dt, 1e-3)
                    print(
                        f"     ✓ chunk-{cidx:03d}  {entry['total_frames']} frames "
                        f"({ep_count} eps)  载入完成 {dt:.1f}s  ≈ {gibps:.2f} GiB/s",
                        flush=True,
                    )
            if local_idx != num_frames:
                raise RuntimeError(
                    f"Loaded {local_idx} frames but expected {num_frames}; "
                    f"meta/episodes.jsonl 与 npz 数据不一致？"
                )
            cache_name = base_cache.name
            seq_lens_name = base_cache.seq_lens_name
            if self.verbose:
                dt_total = time.time() - t0
                gibps = _bytes_to_gib(
                    int(num_frames)
                    * self.num_layers
                    * self.max_seq_len_global
                    * self.hidden_dim
                    * np.dtype(self.cache_np_dtype).itemsize
                ) / max(dt_total, 1e-3)
                print(
                    f"   ✓ rank 0 装载完成: {dt_total:.1f}s  "
                    f"≈ {gibps:.2f} GiB/s (有效写入)"
                )

        # ---------- 广播 SHM 名 → 其他 rank attach ----------
        if self.world_size > 1 and dist.is_initialized():
            try:
                local_rank = int(os.environ.get("LOCAL_RANK", self.rank % max(1, torch.cuda.device_count())))
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
                cache_name = bytes(c for c in name_tensor.cpu().tolist() if c != 0).decode("ascii")
                seq_lens_name = bytes(c for c in seqn_tensor.cpu().tolist() if c != 0).decode("ascii")
                base_cache = SharedVLMCache.attach(
                    name=cache_name,
                    seq_lens_name=seq_lens_name,
                    num_samples=num_frames,
                    sample_shape=sample_shape,
                    dtype=self.cache_np_dtype,
                )

        if base_cache is None:
            raise RuntimeError("Failed to obtain SharedVLMCache after broadcast")

        # 包成 RemappedSharedVLMCache
        shared_cache = RemappedSharedVLMCache(
            base_cache=base_cache,
            vlm_index_remap=vlm_index_remap,
            total_frames=self.total_frames_dataset,
        )

        # 记录当前状态
        self.current_shared_cache = shared_cache
        self.current_episode_indices = episode_indices
        self.current_vlm_remap = vlm_index_remap
        self.current_num_frames = num_frames

        # 准备下一批
        self.current_batch_idx = batch_idx + 1
        self._first_load_done = True

        return episode_indices, vlm_index_remap, num_frames, shared_cache

    def cleanup(self) -> None:
        """训练结束时调用，释放 SHM。"""
        self._release_current_cache(verbose=self.verbose)

    # --------------------------------------------------------------------- #
    # internal helpers
    # --------------------------------------------------------------------- #

    def _shuffle_and_pack(self, seed: int) -> None:
        rng = random.Random(int(seed))
        chunk_list = list(self.all_chunk_indices)
        rng.shuffle(chunk_list)
        self.shuffled_chunk_indices = chunk_list

        # 简单按 chunks_per_batch 切片
        cpb = self.chunks_per_batch
        if cpb <= 0:
            cpb = 1
        batches: List[List[int]] = []
        for i in range(0, len(chunk_list), cpb):
            batches.append(chunk_list[i : i + cpb])
        self.batches = batches

    def _release_current_cache(self, verbose: bool = False) -> None:
        if self.current_shared_cache is None:
            return
        try:
            if verbose:
                print(f"   ♻️  释放上一批 SHM: {self.current_shared_cache.name}")
            if self.rank == 0:
                # 创建者负责 unlink；其他 rank 只 close
                self.current_shared_cache.close()
                self.current_shared_cache.unlink()
            else:
                self.current_shared_cache.close()
        except Exception as e:  # pragma: no cover
            print(f"⚠️ ChunkBatchCache 释放 SHM 时出错（已忽略）: {e}")
        self.current_shared_cache = None
        self.current_episode_indices = []
        self.current_vlm_remap = {}
        self.current_num_frames = 0

    def _print_init_summary(self) -> None:
        ram_gib = _bytes_to_gib(_ram_available_bytes())
        shm_gib = _bytes_to_gib(_shm_available_bytes())
        budget_gib = _bytes_to_gib(self.budget_bytes) if self.budget_bytes > 0 else -1.0
        avg_ram_gib = _bytes_to_gib(
            sum(e["estimated_ram_bytes"] for e in self._chunks.values())
            // max(1, len(self._chunks))
        )
        print(f"📦 [ChunkBatchCache] 初始化完成")
        print(f"   dataset_path:        {self.dataset_path}")
        print(f"   chunks_size (info):  {self.chunks_size}")
        print(f"   total_episodes:      {self.total_episodes_dataset}")
        print(f"   total_frames:        {self.total_frames_dataset}")
        print(f"   selected chunks:     {self.num_chunks_total}")
        print(f"   selected episodes:   {sum(len(e['episodes']) for e in self._chunks.values())}")
        print(f"   selected frames:     {self.total_frames}")
        print(f"   VLM shape per frame: ({self.num_layers}, {self.max_seq_len_global}, {self.hidden_dim})")
        print(f"   on-disk dtype:       {self._on_disk_dtype}")
        print(f"   cache dtype:         {self.cache_dtype_str}  ({np.dtype(self.cache_np_dtype).itemsize} B / scalar)")
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
