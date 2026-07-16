"""
DiskVLMCache —— 不预加载到共享内存，训练时按需 mmap chunk-XXX.npz 读单帧。

设计动机
========
ChunkBatchCache / MultiChunkBatchCache 的"分段进 SHM"方案有一个固有缺陷：
  shuffle 半径 ≤ 一个 segment 的样本数。
切段时数据分布瞬间漂移，loss 会出现突刺；并且段内的样本不会跨段混合，
相当于"短时间内只看到这部分子集"。

本模块提供另一条数据通路：
  1. 完全不创建共享内存；
  2. 所有 ep 一次性激活，DataLoader 看到的就是**全局 sample 池**；
  3. DistributedSampler 在整个池子上 shuffle —— shuffle 半径 = 全部样本；
  4. __getitem__ 时按 vlm_index_local 路由到对应的 chunk-XXX.npz，
     用 _NpzMmapReader 做零拷贝 mmap 读单帧（OS page cache 自然做热点缓存）。

关键性能点
==========
- 用 chunk_batch_cache._NpzMmapReader, npz 必须是 np.savez (ZIP_STORED) 的，
  压缩的 npz 会回退到 np.load (代价是会一次性解压整个 chunk)；
- 每个 DataLoader worker 进程持有一个 _DiskVLMReader 实例
  (主进程 fork 出 worker 时 metadata COW 共享，mmap 句柄按需 lazy 打开)；
- 每个 worker 内部维护一个 LRU dict, 存最近用过的 _NpzMmapReader，
  超出 lru_size 时驱逐最旧的, 自动 close() 释放 mmap；
- 由于 mmap 只占虚拟地址, RSS 由 OS page cache 自然控制, 不会 OOM。

与现有训练代码的兼容
====================
DiskVLMCacheView 鸭子类型对齐 SubsetSharedVLMCacheView, 也即:
  .get_sample(vlm_index_local) -> np.ndarray
  .num_samples / .sample_shape / .dtype / .name ...
这样 LeRobotDataset._load_vlm_hidden_state 那条 shared_vlm_cache 分支零修改，
MultiLeRobotDataset.set_active_episodes 也零修改。

外部主流程入口:
  DiskVLMCacheManager.attach_to_index(multi_dataset_index, ...)
    -> {sub_idx: DiskVLMCacheView}   (传给 MultiLeRobotDataset.set_active_episodes)
"""

from __future__ import annotations

import mmap as _mmap_mod
import os
import re
import struct
import threading
import time
import warnings
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.format import (
    read_array_header_1_0,
    read_array_header_2_0,
    read_magic,
)

try:
    from .multi_dataset_index import GlobalChunk, MultiDatasetIndex, SubsetInfo
except ImportError:
    from multi_dataset_index import (  # type: ignore[no-redef]
        GlobalChunk,
        MultiDatasetIndex,
        SubsetInfo,
    )


_ZIP_LOCAL_HEADER_SIG = 0x04034B50


def _detect_disk_type(path: Path) -> Tuple[str, str]:
    """
    探测 path 所在块设备的类型 (HDD / SSD / 未知)。

    返回 (kind, description)。kind ∈ {"hdd", "ssd", "unknown"}。
    description 包含设备名等便于打印。

    检测思路 (仅 Linux 有效):
      1. os.stat(path).st_dev → (major, minor)
      2. 反查 /sys/dev/block/<major>:<minor>/queue/rotational
      3. = "1" → HDD, "0" → SSD/NVMe
    任一步失败返回 ("unknown", "...").
    """
    try:
        path = Path(path).resolve()
        st = os.stat(path)
        major = os.major(st.st_dev)
        minor = os.minor(st.st_dev)
        rota_path = Path(f"/sys/dev/block/{major}:{minor}/queue/rotational")
        if not rota_path.exists():
            # 走到所属父块设备 (分区 → 整盘)
            partition_dev = Path(f"/sys/dev/block/{major}:{minor}")
            if partition_dev.exists():
                slaves_path = partition_dev / ".."
                # 拼出 /sys/block/<diskname>/queue/rotational
                try:
                    dev_name = (partition_dev / "uevent").read_text().split()
                    for tok in dev_name:
                        if tok.startswith("DEVNAME="):
                            disk_name = tok.split("=", 1)[1]
                            # 去掉分区编号: nvme0n1p1 → nvme0n1; sda1 → sda
                            disk_root = disk_name.rstrip("0123456789")
                            if disk_root.endswith("p"):
                                disk_root = disk_root[:-1]
                            candidate = Path(f"/sys/block/{disk_root}/queue/rotational")
                            if candidate.exists():
                                rota_path = candidate
                                break
                except Exception:
                    pass
        if rota_path.exists():
            rota = rota_path.read_text().strip()
            if rota == "1":
                return "hdd", f"{rota_path} = 1"
            if rota == "0":
                return "ssd", f"{rota_path} = 0"
        return "unknown", f"无法读取 rotational ({path})"
    except Exception as exc:
        return "unknown", f"探测失败: {type(exc).__name__}: {exc}"


class NpzEntryUnreadable(Exception):
    """
    某个 npz entry 既无法 mmap-read, 也无法 np.load-fallback 时抛出。
    调用方应捕获这个异常并返回 zero placeholder, **不要**让它向上传播,
    否则 DataLoader worker 会因为这个崩溃 → 训练中断。
    """

    def __init__(
        self,
        path: str,
        key: str,
        reason: str,
        shape_hint: Optional[Tuple[int, ...]] = None,
        dtype_hint: Optional[np.dtype] = None,
    ) -> None:
        super().__init__(f"NpzEntryUnreadable({path}::{key}): {reason}")
        self.path = path
        self.key = key
        self.reason = reason
        self.shape_hint = shape_hint
        self.dtype_hint = dtype_hint


class _LazyNpzMmapReader:
    """
    懒解析的 npz mmap reader。

    与 chunk_batch_cache._NpzMmapReader 的区别:
      - __init__ 只读 ZIP 中央目录, 建立 {key: ZipInfo} 映射, **不**预扫每个
        entry 的 local header / .npy header；
      - __getitem__(key) 触发对**该 entry** 的 local header + .npy header
        解析 (一次小 seek+read), 然后返回 np.frombuffer 视图到 mmap。
      - 对单 chunk 1000 entries 的场景, init 时间从 ~5s 降到 ~10-50ms；
        单 sample 读取多一次小 seek+read 开销 (~50us), 可忽略。

    限制: 与 _NpzMmapReader 一样, 要求所有 entries 是 ZIP_STORED (np.savez 而非
    _compressed)。压缩 entry 会在 __getitem__ 抛 ValueError。
    """

    def __init__(self, path: os.PathLike) -> None:
        self._path = str(path)
        self._fobj = None
        self._mmap = None
        # key -> ZipInfo (用来定位 local header offset)
        self._zinfos: Dict[str, zipfile.ZipInfo] = {}
        # 已解析过的 key 缓存: key -> (data_offset, np.dtype, shape, fortran_order)
        self._parsed: Dict[str, Tuple[int, np.dtype, Tuple[int, ...], bool]] = {}
        self._parse_lock = threading.Lock()
        # 若 mmap 路径在该 chunk 上失败过一次, 整个 chunk 切到 np.load 路径
        # (慢但绝对正确). 用 NpzFile 句柄复用避免每次重新打 zip 中央目录.
        self._fallback_npz: Optional[Any] = None
        self._fallback_count: int = 0
        self._open()

    def _open(self) -> None:
        f = open(self._path, "rb")
        self._fobj = f
        try:
            # zipfile 只读 central dir, 对 23GB npz 也是几 ms 内完成
            with zipfile.ZipFile(f, mode="r") as zf:
                for zinfo in zf.infolist():
                    if not zinfo.filename.endswith(".npy"):
                        continue
                    key = zinfo.filename[:-4]
                    self._zinfos[key] = zinfo
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_RANDOM)
            except (AttributeError, OSError):
                pass
            self._mmap = _mmap_mod.mmap(f.fileno(), 0, prot=_mmap_mod.PROT_READ)
            try:
                self._mmap.madvise(_mmap_mod.MADV_RANDOM)
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
        return list(self._zinfos.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._zinfos

    def _parse_entry(self, key: str) -> Tuple[int, np.dtype, Tuple[int, ...], bool]:
        cached = self._parsed.get(key)
        if cached is not None:
            return cached
        with self._parse_lock:
            cached = self._parsed.get(key)
            if cached is not None:
                return cached
            zinfo = self._zinfos[key]
            if zinfo.compress_type != zipfile.ZIP_STORED:
                raise ValueError(
                    f"_LazyNpzMmapReader: entry '{zinfo.filename}' in "
                    f"{self._path} is compressed (must be np.savez 非 _compressed)"
                )
            assert self._fobj is not None
            f = self._fobj
            f.seek(zinfo.header_offset)
            hdr = f.read(30)
            if len(hdr) < 30:
                raise ValueError(
                    f"_LazyNpzMmapReader: bad ZIP local header at "
                    f"{zinfo.header_offset} in {self._path}"
                )
            sig = struct.unpack("<I", hdr[0:4])[0]
            if sig != _ZIP_LOCAL_HEADER_SIG:
                raise ValueError(
                    f"_LazyNpzMmapReader: bad ZIP signature 0x{sig:08x} "
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
                    f"_LazyNpzMmapReader: unsupported .npy version {version} "
                    f"in {self._path}::{zinfo.filename}"
                )
            arr_off = f.tell()
            cached = (arr_off, np.dtype(dtype), tuple(int(d) for d in shape), bool(fortran_order))
            self._parsed[key] = cached
            return cached

    def _switch_to_fallback(self, reason: str) -> Any:
        """触发 fallback: 该 chunk 永久切到 np.load 路径并缓存 NpzFile 句柄。"""
        if self._fallback_npz is None:
            warnings.warn(
                f"[_LazyNpzMmapReader] FALLBACK chunk -> np.load: {self._path} "
                f"reason: {reason}",
                RuntimeWarning,
                stacklevel=3,
            )
            self._fallback_npz = np.load(self._path, allow_pickle=False)
        self._fallback_count += 1
        return self._fallback_npz

    def __getitem__(self, key: str) -> np.ndarray:
        """
        正常路径返回 mmap view; 损坏时抛 NpzEntryUnreadable, 由调用方决定
        是返回 zero placeholder 还是 raise。__getitem__ 自己**不**生成 placeholder
        (因为只有调用方知道 frame_in_ep, 才能正确返回单帧形状)。
        """
        if key not in self._zinfos:
            raise KeyError(key)
        # 已经切到 fallback 路径 -> 直接走 np.load (NpzFile 已缓存)
        if self._fallback_npz is not None:
            try:
                return np.asarray(self._fallback_npz[key])
            except Exception as exc:
                raise NpzEntryUnreadable(
                    path=self._path,
                    key=key,
                    reason=f"np.load fallback failed: {type(exc).__name__}: {exc}",
                ) from exc
        if self._mmap is None:
            raise RuntimeError(f"_LazyNpzMmapReader already closed: {self._path}")
        offset, dtype, shape, fortran_order = self._parse_entry(key)
        nelem = 1
        for d in shape:
            nelem *= int(d)
        need_bytes = int(nelem) * int(dtype.itemsize)
        mmap_size = len(self._mmap)
        if offset < 0 or offset + need_bytes > mmap_size:
            try:
                npz = self._switch_to_fallback(
                    f"entry {key}: offset={offset}+need={need_bytes} > "
                    f"mmap_size={mmap_size} (shape={shape}, dtype={dtype})"
                )
                return np.asarray(npz[key])
            except Exception as exc:
                raise NpzEntryUnreadable(
                    path=self._path,
                    key=key,
                    reason=f"mmap-OOB + np.load failed: "
                    f"{type(exc).__name__}: {exc}",
                    shape_hint=shape,
                    dtype_hint=dtype,
                ) from exc
        try:
            arr = np.frombuffer(self._mmap, dtype=dtype, count=nelem, offset=offset)
        except ValueError as ve:
            # 极少数情况边界看似 OK 但仍报 buffer size 错; 同样 fallback
            try:
                npz = self._switch_to_fallback(
                    f"entry {key}: np.frombuffer raised despite bounds-OK: {ve}"
                )
                return np.asarray(npz[key])
            except Exception as exc:
                raise NpzEntryUnreadable(
                    path=self._path,
                    key=key,
                    reason=f"frombuffer + np.load fallback failed: "
                    f"{type(exc).__name__}: {exc}",
                    shape_hint=shape,
                    dtype_hint=dtype,
                ) from exc
        if fortran_order:
            return arr.reshape(shape, order="F")
        return arr.reshape(shape)

    def close(self) -> None:
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BufferError:
                pass
            self._mmap = None
        if self._fobj is not None:
            try:
                self._fobj.close()
            except OSError:
                pass
            self._fobj = None
        if self._fallback_npz is not None:
            try:
                self._fallback_npz.close()
            except Exception:
                pass
            self._fallback_npz = None

    def __enter__(self) -> "_LazyNpzMmapReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


_EP_KEY_RE = re.compile(r"^episode_(\d+)$")


def _list_npz_episode_keys(npz_path: Path) -> set:
    """
    用 zipfile 仅读 central directory 列出 npz 里实际存在的 episode_xxx keys。
    比 _read_npz_shapes_only 更轻 (不读 .npy header), 用于 init 阶段建立
    "每个 chunk 实际落地的 episode_index 集合"。

    单 chunk 通常 ~10ms 冷盘 / ~1ms 热盘。
    """
    eps: set = set()
    try:
        with zipfile.ZipFile(npz_path) as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue
                key = name[:-4]
                m = _EP_KEY_RE.match(key)
                if m is not None:
                    eps.add(int(m.group(1)))
    except (zipfile.BadZipFile, OSError):
        pass
    return eps


# ============================================================================
# 子集级 metadata: vlm_index_local -> (chunk_idx, ep_idx, frame_in_ep)
# ============================================================================
class _SubsetVLMIndexMap:
    """
    单个子集内的「vlm_index_local 路由表」。

    构造时拿到的输入只是子集 meta (ep_local_vlm_start / episode_lengths /
    chunks_size), 不读任何 npz, 全部在 RAM 计算。本结构是 read-only, 主进程
    构造完之后 fork 给 worker 进程时是 COW 共享, 占用极小。
    """

    def __init__(
        self,
        subset: SubsetInfo,
        npz_paths: Dict[int, Path],
        precise_scan: bool = True,
    ) -> None:
        self.sub_idx: int = int(subset.sub_idx)
        self.name: str = subset.name
        self.chunks_size: int = int(subset.chunks_size)

        # ----- 1. 建立 chunk_idx -> 该 chunk 内**实际**存在的 episode 索引集合
        # ChunkBatchCache 在 __init__ 时也会做这一步 (用 _read_npz_shapes_only,
        # 顺便探测 shape)。我们这里只需要 episode keys, 用更轻的
        # zipfile.namelist() 读 central directory, 单 chunk ~10ms 冷盘。
        # precise_scan=False 时退化为"只按 chunk 是否存在过滤" (用于纯单元测试)。
        self._npz_paths: Dict[int, Path] = dict(npz_paths)
        chunk_to_actual_eps: Dict[int, set] = {}
        if precise_scan:
            for cidx, p in self._npz_paths.items():
                chunk_to_actual_eps[cidx] = _list_npz_episode_keys(p)
        else:
            for cidx in self._npz_paths.keys():
                chunk_to_actual_eps[cidx] = set()  # 不过滤

        # ----- 2. 用上述结果精确算出 "可读" 的 episode 集合
        # subset.episode_lengths 含所有 declared episode, 但只有
        #   (ep // chunks_size) in available_chunks  且  ep in chunk_actual_eps
        # 才算真正能读 (chunk 中真有这个 ep)
        eps_all = sorted(subset.ep_local_vlm_start.items(), key=lambda kv: kv[1])
        if not eps_all:
            raise ValueError(f"[DiskVLMCache] subset {subset.name} 没有 ep_local_vlm_start")
        available_chunk_indices = set(self._npz_paths.keys())
        available_eps: List[Tuple[int, int]] = []
        for ep_idx, vlm_start in eps_all:
            cidx = int(ep_idx) // self.chunks_size
            if cidx not in available_chunk_indices:
                continue
            actual_eps = chunk_to_actual_eps.get(cidx)
            if precise_scan and actual_eps is not None and int(ep_idx) not in actual_eps:
                # chunk 落地但 ep 缺失 (部分写入); 跳过
                continue
            available_eps.append((int(ep_idx), int(vlm_start)))

        if not available_eps:
            raise ValueError(
                f"[DiskVLMCache] subset {subset.name}: 没有任何 ep 在已落地的 "
                f"chunk-XXX.npz 中可读 (declared eps={len(eps_all)}, "
                f"available chunks={len(self._npz_paths)})"
            )

        # ----- 3. 仅用 available_eps 建立查询数组 (vlm_starts / ep_for_position /
        # ep_lengths), 让 searchsorted 永远路由到可读的 ep
        self._vlm_starts = np.asarray([vs for _, vs in available_eps], dtype=np.int64)
        self._ep_for_position = np.asarray([e for e, _ in available_eps], dtype=np.int64)
        self._ep_lengths = np.asarray(
            [int(subset.episode_lengths[e]) for e, _ in available_eps], dtype=np.int64
        )
        self._available_eps: set = set(int(e) for e, _ in available_eps)

        # num_samples = 最后一个 available ep 的 vlm_start + 它的长度
        # (这个上限恰好覆盖所有可读的 vlm_index_local, LeRobotDataset 的
        # `if vlm_index < shared_vlm_cache.num_samples` 守卫由此变得严格)
        last_start = int(self._vlm_starts[-1])
        last_len = int(self._ep_lengths[-1])
        self.num_samples: int = last_start + last_len

        # 调试统计 (供 manager 汇总)
        self.total_available_eps: int = len(available_eps)
        self.total_available_frames: int = int(self._ep_lengths.sum())

    def get_npz_path(self, chunk_idx: int) -> Optional[Path]:
        return self._npz_paths.get(int(chunk_idx))

    def available_chunks(self) -> List[int]:
        return sorted(self._npz_paths.keys())

    def locate(self, vlm_index_local: int) -> Tuple[int, int, int]:
        """
        把子集内 vlm_index_local 还原成 (chunk_idx, ep_idx, frame_in_ep)。
        返回的 ep_idx 是子集内 episode_index。
        """
        q = int(vlm_index_local)
        if q < 0:
            raise IndexError(f"vlm_index_local {q} < 0")
        # bisect_right(starts, q) - 1
        pos = int(np.searchsorted(self._vlm_starts, q, side="right")) - 1
        if pos < 0:
            raise IndexError(
                f"vlm_index_local {q} < 第一个 episode 的 vlm_start "
                f"{int(self._vlm_starts[0])} (subset={self.name})"
            )
        ep_start = int(self._vlm_starts[pos])
        ep_len = int(self._ep_lengths[pos])
        frame_in_ep = q - ep_start
        if frame_in_ep >= ep_len:
            # 该 vlm_index_local 超出了「最近一个 episode」的合法范围 →
            # 说明它落在某两个 ep 之间的间隙里 (理论上不该发生)
            raise IndexError(
                f"vlm_index_local {q} not within episode (subset={self.name}, "
                f"ep_start={ep_start}, ep_len={ep_len})"
            )
        ep_idx = int(self._ep_for_position[pos])
        chunk_idx = ep_idx // self.chunks_size
        return chunk_idx, ep_idx, frame_in_ep

    def is_ep_available(self, ep_idx: int) -> bool:
        return int(ep_idx) in self._available_eps


# ============================================================================
# 单 worker 内的 npz mmap LRU 池
# ============================================================================
class _DiskVLMReader:
    """
    每个 DataLoader worker 进程独立持有一个 _DiskVLMReader 实例。

    负责:
      1. 把子集级 metadata (_SubsetVLMIndexMap) 拿来做 vlm_index_local 路由
      2. 维护一个 LRU dict {chunk_idx: _NpzMmapReader}, 满了驱逐最早的
      3. 读单帧时返回**拷贝** (与 SharedVLMCache.get_sample 行为一致, 保证
         np.frombuffer 视图不会跨 DataLoader 进程边界传递)

    线程安全:
      DataLoader worker 默认是单线程的 (worker process), 这里加锁只是为了
      防止用户万一在 num_workers=0 + 多线程模型预取里同时调到; 锁开销可忽略。
    """

    def __init__(
        self,
        index_map: _SubsetVLMIndexMap,
        lru_size: int = -1,
        verbose_first_open: bool = False,
    ) -> None:
        self._idx_map = index_map
        # lru_size <= 0 表示不限 (每个 worker 持有该 subset 的所有 chunk mmap)
        self._lru_size = int(lru_size) if int(lru_size) > 0 else -1
        self._readers: "OrderedDict[int, _LazyNpzMmapReader]" = OrderedDict()
        self._lock = threading.Lock()
        self._pid_when_created = os.getpid()
        self._verbose_first_open = bool(verbose_first_open)
        # 从第一次成功读取的 sample 学到单帧 shape/dtype, 用于损坏时 zero placeholder
        self._sample_shape_dtype: Optional[Tuple[Tuple[int, ...], np.dtype]] = None
        # 报错过的 (chunk, ep) 集合, 同样的坏 entry 不重复 spam warning
        self._reported_bad: set = set()
        self._first_open_done = False

    def get_sample(self, vlm_index_local: int) -> np.ndarray:
        chunk_idx, ep_idx, frame_in_ep = self._idx_map.locate(int(vlm_index_local))
        reader = self._get_reader(chunk_idx)
        ep_key = f"episode_{ep_idx:06d}"
        try:
            # _LazyNpzMmapReader.__getitem__ 返回 np.frombuffer 视图, 形状可能是
            # (n_f, n_l, s_l, h_d) 或 (n_f, s_l, h_d) (无 layer 维)。
            ep_arr = reader[ep_key]
        except NpzEntryUnreadable as exc:
            # 该 entry 损坏到连 np.load 都解不出 -> 返回 zero placeholder
            return self._zero_placeholder_for_bad_entry(chunk_idx, ep_idx, exc)
        try:
            if ep_arr.ndim == 4:
                frame = ep_arr[frame_in_ep]                # (n_l, s_l, h_d)
            elif ep_arr.ndim == 3:
                frame = ep_arr[frame_in_ep][None, :, :]    # (1, s_l, h_d)
            else:
                raise ValueError(
                    f"[DiskVLMCache] unexpected ndim={ep_arr.ndim} in "
                    f"{self._idx_map.name} chunk-{chunk_idx:03d} {ep_key}"
                )
        except IndexError as ie:
            # ep_arr.shape[0] 比 metadata 声称的 ep_len 小: 某些 frame 数据缺失
            self._note_bad_entry(
                chunk_idx,
                ep_idx,
                f"frame_in_ep={frame_in_ep} >= ep_arr.shape[0]={ep_arr.shape[0]}: {ie}",
            )
            return self._zero_placeholder_inferred(ep_arr.dtype, ep_arr.shape)
        out = np.ascontiguousarray(frame)
        # 记录单帧 shape/dtype, 给 placeholder 用
        if self._sample_shape_dtype is None:
            self._sample_shape_dtype = (tuple(out.shape), out.dtype)
        return out

    # ------------------ zero-placeholder helpers ------------------ #
    def _note_bad_entry(self, chunk_idx: int, ep_idx: int, reason: str) -> None:
        bad_key = (int(chunk_idx), int(ep_idx))
        if bad_key in self._reported_bad:
            return
        self._reported_bad.add(bad_key)
        npz_path = self._idx_map.get_npz_path(chunk_idx)
        warnings.warn(
            f"[DiskVLMCache] ⚠️  BAD ENTRY -> zero placeholder: "
            f"subset={self._idx_map.name} chunk-{chunk_idx:03d} "
            f"episode_{ep_idx:06d}  ({npz_path})\n"
            f"    reason: {reason}",
            RuntimeWarning,
            stacklevel=3,
        )

    def _zero_placeholder_for_bad_entry(
        self,
        chunk_idx: int,
        ep_idx: int,
        exc: "NpzEntryUnreadable",
    ) -> np.ndarray:
        self._note_bad_entry(chunk_idx, ep_idx, exc.reason)
        if self._sample_shape_dtype is not None:
            shape, dtype = self._sample_shape_dtype
            return np.zeros(shape, dtype=dtype)
        # 启动期就坏 (还没有 sample 成功过), 给一个保守猜测
        # 训练侧 collate_fn 会做 dtype cast, 这里 dtype 不重要
        if exc.dtype_hint is not None and exc.shape_hint is not None:
            # ep array shape -> 单帧 shape: 去掉第 0 维 + 加 layer 维 (若需要)
            try:
                inferred = exc.shape_hint[1:]
                if len(inferred) == 2:
                    inferred = (1,) + inferred  # 补 layer 维
                return np.zeros(inferred, dtype=exc.dtype_hint)
            except Exception:
                pass
        return np.zeros((1, 80, 2048), dtype=np.float32)

    def _zero_placeholder_inferred(
        self,
        dtype: np.dtype,
        ep_shape: Tuple[int, ...],
    ) -> np.ndarray:
        if self._sample_shape_dtype is not None:
            shape, sample_dtype = self._sample_shape_dtype
            return np.zeros(shape, dtype=sample_dtype)
        try:
            inferred = ep_shape[1:]
            if len(inferred) == 2:
                inferred = (1,) + inferred
            return np.zeros(inferred, dtype=dtype)
        except Exception:
            return np.zeros((1, 80, 2048), dtype=np.float32)

    def _get_reader(self, chunk_idx: int) -> _LazyNpzMmapReader:
        with self._lock:
            r = self._readers.get(chunk_idx)
            if r is not None:
                self._readers.move_to_end(chunk_idx)
                return r
            npz_path = self._idx_map.get_npz_path(chunk_idx)
            if npz_path is None:
                raise KeyError(
                    f"[DiskVLMCache] chunk-{chunk_idx:03d}.npz 不存在于 subset "
                    f"{self._idx_map.name} (available={self._idx_map.available_chunks()[:8]}...)"
                )
            try:
                r = _LazyNpzMmapReader(npz_path)
            except Exception as exc:
                raise RuntimeError(
                    f"[DiskVLMCache] 打开 {npz_path} 失败: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            self._readers[chunk_idx] = r
            if self._verbose_first_open and not self._first_open_done:
                print(
                    f"  📂 [DiskVLMCache pid={os.getpid()}] 首次 mmap open: "
                    f"{npz_path.name} (subset={self._idx_map.name}, "
                    f"lru_size={'unlimited' if self._lru_size < 0 else self._lru_size})",
                    flush=True,
                )
                self._first_open_done = True
            if self._lru_size > 0:
                while len(self._readers) > self._lru_size:
                    _old_idx, old_r = self._readers.popitem(last=False)
                    try:
                        old_r.close()
                    except Exception:
                        pass
            return r

    def close_all(self) -> None:
        with self._lock:
            for r in self._readers.values():
                try:
                    r.close()
                except Exception:
                    pass
            self._readers.clear()


# ============================================================================
# DiskVLMCacheView —— SubsetSharedVLMCacheView 的鸭子替身
# ============================================================================
class DiskVLMCacheView:
    """
    暴露给 LeRobotDataset 看的 shared_vlm_cache 接口。

    每个 DataLoader worker 进程独立持有一个 _DiskVLMReader (lazy-init),
    保证 mmap 句柄不跨进程共享。fork 时 _reader_per_pid 会被 COW 拷贝, 但
    实际句柄是按 pid 重建的。
    """

    def __init__(
        self,
        index_map: _SubsetVLMIndexMap,
        lru_size: int = -1,
        verbose_first_open: bool = False,
    ) -> None:
        self._idx_map = index_map
        self._lru_size = int(lru_size)
        self._verbose_first_open = bool(verbose_first_open)
        # 按 pid 隔离: fork 出 worker 后 pid 变化, 自动新建 reader
        self._reader_per_pid: Dict[int, _DiskVLMReader] = {}
        self._reader_lock = threading.Lock()

    # ------- SharedVLMCache 鸭子接口 (LeRobotDataset 实际只会调 get_sample) -------
    @property
    def name(self) -> str:
        return f"disk_vlm[{self._idx_map.name}]"

    @property
    def seq_lens_name(self) -> str:
        return self.name + ".seqlens"

    @property
    def dtype(self):
        # 训练侧 collate_fn 用 vlm_dtype 再 cast 一次, 这里返回真正读出的 dtype
        # 之前已经探测过 disk on-disk dtype, 但单帧函数不依赖它, 留 None 即可。
        return None

    @property
    def sample_shape(self):
        return None

    @property
    def is_creator(self) -> bool:
        return False

    @property
    def seq_lens(self):
        # LeRobotDataset 不直接读 seq_lens, oft_collate_fn 是按返回 ndarray
        # 的 shape 自动拿到 seq_len 的, 留 None 不影响。
        return None

    @property
    def data(self):
        return None

    @property
    def num_samples(self) -> int:
        return self._idx_map.num_samples

    def get_sample(self, vlm_index_local: int) -> np.ndarray:
        reader = self._get_or_create_reader()
        return reader.get_sample(int(vlm_index_local))

    def _get_or_create_reader(self) -> _DiskVLMReader:
        pid = os.getpid()
        # fast-path: 当前 pid 已有 reader
        r = self._reader_per_pid.get(pid)
        if r is not None and r._pid_when_created == pid:
            return r
        with self._reader_lock:
            r = self._reader_per_pid.get(pid)
            if r is not None and r._pid_when_created == pid:
                return r
            r = _DiskVLMReader(
                index_map=self._idx_map,
                lru_size=self._lru_size,
                verbose_first_open=self._verbose_first_open,
            )
            self._reader_per_pid[pid] = r
            return r

    def close(self) -> None:
        for r in self._reader_per_pid.values():
            r.close_all()
        self._reader_per_pid.clear()

    def unlink(self) -> None:
        # 兼容 SharedVLMCache 接口, disk 模式无 SHM, 啥都不做
        pass


# ============================================================================
# DiskVLMCacheManager —— MultiChunkBatchCache 的同位替身, 不分 segment
# ============================================================================
class DiskVLMCacheManager:
    """
    多源 (or 单源) 的 disk shuffle 模式入口。

    与 MultiChunkBatchCache 的关键差异:
      - 不创建任何 SHM, 不预加载, 不打包 chunk;
      - num_batches 永远等于 1 (整个数据集就是一个"虚拟段");
      - load_next_batch() 一次返回**所有可用 ep_uids** 的列表 + per-subset
        DiskVLMCacheView, 让 MultiLeRobotDataset.set_active_episodes 一次性
        激活全部 episode → DataLoader 看到全局 sample 池 → DistributedSampler
        全局 shuffle。

    构造时只读 index 已有的 metadata, 不打开任何 npz; 真正的 mmap 在 DataLoader
    worker 拿到第一个 sample 时才发生 (per-worker lazy)。
    """

    def __init__(
        self,
        index: MultiDatasetIndex,
        npz_lru_per_worker: int = -1,
        rank: int = 0,
        world_size: int = 1,
        verbose: bool = True,
        seed: int = 42,
        precise_scan: bool = True,
    ) -> None:
        self.index = index
        self.npz_lru_per_worker = int(npz_lru_per_worker)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.verbose = bool(verbose) and self.rank == 0
        self.base_seed = int(seed)
        self.precise_scan = bool(precise_scan)

        if not index.global_chunks:
            raise ValueError("DiskVLMCacheManager: index.global_chunks 为空")

        # 为每个子集预 build:
        #   - _SubsetVLMIndexMap (vlm_index_local → chunk/ep/frame 路由,
        #     init 阶段精确扫描每个 chunk 实际存在的 episode keys)
        #   - chunk_idx → npz_path 列表 (来自 index.global_chunks)
        if self.verbose:
            print(
                f"\n📦 [DiskVLMCacheManager] 扫描 {len(index.global_chunks)} 个 "
                f"chunk-XXX.npz 建立精确 episode 索引 (zip central dir, ~10ms/chunk) ..."
            )
        t0 = time.time()
        self._subset_index_maps: Dict[int, _SubsetVLMIndexMap] = {}
        subset_chunks: Dict[int, Dict[int, Path]] = {}
        for gc in index.global_chunks:
            subset_chunks.setdefault(int(gc.sub_idx), {})[int(gc.chunk_idx)] = Path(gc.npz_path)

        for s in index.subsets:
            sub_idx = int(s.sub_idx)
            npz_paths = subset_chunks.get(sub_idx, {})
            if not npz_paths:
                if self.verbose:
                    print(
                        f"  ⚠️ [DiskVLMCache] subset {s.name} 没有任何 chunk-XXX.npz, 已跳过",
                        flush=True,
                    )
                continue
            try:
                self._subset_index_maps[sub_idx] = _SubsetVLMIndexMap(
                    s, npz_paths, precise_scan=self.precise_scan,
                )
            except ValueError as e:
                if self.verbose:
                    print(f"  ⚠️ [DiskVLMCache] subset {s.name}: {e}", flush=True)
                continue

        if self.verbose:
            print(f"   ✓ chunk 扫描完成 {time.time() - t0:.2f}s")

        if not self._subset_index_maps:
            raise RuntimeError(
                "DiskVLMCacheManager: 所有子集都没有可用 chunk-XXX.npz"
            )

        # 一次性把"所有可用 ep_uid"列出来 —— 这就是新模式下 set_active_episodes
        # 收到的 episode 集合, 让 ConcatDataset 直接覆盖全数据集
        self._all_ep_uids: List[int] = []
        self._total_frames: int = 0
        for s in index.subsets:
            sub_idx = int(s.sub_idx)
            if sub_idx not in self._subset_index_maps:
                continue
            imap = self._subset_index_maps[sub_idx]
            for ep_idx in sorted(s.ep_local_vlm_start.keys()):
                if not imap.is_ep_available(int(ep_idx)):
                    continue
                ep_uid = index.episode_uid(sub_idx, int(ep_idx))
                self._all_ep_uids.append(int(ep_uid))
                self._total_frames += int(s.episode_lengths[int(ep_idx)])

        # 每个 worker 用的 view (LeRobotDataset 在 worker 中调用 get_sample)
        self._subset_views: Dict[int, DiskVLMCacheView] = {
            sub_idx: DiskVLMCacheView(
                index_map=imap,
                lru_size=self.npz_lru_per_worker,
                verbose_first_open=(self.rank == 0),
            )
            for sub_idx, imap in self._subset_index_maps.items()
        }

        self._first_load_done = False
        if self.verbose:
            self._print_init_summary()

    # ---------------------------------------------------------------- #
    # ChunkBatchCache 鸭子接口 (load_segment / load_next_batch / num_batches ...)
    # ---------------------------------------------------------------- #
    @property
    def num_batches(self) -> int:
        return 1

    @property
    def num_segments(self) -> int:
        return 1

    @property
    def total_chunks(self) -> int:
        return sum(len(imap.available_chunks()) for imap in self._subset_index_maps.values())

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def current_subset_caches(self) -> Dict[int, DiskVLMCacheView]:
        return self._subset_views

    @property
    def current_base_cache(self) -> None:
        # 多源场景下 train_multigpu.py 拿 shared_vlm_cache 走的是 None 路径,
        # 真正每子集的 view 在 current_subset_caches 里
        return None

    def set_epoch(self, epoch: int) -> None:
        # 不需要重打包; epoch 级 shuffle 由外层 DistributedSampler.set_epoch 做
        return

    def freeze_packing(self, seed: Optional[int] = None) -> None:
        # 单段模式, 无需冻结
        return

    def load_segment(
        self, segment_idx: int
    ) -> Tuple[List[int], Dict[int, int], int, Optional[DiskVLMCacheView]]:
        if int(segment_idx) != 0:
            raise IndexError(
                f"DiskVLMCacheManager 只有 1 个 segment, 但请求 segment_idx={segment_idx}"
            )
        return self.load_next_batch()

    def load_next_batch(
        self,
    ) -> Tuple[List[int], Dict[int, int], int, Optional[DiskVLMCacheView]]:
        """
        返回 (ep_uids, {}, num_frames, None) —— 与 MultiChunkBatchCache.load_next_batch
        签名一致, 让 train_multigpu.py 复用同一段 unpack 代码。

        多源场景下下游会走 train_dataset.set_active_episodes(ep_uids,
        chunk_batch_cache.current_subset_caches) 这条路, base_cache 字段不需要
        真东西 (None)。

        单源场景下下游会走 LeRobotDataset(..., shared_vlm_cache=batch_shared_cache),
        因此第 4 个返回值在单源时必须是一个真的 DiskVLMCacheView (单子集), 见
        load_next_batch_single_source().
        """
        if self.verbose and not self._first_load_done:
            t0 = time.time()
            print(
                f"\n📂 [DiskVLMCache] 激活全部 {len(self._all_ep_uids)} 个 episode "
                f"({self._total_frames} 帧, {self.total_chunks} 个 chunk, "
                f"mmap lazy)..."
            )
            self._first_load_done = True
            print(f"   ✓ 元数据组装完成 {time.time() - t0:.2f}s, 实际 mmap 在 worker 首次 __getitem__ 时发生")

        # 4-tuple 与 ChunkBatchCache 兼容 (多源单源的下游分支共用此接口):
        #   - 多源: 第 4 项填 None, 真正按子集 view 走 self._subset_views
        #   - 单源: 上层会调 load_next_batch_single_source() 拿到那个唯一 view
        return list(self._all_ep_uids), {}, int(self._total_frames), None

    def load_next_batch_single_source(
        self,
    ) -> Tuple[List[int], Dict[int, int], int, Optional[DiskVLMCacheView]]:
        """
        单源场景专用:
          - shared_vlm_cache 直接给 LeRobotDataset 看的是子集 0 的 DiskVLMCacheView
          - ep_uids 这里其实就是子集 0 内的 ep_idx_local (LeRobotDataset 期望
            episode_indices 是子集 0 内的本地索引)

        返回 (ep_idx_locals, {}, num_frames, view0)。
        """
        if len(self._subset_views) != 1:
            raise RuntimeError(
                f"load_next_batch_single_source 只能在 single-source 用 "
                f"(当前 subsets={len(self._subset_views)})"
            )
        sub_idx = next(iter(self._subset_views.keys()))
        view0 = self._subset_views[sub_idx]
        sinfo = self.index.get_subset(sub_idx)
        ep_locals = [
            int(e) for e in sorted(sinfo.ep_local_vlm_start.keys())
            if self._subset_index_maps[sub_idx].is_ep_available(int(e))
        ]
        return ep_locals, {}, int(self._total_frames), view0

    def cleanup(self) -> None:
        for v in self._subset_views.values():
            try:
                v.close()
            except Exception:
                pass
        self._subset_views.clear()

    # ---------------------------------------------------------------- #
    # 输出
    # ---------------------------------------------------------------- #
    def _print_init_summary(self) -> None:
        # 探测数据所在盘类型, 在 HDD 上给出强烈警告
        disk_kind, disk_desc = _detect_disk_type(Path(self.index.root_dir))
        print(f"📦 [DiskVLMCacheManager] 初始化完成")
        print(f"   root_dir:           {self.index.root_dir}")
        print(f"   subsets:            {len(self._subset_index_maps)}")
        print(f"   total episodes:     {len(self._all_ep_uids)}")
        print(f"   total frames:       {self._total_frames}")
        print(f"   total chunks (npz): {self.total_chunks}")
        print(f"   npz LRU per worker: "
              f"{'unlimited' if self.npz_lru_per_worker <= 0 else self.npz_lru_per_worker}")
        print(f"   mode:               disk-mmap (无 SHM, 全局 shuffle)")
        print(f"   disk type:          {disk_kind.upper()}  ({disk_desc})")
        if disk_kind == "hdd":
            print("=" * 60)
            print("⚠️  ⚠️  ⚠️   警告: 数据位于 HDD (机械硬盘)   ⚠️  ⚠️  ⚠️")
            print(
                "    Disk-shuffle 模式下每个 sample 都是随机读单帧 hidden state, "
                "HDD 随机 IO 性能极低 (实测 ~30 sample/s/worker)。"
            )
            print(
                "    在你的 batch_size × world_size 下大概率喂不饱 GPU, "
                "data_wait_time 会暴涨, 训练速度会比 SHM 模式慢 50-200 倍。"
            )
            print("    建议: 把数据集移到 NVMe (/data_disk2 等), 再启用本模式。")
            print("    如果只是想验证模式正确性, 可以先小规模跑几个 step 看 loss。")
        print("=" * 60)


# ============================================================================
# 单源便捷构造器 (没有 MultiDatasetIndex 时也能用)
# ============================================================================
def build_single_source_disk_cache(
    dataset_path: str,
    episode_indices: Optional[List[int]] = None,
    npz_lru_per_worker: int = -1,
    rank: int = 0,
    world_size: int = 1,
    verbose: bool = True,
) -> Tuple[DiskVLMCacheView, List[int], int, "DiskVLMCacheManager"]:
    """
    给单源 (USE_MULTI_DATASET=false) 训练用的便捷入口。

    返回 (view, ep_idx_locals, total_frames, manager)。调用方可以:

        view, ep_locals, n_frames, mgr = build_single_source_disk_cache(...)
        train_dataset = LeRobotDataset(
            dataset_path=..., episode_indices=ep_locals,
            shared_vlm_cache=view, ...
        )
        atexit.register(mgr.cleanup)

    实现上是建一个**只含一个子集**的临时 MultiDatasetIndex（不写 manifest）,
    然后委托给 DiskVLMCacheManager，最大限度复用同一套路由代码。
    """
    root = Path(dataset_path).resolve()
    parent = root.parent
    if not parent.exists():
        raise FileNotFoundError(f"dataset_path 父目录不存在: {parent}")
    try:
        idx = MultiDatasetIndex.scan(
            root_dir=parent,
            include=[root.name],
            exclude=None,
            require_complete_vlm=False,
            require_uniform_action_dim=True,
            target_action_dim=None,
            allow_state_dim_mismatch=True,
            verbose=verbose and rank == 0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"[build_single_source_disk_cache] 扫描 {root} 失败: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if len(idx.subsets) != 1:
        raise RuntimeError(
            f"build_single_source_disk_cache: 期望扫到 1 个子集, "
            f"实际 {len(idx.subsets)}; root.parent={parent}, target={root.name}"
        )

    mgr = DiskVLMCacheManager(
        index=idx,
        npz_lru_per_worker=npz_lru_per_worker,
        rank=rank,
        world_size=world_size,
        verbose=verbose and rank == 0,
    )
    ep_locals, _remap, n_frames, view = mgr.load_next_batch_single_source()
    if view is None:
        raise RuntimeError("build_single_source_disk_cache: 没拿到 DiskVLMCacheView")

    # 若上层显式传 episode_indices，再做一次过滤
    if episode_indices is not None:
        wanted = set(int(e) for e in episode_indices)
        ep_locals = [e for e in ep_locals if e in wanted]
        # 重新算 num_frames
        sinfo = idx.subsets[0]
        n_frames = sum(int(sinfo.episode_lengths[e]) for e in ep_locals)

    return view, ep_locals, int(n_frames), mgr
