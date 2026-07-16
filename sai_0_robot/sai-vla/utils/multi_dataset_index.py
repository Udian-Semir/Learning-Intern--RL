"""
MultiDatasetIndex —— 扫描"多 LeRobot 子数据集 root 目录"，输出统一索引。

用途
====
现有 train_multigpu.py / ChunkBatchCache / LeRobotDataset 都假设
单个 dataset_path（一个完整的 LeRobot 目录）。要在不物理合并的前提下
做"22 个子集一起 pretrain"，先用本模块把 root 扫一遍，输出一个统一
的索引视图，供后续 multi_chunk_batch_cache / multi_lerobot_dataset 使用。

目录假设
========
root/
  recipe.json                 (可选；本模块不强依赖)
  jaco_play/                  (子集 1：自己有完整 LeRobot 结构)
    meta/{info,episodes,stats,tasks}.json[l]
    data/chunk-{NNN}/episode_{EEEEEE}.parquet
    vlm_hidden_states/chunk-{NNN}.npz
  kuka/                       (子集 2)
    ...
  ...

核心输出
========
MultiDatasetIndex.subsets : List[SubsetInfo]
  每个被收纳的子集的元信息（含 chunks_size、action/state dim、vlm 完整性）。

MultiDatasetIndex.global_chunks : List[GlobalChunk]
  把"所有子集 × 所有 vlm chunk"摊平后的全局 chunk 池。每个元素自带
  sub_idx + 子集内 chunk_idx + npz_path + 该 chunk 内的 episode 列表 +
  各 episode 在子集内的 vlm 起点 + 该 chunk 总 frame 数 + npz 字节大小。
  这是 multi_chunk_batch_cache shuffle 与 budget 打包的最小单元。

MultiDatasetIndex.subset_episode_offset / subset_frame_offset:
  把 (sub_idx, ep_idx_local) 映射成全局唯一的 ep_uid / frame_uid，让
  RemappedSharedVLMCache 的 Dict[int, int] 接口零改动复用。

MultiDatasetIndex.skipped : List[SkippedSubset]
  跳过的子集 + 跳过原因（用户后期可以拍照对账）。

设计哲学
========
- 本模块只做扫描+校验+索引，不做任何 SHM/IO 重活；
- IO 都留给 multi_chunk_batch_cache；
- normalizer/stats 留给 normalization_stats_merge；
- LeRobotDataset 路由留给 multi_lerobot_dataset。
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class SubsetInfo:
    """单个子数据集的元信息（被 MultiDatasetIndex 收录后的形态）。

    重要区分:
        total_episodes / total_frames: 子集 meta 里声明的全量数 (含没 vlm 的)
        eligible_episodes / eligible_frames: 已生成 vlm chunk 实际可训规模
        episode_lengths:  仅含 eligible 范围内的 ep 长度
        ep_local_vlm_start: **仍按全量 episode_index 累加**, 因为 parquet 里
            写好的 vlm_hidden_state_index 是基于全量序列的, 不能动!
    """

    sub_idx: int
    name: str
    dataset_path: Path
    action_dim: int
    state_dim: int
    fps: int
    chunks_size: int
    total_episodes: int
    total_frames: int
    num_data_chunks: int
    num_vlm_chunks: int
    vlm_chunk_indices: List[int] = field(default_factory=list)
    episode_lengths: Dict[int, int] = field(default_factory=dict)
    ep_local_vlm_start: Dict[int, int] = field(default_factory=dict)
    # 新增: 已有 vlm 范围内的可训规模 (vlm 完整时 = total_*; 不完整时 < total_*)
    eligible_episodes: int = 0
    eligible_frames: int = 0


@dataclass
class GlobalChunk:
    """全局 chunk 池中的一项。multi_chunk_batch_cache 直接 shuffle 这个列表。"""

    sub_idx: int
    chunk_idx: int
    npz_path: Path
    npz_size_bytes: int
    episodes: List[int]
    episode_lengths: Dict[int, int]
    total_frames: int

    @property
    def global_id(self) -> Tuple[int, int]:
        return (self.sub_idx, self.chunk_idx)


@dataclass
class SkippedSubset:
    name: str
    reason: str
    detail: str = ""


# ============================================================================
# 内部实现 (低层 helper, 先于主类定义以便 type-check)
# ============================================================================

@dataclass
class _RawSubsetCandidate:
    name: str
    dataset_path: Path
    action_dim: int
    state_dim: int
    fps: int
    chunks_size: int
    total_episodes: int
    total_frames: int
    num_data_chunks: int
    num_vlm_chunks: int
    vlm_chunk_indices: List[int]
    episode_lengths: Dict[int, int]
    ep_local_vlm_start: Dict[int, int]
    eligible_episodes: int = 0
    eligible_frames: int = 0


class _SubsetSkipError(Exception):
    """单个子集扫描时遇到的"已知应跳过"错误。"""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


_CHUNK_NPZ_RE = re.compile(r"^chunk-(\d+)\.npz$")


def _load_recipe_if_exists(root_dir: Path) -> Optional[dict]:
    p = root_dir / "recipe.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _collect_candidate_subset_names(
    root_dir: Path,
    include: Optional[Sequence[str]],
    exclude: Optional[Sequence[str]],
    recipe: Optional[dict],
) -> List[str]:
    """决定要扫描哪些子目录名。"""
    inc = {str(x) for x in include} if include else None
    exc = {str(x) for x in exclude} if exclude else set()

    if inc is not None:
        names = sorted(inc - exc)
    else:
        names = []
        for p in sorted(root_dir.iterdir()):
            if not p.is_dir():
                continue
            n = p.name
            if n.startswith("."):
                continue
            if n in {"meta", "data", "videos", "vlm_hidden_states"}:
                continue
            if n in exc:
                continue
            names.append(n)

    if inc is None and recipe and isinstance(recipe.get("datasets"), list):
        recipe_order: List[str] = []
        seen = set()
        for d in recipe["datasets"]:
            n = d.get("name") if isinstance(d, dict) else None
            if n and n in names and n not in seen:
                recipe_order.append(n)
                seen.add(n)
        for n in names:
            if n not in seen:
                recipe_order.append(n)
        names = recipe_order
    return names


def _verify_npz_deep(npz_path: Path) -> Optional[str]:
    """对单个 npz 做一次"读入第一个 entry 的少量 bytes"的深度校验。
    返回 None 表示 OK; 返回 str 是错误描述。"""
    try:
        import numpy as _np
        with _np.load(npz_path, allow_pickle=False) as data:
            if not data.files:
                return "empty npz (no entries)"
            first_key = sorted(data.files)[0]
            arr = data[first_key]
            if arr.ndim < 2:
                return f"unexpected ndim={arr.ndim}"
        return None
    except (zipfile.BadZipFile, EOFError, OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"


def _scan_one_subset(
    sub_dir: Path,
    require_complete_vlm: bool,
) -> _RawSubsetCandidate:
    """扫描单个子集目录，做基础校验，输出 _RawSubsetCandidate。"""
    name = sub_dir.name

    info_path = sub_dir / "meta" / "info.json"
    episodes_path = sub_dir / "meta" / "episodes.jsonl"
    vlm_dir = sub_dir / "vlm_hidden_states"

    if not info_path.exists():
        raise _SubsetSkipError("missing_meta_info", str(info_path))
    if not episodes_path.exists():
        raise _SubsetSkipError("missing_meta_episodes", str(episodes_path))
    if not vlm_dir.exists():
        raise _SubsetSkipError("missing_vlm_hidden_states_dir", str(vlm_dir))

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    try:
        action_dim = int(info["features"]["action"]["shape"][0])
    except Exception as exc:
        raise _SubsetSkipError("bad_action_feature", repr(exc))
    try:
        state_dim = int(info["features"]["observation.state"]["shape"][0])
    except Exception as exc:
        raise _SubsetSkipError("bad_state_feature", repr(exc))

    fps = int(info.get("fps", 0))
    chunks_size = int(info.get("chunks_size", 1000))
    total_episodes = int(info.get("total_episodes", 0))
    total_frames = int(info.get("total_frames", 0))
    num_data_chunks = int(info.get("total_chunks", 0))

    episode_lengths: Dict[int, int] = {}
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep_idx = int(rec["episode_index"])
            ep_len = int(rec["length"])
            episode_lengths[ep_idx] = ep_len
    if not episode_lengths:
        raise _SubsetSkipError("empty_episodes_jsonl", str(episodes_path))

    ep_local_vlm_start: Dict[int, int] = {}
    running = 0
    for e in sorted(episode_lengths.keys()):
        ep_local_vlm_start[e] = running
        running += episode_lengths[e]

    expected_chunk_indices = sorted({
        ep // chunks_size for ep in episode_lengths.keys()
    })
    actual_chunk_indices: List[int] = []
    broken_chunk_files: List[str] = []
    for entry in sorted(vlm_dir.iterdir()):
        m = _CHUNK_NPZ_RE.match(entry.name)
        if not m:
            continue
        # 轻量完整性校验: 中央目录坏的 zip 直接当作"未生成"
        try:
            if entry.stat().st_size < 32 or not zipfile.is_zipfile(entry):
                broken_chunk_files.append(entry.name)
                continue
        except OSError:
            broken_chunk_files.append(entry.name)
            continue
        actual_chunk_indices.append(int(m.group(1)))
    actual_chunk_indices.sort()

    missing = sorted(set(expected_chunk_indices) - set(actual_chunk_indices))
    if broken_chunk_files:
        print(
            f"  warn [{name}] 检测到 {len(broken_chunk_files)} 个损坏 npz 已剔除: "
            f"{broken_chunk_files[:3]}{' ...' if len(broken_chunk_files) > 3 else ''}",
            flush=True,
        )
    if missing and require_complete_vlm:
        raise _SubsetSkipError(
            "vlm_incomplete",
            (
                f"missing chunk-{missing[0]:03d}.npz ... "
                f"({len(missing)}/{len(expected_chunk_indices)} chunks 缺失)"
            ),
        )

    # eligible_*: vlm 完整时 = total_*; 不完整时 = 已有 vlm chunk 内的 episodes
    actual_set = set(actual_chunk_indices)
    eligible_eps_lengths: Dict[int, int] = {
        ep: ln for ep, ln in episode_lengths.items()
        if (ep // chunks_size) in actual_set
    }
    eligible_episodes = len(eligible_eps_lengths)
    eligible_frames = int(sum(eligible_eps_lengths.values()))

    return _RawSubsetCandidate(
        name=name,
        dataset_path=sub_dir.resolve(),
        action_dim=action_dim,
        state_dim=state_dim,
        fps=fps,
        chunks_size=chunks_size,
        total_episodes=total_episodes,
        total_frames=total_frames,
        num_data_chunks=num_data_chunks,
        num_vlm_chunks=len(actual_chunk_indices),
        vlm_chunk_indices=actual_chunk_indices,
        episode_lengths=episode_lengths,
        ep_local_vlm_start=ep_local_vlm_start,
        eligible_episodes=eligible_episodes,
        eligible_frames=eligible_frames,
    )


# ============================================================================
# 扫描器主类
# ============================================================================

class MultiDatasetIndex:
    """扫描 root 下所有子数据集，产出统一索引。"""

    def __init__(
        self,
        root_dir: Path,
        subsets: List[SubsetInfo],
        skipped: List[SkippedSubset],
        global_chunks: List[GlobalChunk],
        max_state_dim: int,
        max_action_dim: int,
        recipe: Optional[dict] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.subsets = subsets
        self.skipped = skipped
        self.global_chunks = global_chunks
        self.max_state_dim = int(max_state_dim)
        self.max_action_dim = int(max_action_dim)
        self.recipe = recipe

        self._subset_by_idx: Dict[int, SubsetInfo] = {s.sub_idx: s for s in subsets}
        self._subset_by_name: Dict[str, SubsetInfo] = {s.name: s for s in subsets}

        # ep_uid = subset_episode_offset[sub_idx] + ep_idx_local
        # frame_uid = subset_frame_offset[sub_idx] + ep_local_vlm_start[ep_idx_local] + frame_in_ep
        self.subset_episode_offset: List[int] = []
        self.subset_frame_offset: List[int] = []
        ep_running = 0
        fr_running = 0
        for s in self.subsets:
            self.subset_episode_offset.append(ep_running)
            self.subset_frame_offset.append(fr_running)
            ep_running += s.total_episodes
            fr_running += s.total_frames
        self.total_global_episodes = ep_running
        self.total_global_frames = fr_running

        # 已有 vlm 实际可训规模 (vlm 完整时 = total_*)
        self.eligible_global_episodes = sum(s.eligible_episodes for s in self.subsets)
        self.eligible_global_frames = sum(s.eligible_frames for s in self.subsets)

    @classmethod
    def scan(
        cls,
        root_dir,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        require_complete_vlm: bool = True,
        require_uniform_action_dim: bool = True,
        target_action_dim: Optional[int] = None,
        allow_state_dim_mismatch: bool = True,
        verbose: bool = True,
    ) -> "MultiDatasetIndex":
        root_dir = Path(root_dir).resolve()
        if not root_dir.is_dir():
            raise FileNotFoundError(f"root_dir 不存在或不是目录: {root_dir}")

        recipe = _load_recipe_if_exists(root_dir)
        candidates = _collect_candidate_subset_names(
            root_dir, include=include, exclude=exclude, recipe=recipe,
        )

        if verbose:
            print(f"[MultiDatasetIndex] root_dir = {root_dir}")
            print(f"[MultiDatasetIndex] 候选子集数 = {len(candidates)}")

        # 第一遍：每个候选子集做"原始扫描"
        raw_results: List[Tuple[str, _RawSubsetCandidate]] = []
        skipped: List[SkippedSubset] = []
        for name in candidates:
            sub_dir = root_dir / name
            try:
                raw = _scan_one_subset(sub_dir, require_complete_vlm=require_complete_vlm)
                raw_results.append((name, raw))
            except _SubsetSkipError as exc:
                skipped.append(SkippedSubset(name=name, reason=exc.reason, detail=exc.detail))
                if verbose:
                    print(f"  skip [{name}]: {exc.reason}  ({exc.detail})")

        if not raw_results:
            raise RuntimeError(
                f"没有任何子集通过初步扫描（root={root_dir}，候选={len(candidates)}）。"
                "请检查 include/exclude，或确认子集已完成 vlm 抽取。"
            )

        # 决定 target_action_dim（多数派或外部指定）
        if require_uniform_action_dim:
            ad_counter: Dict[int, int] = {}
            for _, raw in raw_results:
                ad_counter[raw.action_dim] = ad_counter.get(raw.action_dim, 0) + 1
            if target_action_dim is None:
                target_action_dim = sorted(
                    ad_counter.items(), key=lambda x: (-x[1], x[0])
                )[0][0]
            if verbose and len(ad_counter) > 1:
                print(
                    f"  [MultiDatasetIndex] action_dim 多数派 = {target_action_dim} "
                    f"(分布: {ad_counter})"
                )

        # 第二遍：按 action_dim / state_dim 一致性过滤
        all_state_dims: List[int] = []
        all_action_dims: List[int] = []
        kept: List[_RawSubsetCandidate] = []
        for name, raw in raw_results:
            if require_uniform_action_dim and raw.action_dim != target_action_dim:
                skipped.append(SkippedSubset(
                    name=name, reason="action_dim_mismatch",
                    detail=f"action_dim={raw.action_dim} != target_action_dim={target_action_dim}",
                ))
                if verbose:
                    print(
                        f"  skip [{name}]: action_dim={raw.action_dim} != target {target_action_dim}"
                    )
                continue

            if (not allow_state_dim_mismatch) and (
                kept and raw.state_dim != kept[0].state_dim
            ):
                skipped.append(SkippedSubset(
                    name=name, reason="state_dim_mismatch_strict",
                    detail=f"state_dim={raw.state_dim} != {kept[0].state_dim} (strict mode)",
                ))
                if verbose:
                    print(
                        f"  skip [{name}]: state_dim={raw.state_dim} != "
                        f"{kept[0].state_dim} (strict)"
                    )
                continue

            kept.append(raw)
            all_state_dims.append(raw.state_dim)
            all_action_dims.append(raw.action_dim)

        if not kept:
            raise RuntimeError(
                "所有候选子集都被维度过滤掉了。请检查 require_uniform_action_dim / "
                "allow_state_dim_mismatch / target_action_dim 设置。"
            )

        # 组装 SubsetInfo + GlobalChunk
        subsets: List[SubsetInfo] = []
        global_chunks: List[GlobalChunk] = []
        for sub_idx, raw in enumerate(kept):
            subset = SubsetInfo(
                sub_idx=sub_idx,
                name=raw.name,
                dataset_path=raw.dataset_path,
                action_dim=raw.action_dim,
                state_dim=raw.state_dim,
                fps=raw.fps,
                chunks_size=raw.chunks_size,
                total_episodes=raw.total_episodes,
                total_frames=raw.total_frames,
                num_data_chunks=raw.num_data_chunks,
                num_vlm_chunks=raw.num_vlm_chunks,
                vlm_chunk_indices=list(raw.vlm_chunk_indices),
                episode_lengths=dict(raw.episode_lengths),
                ep_local_vlm_start=dict(raw.ep_local_vlm_start),
                eligible_episodes=raw.eligible_episodes,
                eligible_frames=raw.eligible_frames,
            )
            subsets.append(subset)

            for cidx in raw.vlm_chunk_indices:
                eps_in_chunk = sorted(
                    e for e in raw.episode_lengths.keys()
                    if (e // raw.chunks_size) == cidx
                )
                if not eps_in_chunk:
                    if verbose:
                        print(
                            f"  warn [{raw.name}] chunk-{cidx:03d}.npz 没有对应 episode，已忽略"
                        )
                    continue
                npz_path = raw.dataset_path / "vlm_hidden_states" / f"chunk-{cidx:03d}.npz"
                gc = GlobalChunk(
                    sub_idx=sub_idx,
                    chunk_idx=cidx,
                    npz_path=npz_path,
                    npz_size_bytes=int(os.path.getsize(npz_path)),
                    episodes=eps_in_chunk,
                    episode_lengths={e: raw.episode_lengths[e] for e in eps_in_chunk},
                    total_frames=int(sum(raw.episode_lengths[e] for e in eps_in_chunk)),
                )
                global_chunks.append(gc)

        max_state_dim = max(all_state_dims)
        max_action_dim = max(all_action_dims)

        index = cls(
            root_dir=root_dir,
            subsets=subsets,
            skipped=skipped,
            global_chunks=global_chunks,
            max_state_dim=max_state_dim,
            max_action_dim=max_action_dim,
            recipe=recipe,
        )
        if verbose:
            index.print_summary()
        return index

    # ------------------------------------------------------------------ #
    # 查询接口
    # ------------------------------------------------------------------ #

    def get_subset(self, sub_idx: int) -> SubsetInfo:
        return self._subset_by_idx[int(sub_idx)]

    def get_subset_by_name(self, name: str) -> SubsetInfo:
        return self._subset_by_name[name]

    def episode_uid(self, sub_idx: int, ep_idx_local: int) -> int:
        """子集内 (sub_idx, ep_idx_local) 转换为全局 ep_uid。"""
        return self.subset_episode_offset[int(sub_idx)] + int(ep_idx_local)

    def frame_uid(self, sub_idx: int, ep_idx_local: int, frame_in_ep: int) -> int:
        """子集内 (sub_idx, ep_idx_local, frame_in_ep) 转换为全局 frame_uid。"""
        s = self._subset_by_idx[int(sub_idx)]
        return (
            self.subset_frame_offset[int(sub_idx)]
            + int(s.ep_local_vlm_start[int(ep_idx_local)])
            + int(frame_in_ep)
        )

    def split_ep_uid(self, ep_uid: int) -> Tuple[int, int]:
        """全局 ep_uid 还原成 (sub_idx, ep_idx_local)。"""
        ep_uid = int(ep_uid)
        lo, hi = 0, len(self.subset_episode_offset) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.subset_episode_offset[mid] <= ep_uid:
                lo = mid
            else:
                hi = mid - 1
        sub_idx = lo
        ep_idx_local = ep_uid - self.subset_episode_offset[sub_idx]
        return sub_idx, ep_idx_local

    # ------------------------------------------------------------------ #
    # 输出
    # ------------------------------------------------------------------ #

    def print_summary(self) -> None:
        print("\n" + "=" * 78)
        print(f"MultiDatasetIndex 汇总  (root={self.root_dir})")
        print("=" * 78)
        print(f"   收录子集数:        {len(self.subsets)}")
        print(f"   跳过子集数:        {len(self.skipped)}")
        print(
            f"   全局 episodes:     {self.eligible_global_episodes} eligible"
            f"  ({self.total_global_episodes} 子集声明)"
        )
        print(
            f"   全局 frames:       {self.eligible_global_frames} eligible"
            f"  ({self.total_global_frames} 子集声明)"
        )
        print(f"   全局 vlm chunks:   {len(self.global_chunks)}")
        print(f"   max action_dim:    {self.max_action_dim}")
        print(f"   max state_dim:     {self.max_state_dim}")
        print("-" * 78)
        print(
            f"   {'#':>2}  {'name':<52} {'a':>2} {'s':>2} {'fps':>3} "
            f"{'eligi_ep':>8} {'eligi_fr':>10} {'chunks':>10}"
        )
        for s in self.subsets:
            partial_flag = (
                ""
                if s.eligible_episodes == s.total_episodes
                else f" (of {s.total_episodes} ep / {s.num_data_chunks} chunks)"
            )
            print(
                f"   {s.sub_idx:>2}  {s.name[:52]:<52} "
                f"{s.action_dim:>2} {s.state_dim:>2} {s.fps:>3} "
                f"{s.eligible_episodes:>8} {s.eligible_frames:>10} "
                f"{s.num_vlm_chunks:>10}"
                f"{partial_flag}"
            )
        if self.skipped:
            print("-" * 78)
            print("   跳过列表:")
            for sk in self.skipped:
                print(f"     - {sk.name:<54} reason={sk.reason}  detail={sk.detail}")
        print("=" * 78 + "\n")

    def save_manifest(self, json_path) -> Path:
        """把索引保存为 JSON，方便审计 / resume 校验。"""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root_dir": str(self.root_dir),
            "max_state_dim": self.max_state_dim,
            "max_action_dim": self.max_action_dim,
            "total_global_episodes": self.total_global_episodes,
            "total_global_frames": self.total_global_frames,
            "subsets": [
                {
                    "sub_idx": s.sub_idx,
                    "name": s.name,
                    "dataset_path": str(s.dataset_path),
                    "action_dim": s.action_dim,
                    "state_dim": s.state_dim,
                    "fps": s.fps,
                    "chunks_size": s.chunks_size,
                    "total_episodes": s.total_episodes,
                    "total_frames": s.total_frames,
                    "num_data_chunks": s.num_data_chunks,
                    "num_vlm_chunks": s.num_vlm_chunks,
                    "vlm_chunk_indices": s.vlm_chunk_indices,
                    "episode_offset_global": self.subset_episode_offset[s.sub_idx],
                    "frame_offset_global": self.subset_frame_offset[s.sub_idx],
                }
                for s in self.subsets
            ],
            "global_chunks": [
                {
                    "sub_idx": gc.sub_idx,
                    "chunk_idx": gc.chunk_idx,
                    "npz_path": str(gc.npz_path),
                    "npz_size_bytes": gc.npz_size_bytes,
                    "num_episodes": len(gc.episodes),
                    "total_frames": gc.total_frames,
                }
                for gc in self.global_chunks
            ],
            "skipped": [
                {"name": sk.name, "reason": sk.reason, "detail": sk.detail}
                for sk in self.skipped
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return json_path


# ============================================================================
# CLI（dry-run 扫描验证）
# ============================================================================

def _build_argparser():
    import argparse

    ap = argparse.ArgumentParser(
        description="Scan a multi-subset LeRobot root and print a summary."
    )
    ap.add_argument("--root", required=True, help="root 目录路径")
    ap.add_argument("--include", default=None, help="逗号分隔的白名单子集名")
    ap.add_argument("--exclude", default=None, help="逗号分隔的黑名单子集名")
    ap.add_argument(
        "--require_complete_vlm", action="store_true", default=True,
        help="vlm 不完整就跳过（默认开启）",
    )
    ap.add_argument(
        "--no_require_complete_vlm", dest="require_complete_vlm", action="store_false",
        help="即使 vlm 不完整也保留（仅用于排查）",
    )
    ap.add_argument(
        "--strict_state_dim", action="store_true", default=False,
        help="严格要求 state_dim 一致（不一致就跳过）",
    )
    ap.add_argument(
        "--target_action_dim", type=int, default=None,
        help="强制 action_dim 等于这个值；留空则取多数派",
    )
    ap.add_argument(
        "--save_manifest", default=None, help="把扫描结果落到这个 JSON 路径",
    )
    ap.add_argument(
        "--deep_verify_npz", action="store_true", default=False,
        help=("对每个 vlm chunk 做深度校验 (读第一个 entry 的 array 头部); "
              "比 is_zipfile 慢, 但能查出 zip entry 数据流截断的 npz。"),
    )
    return ap


def main() -> int:
    args = _build_argparser().parse_args()
    inc = [s.strip() for s in args.include.split(",")] if args.include else None
    exc = [s.strip() for s in args.exclude.split(",")] if args.exclude else None

    index = MultiDatasetIndex.scan(
        root_dir=args.root,
        include=inc,
        exclude=exc,
        require_complete_vlm=args.require_complete_vlm,
        require_uniform_action_dim=True,
        target_action_dim=args.target_action_dim,
        allow_state_dim_mismatch=(not args.strict_state_dim),
        verbose=True,
    )

    if args.deep_verify_npz:
        print("\n[deep_verify_npz] 对所有 chunk npz 做深度校验 (慢, 请耐心) ...")
        bad_list: List[Tuple[str, int, str]] = []
        for gc in index.global_chunks:
            err = _verify_npz_deep(gc.npz_path)
            if err is not None:
                sub_name = index.get_subset(gc.sub_idx).name
                bad_list.append((sub_name, gc.chunk_idx, err))
                print(f"  ❌ [{sub_name}] chunk-{gc.chunk_idx:03d}.npz : {err[:120]}")
        if bad_list:
            print(f"\n共 {len(bad_list)} 个损坏 chunk 已发现, 请重新生成。")
        else:
            print(f"\n✅ 全部 {len(index.global_chunks)} 个 chunk 通过深度校验。")

    if args.save_manifest:
        out = index.save_manifest(args.save_manifest)
        print(f"已写入 manifest: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
