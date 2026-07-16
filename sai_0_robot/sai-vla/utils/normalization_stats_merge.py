"""
normalization_stats_merge —— 多子集 LeRobot 的归一化统计聚合。

输入: utils.multi_dataset_index.MultiDatasetIndex (已扫描完成的子集列表)
输出: MultiDatasetNormalizers (per-subset 或 global normalizer 字典)

策略:
  - per_subset    : 每个子集一份 normalizer (推荐, 机器人形态差异大时)
  - minmax_union  : 全局合并 min/max, 所有子集共享一份

接口与 train_multigpu.py 里的 MinMaxNormalizer 在数值/方法上完全一致, 这里
复制一份只为避免反向 import 大型训练入口。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


# ============================================================================
# MinMaxNormalizer (本地副本, 行为与训练入口里的版本完全一致)
# ============================================================================

class MinMaxNormalizer:
    """Min-Max 归一化器, 输出区间 [-1, 1]。max==min 的维度恒为 0。"""

    def __init__(self, min_vals: torch.Tensor, max_vals: torch.Tensor) -> None:
        self.min_vals = min_vals.detach().clone().to(dtype=torch.float32)
        self.max_vals = max_vals.detach().clone().to(dtype=torch.float32)
        if self.min_vals.shape != self.max_vals.shape:
            raise ValueError(
                f"min/max shape mismatch: {self.min_vals.shape} vs {self.max_vals.shape}"
            )

    @property
    def dim(self) -> int:
        return int(self.min_vals.shape[0])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        mask = min_vals != max_vals
        normalized = torch.zeros_like(x)
        if mask.any():
            normalized[..., mask] = (
                x[..., mask] - min_vals[mask]
            ) / (max_vals[mask] - min_vals[mask])
            normalized[..., mask] = 2.0 * normalized[..., mask] - 1.0
        return normalized

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        min_vals = self.min_vals.to(x.device, dtype=x.dtype)
        max_vals = self.max_vals.to(x.device, dtype=x.dtype)
        return (x + 1.0) / 2.0 * (max_vals - min_vals) + min_vals


# ============================================================================
# stats.json IO + 工具
# ============================================================================

@dataclass
class _SubsetStats:
    state_min: Optional[np.ndarray]
    state_max: Optional[np.ndarray]
    action_min: Optional[np.ndarray]
    action_max: Optional[np.ndarray]


def _read_subset_stats(stats_path: Path) -> _SubsetStats:
    if not stats_path.exists():
        raise FileNotFoundError(f"stats.json not found: {stats_path}")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    state_min = state_max = action_min = action_max = None
    if "observation.state" in stats:
        s = stats["observation.state"]
        if "min" in s and "max" in s:
            state_min = np.asarray(s["min"], dtype=np.float32)
            state_max = np.asarray(s["max"], dtype=np.float32)
    if "action" in stats:
        a = stats["action"]
        if "min" in a and "max" in a:
            action_min = np.asarray(a["min"], dtype=np.float32)
            action_max = np.asarray(a["max"], dtype=np.float32)
    return _SubsetStats(
        state_min=state_min, state_max=state_max,
        action_min=action_min, action_max=action_max,
    )


def _convert_quat_state_to_axisangle_minmax(
    state_min: np.ndarray, state_max: np.ndarray
):
    """与 load_normalization_stats 对齐: 9 维(含四元数)→8 维轴角。"""
    if len(state_min) != 9 or len(state_max) != 9:
        raise ValueError(
            "convert_quat_to_axisangle 要求 state 原始维度 = 9, 当前 = "
            f"{len(state_min)}/{len(state_max)}"
        )
    new_min = np.array(
        [
            state_min[0], state_min[1],
            state_min[2], state_min[3], state_min[4],
            -math.pi, -math.pi, -math.pi,
        ],
        dtype=np.float32,
    )
    new_max = np.array(
        [
            state_max[0], state_max[1],
            state_max[2], state_max[3], state_max[4],
            math.pi, math.pi, math.pi,
        ],
        dtype=np.float32,
    )
    return new_min, new_max


# ============================================================================
# MultiDatasetNormalizers
# ============================================================================

class MultiDatasetNormalizers:
    """把多子集 stats.json 聚合成 normalizer 表。"""

    STRATEGIES = ("per_subset", "minmax_union")

    def __init__(
        self,
        strategy: str,
        target_state_dim: int,
        target_action_dim: int,
        per_subset_state: Dict[int, MinMaxNormalizer],
        per_subset_action: Dict[int, MinMaxNormalizer],
        global_state: Optional[MinMaxNormalizer] = None,
        global_action: Optional[MinMaxNormalizer] = None,
        subset_state_dims: Optional[Dict[int, int]] = None,
        subset_action_dims: Optional[Dict[int, int]] = None,
        subset_names: Optional[Dict[int, str]] = None,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self.STRATEGIES}, got {strategy!r}"
            )
        self.strategy = strategy
        self.target_state_dim = int(target_state_dim)
        self.target_action_dim = int(target_action_dim)
        self._per_state = per_subset_state
        self._per_action = per_subset_action
        self._global_state = global_state
        self._global_action = global_action
        self._subset_state_dims = subset_state_dims or {}
        self._subset_action_dims = subset_action_dims or {}
        self._subset_names = subset_names or {}

    @classmethod
    def build(
        cls,
        index,
        strategy: str = "per_subset",
        target_state_dim: Optional[int] = None,
        target_action_dim: Optional[int] = None,
        convert_quat_to_axisangle: bool = False,
        verbose: bool = True,
    ) -> "MultiDatasetNormalizers":
        if target_state_dim is None:
            target_state_dim = index.max_state_dim
        if target_action_dim is None:
            target_action_dim = index.max_action_dim

        per_state: Dict[int, MinMaxNormalizer] = {}
        per_action: Dict[int, MinMaxNormalizer] = {}
        subset_state_dims: Dict[int, int] = {}
        subset_action_dims: Dict[int, int] = {}
        subset_names: Dict[int, str] = {}

        if verbose:
            print("\n[MultiDatasetNormalizers] 加载各子集 stats.json ...")

        for s in index.subsets:
            stats_path = Path(s.dataset_path) / "meta" / "stats.json"
            try:
                stats = _read_subset_stats(stats_path)
            except FileNotFoundError:
                if verbose:
                    print(
                        f"  warn [{s.name}] stats.json 缺失, 使用默认 [-1,1] passthrough"
                    )
                state_min = np.full(s.state_dim, -1.0, dtype=np.float32)
                state_max = np.full(s.state_dim, 1.0, dtype=np.float32)
                action_min = np.full(s.action_dim, -1.0, dtype=np.float32)
                action_max = np.full(s.action_dim, 1.0, dtype=np.float32)
            else:
                state_min, state_max = stats.state_min, stats.state_max
                action_min, action_max = stats.action_min, stats.action_max
                if state_min is None or state_max is None:
                    state_min = np.full(s.state_dim, -1.0, dtype=np.float32)
                    state_max = np.full(s.state_dim, 1.0, dtype=np.float32)
                if action_min is None or action_max is None:
                    action_min = np.full(s.action_dim, -1.0, dtype=np.float32)
                    action_max = np.full(s.action_dim, 1.0, dtype=np.float32)

            if convert_quat_to_axisangle and len(state_min) == 9:
                state_min, state_max = _convert_quat_state_to_axisangle_minmax(
                    state_min, state_max
                )

            if len(action_min) != s.action_dim:
                raise ValueError(
                    f"[{s.name}] stats action min len={len(action_min)} != "
                    f"info action_dim={s.action_dim}"
                )

            per_state[s.sub_idx] = MinMaxNormalizer(
                torch.from_numpy(state_min), torch.from_numpy(state_max)
            )
            per_action[s.sub_idx] = MinMaxNormalizer(
                torch.from_numpy(action_min), torch.from_numpy(action_max)
            )
            subset_state_dims[s.sub_idx] = int(len(state_min))
            subset_action_dims[s.sub_idx] = int(len(action_min))
            subset_names[s.sub_idx] = s.name

            if verbose:
                print(
                    f"  ok   [{s.name}] state_dim={len(state_min)} "
                    f"action_dim={len(action_min)}"
                )

        global_state = _build_global_minmax(
            per_state, target_dim=target_state_dim, kind="state"
        )
        global_action = _build_global_minmax(
            per_action, target_dim=target_action_dim, kind="action"
        )

        return cls(
            strategy=strategy,
            target_state_dim=target_state_dim,
            target_action_dim=target_action_dim,
            per_subset_state=per_state,
            per_subset_action=per_action,
            global_state=global_state,
            global_action=global_action,
            subset_state_dims=subset_state_dims,
            subset_action_dims=subset_action_dims,
            subset_names=subset_names,
        )

    def state_normalizer(self, sub_idx: int) -> MinMaxNormalizer:
        if self.strategy == "per_subset":
            return self._per_state[int(sub_idx)]
        return self._global_state  # type: ignore[return-value]

    def action_normalizer(self, sub_idx: int) -> MinMaxNormalizer:
        if self.strategy == "per_subset":
            return self._per_action[int(sub_idx)]
        return self._global_action  # type: ignore[return-value]

    def normalize_state(self, sub_idx: int, state: torch.Tensor) -> torch.Tensor:
        norm = self.state_normalizer(sub_idx)
        D_sub = norm.dim
        if state.shape[-1] != D_sub:
            raise ValueError(
                f"normalize_state: sub_idx={sub_idx} 期望最后一维={D_sub}, 实际={state.shape[-1]}"
            )
        normed = norm.normalize(state)
        return _right_pad_last_dim(normed, self.target_state_dim)

    def normalize_action(self, sub_idx: int, action: torch.Tensor) -> torch.Tensor:
        norm = self.action_normalizer(sub_idx)
        D_sub = norm.dim
        if action.shape[-1] != D_sub:
            raise ValueError(
                f"normalize_action: sub_idx={sub_idx} 期望最后一维={D_sub}, 实际={action.shape[-1]}"
            )
        normed = norm.normalize(action)
        return _right_pad_last_dim(normed, self.target_action_dim)

    def export_summary(self) -> dict:
        def _norm_to_dict(n):
            if n is None:
                return None
            return {"min": n.min_vals.tolist(), "max": n.max_vals.tolist(), "dim": n.dim}

        return {
            "strategy": self.strategy,
            "target_state_dim": self.target_state_dim,
            "target_action_dim": self.target_action_dim,
            "per_subset_state": {int(i): _norm_to_dict(n) for i, n in self._per_state.items()},
            "per_subset_action": {int(i): _norm_to_dict(n) for i, n in self._per_action.items()},
            "global_state": _norm_to_dict(self._global_state),
            "global_action": _norm_to_dict(self._global_action),
            "subset_state_dims": dict(self._subset_state_dims),
            "subset_action_dims": dict(self._subset_action_dims),
            "subset_names": dict(self._subset_names),
        }


def _right_pad_last_dim(t: torch.Tensor, target: int) -> torch.Tensor:
    cur = int(t.shape[-1])
    if cur == target:
        return t
    if cur > target:
        raise ValueError(f"_right_pad_last_dim: cur={cur} > target={target}")
    pad_shape = list(t.shape)
    pad_shape[-1] = target - cur
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=-1)


def _build_global_minmax(
    per: Dict[int, MinMaxNormalizer], target_dim: int, kind: str
) -> Optional[MinMaxNormalizer]:
    if not per:
        return None
    g_min = np.full(target_dim, np.inf, dtype=np.float32)
    g_max = np.full(target_dim, -np.inf, dtype=np.float32)
    for n in per.values():
        d = n.dim
        if d > target_dim:
            raise ValueError(
                f"_build_global_minmax({kind}): subset dim={d} > target_dim={target_dim}"
            )
        m_min = n.min_vals.cpu().numpy()
        m_max = n.max_vals.cpu().numpy()
        g_min[:d] = np.minimum(g_min[:d], m_min)
        g_max[:d] = np.maximum(g_max[:d], m_max)
    inf_mask = np.isinf(g_min) | np.isinf(g_max)
    g_min[inf_mask] = -1.0
    g_max[inf_mask] = 1.0
    return MinMaxNormalizer(torch.from_numpy(g_min), torch.from_numpy(g_max))
