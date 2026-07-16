#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时显示 LIBERO-plus 评估成功率的 dashboard.

工作方式:
  - 扫 ``experiments/`` 下所有 ``eval_results_<suite>_shard*_of_*.json``
  - 对每个 (实验目录, suite) 聚合所有 shard 的数据
  - 终端清屏循环刷新, 显示:
      • 各 suite 的总体 SR (跑完的打 ✅, 进行中打 ⏳)
      • 当前 ACTIVE suite 的逐 shard 进度 + SR
      • 当前 ACTIVE suite 的分 category SR
      • 当前 ACTIVE suite 的分 difficulty SR
  - 按 Ctrl+C 退出

用法:
  # 自动找 experiments 下所有最近的实验
  python -m eval.Sai0_1.libero_plus.OFT1_0.tools.watch_eval_progress

  # 只显示某个 step 的所有 suite
  python -m eval.Sai0_1.libero_plus.OFT1_0.tools.watch_eval_progress --filter step_180000

  # 单次输出 (cron / pipe 友好)
  python -m eval.Sai0_1.libero_plus.OFT1_0.tools.watch_eval_progress --once

  # 自定义刷新间隔 (秒)
  python -m eval.Sai0_1.libero_plus.OFT1_0.tools.watch_eval_progress --interval 3
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LIBERO_PLUS_CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)

LIBERO_PLUS_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# 每个 suite 完整的 task 总数 (来自 task_classification.json), 用于估算总进度
SUITE_TOTAL_TASKS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}

EXPERIMENTS_ROOT_DEFAULT = (
    Path(__file__).resolve().parents[1] / "experiments"
)

ANSI_CLEAR = "\033[2J\033[H"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_CYAN = "\033[36m"
ANSI_MAGENTA = "\033[35m"
ANSI_RED = "\033[31m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def _c(text: str, *codes: str, enable: bool = True) -> str:
    if not enable or not codes:
        return text
    return "".join(codes) + text + ANSI_RESET


def _color_for_sr(sr: float, enable: bool) -> str:
    """根据成功率上色 (>=60% 绿, >=30% 黄, 其他红)."""
    if not enable:
        return ""
    if sr >= 0.60:
        return ANSI_GREEN
    if sr >= 0.30:
        return ANSI_YELLOW
    return ANSI_RED


def _fmt_pct(numer: int, denom: int, enable_color: bool = True) -> str:
    if denom <= 0:
        return _c("  ?  ", ANSI_DIM, enable=enable_color)
    sr = numer / denom
    color = _color_for_sr(sr, enable_color)
    return _c(f"{sr * 100:6.2f}%", color, enable=enable_color)


def _list_final_shard_paths(exp_dir: Path, suite: str) -> List[str]:
    """列出 shard JSON 路径, 排除 worker 原子写过程中临时存在的 ``.tmp.json``."""
    pattern = str(exp_dir / f"eval_results_{suite}_shard*_of_*.json")
    return [p for p in sorted(glob.glob(pattern)) if not p.endswith(".tmp.json")]


_SHARD_NAME_RE = re.compile(
    r"eval_results_.+_shard(\d+)_of_(\d+)\.json$"
)


def _parse_shard_index_from_name(path: str) -> Tuple[int, int]:
    """从 ``eval_results_<suite>_shard{K}_of_{N}.json`` 文件名解析 (K, N).

    早期 summary_payload 漏写 shard_index/num_shards 字段时, 用这个兜底.
    """
    m = _SHARD_NAME_RE.search(os.path.basename(path))
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1


def _read_shard_jsons(
    exp_dir: Path, suite: str
) -> List[Tuple[Dict[str, Any], int, int]]:
    """返回 [(json_payload, shard_index, num_shards)], 解析后的 idx/total 一定有效."""
    out: List[Tuple[Dict[str, Any], int, int]] = []
    for path in _list_final_shard_paths(exp_dir, suite):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            # shard 文件可能正在被 worker 写入 / 原子 rename 中, 下一轮再读
            continue

        # 优先用 payload 里写的字段; 否则 fallback 到文件名解析
        idx_raw = payload.get("shard_index", -1)
        n_raw = payload.get("num_shards", -1)
        try:
            idx = int(idx_raw)
            n = int(n_raw)
        except (TypeError, ValueError):
            idx, n = -1, -1
        if idx < 0 or n <= 0:
            idx_fb, n_fb = _parse_shard_index_from_name(path)
            if idx < 0:
                idx = idx_fb
            if n <= 0:
                n = n_fb

        out.append((payload, idx, n))
    return out


def _aggregate_shards(
    shards: List[Tuple[Dict[str, Any], int, int]]
) -> Dict[str, Any]:
    """跨 shard 聚合 metrics + 单 shard 详情."""
    overall_ep = overall_succ = 0
    cat_buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ep": 0, "succ": 0})
    diff_buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"ep": 0, "succ": 0})
    shard_list: List[Dict[str, Any]] = []

    for sh, idx, ns in shards:
        m = sh.get("metrics", {})
        ov = m.get("overall", {}) or {}
        ep = int(ov.get("episodes", 0))
        succ = int(ov.get("successes", 0))
        overall_ep += ep
        overall_succ += succ

        for cat, v in (m.get("by_category", {}) or {}).items():
            cat_buckets[cat]["ep"] += int(v.get("episodes", 0))
            cat_buckets[cat]["succ"] += int(v.get("successes", 0))
        for diff, v in (m.get("by_difficulty", {}) or {}).items():
            diff_buckets[diff]["ep"] += int(v.get("episodes", 0))
            diff_buckets[diff]["succ"] += int(v.get("successes", 0))

        shard_list.append(
            {
                "shard_index": idx,
                "num_shards": ns,
                "episodes": ep,
                "successes": succ,
            }
        )

    shard_list.sort(key=lambda s: s.get("shard_index", -1))

    return {
        "overall": {"episodes": overall_ep, "successes": overall_succ},
        "by_category": {k: dict(v) for k, v in cat_buckets.items()},
        "by_difficulty": {k: dict(v) for k, v in diff_buckets.items()},
        "shards": shard_list,
    }


def _suite_dirs_for_filter(
    experiments_root: Path,
    name_filter: Optional[str],
    explicit_dirs: List[Path],
) -> Dict[str, Path]:
    """对每个 suite, 找最新一个 (按 mtime) 匹配 filter 的实验目录.

    返回 {suite_name: experiment_dir}. 未找到的 suite 不在 dict 里.
    """
    if explicit_dirs:
        candidates = list(explicit_dirs)
    elif experiments_root.is_dir():
        candidates = [p for p in experiments_root.iterdir() if p.is_dir()]
    else:
        candidates = []

    if name_filter:
        candidates = [p for p in candidates if name_filter in p.name]

    out: Dict[str, Tuple[Path, float]] = {}
    for d in candidates:
        for suite in LIBERO_PLUS_SUITES:
            shards = _list_final_shard_paths(d, suite)
            if not shards:
                continue
            mtimes: List[float] = []
            for s in shards:
                try:
                    mtimes.append(os.path.getmtime(s))
                except (OSError, FileNotFoundError):
                    # 文件可能在原子 rename 期间瞬时消失, 忽略
                    continue
            if not mtimes:
                continue
            latest_mtime = max(mtimes)
            prev = out.get(suite)
            if prev is None or latest_mtime > prev[1]:
                out[suite] = (d, latest_mtime)

    return {k: v[0] for k, v in out.items()}


def _is_active(exp_dir: Path) -> bool:
    """exp 目录里有 worker_shard*.log 在过去 60s 内被写过 -> 算 active."""
    pattern = str(exp_dir / "worker_shard*.log")
    now = time.time()
    for log in glob.glob(pattern):
        try:
            if now - os.path.getmtime(log) < 60:
                return True
        except OSError:
            continue
    return False


def _extract_step_tag(name: str) -> str:
    m = re.search(r"step_\d+", name)
    return m.group(0) if m else "step_?"


def _render_section_header(title: str, color: bool) -> str:
    return _c(f"  {title}", ANSI_BOLD, ANSI_CYAN, enable=color)


def _render_suite_detail(
    suite: str,
    exp_dir: Path,
    is_active: bool,
    *,
    color: bool,
    show_per_shard: bool = True,
) -> List[str]:
    """渲染单个 suite 的详细 (per-shard / by_category / by_difficulty) 区块."""
    shards = _read_shard_jsons(exp_dir, suite)
    agg = _aggregate_shards(shards)
    suite_target = SUITE_TOTAL_TASKS.get(suite, 0)

    num_shards_total = max(
        (s.get("num_shards", 0) for s in agg["shards"] if s.get("num_shards", 0) > 0),
        default=len(agg["shards"]),
    )
    if num_shards_total <= 0:
        num_shards_total = max(len(agg["shards"]), 1)

    out: List[str] = []
    suite_label = _c(
        f"{suite}  ({exp_dir.name})",
        ANSI_BOLD,
        ANSI_MAGENTA,
        enable=color,
    )
    active_tag = (
        _c(" [ACTIVE]", ANSI_YELLOW, ANSI_BOLD, enable=color)
        if is_active
        else ""
    )
    out.append(_render_section_header(f"详情: {suite_label}{active_tag}", color))
    out.append("")

    # 1. Per-shard
    if show_per_shard and agg["shards"]:
        out.append(
            f"  {'Shard':<10}{'Done':>9}{'Target':>10}{'Cov':>9}"
            f"{'Succ':>8}{'SR':>11}"
        )
        out.append("  " + "-" * 64)
        per_shard_target = (
            (suite_target + num_shards_total - 1) // num_shards_total
            if num_shards_total
            else 0
        )
        for sh in agg["shards"]:
            idx = sh.get("shard_index", -1)
            ns = sh.get("num_shards", num_shards_total)
            ep = sh["episodes"]
            succ = sh["successes"]
            cov = (ep / per_shard_target) if per_shard_target else 0.0
            out.append(
                f"  shard {idx}/{ns} {ep:>9d}{per_shard_target:>10d}"
                f"{cov * 100:>8.1f}%{succ:>8d}    {_fmt_pct(succ, ep, color)}"
            )
        out.append("")

    # 2. By category
    cat_data = agg["by_category"]
    if cat_data:
        out.append(_render_section_header("分 Category", color))
        out.append(f"  {'Category':<25}{'Episodes':>11}{'Succ':>8}{'SR':>11}")
        out.append("  " + "-" * 64)
        ordered_cats = list(LIBERO_PLUS_CATEGORIES) + sorted(
            c for c in cat_data if c not in LIBERO_PLUS_CATEGORIES
        )
        for cat in ordered_cats:
            v = cat_data.get(cat)
            if not v or v["ep"] == 0:
                continue
            out.append(
                f"  {cat:<25}{v['ep']:>11d}{v['succ']:>8d}    "
                f"{_fmt_pct(v['succ'], v['ep'], color)}"
            )
        out.append("")

    # 3. By difficulty
    diff_data = agg["by_difficulty"]
    if diff_data:
        out.append(
            _render_section_header(
                "分 Difficulty  (L1=最轻微扰动 → L5=最强扰动)", color
            )
        )
        out.append(f"  {'Difficulty':<25}{'Episodes':>11}{'Succ':>8}{'SR':>11}")
        out.append("  " + "-" * 64)
        for diff in sorted(diff_data.keys()):
            v = diff_data[diff]
            if v["ep"] == 0:
                continue
            out.append(
                f"  {diff:<25}{v['ep']:>11d}{v['succ']:>8d}    "
                f"{_fmt_pct(v['succ'], v['ep'], color)}"
            )
        out.append("")

    return out


def render_dashboard(
    suite_to_dir: Dict[str, Path],
    *,
    color: bool,
    detail_mode: str = "all",
) -> str:
    lines: List[str] = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append(
        _c("=" * 88, ANSI_BOLD, ANSI_CYAN, enable=color)
    )
    lines.append(
        _c(
            f"  📊 LIBERO-plus 评估实时进度  [{now_str}]",
            ANSI_BOLD,
            ANSI_CYAN,
            enable=color,
        )
    )
    lines.append(_c("=" * 88, ANSI_BOLD, ANSI_CYAN, enable=color))
    lines.append("")

    if not suite_to_dir:
        lines.append(
            _c("  (尚未发现任何评估实验目录)", ANSI_DIM, enable=color)
        )
        lines.append("")
        return "\n".join(lines)

    # ---------- 1. Suite 总览 ----------
    lines.append(_render_section_header("Suite 总览", color))
    lines.append(
        f"  {'Suite':<18}{'Step':<14}{'Status':<10}"
        f"{'Episodes':>11}{'Target':>9}{'Cov':>9}{'SR':>10}"
    )
    lines.append("  " + "-" * 84)

    grand_ep = grand_succ = 0
    grand_target = 0
    active_suite: Optional[str] = None
    active_dir: Optional[Path] = None

    for suite in LIBERO_PLUS_SUITES:
        exp_dir = suite_to_dir.get(suite)
        if exp_dir is None:
            lines.append(
                f"  {suite:<18}"
                + _c(f"{'(无实验)':<14}{'-':<10}{'':>11}{'':>9}{'':>9}{'':>10}",
                     ANSI_DIM, enable=color)
            )
            continue

        step_tag = _extract_step_tag(exp_dir.name)
        shards = _read_shard_jsons(exp_dir, suite)
        agg = _aggregate_shards(shards)
        ep = agg["overall"]["episodes"]
        succ = agg["overall"]["successes"]
        target = SUITE_TOTAL_TASKS.get(suite, 0)
        cov = (ep / target) if target else 0.0
        active = _is_active(exp_dir)

        if active and active_suite is None:
            active_suite = suite
            active_dir = exp_dir

        if active:
            status = _c("⏳ 进行中", ANSI_YELLOW, ANSI_BOLD, enable=color)
        elif ep >= target and target > 0:
            status = _c("✅ 完成  ", ANSI_GREEN, ANSI_BOLD, enable=color)
        else:
            status = _c("⏸  停止  ", ANSI_DIM, enable=color)

        lines.append(
            f"  {suite:<18}{step_tag:<14}{status:<22}"
            f"{ep:>11d}{target:>9d}{cov * 100:>8.1f}%"
            f"  {_fmt_pct(succ, ep, color)}"
        )

        grand_ep += ep
        grand_succ += succ
        grand_target += target

    lines.append("  " + "-" * 84)
    g_cov = (grand_ep / grand_target) if grand_target else 0.0
    lines.append(
        f"  {_c('TOTAL', ANSI_BOLD, enable=color):<27}"
        f"{'':<10}"
        f"{grand_ep:>11d}{grand_target:>9d}{g_cov * 100:>8.1f}%"
        f"  {_fmt_pct(grand_succ, grand_ep, color)}"
    )
    lines.append("")

    # ---------- 2. 各 suite 详情 ----------
    # 决定展开哪些 suite
    if detail_mode == "active":
        focus = active_suite
        if focus is None:
            for s in LIBERO_PLUS_SUITES:
                if s in suite_to_dir:
                    focus = s
                    break
        suites_to_render = [focus] if focus else []
    elif detail_mode in LIBERO_PLUS_SUITES:
        suites_to_render = [detail_mode] if detail_mode in suite_to_dir else []
    else:  # "all" / "compact" 之类未识别值都视作 all
        suites_to_render = [s for s in LIBERO_PLUS_SUITES if s in suite_to_dir]

    for s in suites_to_render:
        d = suite_to_dir.get(s)
        if d is None:
            continue
        is_active = (s == active_suite)
        lines.extend(
            _render_suite_detail(s, d, is_active, color=color, show_per_shard=True)
        )

    lines.append(_c("=" * 88, ANSI_DIM, enable=color))
    lines.append(
        _c(
            "  按 Ctrl+C 退出  |  --interval N 改刷新间隔  |  --once 单次输出  |  "
            "--detail active|all|<suite>",
            ANSI_DIM,
            enable=color,
        )
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LIBERO-plus 评估实时进度 dashboard"
    )
    parser.add_argument(
        "--experiments_root",
        type=str,
        default=str(EXPERIMENTS_ROOT_DEFAULT),
        help="experiments 目录, 默认 eval/Sai0_1/libero_plus/OFT1_0/experiments",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="实验目录名 substring 过滤 (例如 step_180000)",
    )
    parser.add_argument(
        "--exp",
        type=str,
        nargs="*",
        default=[],
        help="显式指定实验目录, 可多个; 给了就忽略 --experiments_root 自动扫描",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="刷新间隔 (秒), 默认 5",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="只输出一次后退出 (cron / pipe 用)",
    )
    parser.add_argument(
        "--no_color",
        action="store_true",
        help="禁用 ANSI 颜色 (输出到文件时建议加)",
    )
    parser.add_argument(
        "--detail",
        type=str,
        default="all",
        help=(
            "详情展开范围: 'all' (默认, 展开所有有数据的 suite) | "
            "'active' (只展开当前正在跑的 suite) | "
            "<suite_name> (只展开指定 suite, 例如 libero_spatial)"
        ),
    )
    args = parser.parse_args()

    color = _supports_color() and not args.no_color
    experiments_root = Path(args.experiments_root)
    explicit_dirs = [Path(p) for p in args.exp]
    name_filter = args.filter or None
    detail_mode = args.detail

    try:
        while True:
            suite_to_dir = _suite_dirs_for_filter(
                experiments_root, name_filter, explicit_dirs
            )
            text = render_dashboard(
                suite_to_dir, color=color, detail_mode=detail_mode
            )
            if args.once:
                # 不清屏, 直接打印一次
                print(text)
                return 0

            if color:
                sys.stdout.write(ANSI_CLEAR)
            sys.stdout.write(text)
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(max(0.5, args.interval))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
