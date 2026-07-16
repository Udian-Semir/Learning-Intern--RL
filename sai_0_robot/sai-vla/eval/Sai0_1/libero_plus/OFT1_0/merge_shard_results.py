"""把多 GPU shard 跑出来的 ``eval_results_<suite>_shard{K}_of_{N}.json`` 合并成
一份总 JSON ``eval_results_<suite>_merged.json``,并重新计算 overall /
by_category / by_difficulty 指标。

用法 (从 ``run_eval_libero_plus_multi_gpu.sh`` 中自动调用):

    python -m eval.Sai0_1.libero_plus.OFT1_0.merge_shard_results \
        --video_dir /path/to/experiment_dir \
        --task_suite_name libero_spatial \
        --num_shards 8

也可以直接传一组 shard 文件:

    python -m eval.Sai0_1.libero_plus.OFT1_0.merge_shard_results \
        --shard_files a.json b.json c.json \
        --output merged.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

LIBERO_PLUS_CATEGORIES = (
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
    "Objects Layout",
)


def _aggregate(task_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    overall_eps = 0
    overall_succ = 0
    cat_buckets: Dict[str, Dict[str, int]] = {
        c: {"ep": 0, "succ": 0} for c in LIBERO_PLUS_CATEGORIES
    }
    cat_buckets["Unknown"] = {"ep": 0, "succ": 0}
    diff_buckets: Dict[str, Dict[str, int]] = {}

    for r in task_results.values():
        cat = r.get("category", "Unknown")
        diff = r.get("difficulty_level")
        ep = r.get("episodes", 0)
        succ = r.get("successes", 0)
        overall_eps += ep
        overall_succ += succ

        cat_buckets.setdefault(cat, {"ep": 0, "succ": 0})
        cat_buckets[cat]["ep"] += ep
        cat_buckets[cat]["succ"] += succ

        diff_key = "None" if diff is None else f"L{diff}"
        diff_buckets.setdefault(diff_key, {"ep": 0, "succ": 0})
        diff_buckets[diff_key]["ep"] += ep
        diff_buckets[diff_key]["succ"] += succ

    def _norm(b: Dict[str, int]) -> Dict[str, Any]:
        rate = b["succ"] / b["ep"] if b["ep"] else 0.0
        return {"episodes": b["ep"], "successes": b["succ"], "success_rate": rate}

    return {
        "overall": {
            "episodes": overall_eps,
            "successes": overall_succ,
            "success_rate": (overall_succ / overall_eps) if overall_eps else 0.0,
        },
        "by_category": {c: _norm(v) for c, v in cat_buckets.items() if v["ep"]},
        "by_difficulty": {k: _norm(v) for k, v in diff_buckets.items() if v["ep"]},
    }


def _print_summary(metrics: Dict[str, Any], suite_name: str, num_shards: int) -> None:
    print("\n" + "=" * 76)
    print(f"📊 LIBERO-plus 评估结果汇总 (合并 {num_shards} 张 GPU 的 shard) - {suite_name}")
    print("=" * 76)
    o = metrics["overall"]
    print(
        f"  总 episodes: {o['episodes']}, 成功: {o['successes']}, "
        f"Overall SR: {o['success_rate'] * 100:.2f}%"
    )
    print("-" * 76)
    print(f"  {'Category':<25}{'Episodes':>10}{'Succ':>8}{'SR':>10}")
    for cat in list(LIBERO_PLUS_CATEGORIES) + ["Unknown"]:
        if cat in metrics["by_category"]:
            v = metrics["by_category"][cat]
            print(
                f"  {cat:<25}{v['episodes']:>10}{v['successes']:>8}"
                f"{v['success_rate'] * 100:>9.2f}%"
            )
    print("-" * 76)
    print(f"  {'Difficulty':<25}{'Episodes':>10}{'Succ':>8}{'SR':>10}")
    for k in sorted(metrics["by_difficulty"].keys()):
        v = metrics["by_difficulty"][k]
        print(
            f"  {k:<25}{v['episodes']:>10}{v['successes']:>8}"
            f"{v['success_rate'] * 100:>9.2f}%"
        )
    print("=" * 76)


def _resolve_shard_paths(args: argparse.Namespace) -> List[Path]:
    if args.shard_files:
        return [Path(p) for p in args.shard_files]
    if args.video_dir and args.task_suite_name and args.num_shards:
        video_dir = Path(args.video_dir)
        return [
            video_dir / (
                f"eval_results_{args.task_suite_name}_shard{i}_of_{args.num_shards}.json"
            )
            for i in range(args.num_shards)
        ]
    raise ValueError(
        "必须提供 (--video_dir + --task_suite_name + --num_shards) 或 --shard_files"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LIBERO-plus shard results")
    parser.add_argument("--video_dir", type=str, default="",
                        help="实验输出目录 (与 run_eval_libero_plus_multi_gpu.sh 中的 VIDEO_DIR 一致)")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        help="评估的 task suite 名")
    parser.add_argument("--num_shards", type=int, default=0,
                        help="GPU 数 (= shard 数). 用于自动定位 shard 文件名")
    parser.add_argument("--shard_files", type=str, nargs="*", default=None,
                        help="(可选) 直接指定一组 shard JSON 路径, 优先级高于 --num_shards")
    parser.add_argument("--output", type=str, default="",
                        help="(可选) 输出 JSON 路径. 默认为 ${video_dir}/eval_results_${suite}_merged.json")
    parser.add_argument("--allow_partial", action="store_true",
                        help="某些 shard 文件不存在时仍合并已有的 (默认会报错)")
    args = parser.parse_args()

    shard_paths = _resolve_shard_paths(args)
    print(f"🔍 待合并的 shard 文件 ({len(shard_paths)} 个):")
    valid_paths: List[Path] = []
    for p in shard_paths:
        exists = p.exists()
        marker = "✓" if exists else "✗"
        print(f"   {marker} {p}")
        if exists:
            valid_paths.append(p)

    if len(valid_paths) < len(shard_paths) and not args.allow_partial:
        raise FileNotFoundError(
            f"只找到 {len(valid_paths)}/{len(shard_paths)} 个 shard 结果, "
            f"如果是有意只合并部分, 加 --allow_partial"
        )
    if not valid_paths:
        raise RuntimeError("没有可用的 shard 结果文件")

    merged_task_results: Dict[str, Dict[str, Any]] = {}
    meta = None
    overlap_keys = set()
    for p in valid_paths:
        with p.open("r") as f:
            data = json.load(f)
        if meta is None:
            meta = {
                k: data.get(k)
                for k in (
                    "suite", "vlm_type", "vlm_model_path",
                    "checkpoint_path", "dataset_path", "config",
                )
            }
        shard_results = data.get("task_results", {})
        for k, v in shard_results.items():
            if k in merged_task_results:
                # 不同 shard 同一个 task_id 重复 - 理论上不应发生 (stride 切分互不相交),
                # 真发生了取已有的, 不覆盖
                overlap_keys.add(k)
                continue
            merged_task_results[k] = v

    if overlap_keys:
        print(
            f"⚠️ 发现 {len(overlap_keys)} 个 task_id 在多个 shard 中重复, "
            "已忽略后来者(可能是 shard 切分参数被改过)"
        )

    final_metrics = _aggregate(merged_task_results)

    if args.output:
        out_path = Path(args.output)
    elif args.video_dir:
        out_path = (
            Path(args.video_dir)
            / f"eval_results_{args.task_suite_name}_merged.json"
        )
    else:
        out_path = Path(f"eval_results_{args.task_suite_name}_merged.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **(meta or {}),
        "merged_from_shards": [str(p) for p in valid_paths],
        "task_results": merged_task_results,
        "metrics": final_metrics,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    _print_summary(final_metrics, args.task_suite_name, len(valid_paths))
    print(f"\n💾 合并结果已保存到: {out_path}")


if __name__ == "__main__":
    main()
