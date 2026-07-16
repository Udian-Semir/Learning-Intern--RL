#!/usr/bin/env python3
"""
Qwen3-VL 多数据集多 GPU Hidden States 提取脚本

功能:
1. 自动枚举 `dataset_root` 下的多个 LeRobot 子数据集
2. 每个 GPU 启动一个独立 worker，并在该 worker 内常驻加载一个完整模型
3. 每个子数据集自动检测自己的 image keys，也支持手动覆盖
4. 多个数据集按队列分发到不同 GPU 上并行提取
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from utils.extract_vlm_hidden_state.S0_1.qwen.qwen_extract_vlm_hidden_states import (
    DEFAULT_QWEN_MODEL_PATH,
    resolve_image_keys,
    validate_layers,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 多数据集多 GPU Hidden States 提取工具"
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="包含多个 LeRobot 子数据集的根目录",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        default=None,
        help="仅处理指定子数据集，逗号分隔；不传则优先按 recipe.json 枚举",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="统一输出根目录；不传则输出到各自数据集目录下的 vlm_hidden_states",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0",
        help="参与推理的 GPU 编号，逗号分隔，例如: 0,1,2,3",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_QWEN_MODEL_PATH,
        help=f"Qwen3-VL 模型路径 (默认: {DEFAULT_QWEN_MODEL_PATH})",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="14",
        help="提取层号，逗号分隔 (默认: 14)",
    )
    parser.add_argument(
        "--image_keys",
        type=str,
        default=None,
        help="手动指定所有数据集共用的图像键；不传则每个数据集自动检测",
    )
    parser.add_argument(
        "--auto_image_keys",
        action="store_true",
        default=True,
        help="自动从每个数据集检测图像视角键名 (默认: True)",
    )
    parser.add_argument(
        "--no_auto_image_keys",
        action="store_false",
        dest="auto_image_keys",
        help="关闭自动检测，未指定 --image_keys 时回退到默认值",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="模型推理数据类型 (默认: bfloat16)",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="保存 hidden states 的数据类型 (默认: float32)",
    )
    parser.add_argument(
        "--flip_images",
        action="store_true",
        default=True,
        help="翻转图像 180 度 (默认: True)",
    )
    parser.add_argument(
        "--no_flip_images",
        action="store_false",
        dest="flip_images",
        help="不翻转图像",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="action",
        help="Prompt 模板名称或自定义模板",
    )
    parser.add_argument(
        "--content_order",
        type=str,
        default="images_first",
        choices=["images_first", "text_first", "interleaved", "single_image"],
        help="内容顺序 (默认: images_first)",
    )
    parser.add_argument(
        "--lowercase_instruction",
        action="store_true",
        default=True,
        help="将指令转为小写 (默认: True)",
    )
    parser.add_argument(
        "--no_lowercase_instruction",
        action="store_false",
        dest="lowercase_instruction",
        help="不转换指令为小写",
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        default=True,
        help="添加 generation prompt (默认: True)",
    )
    parser.add_argument(
        "--no_generation_prompt",
        action="store_false",
        dest="add_generation_prompt",
        help="不添加 generation prompt",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=None,
        help="起始帧索引 (用于断点续传)",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="结束帧索引 (用于断点续传)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="数据预加载 worker 数量 (默认: 4)",
    )
    parser.add_argument(
        "--prefetch_size",
        type=int,
        default=8,
        help="预加载队列大小 (默认: 8)",
    )
    parser.add_argument(
        "--max_datasets",
        type=int,
        default=None,
        help="仅处理前 N 个数据集，便于调试",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印数据集和检测到的 image keys，不真正推理",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="日志文件路径，记录每个数据集的完成状态；启动时自动跳过已完成的数据集",
    )
    parser.add_argument(
        "--no_skip_completed",
        action="store_true",
        help="即使日志中标记为完成，也重新处理（不跳过）",
    )

    return parser.parse_args()


def parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_completed_datasets(log_file: Optional[str]) -> Set[str]:
    """从日志文件中读取已标记为 ok 的数据集名称。"""
    completed = set()
    if log_file is None:
        return completed
    log_path = Path(log_file)
    if not log_path.exists():
        return completed
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
                if entry.get("status") == "ok":
                    completed.add(entry["dataset_name"])
            except json.JSONDecodeError:
                continue
    return completed


def append_log(log_file: Optional[str], entry: Dict):
    """将一条结果追加写入日志文件（JSON Lines 格式）。"""
    if log_file is None:
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry_with_ts = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **entry}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry_with_ts, ensure_ascii=False) + "\n")


def resolve_model_path_for_workers(model_path: str) -> str:
    """
    将模型路径解析为 worker 可直接使用的本地路径。

    目的:
    - 如果传入的已经是本地目录，直接返回
    - 如果传入的是 HuggingFace repo id，则在主进程里先解析到本地 snapshot
      避免每个多进程 worker 都单独走 huggingface_hub/httpx 初始化
    """
    local_path = Path(model_path).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    if "/" in model_path:
        org, name = model_path.split("/", 1)
        hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        hub_root = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
        repo_cache_dir = hub_root / f"models--{org}--{name}" / "snapshots"
        if repo_cache_dir.exists():
            snapshots = sorted([p for p in repo_cache_dir.iterdir() if p.is_dir()])
            if snapshots:
                resolved = str(snapshots[-1].resolve())
                print(f"[INFO] 使用本地 HuggingFace 缓存: {resolved}")
                return resolved

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            f"模型路径 `{model_path}` 不是本地目录，且当前环境缺少 huggingface_hub。"
            "请改为传本地模型目录，或先安装 huggingface_hub，或确保模型已缓存到标准 HF cache。"
        ) from e

    try:
        resolved = snapshot_download(repo_id=model_path, local_files_only=True)
        print(f"[INFO] 使用本地 HuggingFace 缓存: {resolved}")
        return resolved
    except Exception:
        print(f"[INFO] 本地缓存未命中，准备在主进程解析模型: {model_path}")

    resolved = snapshot_download(repo_id=model_path)
    print(f"[INFO] 模型已解析到本地目录: {resolved}")
    return resolved


def discover_dataset_paths(dataset_root: Path, dataset_names: Optional[List[str]] = None) -> List[Path]:
    if dataset_names:
        dataset_paths = [dataset_root / name for name in dataset_names]
    else:
        recipe_path = dataset_root / "recipe.json"
        dataset_paths = []
        if recipe_path.exists():
            with open(recipe_path, "r") as f:
                recipe = json.load(f)
            dataset_paths = [dataset_root / item["name"] for item in recipe.get("datasets", [])]
        else:
            dataset_paths = sorted(
                [
                    path for path in dataset_root.iterdir()
                    if path.is_dir() and (path / "meta" / "info.json").exists()
                ],
                key=lambda p: p.name,
            )

    valid_dataset_paths = []
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            print(f"⚠️ 跳过不存在的数据集: {dataset_path}")
            continue
        if not (dataset_path / "meta" / "info.json").exists():
            print(f"⚠️ 跳过非 LeRobot 数据集目录: {dataset_path}")
            continue
        valid_dataset_paths.append(dataset_path)

    return valid_dataset_paths


def build_output_dir(dataset_path: Path, output_root: Optional[str]) -> str:
    if not output_root:
        return str(dataset_path / "vlm_hidden_states")
    return str(Path(output_root) / dataset_path.name / "vlm_hidden_states")


def process_single_dataset(dataset_path: str, backbone, worker_args: Dict, gpu_id: int) -> Dict:
    from VLMs.S0_1.backbone.model_selector import (
        HiddenStateExtractor,
        get_all_frames_info,
        load_dataset_info,
    )

    dataset_path_obj = Path(dataset_path)
    image_keys = resolve_image_keys(
        dataset_path=dataset_path,
        image_keys_arg=worker_args["image_keys"],
        auto_image_keys=worker_args["auto_image_keys"],
    )
    output_dir = build_output_dir(dataset_path_obj, worker_args["output_root"])

    print(
        f"\n[GPU {gpu_id}] 开始处理数据集: {dataset_path_obj.name}\n"
        f"[GPU {gpu_id}] image_keys={image_keys}, output_dir={output_dir}"
    )

    info, tasks = load_dataset_info(dataset_path)
    frames_info = get_all_frames_info(dataset_path, info)

    start_idx = worker_args["start_idx"] if worker_args["start_idx"] is not None else 0
    end_idx = worker_args["end_idx"] if worker_args["end_idx"] is not None else len(frames_info)
    frames_to_process = frames_info[start_idx:end_idx]

    extractor = HiddenStateExtractor(
        backbone=backbone,
        dataset_path=dataset_path,
        output_dir=output_dir,
        tasks=tasks,
        flip_images=worker_args["flip_images"],
        num_workers=worker_args["num_workers"],
        prefetch_size=worker_args["prefetch_size"],
        save_to_file=True,
        image_keys=image_keys,
        save_dtype=worker_args["save_dtype"],
    )

    chunks_size = info.get("chunks_size", 1000)

    start_time = time.time()
    processed, errors = extractor.extract_per_episode(frames_to_process, chunks_size=chunks_size)
    elapsed = time.time() - start_time

    return {
        "dataset_name": dataset_path_obj.name,
        "dataset_path": dataset_path,
        "gpu_id": gpu_id,
        "image_keys": image_keys,
        "processed": processed,
        "errors": errors,
        "elapsed_sec": elapsed,
        "status": "ok",
    }


def gpu_worker(gpu_id: int, job_queue, result_queue, worker_args: Dict):
    from VLMs.S0_1.backbone.model_selector import (
        PROMPT_TEMPLATES,
        create_vlm_backbone,
    )

    device = f"cuda:{gpu_id}"
    prompt_template = worker_args["prompt_template"]
    if prompt_template in PROMPT_TEMPLATES:
        prompt_template = PROMPT_TEMPLATES[prompt_template]

    try:
        print(f"[GPU {gpu_id}] 正在加载模型到 {device} ...")
        backbone = create_vlm_backbone(
            model_type="qwen3_vl",
            model_path=worker_args["model_path"],
            device=device,
            layers=worker_args["layers"],
            prompt_template=prompt_template,
            content_order=worker_args["content_order"],
            flip_images=worker_args["flip_images"],
            dtype=worker_args["dtype"],
            verbose=worker_args["verbose"],
            lowercase_instruction=worker_args["lowercase_instruction"],
            add_generation_prompt=worker_args["add_generation_prompt"],
        )
        print(f"[GPU {gpu_id}] 模型加载完成")
    except Exception as e:
        result_queue.put(
            {
                "dataset_name": None,
                "dataset_path": None,
                "gpu_id": gpu_id,
                "status": "worker_init_failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        try:
            dataset_path = job_queue.get(timeout=10)
        except Exception:
            continue
        if dataset_path is None:
            print(f"[GPU {gpu_id}] worker 正常结束")
            break

        try:
            result = process_single_dataset(dataset_path, backbone, worker_args, gpu_id)
        except Exception as e:
            result = {
                "dataset_name": Path(dataset_path).name,
                "dataset_path": dataset_path,
                "gpu_id": gpu_id,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        result_queue.put(result)

    result_queue.put({
        "dataset_name": None,
        "gpu_id": gpu_id,
        "status": "worker_done",
    })


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 不存在: {dataset_root}")

    gpu_ids = [int(x) for x in parse_csv_list(args.gpu_ids)]
    if not gpu_ids:
        raise ValueError("至少需要通过 --gpu_ids 指定一个 GPU")

    resolved_model_path = resolve_model_path_for_workers(args.model_path)

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    validate_layers(layers, resolved_model_path)

    dataset_names = parse_csv_list(args.dataset_names)
    dataset_paths = discover_dataset_paths(dataset_root, dataset_names or None)
    if args.max_datasets is not None:
        dataset_paths = dataset_paths[: args.max_datasets]

    if not dataset_paths:
        raise ValueError(f"在 {dataset_root} 下未找到可处理的数据集")

    completed_datasets = set()
    if args.log_file and not args.no_skip_completed:
        completed_datasets = load_completed_datasets(args.log_file)
        if completed_datasets:
            print(f"\n📋 从日志中检测到 {len(completed_datasets)} 个已完成数据集，将自动跳过")

    total_before_skip = len(dataset_paths)
    if completed_datasets:
        dataset_paths = [p for p in dataset_paths if p.name not in completed_datasets]

    print("\n" + "=" * 80)
    print("Qwen3-VL 多数据集多 GPU Hidden States 提取")
    print("=" * 80)
    print(f"数据集根目录: {dataset_root}")
    print(f"数据集总数: {total_before_skip}")
    if completed_datasets:
        print(f"已完成跳过: {len(completed_datasets)}")
    print(f"待处理数量: {len(dataset_paths)}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"模型路径: {args.model_path}")
    print(f"Worker 本地模型路径: {resolved_model_path}")
    print(f"提取层: {layers}")
    print(f"输出根目录: {args.output_root or '各数据集目录/vlm_hidden_states'}")
    if args.log_file:
        print(f"日志文件: {args.log_file}")
    print("=" * 80)

    for dataset_path in dataset_paths:
        image_keys = resolve_image_keys(
            dataset_path=str(dataset_path),
            image_keys_arg=args.image_keys,
            auto_image_keys=args.auto_image_keys,
        )
        print(f"  - {dataset_path.name}: image_keys={image_keys}")

    if not dataset_paths:
        print("\n所有数据集均已完成，无需处理。")
        return

    if args.dry_run:
        print("\nDry run 结束，未执行推理。")
        return

    worker_args = {
        "model_path": resolved_model_path,
        "layers": layers,
        "image_keys": args.image_keys,
        "auto_image_keys": args.auto_image_keys,
        "output_root": args.output_root,
        "dtype": args.dtype,
        "save_dtype": args.save_dtype,
        "flip_images": args.flip_images,
        "prompt_template": args.prompt_template,
        "content_order": args.content_order,
        "lowercase_instruction": args.lowercase_instruction,
        "add_generation_prompt": args.add_generation_prompt,
        "start_idx": args.start_idx,
        "end_idx": args.end_idx,
        "num_workers": args.num_workers,
        "prefetch_size": args.prefetch_size,
        "verbose": args.verbose,
    }

    ctx = mp.get_context("spawn")
    job_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for dataset_path in dataset_paths:
        job_queue.put(str(dataset_path))
    for _ in gpu_ids:
        job_queue.put(None)

    workers: Dict[int, mp.Process] = {}
    for gpu_id in gpu_ids:
        proc = ctx.Process(
            target=gpu_worker,
            args=(gpu_id, job_queue, result_queue, worker_args),
            daemon=False,
        )
        proc.start()
        workers[gpu_id] = proc

    results = []
    expected_results = len(dataset_paths)
    worker_init_failures = 0
    workers_done = set()

    def _check_dead_workers() -> List[int]:
        """检测意外退出的 worker (OOM / CUDA crash / segfault 等)。"""
        dead = []
        for gid, proc in workers.items():
            if gid in workers_done:
                continue
            if not proc.is_alive():
                dead.append(gid)
        return dead

    while len(results) < expected_results:
        try:
            result = result_queue.get(timeout=30)
        except Exception:
            dead = _check_dead_workers()
            if dead:
                for gid in dead:
                    exit_code = workers[gid].exitcode
                    workers_done.add(gid)
                    msg = (
                        f"\n💀 GPU {gid} worker 意外退出 (exit_code={exit_code})，"
                        f"可能被 OOM Killer 杀掉或 CUDA 崩溃"
                    )
                    print(msg)
                    append_log(args.log_file, {
                        "dataset_name": None,
                        "gpu_id": gid,
                        "status": "worker_crashed",
                        "exit_code": exit_code,
                    })

                alive_workers = [
                    gid for gid in workers if gid not in workers_done
                ]
                if not alive_workers:
                    print("\n❌ 所有 worker 均已退出，中止等待")
                    break
                print(
                    f"  仍有 {len(alive_workers)} 个 worker 存活: GPU {alive_workers}，继续等待..."
                )
            continue

        if result["status"] == "worker_init_failed":
            worker_init_failures += 1
            workers_done.add(result["gpu_id"])
            print(f"\n❌ GPU {result['gpu_id']} 初始化失败: {result['error']}")
            print(result["traceback"])
            if worker_init_failures == len(gpu_ids):
                break
            continue

        if result["status"] == "worker_done":
            workers_done.add(result["gpu_id"])
            print(f"\n[GPU {result['gpu_id']}] worker 已正常退出")
            if len(workers_done) == len(gpu_ids):
                break
            continue

        results.append(result)
        log_entry = {
            "dataset_name": result.get("dataset_name"),
            "dataset_path": result.get("dataset_path"),
            "gpu_id": result.get("gpu_id"),
            "status": result["status"],
        }
        if result["status"] == "ok":
            log_entry.update({
                "processed": result["processed"],
                "errors": result["errors"],
                "elapsed_sec": round(result["elapsed_sec"], 1),
            })
            print(
                f"\n✅ 数据集完成: {result['dataset_name']} | GPU {result['gpu_id']} | "
                f"processed={result['processed']} | errors={result['errors']} | "
                f"耗时={result['elapsed_sec']:.1f}s"
            )
        else:
            log_entry["error"] = result.get("error", "unknown")
            print(f"\n❌ 数据集失败: {result['dataset_name']} | GPU {result['gpu_id']}")
            if "traceback" in result:
                print(result["traceback"])
        append_log(args.log_file, log_entry)

    for proc in workers.values():
        proc.join(timeout=60)
        if proc.is_alive():
            print(f"⚠️ Worker PID {proc.pid} 未在超时内退出，强制终止")
            proc.kill()
            proc.join(timeout=10)

    ok_results = [item for item in results if item["status"] == "ok"]
    failed_results = [item for item in results if item["status"] != "ok"]
    missing_count = expected_results - len(results)
    crashed_workers = [
        gid for gid in workers_done
        if workers[gid].exitcode not in (None, 0)
    ]

    print("\n" + "=" * 80)
    print("全部任务完成")
    print("=" * 80)
    print(f"成功数据集: {len(ok_results)}")
    print(f"失败数据集: {len(failed_results)}")
    if missing_count > 0:
        print(f"未完成数据集 (worker crash): {missing_count}")
    print(f"总数据集数: {len(dataset_paths)}")

    if crashed_workers:
        print(f"\n💀 崩溃的 worker:")
        for gid in crashed_workers:
            print(f"  - GPU {gid} (exit_code={workers[gid].exitcode})")
        print("  提示: exit_code=-9 通常是 OOM Killer，检查系统内存/GPU 显存是否足够")

    if ok_results:
        total_processed = sum(item["processed"] for item in ok_results)
        total_errors = sum(item["errors"] for item in ok_results)
        print(f"成功处理帧数: {total_processed}")
        print(f"帧级错误数: {total_errors}")

    if failed_results:
        print("\n失败数据集:")
        for item in failed_results:
            print(f"  - {item['dataset_name']} (GPU {item['gpu_id']}): {item.get('error', 'unknown error')}")

    if failed_results or missing_count > 0:
        print("\n💡 提示: 重新运行时已完成的数据集会自动跳过 (断点续传)")
        sys.exit(1)


if __name__ == "__main__":
    main()
