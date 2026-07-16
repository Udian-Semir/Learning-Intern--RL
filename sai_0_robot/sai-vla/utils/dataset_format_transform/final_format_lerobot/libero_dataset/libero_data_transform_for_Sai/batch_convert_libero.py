#!/usr/bin/env python3
"""
批量转换所有LIBERO数据集为LeRobot格式

支持的数据集:
- libero_10: 10个任务
- libero_90: 90个任务
- libero_goal: 10个任务
- libero_object: 10个任务
- libero_spatial: 10个任务

使用方式:
    python batch_convert_libero.py --all
    python batch_convert_libero.py --all-merge
    python batch_convert_libero.py --all-merge-exclude libero_90
    python batch_convert_libero.py --dataset libero_goal
    python batch_convert_libero.py --dataset libero_goal --output-dir ./my_output
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json
import logging
import numpy as np

# Import functions from libero_to_lerobot for merge conversion
from libero_to_lerobot import (
    read_hdf5_structure,
    extract_demo_data,
    save_episode_parquet,
    save_videos,
    save_metadata,
)
from tqdm import tqdm

def run_conversion(input_dir, output_dir, dataset_name):
    """运行单个数据集的转换"""
    print("\n" + "=" * 70)
    print(f"开始转换 {dataset_name.upper()}")
    print("=" * 70)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    start_time = time.time()
    
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "libero_to_lerobot.py"),
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--fps", "10",
            "--chunk-size", "100"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 70)
        print(f"✓ {dataset_name.upper()} 转换完成")
        print(f"耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print("=" * 70)
        
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 转换失败: {e}")
        return False, elapsed


def run_merge_conversion(
    datasets: List[str],
    base_input_dir: Path,
    output_dir: Path,
    fps: float = 10.0,
    chunk_size: int = 100
) -> Tuple[bool, float]:
    """
    合并转换所有数据集到单一输出目录
    
    Args:
        datasets: 要转换的数据集列表
        base_input_dir: 输入数据集的基础目录
        output_dir: 合并输出目录
        fps: 视频帧率
        chunk_size: 每个chunk的episode数量
    
    Returns:
        (success, elapsed_time) 元组
    """
    print("\n" + "=" * 70)
    print("开始合并转换所有数据集")
    print("=" * 70)
    print(f"输出目录: {output_dir}")
    print(f"数据集: {', '.join(datasets)}")
    
    start_time = time.time()
    
    try:
        # 创建输出目录
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有 HDF5 文件及其来源数据集
        all_hdf5_files = []  # List of (hdf5_path, dataset_name)
        for dataset in datasets:
            input_dir = base_input_dir / dataset
            if not input_dir.exists():
                print(f"⚠ 警告: 输入目录不存在，跳过: {input_dir}")
                continue
            
            hdf5_files = sorted(list(input_dir.glob("*.hdf5")))
            for hdf5_file in hdf5_files:
                all_hdf5_files.append((hdf5_file, dataset))
            print(f"  {dataset}: 找到 {len(hdf5_files)} 个任务文件")
        
        if not all_hdf5_files:
            print("✗ 错误: 没有找到任何 HDF5 文件")
            return False, time.time() - start_time
        
        print(f"\n总计: {len(all_hdf5_files)} 个任务文件")
        
        # 全局计数器
        task_names = []
        total_episodes = 0
        total_frames = 0
        episode_metadata = []
        video_shape = None
        
        global_index = 0
        current_chunk = 0
        episodes_in_chunk = 0
        dtype_registry: Dict[str, np.dtype] = {}
        video_codec_used: Optional[str] = None
        
        def update_dtype(key: str, new_dtype: np.dtype) -> None:
            new_dtype = np.dtype(new_dtype)
            existing = dtype_registry.get(key)
            if existing is None:
                dtype_registry[key] = new_dtype
            elif existing != new_dtype:
                raise ValueError(
                    f"Inconsistent dtype for '{key}': existing {existing} vs new {new_dtype}"
                )
        
        # 处理每个任务文件
        for task_idx, (hdf5_file, dataset_name) in enumerate(tqdm(all_hdf5_files, desc="转换任务")):
            task_name = hdf5_file.stem.replace('_demo', '')
            task_names.append(task_name)
            
            # 读取文件结构
            info = read_hdf5_structure(hdf5_file)
            num_demos = info['num_demos']
            
            print(f"\n任务 {task_idx} ({dataset_name}): {task_name} ({num_demos} demos)")
            
            # 处理每个demo
            for demo_idx in range(num_demos):
                demo_key = f"demo_{demo_idx}"
                
                # 检查是否需要新 chunk
                if episodes_in_chunk >= chunk_size:
                    current_chunk += 1
                    episodes_in_chunk = 0
                
                # 创建 chunk 目录
                data_chunk_dir = output_dir / "data" / f"chunk-{current_chunk:03d}"
                video_chunk_dir = output_dir / "videos" / f"chunk-{current_chunk:03d}"
                data_chunk_dir.mkdir(parents=True, exist_ok=True)
                video_chunk_dir.mkdir(parents=True, exist_ok=True)
                
                # 提取 demo 数据
                data = extract_demo_data(hdf5_file, demo_key)
                num_frames = len(data['actions'])
                
                update_dtype("observation.state", data['robot_states'].dtype)
                update_dtype("action", data['actions'].dtype)
                update_dtype("next.reward", data['rewards'].dtype)
                update_dtype("next.done", data['dones'].dtype)
                
                # 保存 parquet
                _, global_index = save_episode_parquet(
                    data, total_episodes, task_idx, task_name,
                    data_chunk_dir, global_index, fps
                )
                
                # 保存视频并获取尺寸
                height, width, codec_used = save_videos(data, total_episodes, video_chunk_dir, fps)
                if video_shape is None:
                    video_shape = (height, width)
                if video_codec_used is None:
                    video_codec_used = codec_used
                elif video_codec_used != codec_used:
                    logging.warning(
                        "Episode %d switched video codec from %s to %s; keeping metadata codec as %s.",
                        total_episodes,
                        video_codec_used,
                        codec_used,
                        video_codec_used,
                    )
                
                # 记录元数据
                episode_metadata.append({
                    "episode_index": total_episodes,
                    "task_index": task_idx,
                    "length": num_frames,
                    "chunk": current_chunk
                })
                
                total_episodes += 1
                total_frames += num_frames
                episodes_in_chunk += 1
        
        # 保存 episodes 元数据
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "episodes.jsonl", 'w') as f:
            for ep in episode_metadata:
                f.write(json.dumps(ep) + '\n')
        
        # 保存元数据
        save_metadata(
            task_names, total_episodes, total_frames,
            current_chunk + 1, chunk_size, fps, output_dir,
            video_shape if video_shape else (128, 128),
            dtype_info=dtype_registry,
            video_codec=(video_codec_used or "ffv1")
        )
        
        # 创建 VLM hidden states 占位目录
        vlm_dir = output_dir / "vlm_hidden_states"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"合并转换完成!")
        print(f"  总任务数: {len(task_names)}")
        print(f"  总 episodes: {total_episodes}")
        print(f"  总帧数: {total_frames}")
        print(f"  输出目录: {output_dir}")
        print(f"  耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print(f"{'='*70}")
        
        return True, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 合并转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False, elapsed

def main():
    parser = argparse.ArgumentParser(
        description="批量转换LIBERO数据集为LeRobot格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换单个数据集
  python batch_convert_libero.py --dataset libero_goal
  
  # 转换所有数据集（分开保存）
  python batch_convert_libero.py --all
  
  # 转换所有数据集（合并到一个目录）
  python batch_convert_libero.py --all-merge
  
  # 合并除指定数据集外的所有数据集
  python batch_convert_libero.py --all-merge-exclude libero_90
  python batch_convert_libero.py --all-merge-exclude libero_90 libero_10
  
  # 自定义输出目录
  python batch_convert_libero.py --all --base-output-dir /data/libero_lerobot
        """
    )
    
    parser.add_argument(
        "--dataset",
        choices=["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"],
        help="要转换的数据集（不指定则需要--all或--all-merge）"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="转换所有LIBERO数据集（每个数据集单独输出目录）"
    )
    parser.add_argument(
        "--all-merge",
        action="store_true",
        help="转换所有LIBERO数据集（合并到单一输出目录）"
    )
    parser.add_argument(
        "--all-merge-exclude",
        nargs="+",
        choices=["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"],
        help="合并除指定数据集外的所有数据集（可指定多个）"
    )
    parser.add_argument(
        "--base-input-dir",
        type=Path,
        default=Path("/data/HuangWenlong/datasets/libero_github"),
        help="LIBERO数据集基础目录 (default: /data/HuangWenlong/datasets/libero_github)"
    )
    parser.add_argument(
        "--base-output-dir",
        type=Path,
        default=Path("/data/HuangWenlong/datasets"),
        help="输出基础目录 (default: /data/HuangWenlong/datasets)"
    )
    parser.add_argument(
        "--merged-output-name",
        type=str,
        default="libero_lerobot_merged",
        help="合并模式下的输出目录名称 (default: libero_lerobot_merged)"
    )
    
    args = parser.parse_args()
    
    # 检查输入
    if not args.dataset and not args.all and not args.all_merge and not args.all_merge_exclude:
        parser.error("必须指定 --dataset、--all、--all-merge 或 --all-merge-exclude")
    
    # 执行转换
    print(f"\n{'=' * 70}")
    print(f"LIBERO -> LeRobot 批量转换工具")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")
    
    # 定义所有数据集
    all_datasets = ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"]
    
    # 合并模式（排除指定数据集）
    if args.all_merge_exclude:
        excluded = set(args.all_merge_exclude)
        datasets_to_merge = [d for d in all_datasets if d not in excluded]
        
        if not datasets_to_merge:
            parser.error("排除后没有剩余数据集可转换")
        
        print(f"\n模式: 合并转换（排除 {', '.join(excluded)}）")
        print(f"将转换: {', '.join(datasets_to_merge)}")
        output_dir = args.base_output_dir / args.merged_output_name
        
        success, elapsed = run_merge_conversion(
            datasets_to_merge,
            args.base_input_dir,
            output_dir,
            fps=10.0,
            chunk_size=100
        )
        
        print("\n" + "=" * 70)
        print("转换完成总结")
        print("=" * 70)
        status = "✓ 成功" if success else "✗ 失败"
        print(f"合并转换（排除 {', '.join(excluded)}）{status} - 耗时: {elapsed/60:.1f}分钟")
        print("=" * 70)
        
        return 0 if success else 1
    
    # 合并模式（所有数据集）
    if args.all_merge:
        print("\n模式: 合并转换（所有数据集合并到单一目录）")
        output_dir = args.base_output_dir / args.merged_output_name
        
        success, elapsed = run_merge_conversion(
            all_datasets,
            args.base_input_dir,
            output_dir,
            fps=10.0,
            chunk_size=100
        )
        
        print("\n" + "=" * 70)
        print("转换完成总结")
        print("=" * 70)
        status = "✓ 成功" if success else "✗ 失败"
        print(f"合并转换 {status} - 耗时: {elapsed/60:.1f}分钟")
        print("=" * 70)
        
        return 0 if success else 1
    
    # 分开模式
    if args.all:
        datasets = all_datasets
    else:
        datasets = [args.dataset]
    
    print(f"\n模式: 分开转换（每个数据集单独输出）")
    
    results = {}
    total_start = time.time()
    
    for dataset in datasets:
        input_dir = args.base_input_dir / dataset
        
        # Determine output directory name
        if dataset == "libero_10":
            output_name = "libero_lerobot_10"
        elif dataset == "libero_90":
            output_name = "libero_lerobot_90"
        else:
            output_name = f"libero_lerobot_{dataset.split('_')[1]}"
        
        output_dir = args.base_output_dir / output_name
        
        # 检查输入目录是否存在
        if not input_dir.exists():
            print(f"\n✗ 输入目录不存在: {input_dir}")
            results[dataset] = (False, 0)
            continue
        
        success, elapsed = run_conversion(input_dir, output_dir, dataset)
        results[dataset] = (success, elapsed)
        
        # 短暂等待
        if dataset != datasets[-1]:
            print("\n等待10秒后开始下一个转换...")
            time.sleep(10)
    
    # 打印总结
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("转换完成总结")
    print("=" * 70)
    
    for dataset, (success, elapsed) in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        print(f"{dataset:20} {status:10} {elapsed/60:6.1f}分钟")
    
    successful = sum(1 for s, _ in results.values() if s)
    print(f"\n总转换时间: {total_elapsed/60:.1f}分钟 ({successful}/{len(datasets)} 成功)")
    print("=" * 70)
    
    return 0 if all(s for s, _ in results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
