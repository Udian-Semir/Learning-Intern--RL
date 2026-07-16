#!/usr/bin/env python3
"""
统计指定任务索引的视频总帧数

该脚本扫描 LeRobot 格式数据集中的视频文件，统计指定任务索引的所有视频的总帧数。

支持的视频格式：.mp4, .avi
"""

import argparse
import cv2
from pathlib import Path
from typing import List, Dict, Optional
import json
import pandas as pd


def count_video_frames(video_path: Path) -> int:
    """
    统计单个视频文件的帧数
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        视频的总帧数，如果读取失败返回 0
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"警告: 无法打开视频文件 {video_path}")
            return 0
        
        # 获取总帧数
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return frame_count
    except Exception as e:
        print(f"错误: 读取视频 {video_path} 时出错: {e}")
        return 0


def find_videos_in_dataset(dataset_path: Path) -> List[Path]:
    """
    在数据集目录中查找所有视频文件
    
    Args:
        dataset_path: 数据集根目录
        
    Returns:
        所有视频文件的路径列表
    """
    video_extensions = {'.mp4', '.avi'}
    video_files = []
    
    # 递归查找所有视频文件
    for ext in video_extensions:
        video_files.extend(dataset_path.rglob(f'*{ext}'))
    
    return sorted(video_files)


def get_task_index_from_video_path(video_path: Path, dataset_path: Path) -> Optional[int]:
    """
    从视频路径中提取任务索引
    
    对于 LeRobot 格式数据集，需要读取对应的 parquet 文件来获取 task_index
    
    Args:
        video_path: 视频文件路径
        dataset_path: 数据集根目录
        
    Returns:
        任务索引，如果无法获取返回 None
    """
    # 从视频路径中提取 episode 索引
    # 例如: videos/chunk-000/observation.images.top/episode_000000.mp4
    filename = video_path.stem  # episode_000000
    try:
        episode_idx = int(filename.split('_')[-1])
    except (ValueError, IndexError):
        return None
    
    # 查找对应的 parquet 文件
    # video_path.parent.parent 是 videos/chunk-XXX
    chunk_name = video_path.parent.parent.name  # chunk-000
    parquet_path = dataset_path / 'data' / chunk_name / f'episode_{episode_idx:06d}.parquet'
    
    if not parquet_path.exists():
        return None
    
    try:
        # 读取 parquet 文件获取 task_index
        df = pd.read_parquet(parquet_path)
        if 'task_index' in df.columns:
            return int(df['task_index'].iloc[0])
    except Exception as e:
        print(f"警告: 读取 parquet 文件 {parquet_path} 时出错: {e}")
    
    return None


def count_frames_by_task(
    dataset_path: Path,
    task_indices: Optional[List[int]] = None
) -> Dict[int, Dict[str, int]]:
    """
    统计指定任务索引的视频总帧数
    
    Args:
        dataset_path: 数据集根目录路径
        task_indices: 要统计的任务索引列表，如果为 None 则统计所有任务
        
    Returns:
        字典，键为任务索引，值为包含统计信息的字典：
        {
            task_index: {
                'total_frames': 总帧数,
                'video_count': 视频数量,
                'videos': [视频路径列表]
            }
        }
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
    
    print(f"正在扫描数据集: {dataset_path}")
    
    # 查找所有视频文件
    video_files = find_videos_in_dataset(dataset_path)
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 统计结果
    task_stats: Dict[int, Dict[str, any]] = {}
    
    # 处理每个视频文件
    for video_path in video_files:
        # 获取任务索引
        task_idx = get_task_index_from_video_path(video_path, dataset_path)
        
        if task_idx is None:
            continue
        
        # 如果指定了任务索引，只处理指定的任务
        if task_indices is not None and task_idx not in task_indices:
            continue
        
        # 统计帧数
        frame_count = count_video_frames(video_path)
        
        # 更新统计信息
        if task_idx not in task_stats:
            task_stats[task_idx] = {
                'total_frames': 0,
                'video_count': 0,
                'videos': []
            }
        
        task_stats[task_idx]['total_frames'] += frame_count
        task_stats[task_idx]['video_count'] += 1
        task_stats[task_idx]['videos'].append(str(video_path.relative_to(dataset_path)))
    
    return task_stats


def main():
    parser = argparse.ArgumentParser(
        description='统计指定任务索引的视频总帧数',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 统计所有任务的视频帧数
  python count_video_frames.py --dataset-path /path/to/dataset
  
  # 统计指定任务的视频帧数
  python count_video_frames.py --dataset-path /path/to/dataset --task-indices 0 1 2
  
  # 保存结果到 JSON 文件
  python count_video_frames.py --dataset-path /path/to/dataset --output result.json
        """
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='数据集根目录路径'
    )
    
    parser.add_argument(
        '--task-indices',
        type=int,
        nargs='+',
        help='要统计的任务索引列表（空格分隔），不指定则统计所有任务'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出 JSON 文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 执行统计
    try:
        task_stats = count_frames_by_task(
            Path(args.dataset_path),
            args.task_indices
        )
        
        # 打印结果
        print("\n" + "="*60)
        print("视频帧数统计结果")
        print("="*60)
        
        if not task_stats:
            print("未找到符合条件的视频")
        else:
            for task_idx in sorted(task_stats.keys()):
                stats = task_stats[task_idx]
                print(f"\n任务索引 {task_idx}:")
                print(f"  视频数量: {stats['video_count']}")
                print(f"  总帧数: {stats['total_frames']}")
                print(f"  平均帧数: {stats['total_frames'] / stats['video_count']:.2f}")
        
        # 计算总计
        total_videos = sum(s['video_count'] for s in task_stats.values())
        total_frames = sum(s['total_frames'] for s in task_stats.values())
        
        print("\n" + "-"*60)
        print(f"总计:")
        print(f"  任务数量: {len(task_stats)}")
        print(f"  视频数量: {total_videos}")
        print(f"  总帧数: {total_frames}")
        print("="*60)
        
        # 保存到 JSON 文件
        if args.output:
            output_path = Path(args.output)
            output_data = {
                'dataset_path': str(args.dataset_path),
                'task_indices': args.task_indices,
                'statistics': task_stats,
                'summary': {
                    'total_tasks': len(task_stats),
                    'total_videos': total_videos,
                    'total_frames': total_frames
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n结果已保存到: {output_path}")
    
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
