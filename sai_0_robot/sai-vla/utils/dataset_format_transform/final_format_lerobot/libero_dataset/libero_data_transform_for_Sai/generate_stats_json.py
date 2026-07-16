#!/usr/bin/env python3
"""
为LeRobot数据集生成stats.json文件

读取数据集中的所有parquet文件，计算数值特征的统计信息
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any


def load_dataset_info(dataset_path: Path) -> dict:
    """加载数据集info.json"""
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"未找到info.json: {info_path}")
    
    with open(info_path, 'r') as f:
        return json.load(f)


def collect_all_data(dataset_path: Path, info: dict) -> pd.DataFrame:
    """收集所有parquet文件的数据"""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"未找到任何parquet文件: {data_dir}")
    
    print(f"\n读取 {len(parquet_files)} 个parquet文件...")
    
    dfs = []
    for parquet_file in tqdm(parquet_files, desc="加载数据"):
        df = pd.read_parquet(parquet_file)
        dfs.append(df)
    
    # 合并所有数据
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"总共加载 {len(full_df)} 行数据")
    
    return full_df


def calculate_stats(data: np.ndarray) -> Dict[str, List[float]]:
    """计算统计信息
    
    Args:
        data: numpy数组，shape为 (n_samples, n_features) 或 (n_samples,)
    
    Returns:
        包含mean, std, min, max, q01, q99的字典
    """
    # 确保是2D数组
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    stats = {
        "mean": np.mean(data, axis=0).astype(float).tolist(),
        "std": np.std(data, axis=0).astype(float).tolist(),
        "min": np.min(data, axis=0).astype(float).tolist(),
        "max": np.max(data, axis=0).astype(float).tolist(),
        "q01": np.percentile(data, 1, axis=0).astype(float).tolist(),
        "q99": np.percentile(data, 99, axis=0).astype(float).tolist(),
    }
    
    return stats


def process_feature(df: pd.DataFrame, feature_name: str, feature_info: dict) -> Dict[str, List[float]]:
    """处理单个特征的统计信息
    
    Args:
        df: 数据DataFrame
        feature_name: 特征名称
        feature_info: 特征信息（从info.json）
    
    Returns:
        统计信息字典
    """
    if feature_name not in df.columns:
        print(f"警告: 特征 {feature_name} 不在数据中")
        return None
    
    dtype = feature_info.get("dtype")
    
    # 只处理数值类型
    if dtype in ["float32", "float64", "int64", "int32", "int16", "int8", "uint8"]:
        column_data = df[feature_name]
        
        # 处理列表类型的数据（如observation.state, action）
        if isinstance(column_data.iloc[0], (list, np.ndarray)):
            # 转换为numpy数组
            data_array = np.array(column_data.tolist())
            return calculate_stats(data_array)
        else:
            # 标量数据
            data_array = column_data.values
            return calculate_stats(data_array)
    
    return None


def generate_stats_json(dataset_path: Path, force: bool = False):
    """生成stats.json文件
    
    Args:
        dataset_path: LeRobot数据集路径
        force: 是否强制覆盖已存在的文件
    """
    dataset_path = Path(dataset_path)
    
    # 自动确定输出路径
    output_path = dataset_path / "meta" / "stats.json"
    
    # 检查是否已存在
    if output_path.exists() and not force:
        print(f"stats.json已存在: {output_path}")
        print("使用 --force 强制覆盖")
        return
    
    # 加载info.json
    print("加载数据集信息...")
    info = load_dataset_info(dataset_path)
    
    # 收集所有数据
    df = collect_all_data(dataset_path, info)
    
    # 计算每个特征的统计信息
    print("\n计算统计信息...")
    stats = {}
    
    features = info.get("features", {})
    for feature_name, feature_info in tqdm(features.items(), desc="处理特征"):
        feature_stats = process_feature(df, feature_name, feature_info)
        
        if feature_stats is not None:
            stats[feature_name] = feature_stats
    
    # 保存stats.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ stats.json已生成: {output_path}")
    print(f"  包含 {len(stats)} 个特征的统计信息")
    
    # 显示示例
    print("\n示例统计信息:")
    for feature_name in list(stats.keys())[:3]:
        print(f"\n{feature_name}:")
        feature_stats = stats[feature_name]
        if len(feature_stats["mean"]) == 1:
            print(f"  mean: {feature_stats['mean'][0]:.6f}")
            print(f"  std:  {feature_stats['std'][0]:.6f}")
            print(f"  min:  {feature_stats['min'][0]:.6f}")
            print(f"  max:  {feature_stats['max'][0]:.6f}")
        else:
            print(f"  mean: {feature_stats['mean'][:3]}... (shape: {len(feature_stats['mean'])})")
            print(f"  std:  {feature_stats['std'][:3]}... (shape: {len(feature_stats['std'])})")


def main():
    parser = argparse.ArgumentParser(
        description="为LeRobot数据集生成stats.json文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 为指定数据集生成stats.json（自动保存到meta/stats.json）
  python generate_stats_json.py \\
    --dataset-path ../../daxiang_aloha_lerobot_dataset
  
  # 强制覆盖已存在的文件
  python generate_stats_json.py \\
    --dataset-path ../../daxiang_aloha_lerobot_dataset \\
    --force

说明:
  - 自动读取所有parquet文件并计算统计信息
  - 生成的stats.json包含: mean, std, min, max, q01 (1%), q99 (99%)
  - 只处理数值类型的特征（float, int等）
  - 自动保存到 <dataset-path>/meta/stats.json
        """
    )
    
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="LeRobot数据集路径（包含meta和data文件夹）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已存在的文件"
    )
    
    args = parser.parse_args()
    
    generate_stats_json(
        dataset_path=args.dataset_path,
        force=args.force
    )


if __name__ == "__main__":
    main()

# python generate_stats_json.py --dataset-path /home/sythoid_01/文档/Huangwenlong/Isaac-GR00T/raw_data_timestamp_align_to_lerobot_data/demo_data_task_description_pickupanapple_images_2/pickupanapple_v1 --force
# python generate_stats_json.py --dataset-path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0/libero_lerobot_spatial --force
# python generate_stats_json.py --dataset-path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0/libero_lerobot_object --force
# python generate_stats_json.py --dataset-path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0/libero_lerobot_goal --force
# python generate_stats_json.py --dataset-path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0/libero_lerobot_90 --force
# python generate_stats_json.py --dataset-path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0/libero_lerobot_10 --force