"""
计算离散化累计误差

给定 parquet 文件，计算原始累计曲线和离散化累计曲线之间的积分面积差值。
支持多个 chunk size，最后画出柱状图对比。

使用示例:
python tools/calc_discrete_error.py \
    --parquet_path /home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_10/libero_lerobot_10_sys0_eagle_-1/data/chunk-000/episode_000000.parquet \
    --chunk_sizes 4 8 16 25 32 \
    --deltas 0.9 0.8 0.9 0.085 0.15 0.3 \
    --columns 0 1 2 3 4 5 \
    --output_dir ./action_plots
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.discrete import discrete_chunk_calculus, undiscrete_chunk_calculus


def load_parquet_actions(parquet_path: str) -> np.ndarray:
    """从parquet文件加载action数据"""
    df = pd.read_parquet(parquet_path)
    
    if 'action' in df.columns:
        actions = np.array(df['action'].tolist())
    else:
        raise ValueError(f"未找到'action'列，可用列: {df.columns.tolist()}")
    
    return actions


def calc_chunk_error(
    col_data: np.ndarray,
    delta: float,
    chunk_size: int,
    beta: float = 0.6,
    alpha: float = 0.4,
) -> tuple:
    """
    计算单列数据的离散化误差
    
    对于每个 chunk：
    1. 计算原始累计曲线（从 chunk 起点开始的 cumsum）
    2. 计算离散化累计曲线（从 chunk 起点开始的 cumsum）
    3. 计算两条曲线的积分面积差（绝对值之和）
    
    Args:
        col_data: 1D array，单列 action 数据
        delta: 离散化步长
        chunk_size: chunk 大小
        beta: 趋势项权重
        alpha: 趋势平滑系数
    
    Returns:
        chunk_errors: 每个 chunk 的误差列表
        total_error: 所有 chunk 误差的累计和
    """
    num_samples = len(col_data)
    
    chunk_errors = []
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_data = col_data[start_idx:end_idx]
        
        # 原始累计曲线（从0开始）
        original_cumsum = np.cumsum(chunk_data)
        
        # 离散化
        discrete_data = discrete_chunk_calculus(
            chunk_data, delta, beta=beta, alpha=alpha
        )
        
        # 反离散化
        undiscrete_data = undiscrete_chunk_calculus(discrete_data, delta)
        
        # 离散化累计曲线（从0开始）
        discrete_cumsum = np.cumsum(undiscrete_data)
        
        # 计算两条曲线的积分面积差（使用绝对差值的和作为近似）
        # 这相当于两条曲线之间围成的面积
        error = np.sum(np.abs(original_cumsum - discrete_cumsum))
        
        chunk_errors.append(error)
    
    total_error = sum(chunk_errors)
    
    return chunk_errors, total_error


def calc_all_columns_error(
    actions: np.ndarray,
    columns: list,
    deltas: list,
    chunk_size: int,
    beta: float = 0.6,
    alpha: float = 0.4,
) -> tuple:
    """
    计算所有指定列的离散化误差
    
    Args:
        actions: action 数组
        columns: 要计算的列索引列表
        deltas: 对应列的 delta 值列表
        chunk_size: chunk 大小
        beta, alpha: 离散化参数
    
    Returns:
        column_errors: 每列的误差字典 {col_idx: (chunk_errors, total_error)}
        total_error_all: 所有列的误差总和
    """
    column_errors = {}
    total_error_all = 0.0
    
    for col_idx, delta in zip(columns, deltas):
        col_data = actions[:, col_idx]
        chunk_errors, total_error = calc_chunk_error(
            col_data, delta, chunk_size, beta, alpha
        )
        column_errors[col_idx] = (chunk_errors, total_error)
        total_error_all += total_error
    
    return column_errors, total_error_all


def main():
    parser = argparse.ArgumentParser(description='计算离散化累计误差')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径')
    parser.add_argument('--chunk_sizes', type=int, nargs='+', default=[4, 8, 16, 32, 64],
                        help='要测试的 chunk size 列表，默认: 4 8 16 32 64')
    parser.add_argument('--deltas', type=float, nargs='+', default=None,
                        help='每列的 delta 值，默认自动估算')
    parser.add_argument('--columns', type=int, nargs='+', default=None,
                        help='要计算的列索引，默认所有列')
    parser.add_argument('--output_dir', type=str, default='./action_plots',
                        help='输出目录')
    # discrete_chunk_calculus 参数
    parser.add_argument('--beta', type=float, default=0.6,
                        help='趋势项权重 (默认 0.6)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='趋势平滑系数 (默认 0.4)')
    
    args = parser.parse_args()
    
    # 加载数据
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        print(f"错误: 文件不存在 {parquet_path}")
        return
    
    print(f"加载: {parquet_path}")
    actions = load_parquet_actions(str(parquet_path))
    print(f"Action shape: {actions.shape}")
    
    # 设置 columns
    if args.columns is None:
        columns = list(range(actions.shape[1]))
    else:
        columns = args.columns
    
    # 设置 deltas
    if args.deltas is None:
        # 自动估算
        deltas = []
        for col_idx in columns:
            col_data = actions[:, col_idx]
            std = np.std(col_data)
            estimated_delta = max(std, 0.01)
            if estimated_delta > 0.1:
                estimated_delta = round(estimated_delta, 1)
            elif estimated_delta > 0.01:
                estimated_delta = round(estimated_delta, 2)
            else:
                estimated_delta = round(estimated_delta, 3)
            deltas.append(max(estimated_delta, 0.001))
        print(f"自动估算的 deltas: {deltas}")
    else:
        deltas = args.deltas
        if len(deltas) != len(columns):
            if len(deltas) < len(columns):
                deltas = deltas + [deltas[-1]] * (len(columns) - len(deltas))
            else:
                deltas = deltas[:len(columns)]
    
    print(f"使用的列: {columns}")
    print(f"使用的 deltas: {deltas}")
    print(f"测试的 chunk sizes: {args.chunk_sizes}")
    
    # 计算每个 chunk size 的误差
    results = {}  # {chunk_size: total_error_all}
    detailed_results = {}  # {chunk_size: {col_idx: (chunk_errors, total_error)}}
    
    print("\n" + "=" * 60)
    print("开始计算...")
    print("=" * 60)
    
    for chunk_size in args.chunk_sizes:
        column_errors, total_error_all = calc_all_columns_error(
            actions, columns, deltas, chunk_size,
            beta=args.beta, alpha=args.alpha
        )
        results[chunk_size] = total_error_all
        detailed_results[chunk_size] = column_errors
        
        print(f"\n--- Chunk Size: {chunk_size} ---")
        print(f"  总误差累计和: {total_error_all:.4f}")
        for col_idx in columns:
            chunk_errors, total_error = column_errors[col_idx]
            num_chunks = len(chunk_errors)
            avg_chunk_error = total_error / num_chunks if num_chunks > 0 else 0
            print(f"    列 {col_idx}: 总误差={total_error:.4f}, "
                  f"chunk数={num_chunks}, 平均chunk误差={avg_chunk_error:.4f}")
    
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    for chunk_size in args.chunk_sizes:
        print(f"  Chunk Size {chunk_size:3d}: 总误差累计和 = {results[chunk_size]:.4f}")
    
    # 画柱状图
    os.makedirs(args.output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    chunk_sizes = list(results.keys())
    total_errors = [results[cs] for cs in chunk_sizes]
    
    bars = ax.bar(range(len(chunk_sizes)), total_errors, color='steelblue', edgecolor='black')
    
    # 在柱状图上标注数值
    for bar, error in zip(bars, total_errors):
        height = bar.get_height()
        ax.annotate(f'{error:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(chunk_sizes)))
    ax.set_xticklabels([str(cs) for cs in chunk_sizes])
    ax.set_xlabel('Chunk Size')
    ax.set_ylabel('Total Error Sum (Integral Area Difference)')
    ax.set_title(f'{parquet_path.stem}\nsum(Difference of Cumsum) vs Chunk Size')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, f'{parquet_path.stem}_chunk_error_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n柱状图已保存: {save_path}")


if __name__ == '__main__':
    main()
