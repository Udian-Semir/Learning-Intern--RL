"""
离散化轨迹追踪分析（多维欧氏距离版本）

分析离散化轨迹在每个 chunk 结束时"追踪"到原始轨迹的第几步，
统计剩余未走完步数的分布。

核心概念:
- 原始轨迹和离散化轨迹都是多维空间（如6维）中的点序列
- 使用欧氏距离判断离散化轨迹是否"追上"原始轨迹的某个点
- 阈值 = sqrt(sum((delta_i/2)^2))，即每个维度 delta/2 的 L2 范数
- 逐步追踪：离散化轨迹每走一步，检查与当前目标原始点的距离
  - 如果距离 <= 阈值，说明追上了，目标切换到下一个原始点
  - 直到 chunk 结束
- 剩余未追上步数 = chunk_size - 已追上的原始点数

输出:
- 横轴：剩余未追上步数（0 到 chunk_size）
- 纵轴：出现的 chunk 次数

使用示例:
python tools/calc_discrete_tracking.py \
    --parquet_path /path/to/episode_000000.parquet \
    --chunk_size 16 \
    --deltas 0.9 0.8 0.9 0.085 0.15 0.3 \
    --columns 0 1 2 3 4 5 \
    --output_dir ./action_plots

python tools/calc_discrete_tracking.py \
    --parquet_path /home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_10/libero_lerobot_10_sys0_eagle_-1/data/chunk-000/episode_000000.parquet \
    --chunk_size 16 \
    --deltas 0.9 0.9 0.9 0.15 0.15 0.15 \
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


def calc_threshold(deltas: list) -> float:
    """
    计算追踪阈值：每个维度 delta/2 的 L2 范数
    
    threshold = sqrt(sum((delta_i / 2)^2))
    
    Args:
        deltas: 每个维度的 delta 值列表
    
    Returns:
        threshold: 欧氏距离阈值
    """
    return np.sqrt(np.sum([(d / 2) ** 2 for d in deltas]))


def calc_chunk_remaining_multidim(
    chunk_actions: np.ndarray,
    deltas: list,
    beta: float = 0.6,
    alpha: float = 0.4,
) -> int:
    """
    计算单个 chunk 的剩余未追上步数（多维欧氏距离版本）
    
    算法：
    1. 计算原始累积位置序列（多维）
    2. 计算离散化累积位置序列（多维）
    3. 阈值 = sqrt(sum((delta_i/2)^2))
    4. 逐步追踪：离散化轨迹每走一步，检查与当前目标原始点的距离
       - 如果距离 <= 阈值，追上了，目标切换到下一个原始点
       - 直到离散化轨迹走完
    5. 剩余未追上步数 = chunk_size - 已追上的原始点数
    
    Args:
        chunk_actions: shape (chunk_size, n_dims) 的 action 数据
        deltas: 每个维度的 delta 值列表
        beta, alpha: 离散化参数
    
    Returns:
        remaining: 剩余未追上的步数
    """
    chunk_size, n_dims = chunk_actions.shape
    
    # 计算阈值
    threshold = calc_threshold(deltas)
    
    # 计算原始累积位置（多维）
    original_pos = np.cumsum(chunk_actions, axis=0)  # shape: (chunk_size, n_dims)
    
    # 对每个维度进行离散化和反离散化
    discrete_pos = np.zeros_like(original_pos)
    for dim_idx in range(n_dims):
        col_data = chunk_actions[:, dim_idx]
        delta = deltas[dim_idx]
        
        # 离散化
        discrete_data = discrete_chunk_calculus(col_data, delta, beta=beta, alpha=alpha)
        
        # 反离散化
        undiscrete_data = undiscrete_chunk_calculus(discrete_data, delta)
        
        # 累积位置
        discrete_pos[:, dim_idx] = np.cumsum(undiscrete_data)
    
    # 逐步追踪
    target_idx = 0  # 当前要追踪的原始轨迹点索引
    
    for i in range(chunk_size):
        # 当前离散化位置
        current_discrete_pos = discrete_pos[i]
        
        # 检查是否追上了当前目标点（以及之后的点）
        while target_idx < chunk_size:
            target_original_pos = original_pos[target_idx]
            
            # 计算欧氏距离
            distance = np.linalg.norm(current_discrete_pos - target_original_pos)
            
            if distance <= threshold:
                # 追上了，目标切换到下一个原始点
                target_idx += 1
            else:
                # 没追上，等待下一步离散化
                break
    
    # 剩余未追上步数
    remaining = chunk_size - target_idx
    return remaining


def calc_all_chunks_remaining(
    actions: np.ndarray,
    columns: list,
    deltas: list,
    chunk_size: int,
    beta: float = 0.6,
    alpha: float = 0.4,
) -> list:
    """
    计算所有 chunk 的剩余未追上步数
    
    Args:
        actions: action 数组
        columns: 要计算的列索引列表
        deltas: 对应列的 delta 值列表
        chunk_size: chunk 大小
        beta, alpha: 离散化参数
    
    Returns:
        remaining_list: 每个 chunk 的剩余步数列表
    """
    num_samples = actions.shape[0]
    remaining_list = []
    
    # 提取指定列的数据
    selected_actions = actions[:, columns]  # shape: (num_samples, n_dims)
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_actions = selected_actions[start_idx:end_idx]
        
        # 如果 chunk 不完整（最后一个 chunk），跳过
        if chunk_actions.shape[0] < chunk_size:
            continue
        
        # 计算该 chunk 的剩余未追上步数
        remaining = calc_chunk_remaining_multidim(
            chunk_actions, deltas, beta=beta, alpha=alpha
        )
        remaining_list.append(remaining)
    
    return remaining_list


def main():
    parser = argparse.ArgumentParser(description='离散化轨迹追踪分析')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='chunk 大小，默认 16')
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
    print(f"Chunk size: {args.chunk_size}")
    
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
    
    # 计算阈值
    threshold = calc_threshold(deltas)
    print(f"追踪阈值 (欧氏距离): {threshold:.4f}")
    
    # 计算所有 chunk 的剩余步数（多维欧氏距离）
    remaining_list = calc_all_chunks_remaining(
        actions, columns, deltas, args.chunk_size,
        beta=args.beta, alpha=args.alpha
    )
    
    print(f"\n总 chunk 数: {len(remaining_list)}")
    
    # 统计剩余步数的分布（范围 0 到 chunk_size）
    remaining_counts = {}
    for remaining in remaining_list:
        remaining_counts[remaining] = remaining_counts.get(remaining, 0) + 1
    
    # 打印统计结果
    print("\n=== 剩余未追上步数分布 ===")
    for remaining in range(args.chunk_size + 1):
        count = remaining_counts.get(remaining, 0)
        if count > 0:
            print(f"  剩余 {remaining:2d} 步: {count:4d} 次 ({100*count/len(remaining_list):.1f}%)")
    
    # 画柱状图
    os.makedirs(args.output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 横轴：0 到 chunk_size
    x_values = list(range(args.chunk_size + 1))
    y_values = [remaining_counts.get(x, 0) for x in x_values]
    
    # 画柱状图
    bars = ax.bar(x_values, y_values, color='steelblue', edgecolor='black', alpha=0.7)
    
    # 在柱状图上标注数值（只标注非0的）
    for bar, count in zip(bars, y_values):
        if count > 0:
            height = bar.get_height()
            ax.annotate(f'{count}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Remaining Steps (not reached)', fontsize=12)
    ax.set_ylabel('Number of Chunks', fontsize=12)
    ax.set_title(f'{parquet_path.stem}\nDiscrete Tracking Analysis (chunk_size={args.chunk_size}, {len(columns)} dims, threshold={threshold:.4f})', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 设置 x 轴刻度为 0 到 chunk_size
    ax.set_xticks(x_values)
    ax.set_xlim(-0.5, args.chunk_size + 0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, f'{parquet_path.stem}_tracking_chunk_{args.chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n柱状图已保存: {save_path}")
    
    # 打印汇总统计
    total_chunks = len(remaining_list)
    perfect_count = remaining_counts.get(0, 0)
    print(f"\n=== 汇总 ===")
    print(f"完美追踪 (剩余0步): {perfect_count}/{total_chunks} ({100*perfect_count/total_chunks:.1f}%)")
    avg_remaining = np.mean(remaining_list)
    print(f"平均剩余步数: {avg_remaining:.2f}")


if __name__ == '__main__':
    main()
