"""
批量离散化轨迹追踪分析

给定数据文件夹，统计所有 parquet 文件的追踪情况，累加到一张图上。

核心概念:
- 原始轨迹和离散化轨迹都是多维空间中的点序列
- 使用欧氏距离判断离散化轨迹是否"追上"原始轨迹的某个点
- 阈值 = sqrt(sum((delta_i/2)^2))
- 逐步追踪：离散化轨迹每走一步，检查与当前目标原始点的距离
- 剩余未追上步数 = chunk_size - 已追上的原始点数

使用示例:
python tools/calc_discrete_tracking_batch.py \
    --data_dir /path/to/data \
    --chunk_size 16 \
    --deltas 0.9 0.8 0.9 0.085 0.15 0.3 \
    --columns 0 1 2 3 4 5 \
    --output_dir ./action_plots

python tools/calc_discrete_tracking_batch.py \
    --data_dir /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset_action_euler_unwrap/data \
    --chunk_size 25 \
    --deltas 3 3 3 0.5 0.5 0.5 \
    --columns 6 7 8 9 10 11 \
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
from tqdm import tqdm

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


def find_all_parquet_files(data_dir: str) -> list:
    """递归查找所有 parquet 文件"""
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.rglob('*.parquet'))
    return parquet_files


def calc_threshold(deltas: list) -> float:
    """
    计算追踪阈值：每个维度 delta/2 的 L2 范数
    
    threshold = sqrt(sum((delta_i / 2)^2))
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
    """
    chunk_size, n_dims = chunk_actions.shape
    
    # 计算阈值
    threshold = calc_threshold(deltas)
    
    # 计算原始累积位置（多维）
    original_pos = np.cumsum(chunk_actions, axis=0)
    
    # 对每个维度进行离散化和反离散化
    discrete_pos = np.zeros_like(original_pos)
    for dim_idx in range(n_dims):
        col_data = chunk_actions[:, dim_idx]
        delta = deltas[dim_idx]
        
        discrete_data = discrete_chunk_calculus(col_data, delta, beta=beta, alpha=alpha)
        undiscrete_data = undiscrete_chunk_calculus(discrete_data, delta)
        discrete_pos[:, dim_idx] = np.cumsum(undiscrete_data)
    
    # 逐步追踪
    target_idx = 0
    
    for i in range(chunk_size):
        current_discrete_pos = discrete_pos[i]
        
        while target_idx < chunk_size:
            target_original_pos = original_pos[target_idx]
            distance = np.linalg.norm(current_discrete_pos - target_original_pos)
            
            if distance <= threshold:
                target_idx += 1
            else:
                break
    
    remaining = chunk_size - target_idx
    return remaining


def calc_file_remaining(
    actions: np.ndarray,
    columns: list,
    deltas: list,
    chunk_size: int,
    beta: float = 0.6,
    alpha: float = 0.4,
) -> list:
    """
    计算单个文件所有 chunk 的剩余未追上步数
    
    Returns:
        remaining_list: 每个 chunk 的剩余步数列表
    """
    num_samples = actions.shape[0]
    remaining_list = []
    
    selected_actions = actions[:, columns]
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_actions = selected_actions[start_idx:end_idx]
        
        if chunk_actions.shape[0] < chunk_size:
            continue
        
        remaining = calc_chunk_remaining_multidim(
            chunk_actions, deltas, beta=beta, alpha=alpha
        )
        remaining_list.append(remaining)
    
    return remaining_list


def main():
    parser = argparse.ArgumentParser(description='批量离散化轨迹追踪分析')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据文件夹路径')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='chunk 大小，默认 16')
    parser.add_argument('--deltas', type=float, nargs='+', default=None,
                        help='每列的 delta 值，默认自动估算')
    parser.add_argument('--columns', type=int, nargs='+', default=None,
                        help='要计算的列索引，默认所有列')
    parser.add_argument('--output_dir', type=str, default='./action_plots',
                        help='输出目录')
    parser.add_argument('--max_files', type=int, default=None,
                        help='最多处理的文件数，默认全部')
    parser.add_argument('--beta', type=float, default=0.6,
                        help='趋势项权重 (默认 0.6)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='趋势平滑系数 (默认 0.4)')
    
    args = parser.parse_args()
    
    # 查找所有 parquet 文件
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 目录不存在 {data_dir}")
        return
    
    parquet_files = find_all_parquet_files(str(data_dir))
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    if args.max_files is not None:
        parquet_files = parquet_files[:args.max_files]
        print(f"限制处理前 {args.max_files} 个文件")
    
    if len(parquet_files) == 0:
        print("没有找到 parquet 文件")
        return
    
    # 从第一个文件获取数据形状
    first_actions = load_parquet_actions(str(parquet_files[0]))
    n_dims = first_actions.shape[1]
    
    # 设置 columns
    if args.columns is None:
        columns = list(range(n_dims))
    else:
        columns = args.columns
    
    # 设置 deltas
    if args.deltas is None:
        deltas = []
        for col_idx in columns:
            col_data = first_actions[:, col_idx]
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
    
    threshold = calc_threshold(deltas)
    print(f"使用的列: {columns}")
    print(f"使用的 deltas: {deltas}")
    print(f"追踪阈值 (欧氏距离): {threshold:.4f}")
    print(f"Chunk size: {args.chunk_size}")
    
    # 累计所有文件的剩余步数
    all_remaining_list = []
    total_chunks = 0
    
    print(f"\n处理 {len(parquet_files)} 个文件...")
    for parquet_path in tqdm(parquet_files, desc="Processing"):
        try:
            actions = load_parquet_actions(str(parquet_path))
            remaining_list = calc_file_remaining(
                actions, columns, deltas, args.chunk_size,
                beta=args.beta, alpha=args.alpha
            )
            all_remaining_list.extend(remaining_list)
            total_chunks += len(remaining_list)
        except Exception as e:
            print(f"\n警告: 处理 {parquet_path} 时出错: {e}")
            continue
    
    print(f"\n总 chunk 数: {total_chunks}")
    print(f"处理的文件数: {len(parquet_files)}")
    
    # 统计剩余步数的分布
    remaining_counts = {}
    for remaining in all_remaining_list:
        remaining_counts[remaining] = remaining_counts.get(remaining, 0) + 1
    
    # 打印统计结果
    print("\n=== 剩余未追上步数分布 ===")
    for remaining in range(args.chunk_size + 1):
        count = remaining_counts.get(remaining, 0)
        if count > 0:
            print(f"  剩余 {remaining:2d} 步: {count:6d} 次 ({100*count/total_chunks:.1f}%)")
    
    # 画柱状图
    os.makedirs(args.output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
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
    ax.set_title(f'Discrete Tracking Analysis (Batch)\n'
                 f'{len(parquet_files)} files, {total_chunks} chunks, '
                 f'chunk_size={args.chunk_size}, {len(columns)} dims, '
                 f'threshold={threshold:.4f}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_xticks(x_values)
    ax.set_xlim(-0.5, args.chunk_size + 0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, f'tracking_batch_chunk_{args.chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n柱状图已保存: {save_path}")
    
    # 打印汇总统计
    perfect_count = remaining_counts.get(0, 0)
    print(f"\n=== 汇总 ===")
    print(f"完美追踪 (剩余0步): {perfect_count}/{total_chunks} ({100*perfect_count/total_chunks:.1f}%)")
    avg_remaining = np.mean(all_remaining_list)
    print(f"平均剩余步数: {avg_remaining:.2f}")


if __name__ == '__main__':
    main()
