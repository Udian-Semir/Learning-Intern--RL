"""
批量计算离散化累计误差

给定数据文件夹，统计所有 parquet 文件的误差累计和。
支持多个 chunk size，画出柱状图和平均值折线图。

文件夹结构示例:
data/
  chunk-000/
    episode_000000.parquet
    episode_000001.parquet
    ...
  chunk-001/
    episode_000100.parquet
    ...

使用示例:
python tools/calc_discrete_error_batch.py \
    --data_dir /path/to/data \
    --chunk_sizes 4 8 16 32 64 \
    --deltas 0.1 0.1 0.1 0.1 0.1 0.1 \
    --columns 0 1 2 3 4 5 \
    --output_dir ./action_plots

python tools/calc_discrete_error_batch.py \
    --data_dir /home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_10/libero_lerobot_10_sys0_eagle_-1/data \
    --chunk_sizes 4 8 16 32 64 \
    --deltas 0.9 0.8 0.9 0.085 0.15 0.3 \
    --columns 0 1 2 3 4 5 \
    --output_dir ./action_plots \
    --max_file 100
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


def calc_file_error(
    actions: np.ndarray,
    columns: list,
    deltas: list,
    chunk_size: int,
    beta: float = 0.6,
    alpha: float = 0.4,
) -> float:
    """
    计算单个文件的离散化误差累计和
    
    Args:
        actions: action 数组
        columns: 要计算的列索引列表
        deltas: 对应列的 delta 值列表
        chunk_size: chunk 大小
        beta, alpha: 离散化参数
    
    Returns:
        total_error: 该文件所有列、所有 chunk 的误差累计和
    """
    total_error = 0.0
    
    for col_idx, delta in zip(columns, deltas):
        col_data = actions[:, col_idx]
        num_samples = len(col_data)
        
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
            
            # 该 chunk 的积分面积差
            error = np.sum(np.abs(original_cumsum - discrete_cumsum))
            total_error += error
    
    return total_error


def main():
    parser = argparse.ArgumentParser(description='批量计算离散化累计误差')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据文件夹路径（包含 parquet 文件）')
    parser.add_argument('--chunk_sizes', type=int, nargs='+', default=[4, 8, 16, 32, 64],
                        help='要测试的 chunk size 列表，默认: 4 8 16 32 64')
    parser.add_argument('--deltas', type=float, nargs='+', default=None,
                        help='每列的 delta 值，默认自动估算（从第一个文件）')
    parser.add_argument('--columns', type=int, nargs='+', default=None,
                        help='要计算的列索引，默认所有列')
    parser.add_argument('--output_dir', type=str, default='./action_plots',
                        help='输出目录')
    parser.add_argument('--max_files', type=int, default=-1,
                        help='最大处理文件数，-1 表示全部处理')
    # discrete_chunk_calculus 参数
    parser.add_argument('--beta', type=float, default=0.6,
                        help='趋势项权重 (默认 0.6)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='趋势平滑系数 (默认 0.4)')
    
    args = parser.parse_args()
    
    # 查找所有 parquet 文件
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 文件夹不存在 {data_dir}")
        return
    
    print(f"扫描文件夹: {data_dir}")
    parquet_files = find_all_parquet_files(str(data_dir))
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    if len(parquet_files) == 0:
        print("错误: 未找到 parquet 文件")
        return
    
    # 限制文件数量
    if args.max_files > 0:
        parquet_files = parquet_files[:args.max_files]
        print(f"限制处理前 {args.max_files} 个文件")
    
    # 从第一个文件获取 action 维度
    first_actions = load_parquet_actions(str(parquet_files[0]))
    action_dim = first_actions.shape[1]
    print(f"Action 维度: {action_dim}")
    
    # 设置 columns
    if args.columns is None:
        columns = list(range(action_dim))
    else:
        columns = args.columns
    
    # 设置 deltas（从第一个文件自动估算）
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
    
    print(f"使用的列: {columns}")
    print(f"使用的 deltas: {deltas}")
    print(f"测试的 chunk sizes: {args.chunk_sizes}")
    
    # 计算每个 chunk size 的误差
    results = {}  # {chunk_size: {'total': float, 'avg': float, 'file_errors': list}}
    
    print("\n" + "=" * 60)
    print("开始计算...")
    print("=" * 60)
    
    for chunk_size in args.chunk_sizes:
        print(f"\n处理 Chunk Size: {chunk_size}")
        
        file_errors = []
        
        for parquet_file in tqdm(parquet_files, desc=f"chunk_size={chunk_size}"):
            try:
                actions = load_parquet_actions(str(parquet_file))
                error = calc_file_error(
                    actions, columns, deltas, chunk_size,
                    beta=args.beta, alpha=args.alpha
                )
                file_errors.append(error)
            except Exception as e:
                print(f"  警告: 处理 {parquet_file.name} 失败: {e}")
                continue
        
        total_error = sum(file_errors)
        avg_error = total_error / len(file_errors) if file_errors else 0
        
        results[chunk_size] = {
            'total': total_error,
            'avg': avg_error,
            'file_count': len(file_errors),
            'file_errors': file_errors
        }
        
        print(f"  文件数: {len(file_errors)}")
        print(f"  总误差累计和: {total_error:.4f}")
        print(f"  平均误差: {avg_error:.4f}")
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)
    print(f"{'Chunk Size':>12} {'文件数':>8} {'总误差':>15} {'平均误差':>15}")
    print("-" * 60)
    for chunk_size in args.chunk_sizes:
        r = results[chunk_size]
        print(f"{chunk_size:>12} {r['file_count']:>8} {r['total']:>15.4f} {r['avg']:>15.4f}")
    
    # 画图
    os.makedirs(args.output_dir, exist_ok=True)
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    chunk_sizes = list(results.keys())
    total_errors = [results[cs]['total'] for cs in chunk_sizes]
    avg_errors = [results[cs]['avg'] for cs in chunk_sizes]
    
    # Bar chart - Total Error
    x = np.arange(len(chunk_sizes))
    width = 0.6
    bars = ax1.bar(x, total_errors, width, color='steelblue', edgecolor='black', 
                   alpha=0.7, label='Total Error Sum')
    
    # Annotate values on bars
    for bar, error in zip(bars, total_errors):
        height = bar.get_height()
        ax1.annotate(f'{error:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='steelblue')
    
    ax1.set_xlabel('Chunk Size', fontsize=12)
    ax1.set_ylabel('Total Error Sum', fontsize=12, color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(cs) for cs in chunk_sizes])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Second Y axis - Average Error Line
    ax2 = ax1.twinx()
    line = ax2.plot(x, avg_errors, 'ro-', linewidth=2, markersize=8, label='Average Error')
    
    # Annotate values on line
    for i, (xi, avg) in enumerate(zip(x, avg_errors)):
        ax2.annotate(f'{avg:.2f}',
                    xy=(xi, avg),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color='red')
    
    ax2.set_ylabel('Average Error (Total / File Count)', fontsize=12, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # 标题
    file_count = results[chunk_sizes[0]]['file_count']
    plt.title(f'{data_dir.name}\nsum(Difference of Cumsum) vs Chunk Size (num files: {file_count})', fontsize=14)
    
    plt.tight_layout()
    
    save_path = os.path.join(args.output_dir, f'{data_dir.name}_batch_chunk_error_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n柱状图已保存: {save_path}")


if __name__ == '__main__':
    main()
