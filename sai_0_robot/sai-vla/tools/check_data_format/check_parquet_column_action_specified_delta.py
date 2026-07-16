"""
检查parquet文件中的action数据，并可视化离散化效果
# constrain_delta 或 chunk_calculus 离散化方法
使用示例:

# !
python tools/check_parquet_column-action_specified-delta.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset/data/chunk-000/episode_000000.parquet \
    --deltas 3 3 3 0.0025 0.0025 0.0025 3 3 3 0.0025 0.0025 0.0025 \
    --columns 0 1 2 3 4 5 6 7 8 9 10 11 12 13\
    --discrete_method constrain_delta \
    --chunk_size 25

python tools/check_parquet_column-action_specified-delta.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset/data/chunk-000/episode_000000.parquet \
    --deltas 3 3 3 1 1 1 3 3 3 1 1 1 \
    --columns 0 1 2 3 4 5 6 7 8 9 10 11 12 13\
    --discrete_method chunk_calculus \
    --beta 0.6 --alpha 0.4 \
    --chunk_size 25


python tools/check_parquet_column-action_specified-delta.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset_action_euler_unwrap/data/chunk-000/episode_000000.parquet \
    --deltas 3 3 3 0.5 0.5 0.5 3 3 3 0.5 0.5 0.5 \
    --columns 0 1 2 3 4 5 6 7 8 9 10 11 12 13\
    --discrete_method chunk_calculus \
    --beta 0.6 --alpha 0.4 \
    --chunk_size 25
# !

1. 直接传入parquet文件（使用默认 chunk_calculus 方法）:
python tools/check_parquet_column-action_specified-delta.py \
    --parquet_path /home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_10/libero_lerobot_10_sys0_eagle_-1/data/chunk-000/episode_000000.parquet \
    --deltas 0.1 0.8 0.9 0.085 0.15 0.3 \
    --columns 0 1 2 3 4 5 \
    --chunk_size 50 \
    --output_dir ./action_plots

2. 使用简单累积误差方法 (constrain_delta):
python tools/check_parquet_column-action_specified-delta.py \
    --parquet_path /data/HuangWenlong/datasets/.../chunk-000/episode_000000.parquet \
    --discrete_method constrain_delta \
    --chunk_size 16

3. 使用微积分方法 (chunk_calculus)，并调整参数:
python tools/check_parquet_column-action_specified-delta.py \
    --parquet_path /data/HuangWenlong/datasets/.../chunk-000/episode_000000.parquet \
    --discrete_method chunk_calculus \
    --beta 0.6 --alpha 0.4 \
    --chunk_size 16

4. 传入文件夹（自动找第一个episode_*.parquet文件）:
python tools/check_parquet_column-action_specified-delta.py \
    --parquet_path /data/HuangWenlong/datasets/.../chunk-000 \
    --chunk_size 16

离散化方法说明:
- constrain_delta: 简单累积误差方法，只使用积分项 (I)
- chunk_calculus: 基于微积分的方法，使用积分项+趋势预测 (I+D)，对快速变化的信号响应更好（默认）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示图像
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.discrete import (
    discrete_chunk_calculus, 
    discrete_constrain_delta,
    undiscrete_chunk_calculus,
    undiscrete_constrain_delta,
    DISCRETE_METHODS,
)


def load_parquet_actions(parquet_path: str) -> tuple:
    """
    从parquet文件加载action数据
    
    Args:
        parquet_path: parquet文件路径
    
    Returns:
        actions: action数据，shape为 (num_samples, action_dim)
        df: 原始DataFrame
    """
    df = pd.read_parquet(parquet_path)
    
    # 查找action列
    if 'action' in df.columns:
        actions = df['action'].tolist()
        # 转换为numpy数组
        actions = np.array(actions)
    else:
        raise ValueError(f"未找到'action'列，可用列: {df.columns.tolist()}")
    
    return actions, df


def discrete_and_undiscrete_chunk(
    actions: np.ndarray,
    column: int,
    delta: float,
    chunk_size: int,
    method: str = "chunk_calculus",
    beta: float = 0.6,
    alpha: float = 0.4,
) -> tuple:
    """
    对指定列按 chunk 进行离散化再反离散化
    
    流程:
    1. 将数据按 chunk_size 分块
    2. 对每个 chunk 使用指定方法离散化 -> {-1, 0, 1}
    3. 反离散化 -> {-delta, 0, delta}
    4. 每个 chunk 从原始轨迹的当前位置开始累加，得到位置
    
    关键：每个 chunk 开始时，从原始轨迹的当前位置（而不是上个 chunk 离散化后的位置）开始，
    这样可以避免离散化误差的累积。
    
    Args:
        actions: action数组 (num_samples, action_dim)
        column: 要处理的列索引
        delta: 离散化步长
        chunk_size: chunk 大小
        method: 离散化方法
            - "constrain_delta": 简单累积误差方法
            - "chunk_calculus": 基于微积分的方法（默认）
        beta: 趋势项权重 (仅 chunk_calculus 使用)
        alpha: 趋势平滑系数 (仅 chunk_calculus 使用)
    
    Returns:
        reconstructed_position: 反离散化后累加得到的位置（每个 chunk 从原始位置开始）
        undiscrete_data: 反离散化后的 delta action {-delta, 0, delta}
    """
    col_data = actions[:, column].copy()
    num_samples = len(col_data)
    
    # 计算原始轨迹位置（作为每个 chunk 的起点参考）
    original_position = np.cumsum(col_data)
    
    # 按 chunk 分块处理
    undiscrete_data = np.zeros(num_samples, dtype=float)
    reconstructed_position = np.zeros(num_samples, dtype=float)
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_data = col_data[start_idx:end_idx]
        
        # Step 1: 离散化，将连续值转换为 {-1, 0, 1}
        if method == "chunk_calculus":
            discrete_data = discrete_chunk_calculus(
                chunk_data, delta, beta=beta, alpha=alpha
            )
        elif method == "constrain_delta":
            discrete_data = discrete_constrain_delta(chunk_data, delta)
        else:
            raise ValueError(f"未知的离散化方法: {method}. 可用方法: {DISCRETE_METHODS}")
        
        # Step 2: 反离散化，将 {-1, 0, 1} 乘以 delta 得到 {-delta, 0, delta}
        # 两种方法的反离散化都是简单乘以 delta
        chunk_undiscrete = undiscrete_chunk_calculus(discrete_data, delta)
        undiscrete_data[start_idx:end_idx] = chunk_undiscrete
        
        # Step 3: 每个 chunk 从原始轨迹的当前位置开始累加
        # chunk 起点 = 原始轨迹在 start_idx-1 的位置（第一个 chunk 起点为 0）
        if start_idx == 0:
            chunk_start_pos = 0.0
        else:
            chunk_start_pos = original_position[start_idx - 1]
        
        # 在 chunk 内累加
        chunk_cumsum = np.cumsum(chunk_undiscrete)
        reconstructed_position[start_idx:end_idx] = chunk_start_pos + chunk_cumsum
    
    return reconstructed_position, undiscrete_data


def estimate_delta_for_column(col_data: np.ndarray) -> float:
    """
    根据数据范围估算合适的delta值
    
    Args:
        col_data: 单列数据
    
    Returns:
        估算的delta值
    """
    # 计算相邻差值
    diffs = np.diff(col_data)
    
    # 使用差值的标准差来估算delta
    if len(diffs) > 0:
        std = np.std(np.abs(diffs))
        mean_abs = np.mean(np.abs(diffs))
        
        # delta 应该是一个合适的步长
        estimated_delta = max(mean_abs, std) if mean_abs > 0 else 0.01
        
        # 四舍五入到合理的值
        if estimated_delta > 0.1:
            estimated_delta = round(estimated_delta, 1)
        elif estimated_delta > 0.01:
            estimated_delta = round(estimated_delta, 2)
        else:
            estimated_delta = round(estimated_delta, 3)
        
        return max(estimated_delta, 0.001)  # 最小值为0.001
    
    return 0.01


def plot_action_comparison(
    actions: np.ndarray,
    columns: list,
    deltas: list,
    chunk_size: int,
    output_dir: str,
    parquet_name: str,
    method: str = "chunk_calculus",
    beta: float = 0.6,
    alpha: float = 0.4,
):
    """
    画出原始action和离散化后再反离散化的对比图（只有两列：delta对比 + 位置对比）
    
    Args:
        actions: action数组 (num_samples, action_dim)
        columns: 要处理的列索引列表
        deltas: 对应的delta值列表
        chunk_size: chunk 大小
        output_dir: 输出目录
        parquet_name: parquet文件名（用于标题）
        method: 离散化方法 ("constrain_delta" 或 "chunk_calculus")
        beta: 趋势项权重 (仅 chunk_calculus 使用)
        alpha: 趋势平滑系数 (仅 chunk_calculus 使用)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_samples = actions.shape[0]
    x = np.arange(num_samples)
    
    # 只为指定的列创建单独的图
    for i, col_idx in enumerate(columns):
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        
        col_data = actions[:, col_idx]
        
        # 使用对应的delta值
        delta = deltas[i]
        
        # 计算原始数据的累加（位置）
        original_position = np.cumsum(col_data)
        
        # 离散化再反离散化（按 chunk 处理）
        reconstructed_position, undiscrete_data = discrete_and_undiscrete_chunk(
            actions, col_idx, delta, chunk_size, method, beta, alpha
        )
        
        # 子图1: delta action对比
        ax1 = axes[0]
        ax1.plot(x, col_data, 'b-', label='Original Delta Action', alpha=0.7, linewidth=1)
        # 每个 chunk 单独画，使用独立水平线段展示离散值
        for chunk_idx, chunk_start in enumerate(range(0, num_samples, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_x = x[chunk_start:chunk_end]
            chunk_undiscrete = undiscrete_data[chunk_start:chunk_end]
            label = f'Discrete (δ={delta}, chunk={chunk_size})' if chunk_idx == 0 else None
            # 用 hlines 画独立的水平线段，每个点一条线段（宽度0.8个单位）
            ax1.hlines(chunk_undiscrete, chunk_x - 0.3, chunk_x + 0.3, colors='r', alpha=0.8, linewidth=2, label=label)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Delta Value')
        ax1.set_title(f'Delta Action Comparison - Feature {col_idx}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加 chunk 分界线
        for chunk_start in range(0, num_samples, chunk_size):
            ax1.axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.5)
        
        # 子图2: 累加位置对比
        ax2 = axes[1]
        ax2.plot(x, original_position, 'b-', label='Original Position (cumsum)', alpha=0.7, linewidth=1.5)
        # 每个 chunk 单独画，使用独立水平线段展示离散累加
        for chunk_idx, chunk_start in enumerate(range(0, num_samples, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_x = x[chunk_start:chunk_end]
            chunk_recon = reconstructed_position[chunk_start:chunk_end]
            label = f'Reconstructed (δ={delta}, chunk={chunk_size})' if chunk_idx == 0 else None
            # 用 hlines 画独立的水平线段（宽度0.8个单位）
            ax2.hlines(chunk_recon, chunk_x - 0.3, chunk_x + 0.3, colors='r', alpha=0.8, linewidth=2, label=label)
        
        # 计算误差
        error = np.abs(original_position - reconstructed_position)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Position Value')
        ax2.set_title(f'Position Comparison - Feature {col_idx} | Max Error: {max_error:.4f}, Mean Error: {mean_error:.4f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加 chunk 分界线
        for chunk_start in range(0, num_samples, chunk_size):
            ax2.axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(output_dir, f'{parquet_name}_feature_{col_idx}_delta_{delta}_chunk_{chunk_size}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Feature {col_idx}] delta={delta}, chunk_size={chunk_size}, max_error={max_error:.6f}, mean_error={mean_error:.6f} -> {save_path}")
    
    # 创建一个汇总图，只包含指定的列，2列布局
    num_cols = len(columns)
    fig, axes = plt.subplots(num_cols, 2, figsize=(24, 5 * num_cols))
    if num_cols == 1:
        axes = axes.reshape(1, -1)
    
    for i, col_idx in enumerate(columns):
        col_data = actions[:, col_idx]
        
        # 使用对应的delta值
        delta = deltas[i]
        
        original_position = np.cumsum(col_data)
        reconstructed_position, undiscrete_data = discrete_and_undiscrete_chunk(
            actions, col_idx, delta, chunk_size, method, beta, alpha
        )
        
        error = np.abs(original_position - reconstructed_position)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        # 第一列：delta对比
        axes[i, 0].plot(x, col_data, 'b-', label='Original', alpha=0.7, linewidth=1)
        # 每个 chunk 单独画，使用独立水平线段展示离散值
        for chunk_idx, chunk_start in enumerate(range(0, num_samples, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_x = x[chunk_start:chunk_end]
            chunk_undiscrete = undiscrete_data[chunk_start:chunk_end]
            label = f'Discrete (δ={delta})' if chunk_idx == 0 else None
            # 用 hlines 画独立的水平线段（宽度0.6个单位）
            axes[i, 0].hlines(chunk_undiscrete, chunk_x - 0.3, chunk_x + 0.3, colors='r', alpha=0.8, linewidth=2, label=label)
        axes[i, 0].set_ylabel(f'Feature {col_idx}')
        axes[i, 0].legend(loc='upper right', fontsize=8)
        axes[i, 0].grid(True, alpha=0.3)
        
        # 添加 chunk 分界线
        for chunk_start in range(0, num_samples, chunk_size):
            axes[i, 0].axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.5)
        
        # 第二列：位置对比（累加）
        axes[i, 1].plot(x, original_position, 'b-', label='Original', alpha=0.7, linewidth=1.5)
        # 每个 chunk 单独画，使用独立水平线段展示离散累加
        for chunk_idx, chunk_start in enumerate(range(0, num_samples, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, num_samples)
            chunk_x = x[chunk_start:chunk_end]
            chunk_recon = reconstructed_position[chunk_start:chunk_end]
            label = 'Reconstructed' if chunk_idx == 0 else None
            # 用 hlines 画独立的水平线段（宽度0.6个单位）
            axes[i, 1].hlines(chunk_recon, chunk_x - 0.3, chunk_x + 0.3, colors='r', alpha=0.8, linewidth=2, label=label)
        axes[i, 1].set_ylabel(f'Feature {col_idx}')
        axes[i, 1].set_title(f'Max Err: {max_error:.4f}, Mean Err: {mean_error:.4f}', fontsize=9)
        axes[i, 1].legend(loc='upper right', fontsize=8)
        axes[i, 1].grid(True, alpha=0.3)
        
        # 添加 chunk 分界线
        for chunk_start in range(0, num_samples, chunk_size):
            axes[i, 1].axvline(x=chunk_start, color='gray', linestyle=':', alpha=0.5)
    
    axes[0, 0].set_title('Delta Action Comparison')
    axes[0, 1].set_title('Position Comparison (Cumsum)')
    axes[-1, 0].set_xlabel('Sample Index')
    axes[-1, 1].set_xlabel('Sample Index')
    
    plt.suptitle(f'{parquet_name} (chunk_size={chunk_size}, method={method})', fontsize=12, y=1.02)
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, f'{parquet_name}_summary_chunk_{chunk_size}.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n汇总图已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='检查parquet文件中的action数据，可视化离散化效果')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径或所在目录。如果是目录，会自动找第一个episode_*.parquet文件')
    parser.add_argument('--deltas', type=float, nargs='+', default=None,
                        help='每个列的delta值，例如: 0.01 0.01 0.01 0.05 0.05 0.05 0.5')
    parser.add_argument('--columns', type=int, nargs='+', default=None,
                        help='要离散化的列索引，例如: 0 1 2 3 4 5 6')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='chunk大小，默认为16')
    parser.add_argument('--output_dir', type=str, default='./action_plots',
                        help='输出图片的目录')
    # 离散化方法选择
    parser.add_argument('--discrete_method', type=str, default='chunk_calculus',
                        choices=['constrain_delta', 'chunk_calculus'],
                        help='离散化方法: constrain_delta (简单累积误差) 或 chunk_calculus (微积分方法，默认)')
    # discrete_chunk_calculus 参数
    parser.add_argument('--beta', type=float, default=0.6,
                        help='趋势项权重，控制微分作用强度 (默认 0.6，仅 chunk_calculus 方法使用)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='趋势平滑系数，alpha 越大响应越快 (默认 0.4，仅 chunk_calculus 方法使用)')
    
    args = parser.parse_args()
    
    # 判断输入是文件还是目录
    input_path = Path(args.parquet_path)
    if not input_path.exists():
        print(f"错误: 路径不存在 {input_path}")
        return
    
    if input_path.is_file():
        # 直接传入parquet文件
        parquet_path = input_path
    else:
        # 传入目录，查找parquet文件
        parquet_dir = input_path
        # 查找第一个 episode_*.parquet 文件
        parquet_files = sorted(parquet_dir.glob('episode_*.parquet'))
        if not parquet_files:
            # 尝试查找所有parquet文件
            parquet_files = sorted(parquet_dir.glob('*.parquet'))
        
        if not parquet_files:
            print(f"错误: 在 {parquet_dir} 中未找到parquet文件")
            return
        
        parquet_path = parquet_files[0]
    
    print(f"加载parquet文件: {parquet_path}")
    
    # 加载数据
    try:
        actions, df = load_parquet_actions(str(parquet_path))
    except Exception as e:
        print(f"加载失败: {e}")
        print(f"尝试查看文件内容...")
        df = pd.read_parquet(str(parquet_path))
        print(f"列名: {df.columns.tolist()}")
        print(f"数据形状: {df.shape}")
        print(df.head())
        return
    
    print(f"Action数据形状: {actions.shape}")
    print(f"Action维度: {actions.shape[1]}")
    print(f"样本数量: {actions.shape[0]}")
    print(f"Chunk大小: {args.chunk_size}")
    print(f"Chunk数量: {(actions.shape[0] + args.chunk_size - 1) // args.chunk_size}")
    
    # 显示每列的统计信息
    print("\n=== Action统计信息 ===")
    for col_idx in range(actions.shape[1]):
        col_data = actions[:, col_idx]
        print(f"  Feature {col_idx}: min={col_data.min():.6f}, max={col_data.max():.6f}, "
              f"mean={col_data.mean():.6f}, std={col_data.std():.6f}")
    
    parquet_name = parquet_path.stem
    
    # 设置columns和deltas
    if args.columns is None:
        columns = list(range(actions.shape[1]))
    else:
        columns = args.columns
    
    if args.deltas is None:
        # 自动估算每列的delta
        deltas = []
        print("\n=== 自动估算Delta值 ===")
        for col_idx in columns:
            delta = estimate_delta_for_column(actions[:, col_idx])
            deltas.append(delta)
            print(f"  Feature {col_idx}: estimated delta = {delta}")
    else:
        deltas = args.deltas
        if len(deltas) != len(columns):
            print(f"警告: deltas数量({len(deltas)})与columns数量({len(columns)})不匹配")
            # 扩展或截断deltas
            if len(deltas) < len(columns):
                deltas = deltas + [deltas[-1]] * (len(columns) - len(deltas))
            else:
                deltas = deltas[:len(columns)]
    
    print(f"\n使用的列: {columns}")
    print(f"使用的delta: {deltas}")
    print(f"离散化方法: {args.discrete_method}")
    if args.discrete_method == "chunk_calculus":
        print(f"  参数: beta={args.beta}, alpha={args.alpha}")
    
    # 画图
    plot_action_comparison(
        actions,
        columns,
        deltas,
        args.chunk_size,
        args.output_dir,
        parquet_name,
        method=args.discrete_method,
        beta=args.beta,
        alpha=args.alpha,
    )
    
    print(f"\n所有图片已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()
