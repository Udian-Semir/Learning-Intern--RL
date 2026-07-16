"""
绘制parquet文件中state数据的图表
每一列state单独一张图，包含两个子图：
- 左图：state原始值 + chunk多项式拟合平滑曲线
- 右图：state的累积和 cumsum(state)

使用示例:
python /home/dev/文档/huangwenlong/sai0-vla/tools/plot_state_columns.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset111_single_512_filter/data/chunk-000/episode_000000.parquet \
    --chunk_size 8 \
    --degree 3
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path


# 获取当前脚本所在目录
SCRIPT_DIR = Path(__file__).parent
SCRIPT_NAME = Path(__file__).stem  # 不含扩展名的文件名


def load_parquet_states(parquet_path: str) -> np.ndarray:
    """
    从parquet文件加载state数据
    
    Args:
        parquet_path: parquet文件路径
    
    Returns:
        states: state数据，shape为 (num_samples, state_dim)
    """
    df = pd.read_parquet(parquet_path)
    
    # 查找state列 (可能是 observation.state 或 state)
    if 'observation.state' in df.columns:
        states = df['observation.state'].tolist()
        states = np.array(states)
    elif 'state' in df.columns:
        states = df['state'].tolist()
        states = np.array(states)
    else:
        raise ValueError(f"在parquet文件中未找到 'observation.state' 或 'state' 列，可用列: {df.columns.tolist()}")
    
    print(f"加载了 {len(states)} 帧数据，state维度: {states.shape[1]}")
    return states


def polyfit_chunk(frames: np.ndarray, values: np.ndarray, degree: int = 3):
    """
    对单个chunk进行多项式拟合（手动实现最小二乘法）
    
    Args:
        frames: 帧索引数组 (x 值)
        values: 数据值数组 (y 值)
        degree: 多项式阶数，默认3阶
    
    Returns:
        fitted_values: 拟合后的值
        coeffs: 多项式系数 [a_n, a_{n-1}, ..., a_1, a_0]（高次到低次）
        r_squared: R²值（拟合优度）
    """
    x = frames.astype(np.float64)
    y = values.astype(np.float64)
    n = len(x)
    
    # 数值稳定性处理：对 x 进行归一化
    x_mean = np.mean(x)
    x_std = np.std(x) if np.std(x) > 0 else 1.0
    x_norm = (x - x_mean) / x_std
    
    # 构建范德蒙矩阵 V[i,j] = x_norm[i]^j
    V = np.zeros((n, degree + 1))
    for j in range(degree + 1):
        V[:, j] = x_norm ** j
    
    # 解正规方程：(V^T * V) * a = V^T * y
    VtV = V.T @ V
    Vty = V.T @ y
    
    # 使用求解线性方程组
    coeffs_norm = np.linalg.solve(VtV, Vty)
    
    # 计算拟合值
    fitted_values = V @ coeffs_norm
    
    # 将归一化系数转换回原始坐标系的系数
    coeffs_original = np.zeros(degree + 1)
    for j in range(degree + 1):
        for k in range(j + 1):
            binom_coeff = math.comb(j, k)
            term = binom_coeff * ((-x_mean / x_std) ** (j - k)) * (1 / x_std) ** k
            coeffs_original[k] += coeffs_norm[j] * term
    
    coeffs = coeffs_original[::-1]
    
    # 计算R²值
    ss_res = np.sum((y - fitted_values) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return fitted_values, coeffs, r_squared


def plot_state_columns(states: np.ndarray, output_dir: str, parquet_name: str,
                       chunk_size: int = None, degree: int = 3):
    """
    绘制每一列state的图表，每张图包含两个子图
    
    Args:
        states: state数据，shape为 (num_frames, state_dim)
        output_dir: 输出目录
        parquet_name: parquet文件名（用于标题）
        chunk_size: 每个chunk的大小，None表示不做拟合
        degree: 多项式拟合阶数
    """
    num_frames, state_dim = states.shape
    frames = np.arange(num_frames)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算累积和
    cumulative_sum = np.cumsum(states, axis=0)
    
    # 计算chunk数量
    if chunk_size is not None:
        num_chunks = (num_frames + chunk_size - 1) // chunk_size
        print(f"总帧数: {num_frames}, chunk_size: {chunk_size}, chunk数量: {num_chunks}")
        
        # 生成颜色映射
        chunk_colors = plt.cm.tab10(np.linspace(0, 1, min(num_chunks, 10)))
        if num_chunks > 10:
            chunk_colors = plt.cm.rainbow(np.linspace(0, 1, num_chunks))
    
    # 为每一列创建单独的图（包含两个子图）
    for col_idx in range(state_dim):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：state值 + 拟合曲线
        ax1.plot(frames, states[:, col_idx], 'b-', linewidth=1, marker='o', 
                 markersize=2, label='Original', alpha=0.7)
        
        # 如果有chunk_size，添加拟合曲线
        fit_r2_list = []
        if chunk_size is not None:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, num_frames)
                
                chunk_frames = frames[start:end]
                chunk_state = states[start:end, col_idx]
                
                fitted_values, _, r_squared = polyfit_chunk(chunk_frames, chunk_state, degree=degree)
                fit_r2_list.append(r_squared)
                
                color = chunk_colors[chunk_idx % len(chunk_colors)]
                ax1.plot(chunk_frames, fitted_values, '-', linewidth=2, color=color,
                         label=f'Chunk {chunk_idx} (R²={r_squared:.3f})' if num_chunks <= 10 else '')
                
                # 绘制chunk边界线
                if chunk_idx < num_chunks - 1:
                    ax1.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel(f'State[{col_idx}]', fontsize=12)
        title_suffix = f' (chunk={chunk_size}, degree={degree})' if chunk_size else ''
        ax1.set_title(f'State Value{title_suffix}', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 左图统计信息
        col_data = states[:, col_idx]
        stats_text = f'Min: {col_data.min():.4f}\nMax: {col_data.max():.4f}\nMean: {col_data.mean():.4f}\nStd: {col_data.std():.4f}'
        if fit_r2_list:
            stats_text += f'\n\nAvg R²: {np.mean(fit_r2_list):.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 添加图例（如果chunk数量不太多）
        if chunk_size is not None and num_chunks <= 10:
            ax1.legend(loc='upper right', fontsize=8)
        
        # 右图：累积和
        ax2.plot(frames, cumulative_sum[:, col_idx], 'r-', linewidth=1, marker='o', markersize=2)
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Cumulative Sum', fontsize=12)
        ax2.set_title(f'Cumulative Sum (cumsum(state))', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 右图统计信息
        cum_sum_data = cumulative_sum[:, col_idx]
        cum_stats_text = f'Total: {cum_sum_data[-1]:.4f}\nMean: {np.mean(cum_sum_data):.4f}'
        ax2.text(0.02, 0.98, cum_stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        fig.suptitle(f'{parquet_name} - State Column {col_idx}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f'state_col_{col_idx:02d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")
    
    # 创建一个组合图，所有列在一起（两列子图）
    fig, axes = plt.subplots(state_dim, 2, figsize=(16, 3 * state_dim), sharex=True)
    if state_dim == 1:
        axes = axes.reshape(1, 2)
    
    for col_idx in range(state_dim):
        # 左列：state值 + 拟合曲线
        ax1 = axes[col_idx, 0]
        ax1.plot(frames, states[:, col_idx], 'b-', linewidth=1, alpha=0.7)
        
        # 添加拟合曲线
        if chunk_size is not None:
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, num_frames)
                
                chunk_frames = frames[start:end]
                chunk_state = states[start:end, col_idx]
                
                fitted_values, _, _ = polyfit_chunk(chunk_frames, chunk_state, degree=degree)
                
                color = chunk_colors[chunk_idx % len(chunk_colors)]
                ax1.plot(chunk_frames, fitted_values, '-', linewidth=2, color=color)
                
                if chunk_idx < num_chunks - 1:
                    ax1.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax1.set_ylabel(f'Col {col_idx}', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        
        # 右列：累积和
        ax2 = axes[col_idx, 1]
        ax2.plot(frames, cumulative_sum[:, col_idx], 'r-', linewidth=1)
        ax2.set_ylabel(f'Cum Sum {col_idx}', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=8)
    
    axes[-1, 0].set_xlabel('Frame', fontsize=12)
    axes[-1, 1].set_xlabel('Frame', fontsize=12)
    title_suffix = f' + Polyfit (chunk={chunk_size}, degree={degree})' if chunk_size else ''
    axes[0, 0].set_title(f'State Value{title_suffix}', fontsize=12)
    axes[0, 1].set_title('Cumulative Sum', fontsize=12)
    fig.suptitle(f'{parquet_name}\nAll State Columns', fontsize=14, y=1.02)
    plt.tight_layout()
    
    combined_path = os.path.join(output_dir, 'state_all_columns.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存组合图: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制parquet文件中state数据的图表')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径')
    parser.add_argument('--chunk_size', type=int, default=None,
                        help='每个chunk的大小（帧数），用于多项式拟合平滑曲线')
    parser.add_argument('--degree', type=int, default=3,
                        help='多项式拟合阶数，默认3阶')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认为当前脚本所在目录下以脚本名命名的文件夹')
    
    args = parser.parse_args()
    
    # 处理路径
    parquet_path = args.parquet_path
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"文件不存在: {parquet_path}")
    
    parquet_name = os.path.basename(parquet_path)
    
    # 设置输出目录：默认为 tools/outputs/<脚本名>/
    if args.output_dir is None:
        output_dir = SCRIPT_DIR.parent / "outputs" / SCRIPT_NAME
    else:
        output_dir = Path(args.output_dir)
    
    print(f"=" * 60)
    print(f"State 曲线可视化")
    print(f"=" * 60)
    print(f"输入文件: {parquet_path}")
    print(f"输出目录: {output_dir}")
    if args.chunk_size:
        print(f"Chunk 大小: {args.chunk_size}")
        print(f"多项式阶数: {args.degree}")
    print()
    
    # 加载数据
    states = load_parquet_states(parquet_path)
    
    # 绘制图表
    plot_state_columns(states, str(output_dir), parquet_name,
                       chunk_size=args.chunk_size, degree=args.degree)
    
    print(f"\n完成！所有图片已保存到: {output_dir}")


if __name__ == '__main__':
    main()
