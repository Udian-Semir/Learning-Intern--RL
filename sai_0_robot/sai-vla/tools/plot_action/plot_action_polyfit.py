"""
绘制parquet文件中action数据的图表，并按chunk进行3阶多项式拟合
每一列action单独一张图，显示：
- 原始action数据
- 每个chunk的3阶多项式拟合曲线（不同颜色区分）
- chunk边界分隔线

使用示例:
python /home/dev/文档/huangwenlong/sai0-vla/tools/plot_action_polyfit.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset111_single_512_filter/data/chunk-000/episode_000000.parquet \
    --chunk_size 8 \
    --degree 4
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


def load_parquet_actions(parquet_path: str) -> np.ndarray:
    """
    从parquet文件加载action数据
    
    Args:
        parquet_path: parquet文件路径
    
    Returns:
        actions: action数据，shape为 (num_samples, action_dim)
    """
    df = pd.read_parquet(parquet_path)
    
    # 查找action列
    if 'action' in df.columns:
        actions = df['action'].tolist()
        actions = np.array(actions)
    else:
        raise ValueError(f"在parquet文件中未找到'action'列，可用列: {df.columns.tolist()}")
    
    print(f"加载了 {len(actions)} 帧数据，action维度: {actions.shape[1]}")
    return actions


def polyfit_chunk(frames: np.ndarray, values: np.ndarray, degree: int = 3):
    """
    对单个chunk进行多项式拟合（手动实现最小二乘法）
    
    数学原理：
    给定数据点 (x_i, y_i)，找 n 阶多项式 p(x) = a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
    使得 Σ(y_i - p(x_i))^2 最小
    
    构建范德蒙矩阵 V：
        V[i,j] = x_i^j, 其中 i=0..m-1, j=0..n
    
    解正规方程：(V^T * V) * a = V^T * y
    系数向量：a = (V^T * V)^(-1) * V^T * y
    
    Args:
        frames: 帧索引数组 (x 值)
        values: action值数组 (y 值)
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
    # V 的形状是 (n, degree+1)
    V = np.zeros((n, degree + 1))
    for j in range(degree + 1):
        V[:, j] = x_norm ** j
    
    # 解正规方程：(V^T * V) * a = V^T * y
    # a = (V^T * V)^(-1) * V^T * y
    VtV = V.T @ V  # (degree+1, degree+1)
    Vty = V.T @ y  # (degree+1,)
    
    # 使用求解线性方程组（比直接求逆更稳定）
    coeffs_norm = np.linalg.solve(VtV, Vty)  # [a_0, a_1, ..., a_n]
    
    # 计算拟合值
    fitted_values = V @ coeffs_norm
    
    # 将归一化系数转换回原始坐标系的系数
    # p(x) = Σ a_j * ((x - x_mean) / x_std)^j
    # 展开后得到原始 x 的系数
    coeffs_original = np.zeros(degree + 1)
    for j in range(degree + 1):
        # 二项式展开: ((x - x_mean) / x_std)^j = Σ C(j,k) * (-x_mean/x_std)^(j-k) * (x/x_std)^k
        for k in range(j + 1):
            binom_coeff = math.comb(j, k)
            term = binom_coeff * ((-x_mean / x_std) ** (j - k)) * (1 / x_std) ** k
            coeffs_original[k] += coeffs_norm[j] * term
    
    # 转换为高次到低次的顺序 [a_n, a_{n-1}, ..., a_0]
    coeffs = coeffs_original[::-1]
    
    # 计算R²值（决定系数）
    ss_res = np.sum((y - fitted_values) ** 2)  # 残差平方和
    ss_tot = np.sum((y - np.mean(y)) ** 2)      # 总平方和
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    
    return fitted_values, coeffs, r_squared


def plot_action_polyfit(actions: np.ndarray, chunk_size: int, degree: int, output_dir: str, parquet_name: str):
    """
    绘制每一列action的图表，包含原始数据和chunk分段多项式拟合
    
    Args:
        actions: action数据，shape为 (num_frames, action_dim)
        chunk_size: 每个chunk的大小
        degree: 多项式拟合阶数
        output_dir: 输出目录
        parquet_name: parquet文件名（用于标题）
    """
    num_frames, action_dim = actions.shape
    frames = np.arange(num_frames)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算chunk数量
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    print(f"总帧数: {num_frames}, chunk_size: {chunk_size}, chunk数量: {num_chunks}")
    
    # 生成颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, min(num_chunks, 10)))
    if num_chunks > 10:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_chunks))
    
    # 为每一列创建单独的图
    for col_idx in range(action_dim):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # 绘制原始action数据
        ax.plot(frames, actions[:, col_idx], 'b-', linewidth=1, marker='o', 
                markersize=2, label='Original', alpha=0.7)
        
        # 存储每个chunk的拟合信息
        fit_info = []
        
        # 按chunk进行拟合
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_frames)
            
            chunk_frames = frames[start:end]
            chunk_action = actions[start:end, col_idx]
            
            # 多项式拟合
            fitted_values, coeffs, r_squared = polyfit_chunk(chunk_frames, chunk_action, degree=degree)
            
            # 绘制拟合曲线
            color = colors[chunk_idx % len(colors)]
            ax.plot(chunk_frames, fitted_values, '-', linewidth=2, color=color,
                    label=f'Chunk {chunk_idx} (R²={r_squared:.4f})')
            
            fit_info.append({
                'chunk_idx': chunk_idx,
                'start': start,
                'end': end,
                'r_squared': r_squared,
                'coeffs': coeffs
            })
            
            # 绘制chunk边界线（除了最后一个chunk）
            if chunk_idx < num_chunks - 1:
                ax.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel(f'Action[{col_idx}]', fontsize=12)
        ax.set_title(f'{parquet_name}\nAction Column {col_idx} - {degree}th Order Polynomial Fit (chunk_size={chunk_size})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # 添加图例（如果chunk数量不太多）
        if num_chunks <= 10:
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.legend(loc='upper right', fontsize=6, ncol=2)
        
        # 添加统计信息
        col_data = actions[:, col_idx]
        avg_r_squared = np.mean([info['r_squared'] for info in fit_info])
        stats_text = f'Data Stats:\n  Min: {col_data.min():.4f}\n  Max: {col_data.max():.4f}\n  Mean: {col_data.mean():.4f}\n\nFit Stats:\n  Avg R²: {avg_r_squared:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f'action_col_{col_idx:02d}_polyfit.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")
    
    # 创建一个组合图，所有列在一起
    fig, axes = plt.subplots(action_dim, 1, figsize=(16, 4 * action_dim), sharex=True)
    if action_dim == 1:
        axes = [axes]
    
    for col_idx in range(action_dim):
        ax = axes[col_idx]
        
        # 绘制原始action数据
        ax.plot(frames, actions[:, col_idx], 'b-', linewidth=1, alpha=0.7, label='Original')
        
        # 按chunk进行拟合
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_frames)
            
            chunk_frames = frames[start:end]
            chunk_action = actions[start:end, col_idx]
            
            fitted_values, _, _ = polyfit_chunk(chunk_frames, chunk_action, degree=degree)
            
            color = colors[chunk_idx % len(colors)]
            ax.plot(chunk_frames, fitted_values, '-', linewidth=2, color=color)
            
            # 绘制chunk边界线
            if chunk_idx < num_chunks - 1:
                ax.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_ylabel(f'Col {col_idx}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    axes[-1].set_xlabel('Frame', fontsize=12)
    fig.suptitle(f'{parquet_name}\nAll Action Columns - {degree}th Order Polynomial Fit (chunk_size={chunk_size})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    combined_path = os.path.join(output_dir, 'action_all_columns_polyfit.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存组合图: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制parquet文件中action数据的图表，并按chunk进行多项式拟合')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径')
    parser.add_argument('--chunk_size', type=int, required=True,
                        help='每个chunk的大小（帧数）')
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
    
    # 加载数据
    actions = load_parquet_actions(parquet_path)
    
    # 绘制图表
    plot_action_polyfit(actions, args.chunk_size, args.degree, str(output_dir), parquet_name)
    
    print(f"\n完成！所有图片已保存到: {output_dir}")


if __name__ == '__main__':
    main()
