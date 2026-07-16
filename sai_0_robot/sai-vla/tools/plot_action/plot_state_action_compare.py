"""
State 拟合与 Action 对比可视化脚本

对比两种 action 拟合方式：
1. 直接对 action 进行多项式拟合
2. 先对 state 进行多项式拟合，再计算 state 差分得到 action delta

绘图内容（左右两个子图）：
左图 - Action 对比：
  - 原始 action 数据 (蓝色点线)
  - action 直接多项式拟合曲线 (绿色)
  - state 拟合后差分得到的 action delta (红色)

右图 - State 重建对比（取第一个点 + 累加 action）：
  - 原始 state (蓝色点线)
  - 原始 action 累加重建的 state (青色)
  - action 拟合后累加重建的 state (绿色)
  - state 拟合曲线 (红色)

使用示例:
python /home/dev/文档/huangwenlong/sai0-vla/tools/plot_state_action_compare.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset111_single_512_filter/data/chunk-000/episode_000000.parquet \
    --chunk_size 8 \
    --degree 3

# 指定特定的 state 和 action 列进行对比
python /home/dev/文档/huangwenlong/sai0-vla/tools/plot_state_action_compare.py \
    --parquet_path /path/to/episode.parquet \
    --chunk_size 8 \
    --state_cols 0 1 2 \
    --action_cols 0 1 2
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


def load_parquet_data(parquet_path: str):
    """
    从parquet文件加载 state 和 action 数据
    
    Args:
        parquet_path: parquet文件路径
    
    Returns:
        states: state数据，shape为 (num_samples, state_dim)
        actions: action数据，shape为 (num_samples, action_dim)
    """
    df = pd.read_parquet(parquet_path)
    
    # 加载 action
    if 'action' in df.columns:
        actions = np.array(df['action'].tolist())
    else:
        raise ValueError(f"在parquet文件中未找到'action'列，可用列: {df.columns.tolist()}")
    
    # 加载 state (可能是 observation.state 或 state)
    states = None
    if 'observation.state' in df.columns:
        states = np.array(df['observation.state'].tolist())
    elif 'state' in df.columns:
        states = np.array(df['state'].tolist())
    else:
        raise ValueError(f"在parquet文件中未找到 'observation.state' 或 'state' 列，可用列: {df.columns.tolist()}")
    
    print(f"加载了 {len(actions)} 帧数据")
    print(f"  State 维度: {states.shape[1]}")
    print(f"  Action 维度: {actions.shape[1]}")
    
    return states, actions


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
    
    # 解正规方程
    VtV = V.T @ V
    Vty = V.T @ y
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


def plot_state_action_compare(states: np.ndarray, actions: np.ndarray, 
                               chunk_size: int, degree: int, 
                               output_dir: str, parquet_name: str,
                               state_cols: list = None, action_cols: list = None):
    """
    绘制 state 拟合差分与 action 拟合的对比图
    
    Args:
        states: state数据，shape为 (num_frames, state_dim)
        actions: action数据，shape为 (num_frames, action_dim)
        chunk_size: 每个chunk的大小
        degree: 多项式拟合阶数
        output_dir: 输出目录
        parquet_name: parquet文件名（用于标题）
        state_cols: 要处理的 state 列索引列表
        action_cols: 要处理的 action 列索引列表
    """
    num_frames = len(actions)
    frames = np.arange(num_frames)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算chunk数量
    num_chunks = (num_frames + chunk_size - 1) // chunk_size
    print(f"总帧数: {num_frames}, chunk_size: {chunk_size}, chunk数量: {num_chunks}")
    
    # 确定要处理的列
    if state_cols is None:
        state_cols = list(range(states.shape[1]))
    if action_cols is None:
        action_cols = list(range(actions.shape[1]))
    
    # 确定对比的列数（取两者的最小值）
    num_compare_cols = min(len(state_cols), len(action_cols))
    print(f"将对比 {num_compare_cols} 个维度")
    print(f"  State 列: {state_cols[:num_compare_cols]}")
    print(f"  Action 列: {action_cols[:num_compare_cols]}")
    
    # 生成颜色映射（用于不同 chunk）
    chunk_colors = plt.cm.tab10(np.linspace(0, 1, min(num_chunks, 10)))
    if num_chunks > 10:
        chunk_colors = plt.cm.rainbow(np.linspace(0, 1, num_chunks))
    
    # 为每对 state-action 列创建对比图
    for i in range(num_compare_cols):
        state_col = state_cols[i]
        action_col = action_cols[i]
        
        # 创建左右两个子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # ============= 左图：Action 对比 =============
        # 绘制原始 action 数据
        ax1.plot(frames, actions[:, action_col], 'b-', linewidth=0.5, marker='o', 
                markersize=1, label='Original Action', alpha=0.7)
        
        # 存储拟合信息和拟合后的数据
        action_fit_r2_list = []
        state_fit_r2_list = []
        
        # 存储所有 chunk 的拟合结果（用于右图累加）
        all_fitted_action = np.zeros(num_frames)
        all_fitted_state = np.zeros(num_frames)
        all_action_delta_from_state = []  # 差分后长度不一样，分开存储
        
        # 按 chunk 进行拟合
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_frames)
            
            chunk_frames = frames[start:end]
            chunk_action = actions[start:end, action_col]
            chunk_state = states[start:end, state_col]
            
            color = chunk_colors[chunk_idx % len(chunk_colors)]
            
            # 方法1: 直接对 action 多项式拟合
            fitted_action, _, action_r2 = polyfit_chunk(chunk_frames, chunk_action, degree=degree)
            action_fit_r2_list.append(action_r2)
            all_fitted_action[start:end] = fitted_action
            
            # 绘制 action 拟合曲线
            ax1.plot(chunk_frames, fitted_action, '-', linewidth=1, color='green',
                    alpha=0.8, label='Action Polyfit' if chunk_idx == 0 else '')
            
            # 方法2: 对 state 多项式拟合，然后计算差分
            fitted_state, _, state_r2 = polyfit_chunk(chunk_frames, chunk_state, degree=degree)
            state_fit_r2_list.append(state_r2)
            all_fitted_state[start:end] = fitted_state
            
            # 计算 state 差分得到 action delta
            action_delta_from_state = np.diff(fitted_state)
            all_action_delta_from_state.append((chunk_frames[1:], action_delta_from_state))
            
            # 差分后的帧索引
            delta_frames = chunk_frames[1:]
            
            # 绘制 state 差分得到的 action delta
            ax1.plot(delta_frames, action_delta_from_state, '-', linewidth=1, color='red',
                    alpha=0.8, label='Action from State Diff' if chunk_idx == 0 else '')
            
            # 绘制 chunk 边界线
            if chunk_idx < num_chunks - 1:
                ax1.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 设置左图
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel(f'Action[{action_col}]', fontsize=12)
        ax1.set_title(f'Action Comparison\n(degree={degree}, chunk_size={chunk_size})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=10)
        
        # 左图统计信息
        action_data = actions[:, action_col]
        avg_action_r2 = np.mean(action_fit_r2_list)
        avg_state_r2 = np.mean(state_fit_r2_list)
        stats_text = (f'Action Data:\n'
                      f'  Min: {action_data.min():.4f}\n'
                      f'  Max: {action_data.max():.4f}\n'
                      f'  Mean: {action_data.mean():.4f}\n\n'
                      f'Avg R² (Action Fit): {avg_action_r2:.4f}\n'
                      f'Avg R² (State Fit): {avg_state_r2:.4f}')
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ============= 右图：State 重建对比 =============
        # 原始 state
        original_state = states[:, state_col]
        state_init = original_state[0]  # 取第一个点
        
        # 绘制原始 state
        ax2.plot(frames, original_state, 'b-', linewidth=0.5, marker='o', 
                markersize=1, label='Original State', alpha=0.7)
        
        # 1. 原始 action 累加重建的 state: state[0] + cumsum(action)
        state_from_original_action = state_init + np.cumsum(actions[:, action_col])
        # 插入初始值，使得第一个点是 state_init
        state_from_original_action = np.insert(state_from_original_action[:-1], 0, state_init)
        ax2.plot(frames, state_from_original_action, 'c-', linewidth=1, alpha=0.8,
                label='State from Original Action')
        
        # 2. action 拟合后累加重建的 state: state[0] + cumsum(fitted_action)
        state_from_fitted_action = state_init + np.cumsum(all_fitted_action)
        state_from_fitted_action = np.insert(state_from_fitted_action[:-1], 0, state_init)
        ax2.plot(frames, state_from_fitted_action, 'g-', linewidth=1, alpha=0.8,
                label='State from Fitted Action')
        
        # 3. state 拟合曲线（直接画，不是累加）
        ax2.plot(frames, all_fitted_state, 'r-', linewidth=1, alpha=0.8,
                label='Fitted State')
        
        # 绘制 chunk 边界线
        for chunk_idx in range(num_chunks - 1):
            end = (chunk_idx + 1) * chunk_size
            ax2.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 设置右图
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel(f'State[{state_col}]', fontsize=12)
        ax2.set_title(f'State Reconstruction (state[0] + cumsum(action))\n(degree={degree}, chunk_size={chunk_size})', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        # 右图统计信息
        state_data = original_state
        # 计算重建误差
        error_original = np.mean(np.abs(state_from_original_action - original_state))
        error_fitted = np.mean(np.abs(state_from_fitted_action - original_state))
        error_state_fit = np.mean(np.abs(all_fitted_state - original_state))
        
        stats_text2 = (f'State Data:\n'
                       f'  Min: {state_data.min():.4f}\n'
                       f'  Max: {state_data.max():.4f}\n'
                       f'  Init: {state_init:.4f}\n\n'
                       f'MAE (Original Action): {error_original:.4f}\n'
                       f'MAE (Fitted Action): {error_fitted:.4f}\n'
                       f'MAE (Fitted State): {error_state_fit:.4f}')
        ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        fig.suptitle(f'{parquet_name}\nState[{state_col}] vs Action[{action_col}]', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f'compare_state{state_col:02d}_action{action_col:02d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")
    
    # 创建组合图（所有对比维度在一起）
    fig, axes = plt.subplots(num_compare_cols, 2, figsize=(20, 4 * num_compare_cols), sharex=True)
    if num_compare_cols == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(num_compare_cols):
        state_col = state_cols[i]
        action_col = action_cols[i]
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        
        # 左列：Action 对比
        ax1.plot(frames, actions[:, action_col], 'b-', linewidth=0.5, alpha=0.7, label='Original Action')
        
        all_fitted_action = np.zeros(num_frames)
        all_fitted_state = np.zeros(num_frames)
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, num_frames)
            
            chunk_frames = frames[start:end]
            chunk_action = actions[start:end, action_col]
            chunk_state = states[start:end, state_col]
            
            # Action 直接拟合
            fitted_action, _, _ = polyfit_chunk(chunk_frames, chunk_action, degree=degree)
            all_fitted_action[start:end] = fitted_action
            ax1.plot(chunk_frames, fitted_action, '-', linewidth=1, color='green', alpha=0.8)
            
            # State 拟合后差分
            fitted_state, _, _ = polyfit_chunk(chunk_frames, chunk_state, degree=degree)
            all_fitted_state[start:end] = fitted_state
            action_delta_from_state = np.diff(fitted_state)
            delta_frames = chunk_frames[1:]
            ax1.plot(delta_frames, action_delta_from_state, '-', linewidth=1, color='red', alpha=0.8)
            
            if chunk_idx < num_chunks - 1:
                ax1.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax2.axvline(x=end - 0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax1.set_ylabel(f'Action[{action_col}]', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=8)
        
        if i == 0:
            ax1.legend(['Original Action', 'Action Polyfit', 'Action from State Diff'], 
                      loc='upper right', fontsize=8)
        
        # 右列：State 重建对比
        original_state = states[:, state_col]
        state_init = original_state[0]
        
        ax2.plot(frames, original_state, 'b-', linewidth=0.5, alpha=0.7)
        
        state_from_original_action = state_init + np.cumsum(actions[:, action_col])
        state_from_original_action = np.insert(state_from_original_action[:-1], 0, state_init)
        ax2.plot(frames, state_from_original_action, 'c-', linewidth=1, alpha=0.8)
        
        state_from_fitted_action = state_init + np.cumsum(all_fitted_action)
        state_from_fitted_action = np.insert(state_from_fitted_action[:-1], 0, state_init)
        ax2.plot(frames, state_from_fitted_action, 'g-', linewidth=1, alpha=0.8)
        
        ax2.plot(frames, all_fitted_state, 'r-', linewidth=1, alpha=0.8)
        
        ax2.set_ylabel(f'State[{state_col}]', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=8)
        
        if i == 0:
            ax2.legend(['Original State', 'State from Orig Action', 'State from Fit Action', 'Fitted State'], 
                      loc='upper right', fontsize=8)
    
    axes[-1, 0].set_xlabel('Frame', fontsize=12)
    axes[-1, 1].set_xlabel('Frame', fontsize=12)
    axes[0, 0].set_title('Action Comparison', fontsize=12)
    axes[0, 1].set_title('State Reconstruction', fontsize=12)
    fig.suptitle(f'{parquet_name}\nState vs Action Comparison (degree={degree}, chunk_size={chunk_size})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    combined_path = os.path.join(output_dir, 'compare_all_columns.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存组合图: {combined_path}")


def main():
    parser = argparse.ArgumentParser(
        description='对比 State 拟合差分与 Action 直接拟合的可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python plot_state_action_compare.py --parquet_path episode.parquet --chunk_size 8
  
  # 指定多项式阶数
  python plot_state_action_compare.py --parquet_path episode.parquet --chunk_size 8 --degree 4
  
  # 指定特定的列进行对比
  python plot_state_action_compare.py --parquet_path episode.parquet --chunk_size 8 \\
      --state_cols 0 1 2 --action_cols 0 1 2
        """
    )
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径')
    parser.add_argument('--chunk_size', type=int, required=True,
                        help='每个chunk的大小（帧数）')
    parser.add_argument('--degree', type=int, default=3,
                        help='多项式拟合阶数，默认3阶')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录，默认为当前脚本所在目录下以脚本名命名的文件夹')
    parser.add_argument('--state_cols', type=int, nargs='+', default=None,
                        help='要处理的 state 列索引 (可选，默认全部)')
    parser.add_argument('--action_cols', type=int, nargs='+', default=None,
                        help='要处理的 action 列索引 (可选，默认全部)')
    
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
    print(f"State 拟合与 Action 对比可视化")
    print(f"=" * 60)
    print(f"输入文件: {parquet_path}")
    print(f"输出目录: {output_dir}")
    print(f"Chunk 大小: {args.chunk_size}")
    print(f"多项式阶数: {args.degree}")
    print()
    
    # 加载数据
    states, actions = load_parquet_data(parquet_path)
    
    # 绘制对比图
    plot_state_action_compare(
        states=states,
        actions=actions,
        chunk_size=args.chunk_size,
        degree=args.degree,
        output_dir=str(output_dir),
        parquet_name=parquet_name,
        state_cols=args.state_cols,
        action_cols=args.action_cols
    )
    
    print(f"\n完成！所有图片已保存到: {output_dir}")


if __name__ == '__main__':
    main()
