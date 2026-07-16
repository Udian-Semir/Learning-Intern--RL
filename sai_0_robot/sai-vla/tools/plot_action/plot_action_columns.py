"""
绘制parquet文件中action数据的图表
每一列action单独一张图，包含两个子图：
- 左图：action值随帧变化
- 右图：action的累积和 cumsum(action)

使用示例:
python /home/dev/文档/huangwenlong/sai0-vla/tools/plot_action_columns.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset111_single_512_filter/data/chunk-000/episode_000000.parquet
"""

import os
import argparse
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


def plot_action_columns(actions: np.ndarray, output_dir: str, parquet_name: str):
    """
    绘制每一列action的图表，每张图包含两个子图
    
    Args:
        actions: action数据，shape为 (num_frames, action_dim)
        output_dir: 输出目录
        parquet_name: parquet文件名（用于标题）
    """
    num_frames, action_dim = actions.shape
    frames = np.arange(num_frames)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算累积和：action值本身的累积和
    cumulative_sum = np.cumsum(actions, axis=0)  # shape: (num_frames, action_dim)
    
    # 为每一列创建单独的图（包含两个子图）
    for col_idx in range(action_dim):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：action值
        ax1.plot(frames, actions[:, col_idx], 'b-', linewidth=1, marker='o', markersize=2)
        ax1.set_xlabel('Frame', fontsize=12)
        ax1.set_ylabel(f'Action[{col_idx}]', fontsize=12)
        ax1.set_title(f'Action Value', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 左图统计信息
        col_data = actions[:, col_idx]
        stats_text = f'Min: {col_data.min():.4f}\nMax: {col_data.max():.4f}\nMean: {col_data.mean():.4f}\nStd: {col_data.std():.4f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 右图：累积和
        ax2.plot(frames, cumulative_sum[:, col_idx], 'r-', linewidth=1, marker='o', markersize=2)
        ax2.set_xlabel('Frame', fontsize=12)
        ax2.set_ylabel('Cumulative Sum', fontsize=12)
        ax2.set_title(f'Cumulative Sum (cumsum(action))', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # 右图统计信息
        cum_sum_data = cumulative_sum[:, col_idx]
        cum_stats_text = f'Total: {cum_sum_data[-1]:.4f}\nMean: {np.mean(cum_sum_data):.4f}'
        ax2.text(0.02, 0.98, cum_stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        
        fig.suptitle(f'{parquet_name} - Action Column {col_idx}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        # 保存图片
        output_path = os.path.join(output_dir, f'action_col_{col_idx:02d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"已保存: {output_path}")
    
    # 创建一个组合图，所有列在一起（两列子图）
    fig, axes = plt.subplots(action_dim, 2, figsize=(16, 3 * action_dim), sharex=True)
    if action_dim == 1:
        axes = axes.reshape(1, 2)
    
    for col_idx in range(action_dim):
        # 左列：action值
        ax1 = axes[col_idx, 0]
        ax1.plot(frames, actions[:, col_idx], 'b-', linewidth=1)
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
    axes[0, 0].set_title('Action Value', fontsize=12)
    axes[0, 1].set_title('Cumulative Sum', fontsize=12)
    fig.suptitle(f'{parquet_name}\nAll Action Columns', fontsize=14, y=1.02)
    plt.tight_layout()
    
    combined_path = os.path.join(output_dir, 'action_all_columns.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存组合图: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制parquet文件中action数据的图表')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径')
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
    plot_action_columns(actions, str(output_dir), parquet_name)
    
    print(f"\n完成！所有图片已保存到: {output_dir}")


if __name__ == '__main__':
    main()
