"""
可视化parquet文件中的action数据

使用示例:
python tools/chek_delta_value.py \
    --parquet_path /data/HuangWenlong/datasets/.../chunk-000/episode_000000.parquet \
    --output_dir ./action_value_plots
"""
# python tools/chek_delta_value.py \
#     --parquet_path /data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_spatial/libero_lerobot_spatial_sys1_qwen_2b_14/data/chunk-000/episode_000000.parquet \
#     --output_dir ./action_value_plots

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示图像
import matplotlib.pyplot as plt
from pathlib import Path


def load_parquet_actions(parquet_path: str) -> np.ndarray:
    """
    从parquet文件加载action数据
    
    Args:
        parquet_path: parquet文件路径
    
    Returns:
        action数据，shape为 (num_samples, action_dim)
    """
    df = pd.read_parquet(parquet_path)
    
    # 查找action列
    if 'action' in df.columns:
        actions = df['action'].tolist()
        actions = np.array(actions)
    else:
        raise ValueError(f"未找到'action'列，可用列: {df.columns.tolist()}")
    
    return actions, df


def plot_action_values(
    actions: np.ndarray,
    output_dir: str,
    parquet_name: str
):
    """
    画出每个action特征的值分布图
    
    Args:
        actions: action数组 (num_samples, action_dim)
        output_dir: 输出目录
        parquet_name: parquet文件名（用于标题）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    action_dim = actions.shape[1]
    num_samples = actions.shape[0]
    x = np.arange(num_samples)
    
    # 为每个特征创建单独的图
    for col_idx in range(action_dim):
        fig, ax = plt.subplots(figsize=(14, 6))
        
        col_data = actions[:, col_idx]
        mean_val = np.mean(col_data)
        std_val = np.std(col_data)
        min_val = np.min(col_data)
        max_val = np.max(col_data)
        
        # 画数据点/线
        ax.plot(x, col_data, 'b-', label='Action Value', alpha=0.7, linewidth=1)
        
        # 画平均值线
        ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.6f}')
        
        # 画标准差范围（可选）
        ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'+1 Std: {mean_val + std_val:.6f}')
        ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, 
                   label=f'-1 Std: {mean_val - std_val:.6f}')
        
        ax.set_xlabel('Sample Index (Row)')
        ax.set_ylabel('Action Value')
        ax.set_title(f'{parquet_name} - Feature {col_idx}\n'
                     f'Min: {min_val:.6f}, Max: {max_val:.6f}, Std: {std_val:.6f}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{parquet_name}_feature_{col_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Feature {col_idx}] mean={mean_val:.6f}, std={std_val:.6f}, "
              f"min={min_val:.6f}, max={max_val:.6f} -> {save_path}")
    
    # 创建汇总图
    fig, axes = plt.subplots(action_dim, 1, figsize=(14, 3 * action_dim))
    if action_dim == 1:
        axes = [axes]
    
    for col_idx in range(action_dim):
        col_data = actions[:, col_idx]
        mean_val = np.mean(col_data)
        
        axes[col_idx].plot(x, col_data, 'b-', alpha=0.7, linewidth=1)
        axes[col_idx].axhline(y=mean_val, color='r', linestyle='--', linewidth=2)
        axes[col_idx].set_ylabel(f'Feature {col_idx}')
        axes[col_idx].grid(True, alpha=0.3)
        
        # 在右侧显示统计信息
        stats_text = f'μ={mean_val:.4f}'
        axes[col_idx].text(0.98, 0.95, stats_text, transform=axes[col_idx].transAxes,
                          fontsize=9, verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Sample Index (Row)')
    plt.suptitle(f'{parquet_name} - All Features', fontsize=12, y=1.02)
    plt.tight_layout()
    
    summary_path = os.path.join(output_dir, f'{parquet_name}_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n汇总图已保存: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化parquet文件中的action数据')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径或所在目录')
    parser.add_argument('--output_dir', type=str, default='./action_value_plots',
                        help='输出图片的目录')
    
    args = parser.parse_args()
    
    # 判断输入是文件还是目录
    input_path = Path(args.parquet_path)
    if not input_path.exists():
        print(f"错误: 路径不存在 {input_path}")
        return
    
    if input_path.is_file():
        parquet_path = input_path
    else:
        # 传入目录，查找parquet文件
        parquet_files = sorted(input_path.glob('episode_*.parquet'))
        if not parquet_files:
            parquet_files = sorted(input_path.glob('*.parquet'))
        
        if not parquet_files:
            print(f"错误: 在 {input_path} 中未找到parquet文件")
            return
        
        parquet_path = parquet_files[0]
    
    print(f"加载parquet文件: {parquet_path}")
    
    # 加载数据
    try:
        actions, df = load_parquet_actions(str(parquet_path))
    except Exception as e:
        print(f"加载失败: {e}")
        df = pd.read_parquet(str(parquet_path))
        print(f"列名: {df.columns.tolist()}")
        print(f"数据形状: {df.shape}")
        print(df.head())
        return
    
    print(f"Action数据形状: {actions.shape}")
    print(f"Action维度: {actions.shape[1]}")
    print(f"样本数量: {actions.shape[0]}")
    
    # 显示每列的统计信息
    print("\n=== Action统计信息 ===")
    for col_idx in range(actions.shape[1]):
        col_data = actions[:, col_idx]
        print(f"  Feature {col_idx}: min={col_data.min():.6f}, max={col_data.max():.6f}, "
              f"mean={col_data.mean():.6f}, std={col_data.std():.6f}")
    
    parquet_name = parquet_path.stem
    
    # 画图
    plot_action_values(actions, args.output_dir, parquet_name)
    
    print(f"\n所有图片已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()

