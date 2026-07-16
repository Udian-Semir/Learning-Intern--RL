"""
画 action 数据前3维的 3D 轨迹图

# !
python tools/plot_3d_trajectory.py \
    --parquet_path /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset_action_euler_unwrap/data/chunk-000/episode_000000.parquet \
    --deltas 3 3 3 \
    --dims 6 7 8 \
    --delta_cols 6 7 8 \
    --chunk_size 25 \
    --output_dir ./action_plots \
    --method chunk_calculus \
    --show 

使用示例:
python tools/plot_3d_trajectory.py \
    --parquet_path /home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_10/libero_lerobot_10_sys0_eagle_-1/data/chunk-000/episode_000000.parquet \
    --deltas 0.8 0.8 0.8 \
    --chunk_size 16 \
    --output_dir ./action_plots

# 只对action的第0和第2列进行delta离散化，第1列保持原始累积
python tools/plot_3d_trajectory.py \
    --parquet_path xxx.parquet \
    --deltas 0.8 0.8 0.8 \
    --dims 0 1 2 \
    --delta_cols 0 2 \
    --chunk_size 16

# 使用简单累积误差方法 (constrain_delta)
python tools/plot_3d_trajectory.py \
    --parquet_path xxx.parquet \
    --deltas 0.8 0.8 0.8 \
    --chunk_size 16 \
    --method constrain_delta

# 使用微积分方法 (chunk_calculus，默认)，可调整 beta 和 alpha 参数
python tools/plot_3d_trajectory.py \
    --parquet_path xxx.parquet \
    --deltas 0.8 0.8 0.8 \
    --chunk_size 16 \
    --method chunk_calculus \
    --beta 0.6 --alpha 0.4

# 弹出可交互的3D窗口（可旋转、缩放），关闭窗口后程序结束
python tools/plot_3d_trajectory.py \
    --parquet_path xxx.parquet \
    --deltas 0.8 0.8 0.8 \
    --chunk_size 16 \
    --show
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.discrete import (
    discrete_chunk_calculus, undiscrete_chunk_calculus,
    discrete_constrain_delta, undiscrete_constrain_delta,
    DISCRETE_METHODS
)


def setup_matplotlib(show: bool = False):
    """根据是否需要显示窗口来设置 matplotlib 后端"""
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    return plt


def load_parquet_actions(parquet_path: str) -> tuple:
    """从parquet文件加载action和state数据"""
    df = pd.read_parquet(parquet_path)
    
    if 'action' in df.columns:
        actions = np.array(df['action'].tolist())
    else:
        raise ValueError(f"未找到'action'列，可用列: {df.columns.tolist()}")
    
    states = None
    if 'observation.state' in df.columns:
        states = np.array(df['observation.state'].tolist())
    elif 'state' in df.columns:
        states = np.array(df['state'].tolist())
    
    return actions, states, df


def discrete_and_undiscrete_chunk(
    col_data: np.ndarray,
    delta: float,
    chunk_size: int,
    method: str = "chunk_calculus",
    beta: float = 0.6,
    alpha: float = 0.4,
) -> tuple:
    """
    对单列数据按 chunk 进行离散化再反离散化
    
    关键：每个 chunk 开始时，从原始轨迹的当前位置（而不是上个 chunk 离散化后的位置）开始，
    这样可以避免离散化误差的累积。
    
    Args:
        col_data: 1D array，单列 action 数据
        delta: 离散化步长
        chunk_size: chunk 大小
        method: 离散化方法 ("constrain_delta" 或 "chunk_calculus")
        beta: 趋势项权重 (仅 chunk_calculus 使用)
        alpha: 趋势平滑系数 (仅 chunk_calculus 使用)
    
    Returns:
        reconstructed_position: 反离散化后累加得到的位置（每个 chunk 从原始位置开始）
        undiscrete_data: 反离散化后的 delta action {-delta, 0, delta}
    """
    num_samples = len(col_data)
    
    # 计算原始轨迹位置（作为每个 chunk 的起点参考）
    original_position = np.cumsum(col_data)
    
    # 按 chunk 分块处理
    undiscrete_data = np.zeros(num_samples, dtype=float)
    reconstructed_position = np.zeros(num_samples, dtype=float)
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_data = col_data[start_idx:end_idx]
        
        # 离散化 - 根据方法选择
        if method == "chunk_calculus":
            discrete_data = discrete_chunk_calculus(
                chunk_data, delta, beta=beta, alpha=alpha
            )
            chunk_undiscrete = undiscrete_chunk_calculus(discrete_data, delta)
        else:  # constrain_delta
            discrete_data = discrete_constrain_delta(chunk_data, delta)
            chunk_undiscrete = undiscrete_constrain_delta(discrete_data, delta)
        
        undiscrete_data[start_idx:end_idx] = chunk_undiscrete
        
        # 每个 chunk 从原始轨迹的当前位置开始累加
        if start_idx == 0:
            chunk_start_pos = 0.0
        else:
            chunk_start_pos = original_position[start_idx - 1]
        
        chunk_cumsum = np.cumsum(chunk_undiscrete)
        reconstructed_position[start_idx:end_idx] = chunk_start_pos + chunk_cumsum
    
    return reconstructed_position, undiscrete_data


def compute_trajectories(
    actions: np.ndarray,
    deltas: list,
    chunk_size: int,
    dims: list,
    delta_cols: list,
    method: str,
    beta: float,
    alpha: float,
) -> tuple:
    """
    计算原始轨迹和离散化轨迹
    
    Returns:
        original_traj: 原始累积轨迹
        discrete_traj: 离散化后的轨迹
        delta_cols: 实际使用的 delta_cols
    """
    # 提取指定维度的 action
    action_xyz = actions[:, dims]
    
    # 原始轨迹（累积delta action）
    original_traj = np.cumsum(action_xyz, axis=0)
    
    # 如果 delta_cols 为 None，默认对所有维度进行离散化
    if delta_cols is None:
        delta_cols = dims.copy() if isinstance(dims, list) else list(dims)
    
    # 使用离散化方法（按 chunk 处理，每个 chunk 从原始位置开始）
    discrete_traj = np.zeros_like(action_xyz)
    for i, (dim, delta) in enumerate(zip(dims, deltas)):
        if dim in delta_cols:
            # 对指定列进行离散化
            reconstructed_pos, _ = discrete_and_undiscrete_chunk(
                action_xyz[:, i], delta, chunk_size, method, beta, alpha
            )
            discrete_traj[:, i] = reconstructed_pos
        else:
            # 不在 delta_cols 中的列，直接使用原始累积轨迹
            discrete_traj[:, i] = original_traj[:, i]
    
    return original_traj, discrete_traj, delta_cols


def plot_3d_trajectory(
    actions: np.ndarray,
    deltas: list,
    chunk_size: int,
    output_dir: str,
    parquet_name: str,
    dims: list = [0, 1, 2],
    delta_cols: list = None,
    method: str = "chunk_calculus",
    beta: float = 0.6,
    alpha: float = 0.4,
    show: bool = False,
):
    """
    画前3维的 3D 轨迹图（原始 vs 离散化）
    
    Args:
        actions: action数组 (num_samples, action_dim)
        deltas: 前3维的delta值
        chunk_size: chunk 大小
        output_dir: 输出目录
        parquet_name: 文件名
        dims: action的维度索引，默认 [0, 1, 2]
        delta_cols: 需要进行delta离散化的action列索引列表（绝对索引）
                   默认None表示对所有dims进行离散化
                   例如 dims=[0,1,2], delta_cols=[0,2] 表示对action的第0列和第2列进行离散化
        method: 离散化方法 ("constrain_delta" 或 "chunk_calculus")
        beta: 趋势项权重 (仅 chunk_calculus 使用)
        alpha: 趋势平滑系数 (仅 chunk_calculus 使用)
        show: 是否弹出可交互的3D窗口（关闭窗口后程序结束）
    """
    plt = setup_matplotlib(show)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 计算轨迹
    original_traj, discrete_traj, delta_cols = compute_trajectories(
        actions, deltas, chunk_size, dims, delta_cols, method, beta, alpha
    )
    
    # 计算轨迹末端误差
    end_error = np.linalg.norm(original_traj[-1] - discrete_traj[-1])
    
    # 创建 3D 图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 原始轨迹
    ax.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2], 
            'b-', label='Original (cumsum)', linewidth=1.5, alpha=0.8)
    
    # 离散化轨迹
    ax.plot(discrete_traj[:, 0], discrete_traj[:, 1], discrete_traj[:, 2], 
            'r--', label=f'Discrete (chunk={chunk_size})', linewidth=1.5, alpha=0.8)
    
    # 标记起点和终点
    ax.scatter(*original_traj[0], color='black', s=100, marker='o', label='Start')
    ax.scatter(*original_traj[-1], color='blue', s=100, marker='x', label='Orig End')
    ax.scatter(*discrete_traj[-1], color='red', s=100, marker='^', label='Disc End')
    
    ax.set_xlabel(f'Dim {dims[0]}')
    ax.set_ylabel(f'Dim {dims[1]}')
    ax.set_zlabel(f'Dim {dims[2]}')
    
    # 标题中显示哪些列被离散化和使用的方法
    ax.set_title(f'{parquet_name}\n3D Trajectory (method: {method}, deltas: {deltas}, delta_cols: {delta_cols}, chunk_size: {chunk_size})\nEnd Error: {end_error:.4f}')
    ax.legend()
    
    plt.tight_layout()
    
    # 保存 PNG 文件
    save_path = os.path.join(output_dir, f'{parquet_name}_3d_trajectory_{method}_chunk_{chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    print(f"3D 轨迹图已保存: {save_path}")
    print(f"  终点误差（欧氏距离）: {end_error:.4f}")
    
    # 如果需要显示交互式窗口
    if show:
        print("显示交互式 3D 窗口，关闭窗口后程序结束...")
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='画 action 前3维的 3D 轨迹图')
    parser.add_argument('--parquet_path', type=str, required=True,
                        help='parquet文件路径')
    parser.add_argument('--deltas', type=float, nargs=3, default=[0.1, 0.1, 0.1],
                        help='前3维的delta值，默认: 0.1 0.1 0.1')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='chunk大小，默认: 16')
    parser.add_argument('--dims', type=int, nargs=3, default=[0, 1, 2],
                        help='action的维度索引，默认: 0 1 2')
    parser.add_argument('--delta_cols', type=int, nargs='+', default=None,
                        help='需要进行delta离散化的action列索引（绝对索引）。'
                             '默认None表示对所有dims列进行离散化。'
                             '例如: --dims 0 1 2 --delta_cols 0 2 表示只对action的第0和第2列进行离散化')
    parser.add_argument('--output_dir', type=str, default='./action_plots',
                        help='输出目录')
    parser.add_argument('--show', action='store_true',
                        help='弹出可交互的3D窗口（可旋转、缩放），关闭窗口后程序结束')
    # 离散化方法选择
    parser.add_argument('--method', type=str, default='chunk_calculus',
                        choices=DISCRETE_METHODS,
                        help='离散化方法: constrain_delta (简单累积误差) 或 chunk_calculus (带趋势预测，默认)')
    # chunk_calculus 方法参数
    parser.add_argument('--beta', type=float, default=0.6,
                        help='趋势项权重 (仅 chunk_calculus 使用，默认 0.6)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='趋势平滑系数 (仅 chunk_calculus 使用，默认 0.4)')
    
    args = parser.parse_args()
    
    parquet_path = Path(args.parquet_path)
    if not parquet_path.exists():
        print(f"错误: 文件不存在 {parquet_path}")
        return
    
    print(f"加载: {parquet_path}")
    actions, _, _ = load_parquet_actions(str(parquet_path))
    
    print(f"Action shape: {actions.shape}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Dims: {args.dims}")
    if args.delta_cols is None:
        print(f"Delta cols: 所有dims列 {args.dims}")
    else:
        print(f"Delta cols: {args.delta_cols}")
    print(f"Method: {args.method}")
    print(f"Show: {args.show}")
    
    parquet_name = parquet_path.stem
    
    plot_3d_trajectory(
        actions, args.deltas, args.chunk_size, args.output_dir, parquet_name,
        dims=args.dims,
        delta_cols=args.delta_cols,
        method=args.method,
        beta=args.beta, alpha=args.alpha,
        show=args.show
    )


if __name__ == '__main__':
    main()
