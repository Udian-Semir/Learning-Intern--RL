"""
离散化 Cross Entropy Loss 基准计算工具

计算内容:
1. 标签分布统计: 统计 {-1, 0, 1} 三个类别的出现频率
2. Cross Entropy Loss 基准值:
   - 理论最优: 0 (完美分类)
   - 分布熵基线: H = -Σ p_i * log(p_i)
   - 随机猜测: log(3) ≈ 1.099
3. 完美拟合离散化的极限 (量化误差):
   - Perfect Fit MAE: 完美分类后的重建误差
   - Perfect Fit RMSE: 均方根误差
   - 理论上界: delta/2
4. 达到 delta/2 精度所需的 CE Loss 阈值:
   - 计算 CE Loss 和重建误差的关系
   - 给出达到 delta/2 精度目标所需的 loss 值
   - 帮助评估训练 loss 是否合理

输出解读:
  - CE Loss = 0: 完美分类，重建误差 = Perfect Fit MAE
  - CE Loss <= X: 重建误差可达到 δ/2 精度目标
  - CE Loss >= 分布熵: 模型只预测先验分布（欠拟合）

使用示例:
python tools/calc_discrete_ce_baseline.py \
    --data_dir /path/to/data \
    --columns 0 1 2 3 4 5 \
    --deltas 3 3 3 0.5 0.5 0.5 \
    --chunk_size 25 \
    --discrete_method chunk_calculus \
    --output_dir ./action_plots

python tools/calc_discrete_ce_baseline.py \
    --data_dir /home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset_action_euler_unwrap/data \
    --columns 0 1 2 3 4 5 6 7 8 9 10 11 \
    --deltas 3 3 3 0.5 0.5 0.5 3 3 3 0.5 0.5 0.5 \
    --chunk_size 25 \
    --discrete_method chunk_calculus \
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
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.discrete import (
    discrete_chunk_calculus,
    discrete_constrain_delta,
    DISCRETE_METHODS,
)


# ============================================================================
# 数据加载函数
# ============================================================================

def load_parquet_actions(parquet_path: str) -> np.ndarray:
    """从 parquet 文件加载 action 数据"""
    df = pd.read_parquet(parquet_path)
    
    if 'action' in df.columns:
        actions = np.array(df['action'].tolist())
    else:
        raise ValueError(f"未找到 'action' 列，可用列: {df.columns.tolist()}")
    
    return actions


def find_all_parquet_files(data_dir: str) -> List[Path]:
    """递归查找所有 parquet 文件"""
    data_path = Path(data_dir)
    parquet_files = sorted(data_path.rglob('*.parquet'))
    return parquet_files


# ============================================================================
# 离散化处理函数
# ============================================================================

def discretize_column(
    col_data: np.ndarray,
    delta: float,
    chunk_size: int,
    method: str = "chunk_calculus",
    beta: float = 0.6,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    对单列数据按 chunk 进行离散化
    
    Args:
        col_data: 1D array，单列 action 数据
        delta: 离散化步长
        chunk_size: chunk 大小
        method: 离散化方法
        beta, alpha: chunk_calculus 方法的参数
    
    Returns:
        discrete_data: 离散化后的数据，值为 {-1, 0, 1}
    """
    num_samples = len(col_data)
    discrete_data = np.zeros(num_samples, dtype=float)
    
    for start_idx in range(0, num_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, num_samples)
        chunk_data = col_data[start_idx:end_idx]
        
        if method == "chunk_calculus":
            chunk_discrete = discrete_chunk_calculus(
                chunk_data, delta, beta=beta, alpha=alpha
            )
        elif method == "constrain_delta":
            chunk_discrete = discrete_constrain_delta(chunk_data, delta)
        else:
            raise ValueError(f"未知的离散化方法: {method}. 可用方法: {DISCRETE_METHODS}")
        
        discrete_data[start_idx:end_idx] = chunk_discrete
    
    return discrete_data


# ============================================================================
# Cross Entropy 基准计算函数
# ============================================================================

def calc_entropy(p: np.ndarray) -> float:
    """
    计算分布的熵 H = -Σ p_i * log(p_i)
    
    Args:
        p: 概率分布数组，和为 1
    
    Returns:
        熵值 (使用自然对数)
    """
    p = p[p > 0]  # 避免 log(0)
    return -np.sum(p * np.log(p))


def calc_label_distribution(discrete_data: np.ndarray) -> Dict:
    """
    计算离散标签的分布
    
    Args:
        discrete_data: 离散化后的数据，值为 {-1, 0, 1}
    
    Returns:
        {
            'counts': {-1: count, 0: count, 1: count},
            'distribution': np.array([p_{-1}, p_0, p_1]),
            'total': 总样本数,
        }
    """
    total = len(discrete_data)
    counts = {
        -1: np.sum(discrete_data == -1),
        0: np.sum(discrete_data == 0),
        1: np.sum(discrete_data == 1),
    }
    
    distribution = np.array([counts[-1], counts[0], counts[1]]) / total
    
    return {
        'counts': counts,
        'distribution': distribution,
        'total': total,
    }


def calc_ce_baseline(distribution: np.ndarray) -> Dict:
    """
    计算 Cross Entropy Loss 基准值
    
    Args:
        distribution: 概率分布 [p_{-1}, p_0, p_1]
    
    Returns:
        {
            'entropy': 分布熵 (作为 CE loss 基线),
            'random_baseline': log(3) (随机猜测基线),
            'optimal': 0 (理论最优),
        }
    """
    entropy = calc_entropy(distribution)
    random_baseline = np.log(3)  # 约 1.099
    
    return {
        'entropy': entropy,
        'random_baseline': random_baseline,
        'optimal': 0.0,
    }


# ============================================================================
# 完美拟合极限计算函数 (量化误差)
# ============================================================================

def calc_perfect_fit_error(
    original: np.ndarray,
    discrete: np.ndarray,
    delta: float,
) -> Dict:
    """
    计算完美拟合离散化时的量化误差
    
    即使 CrossEntropyLoss = 0 (完美分类)，重建值与原始值之间仍有误差。
    
    Args:
        original: 原始连续值
        discrete: 离散化后的值 {-1, 0, 1}
        delta: 离散化步长
    
    Returns:
        {
            'mae': 平均绝对误差,
            'rmse': 均方根误差,
            'max_error': 最大误差,
            'theoretical_bound': delta/2 (单步理论上界),
        }
    """
    # 重建值 = 离散值 × delta
    reconstructed = discrete * delta
    
    # 量化误差
    errors = np.abs(original - reconstructed)
    
    return {
        'mae': np.mean(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'max_error': np.max(errors),
        'theoretical_bound': delta / 2,
    }


# ============================================================================
# 目标精度下的 Loss 阈值计算
# ============================================================================

def calc_target_precision_loss(
    perfect_fit_mae: float,
    target_error: float,
    delta: float,
    distribution: np.ndarray,
) -> Dict:
    """
    计算达到目标精度所需的 CE Loss 阈值
    
    模型假设:
    - 分类准确率为 acc 时，总误差 = Perfect_Fit_MAE + (1-acc) * avg_misclass_error
    - avg_misclass_error 是误分类导致的平均额外误差
    - CE Loss ≈ -log(acc)（简化假设）
    
    Args:
        perfect_fit_mae: 完美分类时的 MAE
        target_error: 目标误差 (如 delta/2)
        delta: 离散化步长
        distribution: 标签分布 [p_{-1}, p_0, p_1]
    
    Returns:
        {
            'target_error': 目标误差值,
            'required_accuracy': 所需的分类准确率,
            'estimated_ce_loss': 估算的 CE Loss 阈值,
            'is_achievable': 是否可达成,
            'reason': 说明,
        }
    """
    # 计算平均误分类额外误差
    # 误分类场景:
    #   真实 -1 -> 预测 0: 额外误差 = delta
    #   真实 -1 -> 预测 1: 额外误差 = 2*delta
    #   真实 0 -> 预测 -1: 额外误差 = delta
    #   真实 0 -> 预测 1: 额外误差 = delta
    #   真实 1 -> 预测 0: 额外误差 = delta
    #   真实 1 -> 预测 -1: 额外误差 = 2*delta
    # 假设误分类均匀分布到其他两个类别，平均额外误差 = 1.5 * delta
    avg_misclass_error = 1.5 * delta
    
    # 如果目标误差小于完美拟合误差，则不可能达到
    if target_error < perfect_fit_mae:
        return {
            'target_error': target_error,
            'required_accuracy': None,
            'estimated_ce_loss': 0.0,
            'is_achievable': False,
            'reason': f'目标误差 {target_error:.4f} 小于完美拟合误差 {perfect_fit_mae:.4f}，需要 CE Loss = 0',
        }
    
    # 计算所需的分类准确率
    # total_error = perfect_fit_mae + (1-acc) * avg_misclass_error
    # target_error = perfect_fit_mae + (1-acc) * avg_misclass_error
    # (1-acc) = (target_error - perfect_fit_mae) / avg_misclass_error
    error_margin = target_error - perfect_fit_mae
    error_rate = error_margin / avg_misclass_error
    required_accuracy = 1.0 - error_rate
    
    # 限制准确率范围
    required_accuracy = max(0.0, min(1.0, required_accuracy))
    
    # 从准确率估算 CE Loss
    # CE Loss ≈ -log(p_correct)
    # 假设 p_correct ≈ accuracy（简化）
    if required_accuracy >= 1.0:
        estimated_ce_loss = 0.0
    elif required_accuracy <= 0.0:
        estimated_ce_loss = float('inf')
    else:
        # 更精确的估算：考虑模型输出的概率分布
        # 假设模型对正确类别输出概率 p，对其他类别输出 (1-p)/2
        # 准确率 ≈ p（当 p 较高时）
        # CE Loss = -log(p)
        estimated_ce_loss = -np.log(required_accuracy)
    
    return {
        'target_error': target_error,
        'required_accuracy': required_accuracy,
        'estimated_ce_loss': estimated_ce_loss,
        'is_achievable': True,
        'reason': f'需要分类准确率 >= {required_accuracy*100:.2f}%',
    }


def calc_loss_error_curve(
    perfect_fit_mae: float,
    delta: float,
    distribution: np.ndarray,
    num_points: int = 20,
) -> Dict:
    """
    计算 CE Loss 和重建误差的关系曲线
    
    Args:
        perfect_fit_mae: 完美分类时的 MAE
        delta: 离散化步长
        distribution: 标签分布
        num_points: 采样点数
    
    Returns:
        {
            'accuracies': 准确率数组,
            'ce_losses': 对应的 CE Loss 数组,
            'errors': 对应的重建误差数组,
        }
    """
    avg_misclass_error = 1.5 * delta
    
    accuracies = np.linspace(0.5, 1.0, num_points)
    ce_losses = []
    errors = []
    
    for acc in accuracies:
        # CE Loss
        if acc >= 1.0:
            ce = 0.0
        else:
            ce = -np.log(acc)
        ce_losses.append(ce)
        
        # 重建误差
        error = perfect_fit_mae + (1 - acc) * avg_misclass_error
        errors.append(error)
    
    return {
        'accuracies': accuracies,
        'ce_losses': np.array(ce_losses),
        'errors': np.array(errors),
    }


# ============================================================================
# 批量处理函数
# ============================================================================

def process_all_files(
    parquet_files: List[Path],
    columns: List[int],
    deltas: List[float],
    chunk_size: int,
    method: str = "chunk_calculus",
    beta: float = 0.6,
    alpha: float = 0.4,
    max_files: int = None,
) -> Dict:
    """
    处理所有 parquet 文件，统计标签分布和量化误差
    
    Args:
        parquet_files: parquet 文件路径列表
        columns: 要处理的列索引
        deltas: 对应列的 delta 值
        chunk_size: chunk 大小
        method: 离散化方法
        beta, alpha: chunk_calculus 参数
        max_files: 最多处理的文件数
    
    Returns:
        {
            'column_stats': {col_idx: {...}},  # 每列的统计结果
            'overall_stats': {...},  # 整体统计
        }
    """
    if max_files is not None:
        parquet_files = parquet_files[:max_files]
    
    # 初始化每列的累计统计
    column_accumulators = {col_idx: {
        'label_counts': {-1: 0, 0: 0, 1: 0},
        'total_samples': 0,
        'sum_abs_error': 0.0,
        'sum_sq_error': 0.0,
        'max_error': 0.0,
        'delta': deltas[i],
    } for i, col_idx in enumerate(columns)}
    
    print(f"\n处理 {len(parquet_files)} 个文件...")
    
    for parquet_path in tqdm(parquet_files, desc="Processing"):
        try:
            actions = load_parquet_actions(str(parquet_path))
            num_samples = actions.shape[0]
            
            for i, col_idx in enumerate(columns):
                if col_idx >= actions.shape[1]:
                    continue
                
                col_data = actions[:, col_idx]
                delta = deltas[i]
                
                # 离散化
                discrete_data = discretize_column(
                    col_data, delta, chunk_size, method, beta, alpha
                )
                
                # 更新标签计数
                acc = column_accumulators[col_idx]
                acc['label_counts'][-1] += np.sum(discrete_data == -1)
                acc['label_counts'][0] += np.sum(discrete_data == 0)
                acc['label_counts'][1] += np.sum(discrete_data == 1)
                acc['total_samples'] += num_samples
                
                # 计算量化误差
                reconstructed = discrete_data * delta
                errors = np.abs(col_data - reconstructed)
                
                acc['sum_abs_error'] += np.sum(errors)
                acc['sum_sq_error'] += np.sum(errors ** 2)
                acc['max_error'] = max(acc['max_error'], np.max(errors))
                
        except Exception as e:
            print(f"\n警告: 处理 {parquet_path} 时出错: {e}")
            continue
    
    # 计算每列的最终统计
    column_stats = {}
    for col_idx in columns:
        acc = column_accumulators[col_idx]
        total = acc['total_samples']
        
        if total == 0:
            continue
        
        # 标签分布
        counts = acc['label_counts']
        distribution = np.array([counts[-1], counts[0], counts[1]]) / total
        
        # CE 基准
        ce_baseline = calc_ce_baseline(distribution)
        
        # 量化误差
        mae = acc['sum_abs_error'] / total
        rmse = np.sqrt(acc['sum_sq_error'] / total)
        
        column_stats[col_idx] = {
            'delta': acc['delta'],
            'total_samples': total,
            'label_counts': counts,
            'distribution': distribution,
            'ce_baseline': ce_baseline,
            'perfect_fit': {
                'mae': mae,
                'rmse': rmse,
                'max_error': acc['max_error'],
                'theoretical_bound': acc['delta'] / 2,
            },
        }
    
    # 计算整体统计
    total_samples_all = sum(column_stats[col]['total_samples'] for col in column_stats)
    
    # 整体标签分布 (加权平均)
    overall_distribution = np.zeros(3)
    for col_idx, stats in column_stats.items():
        weight = stats['total_samples'] / total_samples_all
        overall_distribution += stats['distribution'] * weight
    
    # 整体 CE 基准
    overall_ce = calc_ce_baseline(overall_distribution)
    
    # 整体量化误差 (算术平均)
    overall_mae = np.mean([stats['perfect_fit']['mae'] for stats in column_stats.values()])
    overall_rmse = np.mean([stats['perfect_fit']['rmse'] for stats in column_stats.values()])
    overall_max = max(stats['perfect_fit']['max_error'] for stats in column_stats.values())
    
    # 计算每列达到 delta/2 精度所需的 loss
    for col_idx in column_stats:
        stats = column_stats[col_idx]
        target_error = stats['delta'] / 2
        target_precision = calc_target_precision_loss(
            perfect_fit_mae=stats['perfect_fit']['mae'],
            target_error=target_error,
            delta=stats['delta'],
            distribution=stats['distribution'],
        )
        stats['target_precision'] = target_precision
        
        # 计算 Loss-Error 曲线数据
        stats['loss_error_curve'] = calc_loss_error_curve(
            perfect_fit_mae=stats['perfect_fit']['mae'],
            delta=stats['delta'],
            distribution=stats['distribution'],
        )
    
    # 整体平均 delta
    avg_delta = np.mean([column_stats[col]['delta'] for col in column_stats])
    
    # 整体达到 avg_delta/2 精度所需的 loss
    overall_target_precision = calc_target_precision_loss(
        perfect_fit_mae=overall_mae,
        target_error=avg_delta / 2,
        delta=avg_delta,
        distribution=overall_distribution,
    )
    
    overall_stats = {
        'total_files': len(parquet_files),
        'total_columns': len(columns),
        'distribution': overall_distribution,
        'ce_baseline': overall_ce,
        'perfect_fit': {
            'mae': overall_mae,
            'rmse': overall_rmse,
            'max_error': overall_max,
        },
        'avg_delta': avg_delta,
        'target_precision': overall_target_precision,
    }
    
    return {
        'column_stats': column_stats,
        'overall_stats': overall_stats,
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_results(
    results: Dict,
    columns: List[int],
    output_dir: str,
    chunk_size: int,
):
    """
    可视化统计结果
    
    Args:
        results: process_all_files 的返回结果
        columns: 列索引列表
        output_dir: 输出目录
        chunk_size: chunk 大小 (用于标题)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    column_stats = results['column_stats']
    overall_stats = results['overall_stats']
    
    num_cols = len(columns)
    
    # ========== 图1: 标签分布柱状图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 每列的标签分布
    ax1 = axes[0]
    x = np.arange(num_cols)
    width = 0.25
    
    p_neg1 = [column_stats[col]['distribution'][0] for col in columns]
    p_0 = [column_stats[col]['distribution'][1] for col in columns]
    p_1 = [column_stats[col]['distribution'][2] for col in columns]
    
    bars1 = ax1.bar(x - width, p_neg1, width, label='p(-1)', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x, p_0, width, label='p(0)', color='#3498db', alpha=0.8)
    bars3 = ax1.bar(x + width, p_1, width, label='p(1)', color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Probability')
    ax1.set_title('Label Distribution per Column')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(col) for col in columns])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # 右图: 整体分布 + CE 基准值
    ax2 = axes[1]
    labels = ['p(-1)', 'p(0)', 'p(1)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars = ax2.bar(labels, overall_stats['distribution'], color=colors, alpha=0.8, edgecolor='black')
    
    # 在柱状图上标注数值
    for bar, val in zip(bars, overall_stats['distribution']):
        height = bar.get_height()
        ax2.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Overall Label Distribution\n'
                  f'Entropy H = {overall_stats["ce_baseline"]["entropy"]:.4f}, '
                  f'Random = {overall_stats["ce_baseline"]["random_baseline"]:.4f}')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'ce_baseline_distribution_chunk_{chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n标签分布图已保存: {save_path}")
    
    # ========== 图2: 量化误差对比 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 每列的 MAE 和理论上界
    ax1 = axes[0]
    x = np.arange(num_cols)
    width = 0.35
    
    maes = [column_stats[col]['perfect_fit']['mae'] for col in columns]
    bounds = [column_stats[col]['perfect_fit']['theoretical_bound'] for col in columns]
    
    bars1 = ax1.bar(x - width/2, maes, width, label='Perfect Fit MAE', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, bounds, width, label='Theoretical Bound (δ/2)', color='coral', alpha=0.8)
    
    # 标注数值
    for bar, val in zip(bars1, maes):
        height = bar.get_height()
        ax1.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Error')
    ax1.set_title('Perfect Fit Quantization Error per Column')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(col) for col in columns])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 右图: 整体误差汇总
    ax2 = axes[1]
    metrics = ['MAE', 'RMSE', 'Max Error']
    values = [
        overall_stats['perfect_fit']['mae'],
        overall_stats['perfect_fit']['rmse'],
        overall_stats['perfect_fit']['max_error'],
    ]
    colors = ['steelblue', '#9b59b6', '#e74c3c']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11)
    
    ax2.set_ylabel('Error Value')
    ax2.set_title('Overall Perfect Fit Error\n(Error when CE Loss = 0)')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'ce_baseline_quant_error_chunk_{chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"量化误差图已保存: {save_path}")
    
    # ========== 图3: CE Loss vs 重建误差曲线 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 每列的 Loss-Error 曲线
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(columns)))
    
    for i, col_idx in enumerate(columns):
        if col_idx not in column_stats:
            continue
        
        curve = column_stats[col_idx]['loss_error_curve']
        ax1.plot(curve['ce_losses'], curve['errors'], 
                label=f'Col {col_idx} (δ={column_stats[col_idx]["delta"]:.1f})',
                color=colors[i], linewidth=2, alpha=0.8)
        
        # 标记 delta/2 目标点
        target = column_stats[col_idx]['delta'] / 2
        tp = column_stats[col_idx]['target_precision']
        if tp['is_achievable']:
            ax1.scatter([tp['estimated_ce_loss']], [target], 
                       color=colors[i], s=100, marker='*', zorder=5)
    
    ax1.set_xlabel('CE Loss')
    ax1.set_ylabel('Reconstruction Error (MAE)')
    ax1.set_title('CE Loss vs Reconstruction Error\n(Stars mark δ/2 target)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(0.5, overall_stats['ce_baseline']['entropy'] * 1.1))
    
    # 右图: 整体 Loss-Error 关系 + 关键阈值
    ax2 = axes[1]
    
    # 计算整体曲线
    avg_delta = overall_stats['avg_delta']
    overall_mae = overall_stats['perfect_fit']['mae']
    avg_misclass_error = 1.5 * avg_delta
    
    accuracies = np.linspace(0.5, 1.0, 100)
    ce_losses = [-np.log(acc) if acc < 1.0 else 0.0 for acc in accuracies]
    errors = [overall_mae + (1 - acc) * avg_misclass_error for acc in accuracies]
    
    ax2.plot(ce_losses, errors, 'b-', linewidth=2.5, label='Error vs CE Loss')
    
    # 标记关键点
    # 1. 完美分类点 (CE Loss = 0)
    ax2.scatter([0], [overall_mae], color='green', s=150, marker='o', zorder=5,
               label=f'Perfect Fit (CE=0, MAE={overall_mae:.3f})')
    
    # 2. delta/2 目标点
    tp = overall_stats['target_precision']
    if tp['is_achievable']:
        ax2.scatter([tp['estimated_ce_loss']], [avg_delta/2], 
                   color='red', s=150, marker='*', zorder=5,
                   label=f'Target δ/2 (CE={tp["estimated_ce_loss"]:.3f}, MAE={avg_delta/2:.3f})')
        ax2.axhline(y=avg_delta/2, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=tp['estimated_ce_loss'], color='red', linestyle='--', alpha=0.5)
    
    # 3. 分布熵基线
    entropy = overall_stats['ce_baseline']['entropy']
    error_at_entropy = overall_mae + 0.5 * avg_misclass_error  # 约 50% 准确率
    ax2.axvline(x=entropy, color='orange', linestyle=':', alpha=0.7,
               label=f'Entropy Baseline (CE={entropy:.3f})')
    
    ax2.set_xlabel('CE Loss')
    ax2.set_ylabel('Reconstruction Error (MAE)')
    ax2.set_title(f'Overall CE Loss vs Reconstruction Error\n'
                  f'(avg δ = {avg_delta:.2f})')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(0.8, entropy * 1.2))
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'ce_baseline_loss_error_curve_chunk_{chunk_size}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss-Error 曲线图已保存: {save_path}")


def print_results(results: Dict, columns: List[int], deltas: List[float]):
    """打印详细统计结果"""
    column_stats = results['column_stats']
    overall_stats = results['overall_stats']
    
    print("\n" + "=" * 70)
    print("Cross Entropy Loss 基准分析")
    print("=" * 70)
    
    # ===== 标签分布 =====
    print("\n【标签分布统计】")
    print("-" * 70)
    
    for i, col_idx in enumerate(columns):
        if col_idx not in column_stats:
            continue
        
        stats = column_stats[col_idx]
        dist = stats['distribution']
        entropy = stats['ce_baseline']['entropy']
        
        print(f"  列 {col_idx:2d} (δ={stats['delta']:>5.2f}): "
              f"p(-1)={dist[0]:.3f}, p(0)={dist[1]:.3f}, p(1)={dist[2]:.3f}  "
              f"-> 熵 H={entropy:.4f}")
    
    # ===== CE 基准值 =====
    print("\n【Cross Entropy Loss 基准值】")
    print("-" * 70)
    print(f"  理论最优 (完美分类):     CE Loss = 0")
    print(f"  分布熵基线 (先验预测):   CE Loss = {overall_stats['ce_baseline']['entropy']:.4f}")
    print(f"  随机猜测基线 (均匀分布): CE Loss = {overall_stats['ce_baseline']['random_baseline']:.4f}")
    
    # ===== 完美拟合极限 =====
    print("\n【完美拟合离散化的极限 (量化误差)】")
    print("-" * 70)
    print("  当 CE Loss = 0 时，重建值与原始值之间的误差:")
    print()
    
    for i, col_idx in enumerate(columns):
        if col_idx not in column_stats:
            continue
        
        stats = column_stats[col_idx]
        pf = stats['perfect_fit']
        
        print(f"  列 {col_idx:2d} (δ={stats['delta']:>5.2f}): "
              f"MAE={pf['mae']:.4f}, RMSE={pf['rmse']:.4f}, "
              f"Max={pf['max_error']:.4f}, 理论上界(δ/2)={pf['theoretical_bound']:.4f}")
    
    pf_overall = overall_stats['perfect_fit']
    print()
    print(f"  整体 Perfect Fit MAE:  {pf_overall['mae']:.4f}")
    print(f"  整体 Perfect Fit RMSE: {pf_overall['rmse']:.4f}")
    print(f"  整体 Max Error:        {pf_overall['max_error']:.4f}")
    
    # ===== 达到 delta/2 精度所需的 Loss =====
    print("\n【达到 δ/2 精度所需的 CE Loss】")
    print("-" * 70)
    print("  目标: 使重建误差 MAE 达到 delta/2 (理论上界)")
    print()
    
    for i, col_idx in enumerate(columns):
        if col_idx not in column_stats:
            continue
        
        stats = column_stats[col_idx]
        tp = stats['target_precision']
        target = stats['delta'] / 2
        
        if tp['is_achievable']:
            print(f"  列 {col_idx:2d} (δ={stats['delta']:>5.2f}): "
                  f"目标MAE={target:.4f}, 需要准确率>={tp['required_accuracy']*100:.1f}%, "
                  f"CE Loss <= {tp['estimated_ce_loss']:.4f}")
        else:
            print(f"  列 {col_idx:2d} (δ={stats['delta']:>5.2f}): "
                  f"目标MAE={target:.4f}, {tp['reason']}")
    
    # 整体
    tp_overall = overall_stats['target_precision']
    avg_delta = overall_stats['avg_delta']
    print()
    if tp_overall['is_achievable']:
        print(f"  整体 (平均δ={avg_delta:.2f}): "
              f"目标MAE={avg_delta/2:.4f}, 需要准确率>={tp_overall['required_accuracy']*100:.1f}%, "
              f"CE Loss <= {tp_overall['estimated_ce_loss']:.4f}")
    else:
        print(f"  整体: {tp_overall['reason']}")
    
    # ===== 结论 =====
    print("\n【结论】")
    print("-" * 70)
    print(f"  • CE Loss = 0: 完美分类，重建误差 = Perfect Fit MAE ≈ {pf_overall['mae']:.4f}")
    print(f"  • CE Loss <= {tp_overall['estimated_ce_loss']:.4f}: 重建误差可达到 δ/2 = {avg_delta/2:.4f}")
    print(f"  • 分布熵基线 = {overall_stats['ce_baseline']['entropy']:.4f}: 模型只预测先验分布")
    print(f"  • 随机猜测 = {overall_stats['ce_baseline']['random_baseline']:.4f}: 最差情况")
    print()
    print("  Loss 区间解读:")
    print(f"    [0, {tp_overall['estimated_ce_loss']:.4f}]        -> 优秀 (误差 <= δ/2)")
    print(f"    [{tp_overall['estimated_ce_loss']:.4f}, {overall_stats['ce_baseline']['entropy']:.4f}]  -> 良好 (超过 δ/2 但在学习)")
    print(f"    [{overall_stats['ce_baseline']['entropy']:.4f}, {overall_stats['ce_baseline']['random_baseline']:.4f}]  -> 较差 (接近先验预测)")
    
    print("\n" + "=" * 70)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='离散化 Cross Entropy Loss 基准计算工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tools/calc_discrete_ce_baseline.py \\
      --data_dir /path/to/data \\
      --columns 0 1 2 3 4 5 \\
      --deltas 3 3 3 0.5 0.5 0.5 \\
      --chunk_size 25
        """
    )
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据文件夹路径')
    parser.add_argument('--columns', type=int, nargs='+', default=None,
                        help='要计算的列索引，默认所有列')
    parser.add_argument('--deltas', type=float, nargs='+', default=None,
                        help='每列的 delta 值，默认自动估算')
    
    # 离散化参数
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='chunk 大小，默认 16')
    parser.add_argument('--discrete_method', type=str, default='chunk_calculus',
                        choices=['constrain_delta', 'chunk_calculus'],
                        help='离散化方法，默认 chunk_calculus')
    parser.add_argument('--beta', type=float, default=0.6,
                        help='趋势项权重 (默认 0.6)')
    parser.add_argument('--alpha', type=float, default=0.4,
                        help='趋势平滑系数 (默认 0.4)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./action_plots',
                        help='输出目录')
    parser.add_argument('--max_files', type=int, default=None,
                        help='最多处理的文件数，默认全部')
    parser.add_argument('--no_plot', action='store_true',
                        help='不生成图片')
    
    args = parser.parse_args()
    
    # ========== 查找数据文件 ==========
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 目录不存在 {data_dir}")
        return
    
    parquet_files = find_all_parquet_files(str(data_dir))
    print(f"找到 {len(parquet_files)} 个 parquet 文件")
    
    if len(parquet_files) == 0:
        print("没有找到 parquet 文件")
        return
    
    # ========== 从第一个文件获取数据形状 ==========
    first_actions = load_parquet_actions(str(parquet_files[0]))
    n_dims = first_actions.shape[1]
    print(f"Action 维度: {n_dims}")
    
    # ========== 设置 columns ==========
    if args.columns is None:
        columns = list(range(n_dims))
    else:
        columns = args.columns
    
    # ========== 设置 deltas ==========
    if args.deltas is None:
        # 自动估算
        deltas = []
        print("\n自动估算 delta 值:")
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
            print(f"  列 {col_idx}: delta = {deltas[-1]}")
    else:
        deltas = args.deltas
        if len(deltas) != len(columns):
            if len(deltas) < len(columns):
                deltas = deltas + [deltas[-1]] * (len(columns) - len(deltas))
            else:
                deltas = deltas[:len(columns)]
    
    print(f"\n使用的列: {columns}")
    print(f"使用的 deltas: {deltas}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"离散化方法: {args.discrete_method}")
    
    # ========== 处理所有文件 ==========
    results = process_all_files(
        parquet_files=parquet_files,
        columns=columns,
        deltas=deltas,
        chunk_size=args.chunk_size,
        method=args.discrete_method,
        beta=args.beta,
        alpha=args.alpha,
        max_files=args.max_files,
    )
    
    # ========== 打印结果 ==========
    print_results(results, columns, deltas)
    
    # ========== 可视化 ==========
    if not args.no_plot:
        plot_results(results, columns, args.output_dir, args.chunk_size)


if __name__ == '__main__':
    main()
