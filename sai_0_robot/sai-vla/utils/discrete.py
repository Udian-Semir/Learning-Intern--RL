"""
离散化与反离散化工具函数

离散化流程说明:
================
1. 离散化 (discrete_constrain_delta):
   - 输入: 连续 action chunk 数据
   - 输出: 离散化后的 action chunk，值为 {-1, 0, 1}
   - 用途: 将连续动作转换为三元离散动作（后退、不动、前进）

2. 训练时 (ParaCAT):
   - Ground Truth: 将 {-1, 0, 1} 映射到类别索引 {0, 1, 2}
     -1 -> 类别 0
      0 -> 类别 1
      1 -> 类别 2
   - 输出: ParaCAT 输出 logits (shape: ..., 3)，不经过 softmax
   - Loss: CrossEntropyLoss (内部自动处理 softmax)

3. 推理时:
   - 预测: argmax(logits) 得到类别索引 {0, 1, 2}
   - 映射: 类别索引映射回 {-1, 0, 1}
   - 反离散化: 乘以 delta 还原为连续值 {-delta, 0, delta}

关键公式:
=========
- 离散化: 连续值 -> {-1, 0, 1}  (通过累积误差算法)
- 反离散化: {-1, 0, 1} * delta -> {-delta, 0, delta}
"""

import numpy as np
from typing import List, Optional, Union
import torch


def discrete_constrain_delta(X: np.ndarray, delta: float) -> np.ndarray:
    """
    离散化约束函数 - 将连续 action 转换为三元离散值 {-1, 0, 1}
    
    算法说明:
    - 使用累积误差算法，当累积误差超过 delta/2 时输出 ±1
    - 误差会被保留到下一个时间步，保证整体轨迹一致性
    
    Args:
        X: 1D array，单列 action 数据 (chunk_size,)
        delta: 离散化步长阈值
    
    Returns:
        离散化后的 action，值域为 {-1, 0, 1}
        
    示例:
        X = [0.015, 0.008, -0.012, 0.003], delta = 0.01
        输出可能为 [1, 1, -1, 0]  (具体取决于累积误差)
    """
    X_new = X.copy()
    dx0 = 0  # 累积误差
    for i in range(len(X)):  # 修复: 处理所有元素，包括最后一个
        dx = X_new[i] + dx0
        if dx >= delta / 2:
            X_new[i] = 1      # 输出 +1 (前进)
            dx0 = dx - delta  # 保留超出部分作为下一步的累积误差
        elif dx <= -delta / 2:
            X_new[i] = -1     # 输出 -1 (后退)
            dx0 = dx + delta  # 保留超出部分
        else:
            X_new[i] = 0      # 输出 0 (不动)
            dx0 = dx          # 累积误差继续传递
    return X_new

# ! 没有cumsom累计求和操作
def undiscrete_constrain_delta(X: np.ndarray, delta: float) -> np.ndarray:
    """
    反离散化函数 - 将三元离散值 {-1, 0, 1} 还原为连续值 {-delta, 0, delta}
    
    公式: output = X * delta
    
    Args:
        X: 1D array，离散化后的 action 数据，值为 {-1, 0, 1}
        delta: 离散化步长
    
    Returns:
        反离散化后的 action，值域为 {-delta, 0, delta}
        
    示例:
        X = [1, 1, -1, 0], delta = 0.01
        输出: [0.01, 0.01, -0.01, 0]
    """
    X_new = X.copy()
    X_new = X_new * delta
    return X_new

# !

def discrete_chunk_calculus(
    X_chunk: np.ndarray,
    delta: float,
    beta: float = 0.6,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    基于微积分的离散化函数 - 使用 PID 类控制思想将连续 action 转换为三元离散值 {-1, 0, 1}
    
    算法说明:
    - 使用积分项(累积误差)和微分项(趋势预测)相结合的决策策略
    - 累积误差会传递到下一个时间步，保持轨迹一致性
    - 在 chunk 内追不到目标是合理的，不强制结清
    
    数学原理:
    - 趋势项 v_hat: 指数加权移动平均的一阶差分，预测变化趋势
    - 决策量 d = X[i] + e + beta * v_hat (I + D 前馈)
    - 量化函数 Q3: 三档量化 {-1, 0, 1}
    
    Args:
        X_chunk: 1D array，单列 action 数据 (chunk_size,)
        delta: 离散化步长阈值
        beta: 趋势项权重，控制微分作用强度 (默认 0.6)
        alpha: 趋势平滑系数，alpha 越大响应越快 (默认 0.4)
    
    Returns:
        离散化后的 action，值域为 {-1, 0, 1}
        
    示例:
        X = [0.015, 0.008, -0.012, 0.003], delta = 0.01
        输出可能为 [1, 1, -1, 0]
    
    与 discrete_constrain_delta 的区别:
    - 本函数使用趋势预测，对快速变化的信号响应更好
    """
    X = np.asarray(X_chunk, dtype=float)
    T = len(X)

    def Q3(d):
        """三档量化函数，返回 {-1, 0, 1}"""
        if d >= delta / 2:
            return 1
        if d <= -delta / 2:
            return -1
        return 0

    e = 0.0      # 累积误差 (积分项)
    v_hat = 0.0  # 趋势估计 (微分项的平滑版)

    U = np.zeros(T, dtype=float)

    for i in range(T):
        # 趋势项（平滑微分）
        w = 0.0 if i == 0 else (X[i] - X[i - 1])
        v_hat = (1 - alpha) * v_hat + alpha * w

        # 微积分决策量：I + D 前馈
        d = X[i] + e + beta * v_hat
        ui = Q3(d)

        # 积分更新 - 使用实际的 delta 值计算误差
        e = e + X[i] - ui * delta
        U[i] = ui

    return U


def undiscrete_chunk_calculus(U: np.ndarray, delta: float) -> np.ndarray:
    """
    基于微积分离散化的反离散化函数 - 将 {-1, 0, 1} 还原为 {-delta, 0, delta}
    
    公式: output = U * delta
    
    Args:
        U: 1D array，discrete_chunk_calculus 的输出，值为 {-1, 0, 1}
        delta: 离散化步长
    
    Returns:
        反离散化后的 action，值域为 {-delta, 0, delta}
        
    示例:
        U = [1, 1, -1, 0], delta = 0.01
        输出: [0.01, 0.01, -0.01, 0]
    """
    U_new = np.asarray(U, dtype=float).copy()
    U_new = U_new * delta
    return U_new


def discrete_actions_by_columns_calculus(
    actions: np.ndarray,
    columns: List[int],
    deltas: List[float],
    beta: float = 0.6,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    使用微积分方法对 action chunk 的指定列进行离散化，每列使用不同的 delta
    
    离散化后的值为 {-1, 0, 1}，不同列可以有不同的 delta 阈值
    
    Args:
        actions: shape (chunk_size, action_dim)，连续 action 数据
        columns: 要离散化的列索引列表，例如 [0, 1, 2]
        deltas: 对应列的 delta 阈值列表，例如 [0.01, 0.02, 0.01]
        beta: 趋势项权重 (默认 0.6)
        alpha: 趋势平滑系数 (默认 0.4)
    
    Returns:
        离散化后的 actions，指定列的值为 {-1, 0, 1}，其他列保持原值
        
    示例:
        actions = [[0.015, 0.025], [0.008, -0.015]]
        columns = [0, 1], deltas = [0.01, 0.02]
        输出的 actions[:, 0] 和 actions[:, 1] 会是 {-1, 0, 1} 中的值
    """
    if len(columns) != len(deltas):
        raise ValueError(f"columns 和 deltas 长度必须相同: {len(columns)} != {len(deltas)}")
    
    actions_out = actions.copy().astype(np.float64)
    
    for col_idx, delta in zip(columns, deltas):
        if col_idx < actions.shape[1]:
            actions_out[:, col_idx] = discrete_chunk_calculus(
                actions[:, col_idx], delta, beta, alpha
            )
    
    return actions_out


def undiscrete_actions_by_columns_calculus(
    actions: np.ndarray,
    columns: List[int],
    deltas: List[float]
) -> np.ndarray:
    """
    对使用微积分方法离散化的 action chunk 进行反离散化
    
    将 {-1, 0, 1} 乘以 delta 还原为 {-delta, 0, delta}
    
    Args:
        actions: shape (chunk_size, action_dim)，离散值 {-1, 0, 1}
        columns: 离散化的列索引列表
        deltas: 对应列的 delta 值列表
    
    Returns:
        反离散化后的 actions
        指定列: {-1, 0, 1} * delta = {-delta, 0, delta}
        其他列: 保持原值
        
    示例:
        actions = [[1, -1], [0, 1]]  # 离散值
        columns = [0, 1], deltas = [0.01, 0.02]
        输出: [[0.01, -0.02], [0, 0.02]]
    """
    if len(columns) != len(deltas):
        raise ValueError(f"columns 和 deltas 长度必须相同: {len(columns)} != {len(deltas)}")
    
    actions_out = actions.copy().astype(np.float64)
    
    for col_idx, delta in zip(columns, deltas):
        if col_idx < actions.shape[1]:
            actions_out[:, col_idx] = undiscrete_chunk_calculus(actions[:, col_idx], delta)
    
    return actions_out
# !

def discrete_actions_by_columns(
    actions: np.ndarray,
    columns: List[int],
    deltas: List[float]
) -> np.ndarray:
    """
    对 action chunk 的指定列进行离散化，每列使用不同的 delta
    
    离散化后的值为 {-1, 0, 1}，不同列可以有不同的 delta 阈值
    
    Args:
        actions: shape (chunk_size, action_dim)，连续 action 数据
        columns: 要离散化的列索引列表，例如 [0, 1, 2]
        deltas: 对应列的 delta 阈值列表，例如 [0.01, 0.02, 0.01]
    
    Returns:
        离散化后的 actions，指定列的值为 {-1, 0, 1}，其他列保持原值
        
    示例:
        actions = [[0.015, 0.025], [0.008, -0.015]]
        columns = [0, 1], deltas = [0.01, 0.02]
        输出的 actions[:, 0] 和 actions[:, 1] 都会变成 {-1, 0, 1} 中的值
    """
    if len(columns) != len(deltas):
        raise ValueError(f"columns 和 deltas 长度必须相同: {len(columns)} != {len(deltas)}")
    
    actions_out = actions.copy()
    
    for col_idx, delta in zip(columns, deltas):
        if col_idx < actions.shape[1]:
            actions_out[:, col_idx] = discrete_constrain_delta(actions[:, col_idx], delta)
    
    return actions_out


def undiscrete_actions_by_columns(
    actions: np.ndarray,
    columns: List[int],
    deltas: List[float]
) -> np.ndarray:
    """
    对 action chunk 的指定列进行反离散化
    
    将 {-1, 0, 1} 乘以 delta 还原为 {-delta, 0, delta}
    
    Args:
        actions: shape (chunk_size, action_dim)，离散值 {-1, 0, 1}
        columns: 要反离散化的列索引列表
        deltas: 对应列的 delta 值列表
    
    Returns:
        反离散化后的 actions
        指定列: {-1, 0, 1} * delta = {-delta, 0, delta}
        其他列: 保持原值
        
    示例:
        actions = [[1, -1], [0, 1]]  # 离散值
        columns = [0, 1], deltas = [0.01, 0.02]
        输出: [[0.01, -0.02], [0, 0.02]]
    """
    if len(columns) != len(deltas):
        raise ValueError(f"columns 和 deltas 长度必须相同: {len(columns)} != {len(deltas)}")
    
    actions_out = actions.copy()
    
    for col_idx, delta in zip(columns, deltas):
        if col_idx < actions.shape[1]:
            actions_out[:, col_idx] = undiscrete_constrain_delta(actions[:, col_idx], delta)
    
    return actions_out

# ! 没有用到
def undiscrete_actions_by_columns_torch(
    actions: torch.Tensor,
    columns: List[int],
    deltas: List[float]
) -> torch.Tensor:
    """
    PyTorch 版本: 对 action chunk 的指定列进行反离散化
    
    将 {-1, 0, 1} 乘以 delta 还原为 {-delta, 0, delta}
    用于推理时将 ParaCAT 预测结果还原为连续动作
    
    Args:
        actions: shape (batch_size, chunk_size, action_dim) 或 (chunk_size, action_dim)
                 离散值 {-1, 0, 1}（通常来自 argmax 后的类别映射）
        columns: 要反离散化的列索引列表
        deltas: 对应列的 delta 值列表
    
    Returns:
        反离散化后的 actions
        指定列: {-1, 0, 1} * delta = {-delta, 0, delta}
        
    推理流程示例:
        1. ParaCAT 输出 logits: (B, chunk, action_dim, 3)
        2. argmax 得到类别: (B, chunk, action_dim)，值为 {0, 1, 2}
        3. 类别映射为离散值: {0, 1, 2} -> {-1, 0, 1}
        4. 调用本函数: {-1, 0, 1} * delta -> {-delta, 0, delta}
    """
    if len(columns) != len(deltas):
        raise ValueError(f"columns 和 deltas 长度必须相同: {len(columns)} != {len(deltas)}")
    
    actions_out = actions.clone()
    
    # 处理不同的输入维度
    if actions.dim() == 2:
        # (chunk_size, action_dim)
        for col_idx, delta in zip(columns, deltas):
            if col_idx < actions.shape[1]:
                actions_out[:, col_idx] = actions[:, col_idx] * delta
    elif actions.dim() == 3:
        # (batch_size, chunk_size, action_dim)
        for col_idx, delta in zip(columns, deltas):
            if col_idx < actions.shape[2]:
                actions_out[:, :, col_idx] = actions[:, :, col_idx] * delta
    else:
        raise ValueError(f"Unsupported actions shape: {actions.shape}")
    
    return actions_out


def class_idx_to_discrete(class_idx: torch.Tensor) -> torch.Tensor:
    """
    将类别索引 {0, 1, 2} 映射为离散值 {-1, 0, 1}
    
    用于推理时将 argmax 结果转换为离散动作值
    
    映射关系:
        0 -> -1 (后退)
        1 ->  0 (不动)
        2 -> +1 (前进)
    
    Args:
        class_idx: 类别索引 tensor，值为 {0, 1, 2}
    
    Returns:
        离散值 tensor，值为 {-1, 0, 1}
    """
    return class_idx - 1


def discrete_to_class_idx(discrete_val: torch.Tensor) -> torch.Tensor:
    """
    将离散值 {-1, 0, 1} 映射为类别索引 {0, 1, 2}
    
    用于训练时将 Ground Truth 转换为分类标签
    
    映射关系:
        -1 -> 0 (后退类)
         0 -> 1 (不动类)
        +1 -> 2 (前进类)
    
    Args:
        discrete_val: 离散值 tensor，值为 {-1, 0, 1}
    
    Returns:
        类别索引 tensor，值为 {0, 1, 2}
    """
    return (discrete_val + 1).long()


# ============================================================================
# LIBERO Gripper 专用函数
# ============================================================================
# LIBERO 数据集的 gripper 动作原始值已经是 {-1, 0, 1}:
#   -1: 关闭夹爪 (close)
#    0: 保持当前状态 (maintain)
#   +1: 打开夹爪 (open)
#
# 与位置/旋转列不同，gripper 列不需要经过 discrete_constrain_delta 离散化，
# 只需要直接 +1 转换为类别索引 {0, 1, 2} 用于 CrossEntropyLoss。
# ============================================================================


def libero_gripper_to_class_idx(gripper_val: np.ndarray) -> np.ndarray:
    """
    将 LIBERO gripper 原始值 {-1, 0, 1} 直接转换为类别索引 {0, 1, 2}
    
    训练时使用：gripper 本身已是离散值，无需 discrete_constrain_delta 处理
    
    映射关系:
        -1 -> 0 (关闭夹爪类)
         0 -> 1 (保持状态类)
        +1 -> 2 (打开夹爪类)
    
    Args:
        gripper_val: gripper 值，{-1, 0, 1}
    
    Returns:
        类别索引 {0, 1, 2}，可直接用于 CrossEntropyLoss
    """
    return (gripper_val + 1).astype(np.int64)


def libero_gripper_from_class_idx(class_idx: np.ndarray) -> np.ndarray:
    """
    将类别索引 {0, 1, 2} 转换回 LIBERO gripper 值 {-1, 0, 1}
    
    推理时使用：argmax 结果 -1 即可，直接传给 LIBERO 环境执行
    
    映射关系:
        0 -> -1 (关闭夹爪)
        1 ->  0 (保持状态)
        2 -> +1 (打开夹爪)
    
    Args:
        class_idx: 类别索引 {0, 1, 2}
    
    Returns:
        gripper 值 {-1, 0, 1}，可直接传给 LIBERO 环境
    """
    return (class_idx - 1).astype(np.float32)


# ============================================================================
# 离散化方法选择器
# ============================================================================

# 可用的离散化方法名称
DISCRETE_METHODS = ["constrain_delta", "chunk_calculus"]


def get_discrete_function(method: str = "constrain_delta"):
    """
    根据方法名称获取离散化函数
    
    Args:
        method: 离散化方法名称
            - "constrain_delta": 简单累积误差方法 (discrete_constrain_delta)
            - "chunk_calculus": 基于微积分的方法 (discrete_chunk_calculus)，带趋势预测
    
    Returns:
        离散化函数
        
    Raises:
        ValueError: 如果方法名称无效
        
    示例:
        >>> discrete_fn = get_discrete_function("constrain_delta")
        >>> result = discrete_fn(data, delta)
        
        >>> discrete_fn = get_discrete_function("chunk_calculus")
        >>> result = discrete_fn(data, delta, beta=0.6, alpha=0.4)
    """
    method_map = {
        "constrain_delta": discrete_constrain_delta,
        "chunk_calculus": discrete_chunk_calculus,
    }
    
    if method not in method_map:
        raise ValueError(
            f"未知的离散化方法: {method}. "
            f"可用方法: {list(method_map.keys())}"
        )
    
    return method_map[method]


def discrete_with_method(
    X: np.ndarray,
    delta: float,
    method: str = "constrain_delta",
    beta: float = 0.6,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    使用指定方法进行离散化
    
    这是一个便捷函数，根据 method 参数自动选择离散化方法。
    
    Args:
        X: 1D array，单列 action 数据 (chunk_size,)
        delta: 离散化步长阈值
        method: 离散化方法
            - "constrain_delta": 简单累积误差方法
            - "chunk_calculus": 基于微积分的方法，带趋势预测
        beta: 趋势项权重 (仅 chunk_calculus 使用)
        alpha: 趋势平滑系数 (仅 chunk_calculus 使用)
    
    Returns:
        离散化后的 action，值域为 {-1, 0, 1}
        
    示例:
        # 使用简单方法
        >>> result = discrete_with_method(data, delta=0.01, method="constrain_delta")
        
        # 使用微积分方法
        >>> result = discrete_with_method(data, delta=0.01, method="chunk_calculus", beta=0.6)
    """
    if method == "constrain_delta":
        return discrete_constrain_delta(X, delta)
    elif method == "chunk_calculus":
        return discrete_chunk_calculus(X, delta, beta=beta, alpha=alpha)
    else:
        raise ValueError(f"未知的离散化方法: {method}. 可用方法: {DISCRETE_METHODS}")


def discrete_actions_by_columns_with_method(
    actions: np.ndarray,
    columns: List[int],
    deltas: List[float],
    method: str = "constrain_delta",
    beta: float = 0.6,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    使用指定方法对 action chunk 的指定列进行离散化
    
    Args:
        actions: shape (chunk_size, action_dim)，连续 action 数据
        columns: 要离散化的列索引列表
        deltas: 对应列的 delta 阈值列表
        method: 离散化方法 ("constrain_delta" 或 "chunk_calculus")
        beta: 趋势项权重 (仅 chunk_calculus 使用)
        alpha: 趋势平滑系数 (仅 chunk_calculus 使用)
    
    Returns:
        离散化后的 actions，指定列的值为 {-1, 0, 1}，其他列保持原值
    """
    if len(columns) != len(deltas):
        raise ValueError(f"columns 和 deltas 长度必须相同: {len(columns)} != {len(deltas)}")
    
    actions_out = actions.copy().astype(np.float64)
    
    for col_idx, delta in zip(columns, deltas):
        if col_idx < actions.shape[1]:
            actions_out[:, col_idx] = discrete_with_method(
                actions[:, col_idx], delta, method=method, beta=beta, alpha=alpha
            )
    
    return actions_out

