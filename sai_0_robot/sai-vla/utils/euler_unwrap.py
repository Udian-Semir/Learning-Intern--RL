"""
欧拉角解卷绕处理模块

处理欧拉角在 -180/180 度边界的跳变问题，使角度变化连续。

问题背景:
    欧拉角范围是 -180 到 180 度，当角度跨越边界时会发生跳变：
    - 例如：从 179° 变化到 -179°，实际只转了 2°，但数值变化了 358°
    - 这会导致 delta 计算出现错误的大幅跳变

解决方案:
    使用 np.unwrap 函数对角度序列进行"解卷绕"处理，使变化连续：
    - 原始数据: [170, 175, 180, -175, -170]  # 发生跳变
    - 解卷绕后: [170, 175, 180,  185,  190]  # 连续变化
"""

import numpy as np
from typing import List


def unwrap_euler_angles(
    states: np.ndarray, 
    indices: List[int],
    unit: str = "degree"
) -> np.ndarray:
    """
    对指定索引的欧拉角进行解卷绕处理
    
    当欧拉角从 179° 跳变到 -179° 时，实际只转了 2°，
    但数值变化了 358°。此函数将角度序列"解卷绕"，使变化连续。
    
    Args:
        states: state 数组，shape 为 (num_frames, state_dim)
        indices: 欧拉角所在的索引列表（基于组合后 state 的全局索引）
        unit: 角度单位，"degree" 或 "radian"，默认 "degree"
    
    Returns:
        处理后的 state 数组（副本，不修改原数组）
    
    Example:
        >>> states = np.array([[170], [175], [180], [-175], [-170]], dtype=np.float32)
        >>> result = unwrap_euler_angles(states, [0])
        >>> print(result.flatten())  # [170. 175. 180. 185. 190.]
    
    Note:
        - 解卷绕后的角度值可能超出 [-180, 180] 范围，这是正常的
        - 每个 episode 应独立调用此函数，不要跨 episode 处理
    """
    if len(indices) == 0:
        return states
    
    states = states.copy()  # 避免修改原数据
    
    for idx in indices:
        if idx < 0 or idx >= states.shape[1]:
            raise IndexError(
                f"欧拉角索引 {idx} 超出 state 维度范围 [0, {states.shape[1]})"
            )
        
        if unit == "degree":
            # 度 -> 弧度 -> unwrap -> 度
            angles_rad = np.deg2rad(states[:, idx])
            unwrapped_rad = np.unwrap(angles_rad)
            states[:, idx] = np.rad2deg(unwrapped_rad)
        else:
            # 直接 unwrap（假设是弧度）
            states[:, idx] = np.unwrap(states[:, idx])
    
    return states


def parse_euler_indices(indices_str: str) -> List[int]:
    """
    解析欧拉角索引字符串
    
    Args:
        indices_str: 索引字符串，支持格式：
            - "3,4,5,15,16,17"
            - "[3,4,5,15,16,17]"
            - "3, 4, 5"（带空格）
    
    Returns:
        索引列表
    
    Example:
        >>> parse_euler_indices("3,4,5,15,16,17")
        [3, 4, 5, 15, 16, 17]
        >>> parse_euler_indices("[3, 4, 5]")
        [3, 4, 5]
        >>> parse_euler_indices("")
        []
    """
    if not indices_str:
        return []
    
    # 移除方括号和空白
    indices_str = indices_str.strip().strip("[]")
    
    if not indices_str:
        return []
    
    # 分割并转换为整数
    indices = [int(x.strip()) for x in indices_str.split(",") if x.strip()]
    
    return indices
