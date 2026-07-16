"""
Hand Binary Processor - 手部数据二值化处理

将手部电机位置数据转换为二值化的夹爪状态:
- 6个值取平均 > 442  -> 1 (张开)
- 6个值取平均 <= 442 -> 0 (闭合)

输入 6 维，输出 1 维
"""

import numpy as np
from typing import Optional
from .base_processor import BaseProcessor


class HandBinaryProcessor(BaseProcessor):
    """
    手部数据二值化处理器
    
    将手部电机位置数据（6维）转换为二值化状态（1维）。
    
    处理逻辑:
    1. 将 6 个值相加取平均
    2. 平均值 > threshold (默认442) -> 输出 1 (张开)
    3. 平均值 <= threshold -> 输出 0 (闭合)
    
    输入: (num_frames, 6)
    输出: (num_frames, 1)
    """
    
    def __init__(self, threshold: float = 442.0):
        """
        初始化手部处理器
        
        Args:
            threshold: 二值化阈值，默认为 442
        """
        self.threshold = threshold
    
    @property
    def name(self) -> str:
        return "hand_binary"
    
    def get_output_dim(self, input_dim: int) -> int:
        """
        计算输出维度
        
        无论输入多少维，输出都是 1 维
        """
        return 1
    
    def process(
        self, 
        states: np.ndarray, 
        indices: Optional[tuple] = None
    ) -> np.ndarray:
        """
        将手部数据二值化
        
        Args:
            states: state 数据数组，shape 为 (num_frames, 6) 或 (num_frames, state_dim)
            indices: 可选的索引范围 (start, end)，指定要处理的维度
        
        Returns:
            二值化后的 action 数据，shape 为 (num_frames, 1)
        """
        if states.ndim == 1:
            states = states.reshape(-1, 1)
        
        # 如果指定了索引范围，只处理该范围的维度
        if indices is not None:
            start, end = indices
            data = states[:, start:end]
        else:
            data = states
        
        # 6 个值取平均
        mean_values = np.mean(data, axis=1, keepdims=True)
        
        # 二值化处理: 平均值 > threshold -> 1, 否则 -> 0
        binary_data = np.where(mean_values > self.threshold, 1.0, 0.0).astype(np.float32)
        
        return binary_data
