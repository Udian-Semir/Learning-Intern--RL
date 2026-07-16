"""
Hidden States 合并工具
用于在序列长度维度上拼接多个 hidden states
Merge multiple hidden states along sequence length dimension
"""
import torch
import numpy as np
from typing import Union, List


def merge_hidden_states_in_seq_dim(
    hidden_states_list: List[Union[torch.Tensor, np.ndarray]],
    dim: int = 1
) -> Union[torch.Tensor, np.ndarray]:
    """
    在序列长度维度上合并多个 hidden states
    Merge multiple hidden states along the sequence length dimension
    
    Args:
        hidden_states_list: hidden states 列表，要求所有元素形状相同
        dim: 拼接维度，默认为 0 或 1（取决于输入维度）
    
    Returns:
        合并后的 hidden states
        
    Example:
        >>> # 2D tensors (seq_len, hidden_dim) - 在序列维度拼接
        >>> h1 = torch.randn(512, 2048)  # layer 0
        >>> h2 = torch.randn(512, 2048)  # layer 1
        >>> h3 = torch.randn(512, 2048)  # layer 2
        >>> merged = merge_hidden_states_in_seq_dim([h1, h2, h3], dim=0)
        >>> print(merged.shape)  # torch.Size([1536, 2048]) - 512*3=1536
        >>> 
        >>> # ! 我们场景下用的是 -> 3D tensors (batch, seq_len, hidden_dim) - 在序列维度拼接
        >>> h1 = torch.randn(4, 512, 2048)
        >>> h2 = torch.randn(4, 512, 2048)
        >>> h3 = torch.randn(4, 512, 2048)
        >>> merged = merge_hidden_states_in_seq_dim([h1, h2, h3], dim=1)
        >>> print(merged.shape)  # torch.Size([4, 1536, 2048]) - 512*3=1536
    """
    if not hidden_states_list:
        raise ValueError("hidden_states_list 不能为空")
    
    # 检查是否所有元素都是相同类型
    is_torch = isinstance(hidden_states_list[0], torch.Tensor)
    is_numpy = isinstance(hidden_states_list[0], np.ndarray)
    
    if not (is_torch or is_numpy):
        raise TypeError(f"不支持的类型: {type(hidden_states_list[0])}")
    
    # 检查所有 hidden states 形状是否相同
    first_shape = hidden_states_list[0].shape
    for i, hs in enumerate(hidden_states_list[1:], 1):
        if hs.shape != first_shape:
            raise ValueError(
                f"第 {i} 个 hidden state 形状 {hs.shape} "
                f"与第 0 个 {first_shape} 不匹配"
            )
    
    # 在序列维度拼接
    if is_torch:
        return torch.cat(hidden_states_list, dim=dim)
    else:
        return np.concatenate(hidden_states_list, axis=dim)
