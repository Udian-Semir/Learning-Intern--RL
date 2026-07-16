"""
ParaCAT Action Head Module

A cross-attention and self-attention based action head that:
1. Uses learnable query tokens to attend to Pons adapter output
2. Processes through self-attention transformer blocks
3. Applies MLP layers to predict actions

Output shape: (batch_size, chunk_size, action_dim, 3)
"""

from .paracat_action_head import (
    SelfAttentionBlock,
    ParaCATActionHead,
    create_paracat_action_head,
)

__all__ = [
    "SelfAttentionBlock",
    "ParaCATActionHead",
    "create_paracat_action_head",
]

