"""
Pons Adapter Module

A cross-attention based adapter that uses learnable query tokens
to extract information from VLM hidden states.

Architecture:
- Merges multiple VLM hidden states along sequence dimension
- Uses learnable query tokens with position encoding
- Applies cross-attention blocks (Q queries VLM KV)
- Outputs compressed features of shape (batch_size, q_seq_len, hidden_dim)
"""

from .pons_adapter import (
    PonsCrossAttentionBlock,
    PonsAdapter,
    create_pons_adapter,
)

__all__ = [
    "PonsCrossAttentionBlock",
    "PonsAdapter",
    "create_pons_adapter",
]

