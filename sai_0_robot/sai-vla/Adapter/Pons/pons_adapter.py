"""
Pons Adapter Implementation

This module implements a cross-attention based adapter that:
1. Merges multiple VLM hidden states along the sequence dimension
2. Uses learnable query tokens to attend to the merged VLM features
3. Outputs compressed features for downstream tasks

Key Components:
- PonsCrossAttentionBlock: Single cross-attention block with FFN
- PonsAdapter: Main adapter class with multiple blocks
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Union


class PonsCrossAttentionBlock(nn.Module):
    """
    A single Cross-Attention block for the Pons adapter.
    
    Structure:
    ==========
    1. Cross-Attention: Query attends to Key/Value from VLM hidden states
    2. LayerNorm + Residual
    3. Feed-Forward Network (FFN)
    4. LayerNorm + Residual
    
    Data Flow:
    ==========
    
    Query (learnable)     VLM Hidden States
          │                      │
          ▼                      ▼
    ┌─────────────────────────────────┐
    │     Cross-Attention             │
    │  Q=query, K=vlm, V=vlm          │
    └─────────────────────────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  + Residual   │
          │  + LayerNorm  │
          └───────────────┘
                  │
                  ▼
          ┌───────────────┐
          │     FFN       │
          └───────────────┘
                  │
                  ▼
          ┌───────────────┐
          │  + Residual   │
          │  + LayerNorm  │
          └───────────────┘
                  │
                  ▼
              Output
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_expansion: int = 4
    ):
        """
        Initialize the Cross-Attention block.
        
        Args:
            hidden_dim: Dimension of hidden states (must match VLM hidden dim)
            num_heads: Number of attention heads
            dropout: Dropout probability
            ffn_expansion: FFN hidden dimension multiplier (ffn_dim = hidden_dim * ffn_expansion)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Cross-Attention: Query attends to VLM Key/Value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        kv_states: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the cross-attention block.
        
        Args:
            query: Query tensor of shape (batch_size, q_seq_len, hidden_dim)
                   This is the learnable query tokens
            kv_states: Key/Value tensor of shape (batch_size, kv_seq_len, hidden_dim)
                       This is the merged VLM hidden states
            kv_padding_mask: Optional padding mask for KV states
                             Shape: (batch_size, kv_seq_len)
                             True = position to be masked (padding), False = valid
        
        Returns:
            Output tensor of shape (batch_size, q_seq_len, hidden_dim)
        """
        # Cross-Attention with residual connection
        # Query attends to VLM Key/Value
        cross_attn_output, _ = self.cross_attention(
            query=query,
            key=kv_states,
            value=kv_states,
            key_padding_mask=kv_padding_mask
        )
        query = self.cross_attn_norm(query + cross_attn_output)
        
        # FFN with residual connection
        ffn_output = self.ffn(query)
        query = self.ffn_norm(query + ffn_output)
        
        return query


class PonsAdapter(nn.Module):
    """
    Pons Adapter - Cross-Attention based feature aggregator.
    
    This adapter uses learnable query tokens to extract and compress
    information from VLM hidden states through cross-attention.
    
    Architecture Overview:
    =====================
    
    Input: List of VLM Hidden States [(B, S, D), (B, S, D), ...]
                         │
                         ▼
              ┌─────────────────────┐
              │  Merge (concat on   │
              │  seq_len dimension) │
              └─────────────────────┘
                         │
                         ▼
              Merged: (B, S*N, D)
                         │
                         │
    Learnable Query      │
    (1, Q_len, D)        │
         │               │
         ▼               │
    ┌─────────────┐      │
    │ + Position  │      │
    │   Encoding  │      │
    └─────────────┘      │
         │               │
         ▼               ▼
    ┌─────────────────────────┐
    │  Cross-Attention Block  │ × num_blocks
    │  Q=query, K=VLM, V=VLM  │
    └─────────────────────────┘
                 │
                 ▼
         Output: (B, Q_len, D)
    
    Key Features:
    - Merges multiple VLM hidden states along seq_len dimension
    - Learnable query tokens with learnable position encoding
    - Multiple cross-attention blocks (configurable)
    - Hidden dimension auto-adapts to VLM hidden dim
    """
    
    def __init__(
        self,
        q_seq_len: int,
        hidden_dim: Optional[int] = None,
        num_blocks: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_expansion: int = 4
    ):
        """
        Initialize the Pons Adapter.
        
        Args:
            q_seq_len: Length of the learnable query sequence (required)
            hidden_dim: Hidden dimension size. If None, will be inferred from
                        the first forward pass based on VLM hidden states.
                        Once set, it cannot be changed.
            num_blocks: Number of cross-attention blocks (default: 2)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            ffn_expansion: FFN hidden dimension multiplier (default: 4)
        """
        super().__init__()
        
        self.q_seq_len = q_seq_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.ffn_expansion = ffn_expansion
        
        # Hidden dim can be set now or inferred later
        self._hidden_dim = hidden_dim
        self._initialized = hidden_dim is not None
        
        if self._initialized:
            self._build_modules(hidden_dim)
    
    def _build_modules(self, hidden_dim: int):
        """
        Build all learnable modules once hidden_dim is known.
        
        Args:
            hidden_dim: The hidden dimension to use
        """
        self._hidden_dim = hidden_dim
        
        # Learnable query tokens: (1, q_seq_len, hidden_dim)
        # Will be expanded to batch size during forward
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.q_seq_len, hidden_dim) * 0.02
        )
        
        # Learnable position encoding for query tokens
        self.query_pos_encoding = nn.Parameter(
            torch.randn(1, self.q_seq_len, hidden_dim) * 0.02
        )
        
        # Cross-attention blocks
        self.blocks = nn.ModuleList([
            PonsCrossAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                ffn_expansion=self.ffn_expansion
            )
            for _ in range(self.num_blocks)
        ])
        
        self._initialized = True
    
    @property
    def hidden_dim(self) -> Optional[int]:
        """Return the hidden dimension, or None if not yet initialized."""
        return self._hidden_dim
    
    def merge_vlm_hidden_states(
        self,
        vlm_hidden_states: Union[torch.Tensor, List[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Merge multiple VLM hidden states along the sequence dimension.
        
        Merging Logic:
        ==============
        Input: List of tensors, each of shape (batch_size, seq_len, hidden_dim)
        
        Example (3 layers, seq_len=592):
        vlm_hidden_states = [
            layer_0: (B, 592, D),
            layer_1: (B, 592, D),
            layer_2: (B, 592, D),
        ]
        
        Operation: torch.cat(vlm_hidden_states, dim=1)
        
        Output: (B, 592*3, D) = (B, 1776, D)
        
        Visualization:
        [layer_0: (B, 592, D)] ─┐
        [layer_1: (B, 592, D)] ─┼─→ cat(dim=1) ─→ (B, 1776, D)
        [layer_2: (B, 592, D)] ─┘
        
        Args:
            vlm_hidden_states: Either a single tensor (B, S, D) or a list of tensors
                               [(B, S, D), (B, S, D), ...]
        
        Returns:
            Merged tensor of shape (batch_size, total_seq_len, hidden_dim)
            If single tensor input, returns it unchanged.
        """
        # Handle single tensor input
        if isinstance(vlm_hidden_states, torch.Tensor):
            return vlm_hidden_states
        
        # Handle list input
        if len(vlm_hidden_states) == 1:
            # Single element list, no merging needed
            return vlm_hidden_states[0]
        
        # Multiple tensors: concatenate along seq_len dimension (dim=1)
        return torch.cat(vlm_hidden_states, dim=1)
    
    def forward(
        self,
        vlm_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the Pons adapter.
        
        Args:
            vlm_hidden_states: VLM hidden states, either:
                - Single tensor of shape (batch_size, seq_len, hidden_dim)
                - List of tensors [(B, S, D), ...] to be merged
            attention_mask: Optional attention mask for VLM hidden states
                            Shape: (batch_size, total_seq_len)
                            Values: 1 = valid token, 0 = padding token
        
        Returns:
            Output tensor of shape (batch_size, q_seq_len, hidden_dim)
        """
        # Step 1: Merge VLM hidden states
        merged_states = self.merge_vlm_hidden_states(vlm_hidden_states)
        batch_size, total_seq_len, hidden_dim = merged_states.shape
        device = merged_states.device
        
        # Step 2: Initialize modules if not done yet (lazy initialization)
        if not self._initialized:
            self._build_modules(hidden_dim)
            # Move newly created parameters to the correct device
            self.to(device)
        
        # Verify hidden_dim matches
        if hidden_dim != self._hidden_dim:
            raise ValueError(
                f"VLM hidden dim ({hidden_dim}) doesn't match "
                f"adapter hidden dim ({self._hidden_dim})"
            )
        
        # Step 3: Prepare query tokens with position encoding
        # Expand query tokens to batch size: (1, Q, D) -> (B, Q, D)
        query = self.query_tokens.expand(batch_size, -1, -1)
        # Add position encoding
        query = query + self.query_pos_encoding
        
        # Step 4: Prepare attention mask
        # Convert from (1=valid, 0=padding) to key_padding_mask (True=padding, False=valid)
        kv_padding_mask = None
        if attention_mask is not None:
            kv_padding_mask = (attention_mask == 0)
        
        # Step 5: Pass through cross-attention blocks
        for block in self.blocks:
            query = block(
                query=query,
                kv_states=merged_states,
                kv_padding_mask=kv_padding_mask
            )
        
        # Output shape: (batch_size, q_seq_len, hidden_dim)
        return query
    
    def print_model_summary(self, print_details: bool = True) -> dict:
        """
        打印模型参数量统计信息
        
        详细显示各个组件的参数量，包括:
        - Query Tokens 和 Position Encoding
        - Cross-Attention Blocks
        
        Args:
            print_details: 是否打印详细信息 (默认 True)
        
        Returns:
            dict: 包含参数统计信息的字典
        """
        if not self._initialized:
            print("⚠️ 模型尚未初始化，请先执行一次 forward 或显式设置 hidden_dim")
            return {}
        
        def count_params(module):
            """统计模块的总参数量和可训练参数量"""
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable
        
        def format_num(n):
            """格式化数字，添加千位分隔符"""
            return f"{n:,}"
        
        # =====================================================================
        # 1. Query Tokens 参数量
        # =====================================================================
        # query_tokens: (1, q_seq_len, hidden_dim)
        query_tokens_params = self.query_tokens.numel()
        # query_pos_encoding: (1, q_seq_len, hidden_dim)
        query_pos_encoding_params = self.query_pos_encoding.numel()
        
        # =====================================================================
        # 2. Cross-Attention Blocks 参数量
        # =====================================================================
        blocks_total = 0
        blocks_trainable = 0
        for block in self.blocks:
            t, tr = count_params(block)
            blocks_total += t
            blocks_trainable += tr
        
        # =====================================================================
        # 汇总
        # =====================================================================
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 构建结果字典
        result = {
            "总参数量": total_params,
            "可训练参数量": trainable_params,
            "query_tokens": query_tokens_params,
            "query_pos_encoding": query_pos_encoding_params,
            "cross_attention_blocks": blocks_total,
        }
        
        if print_details:
            print("\n" + "=" * 70)
            print("Pons Adapter 参数量统计")
            print("=" * 70)
            
            # 配置信息
            print(f"\n📋 模型配置:")
            print(f"   • q_seq_len: {self.q_seq_len}")
            print(f"   • hidden_dim: {self._hidden_dim}")
            print(f"   • num_heads: {self.num_heads}")
            print(f"   • num_blocks: {self.num_blocks}")
            print(f"   • ffn_expansion: {self.ffn_expansion}")
            
            # 参数量详细分解
            print(f"\n📊 参数量分解:")
            print("-" * 70)
            
            # Query Tokens
            print(f"\n1️⃣ Query Tokens & Position Encoding:")
            print(f"   • query_tokens:       {format_num(query_tokens_params):>15} params")
            print(f"     计算: 1 × {self.q_seq_len} × {self._hidden_dim} = {format_num(query_tokens_params)}")
            print(f"   • query_pos_encoding: {format_num(query_pos_encoding_params):>15} params")
            print(f"     计算: 1 × {self.q_seq_len} × {self._hidden_dim} = {format_num(query_pos_encoding_params)}")
            query_subtotal = query_tokens_params + query_pos_encoding_params
            print(f"   ─────────────────────────────────────────")
            print(f"   小计:                 {format_num(query_subtotal):>15} params")
            
            # Cross-Attention Blocks
            print(f"\n2️⃣ Cross-Attention Blocks (× {self.num_blocks}):")
            if len(self.blocks) > 0:
                single_block_params, _ = count_params(self.blocks[0])
                print(f"   • 单个 block:         {format_num(single_block_params):>15} params")
                print(f"     包含: MultiheadAttention + LayerNorm + FFN")
                print(f"   • 总计 ({self.num_blocks} blocks):    {format_num(blocks_total):>15} params")
                print(f"     计算: {format_num(single_block_params)} × {self.num_blocks} = {format_num(blocks_total)}")
            
            # 总计
            print(f"\n" + "=" * 70)
            print(f"📈 总计:")
            print(f"   • 总参数量:           {format_num(total_params):>15} params")
            print(f"   • 可训练参数量:       {format_num(trainable_params):>15} params")
            print(f"   • 参数量 (MB):        {total_params * 4 / 1024 / 1024:>15.2f} MB (float32)")
            print(f"   • 参数量 (MB):        {total_params * 2 / 1024 / 1024:>15.2f} MB (float16/bf16)")
            print("=" * 70)
            
            # 验证总和
            computed_total = query_subtotal + blocks_total
            if computed_total == total_params:
                print(f"✓ 参数量验证通过: {format_num(query_subtotal)} + {format_num(blocks_total)} = {format_num(total_params)}")
            else:
                print(f"⚠️ 参数量不匹配: 计算值 {format_num(computed_total)} ≠ 实际值 {format_num(total_params)}")
        
        return result
    
    def get_attention_weights(
        self,
        vlm_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        block_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention weights from a specific cross-attention block.
        Useful for visualization and analysis.
        
        Args:
            vlm_hidden_states: VLM hidden states
            attention_mask: Optional attention mask
            block_idx: Which block to get attention from (-1 for last block)
        
        Returns:
            Attention weights tensor
        """
        if block_idx < 0:
            block_idx = len(self.blocks) + block_idx
        
        if block_idx >= len(self.blocks) or block_idx < 0:
            raise ValueError(f"Invalid block index: {block_idx}")
        
        # Merge and prepare
        merged_states = self.merge_vlm_hidden_states(vlm_hidden_states)
        batch_size = merged_states.shape[0]
        device = merged_states.device
        
        if not self._initialized:
            self._build_modules(merged_states.shape[-1])
            self.to(device)
        
        # Prepare query
        query = self.query_tokens.expand(batch_size, -1, -1)
        query = query + self.query_pos_encoding
        
        # Prepare mask
        kv_padding_mask = None
        if attention_mask is not None:
            kv_padding_mask = (attention_mask == 0)
        
        # Forward through blocks up to target
        for i, block in enumerate(self.blocks):
            if i == block_idx:
                # Get attention weights from this block
                _, attn_weights = block.cross_attention(
                    query=query,
                    key=merged_states,
                    value=merged_states,
                    key_padding_mask=kv_padding_mask,
                    average_attn_weights=False
                )
                return attn_weights
            else:
                query = block(query, merged_states, kv_padding_mask)
        
        return None


def create_pons_adapter(
    q_seq_len: int,
    hidden_dim: Optional[int] = None,
    num_blocks: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    **kwargs
) -> PonsAdapter:
    """
    Factory function to create a Pons Adapter.
    
    Args:
        q_seq_len: Length of the learnable query sequence (required)
        hidden_dim: Hidden dimension. If None, auto-inferred from VLM.
        num_blocks: Number of cross-attention blocks (default: 2)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout probability (default: 0.1)
        **kwargs: Additional arguments passed to PonsAdapter
    
    Returns:
        Configured PonsAdapter instance
    
    Example:
        >>> adapter = create_pons_adapter(q_seq_len=64, num_blocks=4)
        >>> # hidden_dim will be auto-inferred from first forward pass
        >>> output = adapter(vlm_hidden_states)  # (B, 64, D)
    """
    return PonsAdapter(
        q_seq_len=q_seq_len,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        dropout=dropout,
        **kwargs
    )


# ============================================================================
# Testing and Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Pons Adapter - Testing and Example Usage")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Test parameters
    batch_size = 4
    seq_len = 592
    hidden_dim = 2560
    num_vlm_layers = 3
    q_seq_len = 64
    
    # -------------------------------------------------------------------------
    # Test 1: Auto-inferred hidden_dim with multiple VLM layers
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 1: Multiple VLM hidden states (auto-inferred hidden_dim)")
    print("-" * 70)
    
    # Create adapter without specifying hidden_dim
    adapter = create_pons_adapter(
        q_seq_len=q_seq_len,
        num_blocks=2,
        num_heads=8
    ).to(device)
    
    # Create dummy VLM hidden states (3 layers)
    vlm_hidden_states = [
        torch.randn(batch_size, seq_len, hidden_dim, device=device)
        for _ in range(num_vlm_layers)
    ]
    
    print(f"Input VLM shapes: {[x.shape for x in vlm_hidden_states]}")
    print(f"Expected merged shape: ({batch_size}, {seq_len * num_vlm_layers}, {hidden_dim})")
    
    with torch.no_grad():
        output = adapter(vlm_hidden_states)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {q_seq_len}, {hidden_dim})")
    assert output.shape == (batch_size, q_seq_len, hidden_dim), "Shape mismatch!"
    print("✓ Test 1 passed!")
    
    # -------------------------------------------------------------------------
    # Test 2: Single VLM hidden state (no merging needed)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 2: Single VLM hidden state (no merging)")
    print("-" * 70)
    
    adapter_single = create_pons_adapter(
        q_seq_len=32,
        hidden_dim=hidden_dim,  # Explicitly set
        num_blocks=3
    ).to(device)
    
    single_vlm_state = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    print(f"Input VLM shape: {single_vlm_state.shape}")
    
    with torch.no_grad():
        output_single = adapter_single(single_vlm_state)
    
    print(f"Output shape: {output_single.shape}")
    print(f"Expected output shape: ({batch_size}, 32, {hidden_dim})")
    assert output_single.shape == (batch_size, 32, hidden_dim), "Shape mismatch!"
    print("✓ Test 2 passed!")
    
    # -------------------------------------------------------------------------
    # Test 3: With attention mask
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 3: With attention mask")
    print("-" * 70)
    
    adapter_mask = create_pons_adapter(
        q_seq_len=16,
        hidden_dim=hidden_dim,
        num_blocks=2
    ).to(device)
    
    # Create attention mask (1=valid, 0=padding)
    # Simulate variable length sequences
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    attention_mask[0, 400:] = 0  # First sample has shorter sequence
    attention_mask[1, 500:] = 0  # Second sample has shorter sequence
    
    vlm_state = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    print(f"Input VLM shape: {vlm_state.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    
    with torch.no_grad():
        output_mask = adapter_mask(vlm_state, attention_mask=attention_mask)
    
    print(f"Output shape: {output_mask.shape}")
    assert output_mask.shape == (batch_size, 16, hidden_dim), "Shape mismatch!"
    print("✓ Test 3 passed!")
    
    # -------------------------------------------------------------------------
    # Test 4: Different hidden dimensions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 4: Different VLM hidden dimensions")
    print("-" * 70)
    
    for test_hidden_dim in [1024, 2048, 4096]:
        adapter_dim = create_pons_adapter(
            q_seq_len=8,
            num_blocks=1
        ).to(device)
        
        vlm_state_dim = torch.randn(2, 100, test_hidden_dim, device=device)
        
        with torch.no_grad():
            output_dim = adapter_dim(vlm_state_dim)
        
        print(f"  hidden_dim={test_hidden_dim}: output shape = {output_dim.shape}")
        assert output_dim.shape == (2, 8, test_hidden_dim), "Shape mismatch!"
    
    print("✓ Test 4 passed!")
    
    # -------------------------------------------------------------------------
    # Test 5: Get attention weights
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 5: Get attention weights")
    print("-" * 70)
    
    adapter_attn = create_pons_adapter(
        q_seq_len=16,
        hidden_dim=hidden_dim,
        num_blocks=2,
        num_heads=8
    ).to(device)
    
    vlm_state_attn = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    
    with torch.no_grad():
        attn_weights = adapter_attn.get_attention_weights(
            vlm_state_attn,
            block_idx=-1  # Last block
        )
    
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Expected: (batch={batch_size}, heads=8, q_len=16, kv_len={seq_len})")
    print("✓ Test 5 passed!")
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Test 6: 使用 print_model_summary() 方法打印详细参数统计
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 6: print_model_summary() 方法")
    print("-" * 70)
    
    # 使用新的方法打印详细的参数统计
    adapter.print_model_summary()

