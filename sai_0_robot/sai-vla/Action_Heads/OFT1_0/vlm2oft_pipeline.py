"""
VLM to OFT Pipeline Implementation

This module implements a pipeline that processes VLM (Vision Language Model) hidden states
and proprioception data through transformer blocks to generate action predictions.

Pipeline Flow:
1. Concatenate VLM hidden states from multiple layers
2. Process proprioception through ProprioProjector_Changed
3. Concatenate VLM and proprioception features
4. Pass through transformer blocks
5. Extract final features and predict actions using L1RegressionActionHead
"""

import torch
import torch.nn as nn
import math
import os
import sys
from typing import List, Optional

# Handle both relative and absolute imports
try:
    # Try relative imports first (when used as a module)
    from .constants import (
        NUM_VLM_HIDDEN_LAYERS, 
        LLM_OUTPUT_DIM_MLP_INPUT_DIM,
        PROPRIO_DIM,
        ACTION_DIM,
        NUM_ACTIONS_CHUNK
    )
    from .models.action_head.projectors_oft_orig import ProprioProjector_Changed
    from .models.action_head.action_heads_oft_orig import L1RegressionActionHead
except ImportError:
    # Fall back to absolute imports (when run as main script or imported from elsewhere)
    # Add current directory to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    from constants import (
        NUM_VLM_HIDDEN_LAYERS, 
        LLM_OUTPUT_DIM_MLP_INPUT_DIM,
        PROPRIO_DIM,
        ACTION_DIM,
        NUM_ACTIONS_CHUNK
    )
    from models.action_head.projectors_oft_orig import ProprioProjector_Changed
    from models.action_head.action_heads_oft_orig import L1RegressionActionHead


class TransformerBlock(nn.Module):
    """
    A single transformer block with multi-head self-attention and feed-forward network.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
            key_padding_mask: Optional padding mask of shape (batch_size, seq_len)
                              True = position to be masked (padding), False = valid position
                              This is used to ignore padding tokens in attention
            
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Self-attention with residual connection
        # MultiheadAttention expects (query, key, value) - in self-attention, all three are the same input x
        # key_padding_mask: (batch_size, seq_len), True for positions to mask (padding)
        attn_output, _ = self.attention(
            query=x,   
            key=x,    
            value=x,   
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TransformerBlock_encoder_decoder(nn.Module):
    """
    A transformer block with both self-attention (encoder) and cross-attention (decoder).
    
    结构说明:
    =========
    1. Self-Attention (Encoder部分): 对输入序列自身进行注意力计算
    2. Cross-Attention (Decoder部分): 对 encoder_hidden_states 进行注意力计算
    3. Feed-Forward Network: 前馈神经网络
    
    与纯 Encoder Block 的区别:
    - Encoder Block: 只有 self-attention
    - Encoder-Decoder Block: self-attention + cross-attention
    
    典型应用场景:
    - 序列到序列任务 (Seq2Seq)
    - 需要融合两个不同来源信息的场景
    - VLM hidden states 作为 encoder 输出，action query 作为 decoder 输入
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.1,
        cross_attention_dim: Optional[int] = None
    ):
        """
        初始化 Encoder-Decoder Transformer Block.
        
        Args:
            hidden_dim: 隐藏层维度 (decoder 输入维度)
            num_heads: 注意力头数量
            dropout: Dropout 比例
            cross_attention_dim: Cross-attention 的 key/value 维度
                                 如果为 None，则与 hidden_dim 相同
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else hidden_dim
        
        # ========== Self-Attention (Encoder 部分) ==========
        # query, key, value 都来自同一个输入
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # ========== Cross-Attention (Decoder 部分) ==========
        # query 来自 decoder 输入，key/value 来自 encoder 输出
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kdim=self.cross_attention_dim,  # key 的维度
            vdim=self.cross_attention_dim,  # value 的维度
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        
        # ========== Feed-Forward Network ==========
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_hidden_states: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through encoder-decoder transformer block.
        
        Args:
            x: Decoder 输入张量，形状 (batch_size, tgt_seq_len, hidden_dim)
               例如: action query tokens
            encoder_hidden_states: Encoder 输出张量，形状 (batch_size, src_seq_len, cross_attention_dim)
               例如: VLM hidden states
            self_attention_mask: Self-attention 的 mask (可选)
            cross_attention_mask: Cross-attention 的 mask (可选)
            use_causal_mask: 是否使用 causal mask (自回归生成时需要)
            
        Returns:
            输出张量，形状 (batch_size, tgt_seq_len, hidden_dim)
            
        数据流示意图:
        =============
        
        x (decoder input)
            │
            ▼
        ┌─────────────────────┐
        │   Self-Attention    │  query=x, key=x, value=x
        │   + Residual + LN   │
        └─────────────────────┘
            │
            ▼
        ┌─────────────────────┐
        │   Cross-Attention   │  query=x, key=encoder, value=encoder
        │   + Residual + LN   │
        └─────────────────────┘
            │
            ▼
        ┌─────────────────────┐
        │   Feed-Forward      │
        │   + Residual + LN   │
        └─────────────────────┘
            │
            ▼
        output
        """
        batch_size, tgt_seq_len, _ = x.shape
        
        # ========== Step 1: Self-Attention ==========
        # 生成 causal mask (可选，用于自回归生成)
        if use_causal_mask:
            causal_mask = torch.triu(
                torch.ones(tgt_seq_len, tgt_seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            if self_attention_mask is not None:
                self_attention_mask = self_attention_mask | causal_mask
            else:
                self_attention_mask = causal_mask
        
        self_attn_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attention_mask
        )
        x = self.self_attn_norm(x + self_attn_output)
        
        # ========== Step 2: Cross-Attention ==========
        # query 来自 decoder (x)，key/value 来自 encoder (encoder_hidden_states)
        cross_attn_output, _ = self.cross_attention(
            query=x,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            attn_mask=cross_attention_mask
        )
        x = self.cross_attn_norm(x + cross_attn_output)
        
        # ========== Step 3: Feed-Forward Network ==========
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x
    
    def forward_self_attention_only(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        只执行 self-attention，跳过 cross-attention.
        当不需要 encoder 输出时使用此方法.
        
        Args:
            x: 输入张量，形状 (batch_size, seq_len, hidden_dim)
            attention_mask: 注意力 mask (可选)
            
        Returns:
            输出张量，形状 (batch_size, seq_len, hidden_dim)
        """
        # Self-Attention
        self_attn_output, _ = self.self_attention(
            query=x, key=x, value=x,
            attn_mask=attention_mask
        )
        x = self.self_attn_norm(x + self_attn_output)
        
        # Feed-Forward (跳过 cross-attention)
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x


class VLM2OFTPipeline(nn.Module):
    """
    Complete pipeline from VLM hidden states to action predictions.
    
    This pipeline:
    1. Processes multiple VLM hidden layers by concatenation
    2. Projects proprioception data using ProprioProjector_Changed
    3. Concatenates VLM and proprio features
    4. Processes through configurable number of transformer blocks
    5. Extracts final features and predicts actions
    """
    
    def __init__(
        self, 
        num_transformer_blocks: int = 2,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        action_head_hidden_dim: int = 4096,
        num_vlm_layers: int = None,
        vlm_output_dim: int = None
    ):
        """
        Initialize the VLM2OFT pipeline.
        
        Args:
            hidden_dim: Hidden dimension size (should match VLM output dim)
            num_transformer_blocks: Number of transformer blocks to use
            num_attention_heads: Number of attention heads in each transformer block
            dropout: Dropout rate for transformer blocks
            action_head_hidden_dim: Hidden dimension for the action head MLP
            num_vlm_layers: Number of VLM hidden layers (default: from constants.py)
            vlm_output_dim: VLM output dimension (default: from constants.py)
        """
        super().__init__()
        
        self.hidden_dim = vlm_output_dim if vlm_output_dim is not None else LLM_OUTPUT_DIM_MLP_INPUT_DIM
        self.num_vlm_layers = num_vlm_layers if num_vlm_layers is not None else NUM_VLM_HIDDEN_LAYERS
        
        # Proprioception projector
        self.proprio_projector = ProprioProjector_Changed(
            llm_dim=self.hidden_dim,
            proprio_dim=PROPRIO_DIM
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout
            ) for _ in range(num_transformer_blocks)
        ])
        
        # Action head - pass vlm_output_dim as input_dim to action head
        self.action_head = L1RegressionActionHead(
            hidden_dim=action_head_hidden_dim,
            action_dim=NUM_ACTIONS_CHUNK * ACTION_DIM,
            input_dim=self.hidden_dim  # Pass vlm_output_dim to action head
        )
        
        # Optional final projection (can be None)
        self.final_projection = None
        
    
    def process_vlm_hidden_states(self, vlm_hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Process and concatenate VLM hidden states from multiple layers.
        
        拼接 VLM 多层 hidden states 的逻辑说明:
        =====================================================
        输入 vlm_hidden_states 是一个列表 (List)，包含多个 VLM 层的 hidden states
        
        示例输入结构 (假设 NUM_VLM_HIDDEN_LAYERS=3):
        vlm_hidden_states = [
            layer_0: Tensor(batch_size=2, seq_len=592, hidden_dim=2560),  # 第0层
            layer_1: Tensor(batch_size=2, seq_len=592, hidden_dim=2560),  # 第1层  
            layer_2: Tensor(batch_size=2, seq_len=592, hidden_dim=2560),  # 第2层
        ]
        
        拼接操作: torch.cat(vlm_hidden_states, dim=1)
        - 沿着 dim=1 (seq_len 维度) 进行拼接
        - 将 3 个 tensor 的序列维度拼接在一起
        
        输出结构:
        concatenated = Tensor(batch_size=2, seq_len=1776, hidden_dim=2560)
                                          ↑
                                    592 * 3 = 1776
        
        可视化:
        [layer_0: (2, 592, 2560)] ─┐
        [layer_1: (2, 592, 2560)] ─┼─→ cat(dim=1) ─→ (2, 1776, 2560)
        [layer_2: (2, 592, 2560)] ─┘
        =====================================================
        
        Args:
            vlm_hidden_states: List of tensors, each of shape (batch_size, seq_len, hidden_dim)
                              Should have NUM_VLM_HIDDEN_LAYERS elements
            
        Returns:
            Concatenated tensor of shape (batch_size, seq_len * NUM_VLM_HIDDEN_LAYERS, hidden_dim)
        """
        if len(vlm_hidden_states) != self.num_vlm_layers:
            raise ValueError(
                f"Expected {self.num_vlm_layers} VLM hidden states, "
                f"but got {len(vlm_hidden_states)}"
            )
        
        # Concatenate along sequence dimension
        # From [(B, seq_len, H), ...] to (B, seq_len * num_layers, H)
        concatenated = torch.cat(vlm_hidden_states, dim=1)
        
        return concatenated
    
    def forward(
        self, 
        vlm_hidden_states: List[torch.Tensor], 
        proprioception: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the complete pipeline.
        
        Args:
            vlm_hidden_states: List of VLM hidden states from different layers
                              Each tensor has shape (batch_size, seq_len, hidden_dim)
            proprioception: Proprioception data of shape (batch_size, proprio_dim)
            attention_mask: Optional attention mask for VLM hidden states
                           Shape: (batch_size, seq_len * num_layers)
                           Values: 1 = valid token, 0 = padding token
            
        Returns:
            Action predictions of shape (batch_size, 1, NUM_ACTIONS_CHUNK * ACTION_DIM)
        """
        batch_size = vlm_hidden_states[0].shape[0]
        device = vlm_hidden_states[0].device
        
        # Step 1: Process VLM hidden states
        # Shape: (batch_size, seq_len * NUM_VLM_HIDDEN_LAYERS, hidden_dim)
        vlm_features = self.process_vlm_hidden_states(vlm_hidden_states)
        vlm_seq_len = vlm_features.shape[1]
        
        # Step 2: Process proprioception
        # Shape: (batch_size, 1, hidden_dim)
        proprio_features = self.proprio_projector(proprioception)
        
        # Step 3: Concatenate VLM and proprioception features
        # Shape: (batch_size, seq_len * NUM_VLM_HIDDEN_LAYERS + 1, hidden_dim)
        combined_features = torch.cat([vlm_features, proprio_features], dim=1)
        
        # Step 4: Process attention mask for transformer blocks
        # Convert from (1=valid, 0=padding) to key_padding_mask format (True=padding, False=valid)
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask: (batch_size, vlm_seq_len), 1=valid, 0=padding
            # Add mask for proprio token (always valid, so add 1)
            proprio_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            combined_mask = torch.cat([attention_mask, proprio_mask], dim=1)  # (batch, vlm_seq_len + 1)
            # Convert to key_padding_mask: True = position to mask (padding), False = valid
            key_padding_mask = (combined_mask == 0)  # (batch, total_seq_len)
        
        # Step 5: Pass through transformer blocks
        x = combined_features
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, key_padding_mask=key_padding_mask)
        
        # Step 6: Extract the last token (corresponding to proprioception)
        # Shape: (batch_size, 1, hidden_dim)
        final_features = x[:, -1:, :]  # Take the last token
        
        # Step 7: Optional final projection
        if self.final_projection is not None:
            final_features = self.final_projection(final_features)
        
        # Step 8: Predict actions
        # Reshape for action head: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        final_features_reshaped = final_features.squeeze(1)
        
        # Get action predictions
        action_predictions = self.action_head.predict_action(
            final_features_reshaped.unsqueeze(1)  # Add back sequence dimension for action head
        )
        
        return action_predictions
    
    def get_attention_weights(
        self, 
        vlm_hidden_states: List[torch.Tensor], 
        proprioception: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Get attention weights from a specific transformer layer for analysis.
        
        Args:
            vlm_hidden_states: VLM hidden states
            proprioception: Proprioception data
            attention_mask: Optional attention mask (1=valid, 0=padding)
            layer_idx: Which transformer layer to get attention from (-1 for last layer)
            
        Returns:
            Attention weights from the specified layer
        """
        if layer_idx < 0:
            layer_idx = len(self.transformer_blocks) + layer_idx
            
        if layer_idx >= len(self.transformer_blocks) or layer_idx < 0:
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        batch_size = vlm_hidden_states[0].shape[0]
        device = vlm_hidden_states[0].device
        
        # Process inputs
        vlm_features = self.process_vlm_hidden_states(vlm_hidden_states)
        proprio_features = self.proprio_projector(proprioception)
        combined_features = torch.cat([vlm_features, proprio_features], dim=1)
        
        # Process attention mask
        key_padding_mask = None
        if attention_mask is not None:
            proprio_mask = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=device)
            combined_mask = torch.cat([attention_mask, proprio_mask], dim=1)
            key_padding_mask = (combined_mask == 0)
        
        # Forward through transformer blocks up to the specified layer
        x = combined_features
        for i, transformer_block in enumerate(self.transformer_blocks):
            if i == layer_idx:
                # Get attention weights from this layer
                attn_output, attn_weights = transformer_block.attention(x, x, x, key_padding_mask=key_padding_mask)
                return attn_weights
            else:
                x = transformer_block(x, key_padding_mask=key_padding_mask)
        
        return None


# Convenience function to create a standard pipeline
def create_vlm2oft_pipeline(
    num_transformer_blocks: int = 2,
    num_attention_heads: int = 8,
    dropout: float = 0.1,
    num_vlm_layers: int = None,
    vlm_output_dim: int = None,
    **kwargs
) -> VLM2OFTPipeline:
    """
    Create a VLM2OFT pipeline with standard configuration.
    
    Args:
        num_transformer_blocks: Number of transformer blocks
        num_attention_heads: Number of attention heads
        dropout: Dropout rate
        num_vlm_layers: Number of VLM hidden layers (default: from constants.py)
        vlm_output_dim: VLM output dimension (default: from constants.py)
        **kwargs: Additional arguments for VLM2OFTPipeline
        
    Returns:
        Configured VLM2OFTPipeline instance
    """
    return VLM2OFTPipeline(
        num_transformer_blocks=num_transformer_blocks,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        num_vlm_layers=num_vlm_layers,
        vlm_output_dim=vlm_output_dim,
        **kwargs
    )


if __name__ == "__main__":

    # Example usage and testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create pipeline
    pipeline = create_vlm2oft_pipeline(
        num_transformer_blocks=2,
        num_attention_heads=8
    ).to(device)
    
    # Create dummy data
    batch_size = 16
    seq_len = 592
    hidden_dim = LLM_OUTPUT_DIM_MLP_INPUT_DIM
    
    # Dummy VLM hidden states (3 layers as per NUM_VLM_HIDDEN_LAYERS)
    vlm_hidden_states = [
        torch.randn(batch_size, seq_len, hidden_dim, device=device)
        for _ in range(NUM_VLM_HIDDEN_LAYERS)
    ]
    
    # Dummy proprioception data
    proprioception = torch.randn(batch_size, PROPRIO_DIM, device=device)
    
    # Forward pass
    with torch.no_grad():
        action_predictions = pipeline(vlm_hidden_states, proprioception)
        print(f"Input VLM shapes: {[x.shape for x in vlm_hidden_states]}")
        print(f"Input proprio shape: {proprioception.shape}")
        print(f"Output action shape: {action_predictions.shape}")
        print(f"Expected output shape: ({batch_size}, 1, {NUM_ACTIONS_CHUNK * ACTION_DIM})")