"""
ParaCAT Action Head Implementation

This module implements a cross-attention and self-attention based action head:
1. Learnable query tokens (q_len = chunk_size * action_dim) attend to Pons output
2. Self-attention transformer blocks process the attended features
3. MLP layers predict final action values with output dim = 3
4. Output is reshaped to (batch_size, chunk_size, action_dim, 3)

Key Components:
- SelfAttentionBlock: Single self-attention block with FFN (no mask)
- ParaCATActionHead: Main action head class
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SelfAttentionBlock(nn.Module):
    """
    A single Self-Attention block for ParaCAT action head.
    
    Structure:
    ==========
    1. Self-Attention (no mask, all positions attend to all)
    2. LayerNorm + Residual
    3. Feed-Forward Network (FFN)
    4. LayerNorm + Residual
    
    Data Flow:
    ==========
    
    Input (B, seq_len, D)
          │
          ▼
    ┌─────────────────┐
    │  Self-Attention │
    │   (no mask)     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  + Residual     │
    │  + LayerNorm    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │      FFN        │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  + Residual     │
    │  + LayerNorm    │
    └────────┬────────┘
             │
             ▼
    Output (B, seq_len, D)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_expansion: int = 4
    ):
        """
        Initialize the Self-Attention block.
        
        Args:
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
            ffn_expansion: FFN hidden dimension multiplier
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-Attention (no mask needed)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_expansion, hidden_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the self-attention block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_dim)
        """
        # Self-Attention with residual connection (no mask - all attend to all)
        self_attn_output, _ = self.self_attention(
            query=x,
            key=x,
            value=x,
            need_weights=False
        )
        x = self.self_attn_norm(x + self_attn_output)
        
        # FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.ffn_norm(x + ffn_output)
        
        return x


class CrossAttentionBlock(nn.Module):
    """
    A single Cross-Attention block for querying Pons output.
    
    Query tokens attend to Pons output (Key/Value).
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
            hidden_dim: Dimension of hidden states
            num_heads: Number of attention heads
            dropout: Dropout probability
            ffn_expansion: FFN hidden dimension multiplier
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Cross-Attention
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
        kv_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the cross-attention block.
        
        Args:
            query: Query tensor of shape (batch_size, q_seq_len, hidden_dim)
            kv_states: Key/Value tensor (Pons output) of shape (batch_size, kv_seq_len, hidden_dim)
        
        Returns:
            Output tensor of shape (batch_size, q_seq_len, hidden_dim)
        """
        # Cross-Attention with residual connection
        cross_attn_output, _ = self.cross_attention(
            query=query,
            key=kv_states,
            value=kv_states,
            need_weights=False
        )
        query = self.cross_attn_norm(query + cross_attn_output)
        
        # FFN with residual connection
        ffn_output = self.ffn(query)
        query = self.ffn_norm(query + ffn_output)
        
        return query


class MLPBlock(nn.Module):
    """
    A single MLP block with LayerNorm, Linear, and optional activation.
    
    注意: ParaCAT 的最后一层 MLP 使用 use_activation=False
    这样输出的是原始 logits，可以直接传入 CrossEntropyLoss
    (CrossEntropyLoss 内部会自动计算 softmax)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        use_activation: bool = True,
        dropout: float = 0.1
    ):
        """
        Initialize the MLP block.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            use_activation: Whether to use activation function
                           设为 False 时输出纯 logits (用于分类最后一层)
            dropout: Dropout probability
        """
        super().__init__()
        self.use_activation = use_activation
        
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        ]
        
        if use_activation:
            layers.extend([
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., input_dim)
        
        Returns:
            Output tensor of shape (..., output_dim)
        """
        return self.mlp(x)


class ParaCATActionHead(nn.Module):
    """
    ParaCAT Action Head - Cross-Attention + Self-Attention + MLP based action predictor.
    
    用于离散化动作预测的 Action Head，输出三分类 logits。
    
    This action head:
    1. Uses learnable query tokens (q_len = chunk_size * action_dim)
    2. Attends to Pons adapter output via cross-attention
    3. Processes through self-attention transformer blocks
    4. Applies MLP layers to predict final actions
    5. Outputs shape: (batch_size, chunk_size, action_dim, 3)
    
    输出说明:
    =========
    - 输出是 **logits**，不是 softmax 概率！
    - 最后一维大小为 3，对应三个类别:
        类别 0: 后退 (-1)
        类别 1: 不动 (0)
        类别 2: 前进 (+1)
    
    训练时:
    - Ground Truth: 离散值 {-1, 0, 1} 转换为类别索引 {0, 1, 2}
    - Loss: CrossEntropyLoss (内部自动处理 softmax)
    
    推理时:
    - 步骤 1: argmax(logits, dim=-1) 得到类别索引 {0, 1, 2}
    - 步骤 2: 类别索引 - 1 = 离散值 {-1, 0, 1}
    - 步骤 3: 离散值 * delta = 连续动作 {-delta, 0, delta}
    
    Architecture Overview:
    =====================
    
    Pons Output (B, pons_len, D)
           │
           │     Learnable Query (1, chunk*action_dim, D)
           │            │
           │            ▼
           │     ┌──────────────────┐
           │     │ + Position Enc   │
           │     └────────┬─────────┘
           │              │
           ▼              ▼
    ┌──────────────────────────────┐
    │      Cross-Attention         │
    │   Q=query, K=pons, V=pons    │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │   Self-Attention Blocks × N  │
    │       (no mask)              │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │      MLP Layers × M          │
    │   D -> expand -> ... -> 3    │
    │   (最后一层无激活函数)         │
    └──────────────┬───────────────┘
                   │
                   ▼
    ┌──────────────────────────────┐
    │          Reshape             │
    │  (B, chunk*action, 3) ->     │
    │  (B, chunk, action, 3)       │
    │                              │
    │  输出: logits (未经 softmax)  │
    └──────────────────────────────┘
    """
    
    def __init__(
        self,
        chunk_size: int,
        action_dim: int,
        hidden_dim: Optional[int] = None,
        num_transformer_blocks: int = 2,
        num_mlp_layers: int = 2,
        mlp_expand_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        ffn_expansion: int = 4
    ):
        """
        Initialize the ParaCAT Action Head.
        
        Args:
            chunk_size: Size of action chunk (required)
            action_dim: Dimension of action space (required)
            hidden_dim: Hidden dimension. If None, auto-inferred from Pons output.
            num_transformer_blocks: Number of self-attention blocks (default: 2)
            num_mlp_layers: Number of MLP layers (default: 2)
            mlp_expand_dim: MLP intermediate expansion dimension (default: 1024)
            num_heads: Number of attention heads (default: 8)
            dropout: Dropout probability (default: 0.1)
            ffn_expansion: FFN hidden dimension multiplier (default: 4)
        """
        super().__init__()
        
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        # Query sequence length = CHUNK_SIZE * ACTION_DIM
        # 每个 query token 对应输出的一个 (时间步, 动作维度) 位置
        # 最终输出会 reshape 为 (B, chunk_size, action_dim, 3)
        self.q_seq_len = chunk_size * action_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.num_mlp_layers = num_mlp_layers
        self.mlp_expand_dim = mlp_expand_dim
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
        # 注意: q_seq_len = chunk_size * action_dim
        # 例如: chunk_size=16, action_dim=7 -> q_seq_len=112
        # 每个 query token 对应一个 (时间步, 动作维度) 的组合
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.q_seq_len, hidden_dim) * 0.02
        )
        
        # Learnable position encoding for query tokens
        # 形状: (1, q_seq_len, hidden_dim)
        # 其中 q_seq_len = CHUNK_SIZE * ACTION_DIM
        # 
        # 这是可学习的位置编码，用于区分不同位置的 query tokens
        # 在 forward 中会与 query_tokens 相加: query = query_tokens + query_pos_encoding
        self.query_pos_encoding = nn.Parameter(
            torch.randn(1, self.q_seq_len, hidden_dim) * 0.02
        )
        
        # Cross-attention block to attend to Pons output
        self.cross_attention = CrossAttentionBlock(
            hidden_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            ffn_expansion=self.ffn_expansion
        )
        
        # Self-attention transformer blocks
        self.self_attention_blocks = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim=hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                ffn_expansion=self.ffn_expansion
            )
            for _ in range(self.num_transformer_blocks)
        ])
        
        # MLP layers
        # First layer: hidden_dim -> mlp_expand_dim
        # Middle layers: mlp_expand_dim -> mlp_expand_dim
        # Last layer: mlp_expand_dim -> 3 (no activation)
        self.mlp_layers = nn.ModuleList()
        
        if self.num_mlp_layers >= 1:
            # First MLP layer
            self.mlp_layers.append(
                MLPBlock(
                    input_dim=hidden_dim,
                    output_dim=self.mlp_expand_dim,
                    use_activation=True,
                    dropout=self.dropout
                )
            )
            
            # Middle MLP layers
            for _ in range(self.num_mlp_layers - 2):
                self.mlp_layers.append(
                    MLPBlock(
                        input_dim=self.mlp_expand_dim,
                        output_dim=self.mlp_expand_dim,
                        use_activation=True,
                        dropout=self.dropout
                    )
                )
            
            # Last MLP layer (output dim = 3, no activation)
            if self.num_mlp_layers >= 2:
                self.mlp_layers.append(
                    MLPBlock(
                        input_dim=self.mlp_expand_dim,
                        output_dim=3,
                        use_activation=False,
                        dropout=self.dropout
                    )
                )
            else:
                # If only 1 MLP layer, go directly from hidden_dim to 3
                self.mlp_layers = nn.ModuleList([
                    MLPBlock(
                        input_dim=hidden_dim,
                        output_dim=3,
                        use_activation=False,
                        dropout=self.dropout
                    )
                ])
        
        self._initialized = True
    
    @property
    def hidden_dim(self) -> Optional[int]:
        """Return the hidden dimension, or None if not yet initialized."""
        return self._hidden_dim
    
    def forward(self, pons_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ParaCAT action head.
        
        返回的是 logits，不是 softmax 概率！
        最后一维的 3 对应三个类别:
            - dim[..., 0]: 类别 0 的 logit (对应离散值 -1, 后退)
            - dim[..., 1]: 类别 1 的 logit (对应离散值  0, 不动)
            - dim[..., 2]: 类别 2 的 logit (对应离散值 +1, 前进)
        
        Args:
            pons_output: Output from Pons adapter
                         Shape: (batch_size, pons_seq_len, hidden_dim)
        
        Returns:
            Action logits of shape (batch_size, chunk_size, action_dim, 3)
            
            训练时: 直接传入 CrossEntropyLoss
            推理时: argmax(logits, dim=-1) - 1 得到离散值 {-1, 0, 1}
        """
        batch_size, pons_seq_len, hidden_dim = pons_output.shape
        device = pons_output.device
        
        # Step 1: Initialize modules if not done yet (lazy initialization)
        if not self._initialized:
            self._build_modules(hidden_dim)
            self.to(device)
        
        # Verify hidden_dim matches
        if hidden_dim != self._hidden_dim:
            raise ValueError(
                f"Pons output hidden dim ({hidden_dim}) doesn't match "
                f"action head hidden dim ({self._hidden_dim})"
            )
        
        # Step 2: Prepare query tokens with position encoding
        # q_seq_len = CHUNK_SIZE * ACTION_DIM
        # Expand to batch size: (1, q_seq_len, D) -> (B, q_seq_len, D)
        query = self.query_tokens.expand(batch_size, -1, -1)
        # 添加可学习的位置编码，使模型能够区分不同位置的 query tokens
        query = query + self.query_pos_encoding
        
        # Step 3: Cross-attention to Pons output
        # Q queries Pons K/V
        query = self.cross_attention(query, pons_output)
        
        # Step 4: Pass through self-attention blocks
        for sa_block in self.self_attention_blocks:
            query = sa_block(query)
        
        # Step 5: Pass through MLP layers
        x = query  # (B, q_seq_len, hidden_dim)
        for mlp_layer in self.mlp_layers:
            x = mlp_layer(x)
        
        # x shape: (B, chunk_size * action_dim, 3)
        # 注意: 这是 logits，不经过 softmax
        
        # Step 6: Reshape to (B, chunk_size, action_dim, 3)
        output = x.view(batch_size, self.chunk_size, self.action_dim, 3)
        
        return output
    
    def predict_discrete_action(self, pons_output: torch.Tensor) -> torch.Tensor:
        """
        推理专用: 从 Pons 输出预测离散动作值 {-1, 0, 1}
        
        流程:
        1. forward() 得到 logits: (B, chunk, action_dim, 3)
        2. argmax 得到类别索引: (B, chunk, action_dim)，值为 {0, 1, 2}
        3. 类别索引 - 1 = 离散值 {-1, 0, 1}
        
        Args:
            pons_output: Output from Pons adapter
        
        Returns:
            Discrete action values of shape (batch_size, chunk_size, action_dim)
            值为 {-1, 0, 1}
        """
        logits = self.forward(pons_output)  # (B, chunk, action_dim, 3)
        class_idx = torch.argmax(logits, dim=-1)  # (B, chunk, action_dim)
        discrete_actions = class_idx - 1  # 映射: {0,1,2} -> {-1,0,1}
        return discrete_actions
    
    def predict_action(self, pons_output: torch.Tensor) -> torch.Tensor:
        """
        Alias for forward() for compatibility with other action heads.
        返回 logits (未经 softmax)
        
        Args:
            pons_output: Output from Pons adapter
        
        Returns:
            Action logits of shape (batch_size, chunk_size, action_dim, 3)
        """
        return self.forward(pons_output)
    
    def print_model_summary(self, print_details: bool = True) -> dict:
        """
        打印模型参数量统计信息
        
        详细显示各个组件的参数量，包括:
        - Query Tokens 和 Position Encoding
        - Cross-Attention Block
        - Self-Attention Blocks
        - MLP Layers
        
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
        # query_tokens: (1, q_seq_len, hidden_dim) = (1, chunk_size * action_dim, hidden_dim)
        query_tokens_params = self.query_tokens.numel()
        # query_pos_encoding: (1, q_seq_len, hidden_dim)
        query_pos_encoding_params = self.query_pos_encoding.numel()
        
        # =====================================================================
        # 2. Cross-Attention Block 参数量
        # =====================================================================
        cross_attn_total, cross_attn_trainable = count_params(self.cross_attention)
        
        # =====================================================================
        # 3. Self-Attention Blocks 参数量
        # =====================================================================
        self_attn_total = 0
        self_attn_trainable = 0
        for sa_block in self.self_attention_blocks:
            t, tr = count_params(sa_block)
            self_attn_total += t
            self_attn_trainable += tr
        
        # =====================================================================
        # 4. MLP Layers 参数量
        # =====================================================================
        mlp_total = 0
        mlp_trainable = 0
        for mlp_layer in self.mlp_layers:
            t, tr = count_params(mlp_layer)
            mlp_total += t
            mlp_trainable += tr
        
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
            "cross_attention": cross_attn_total,
            "self_attention_blocks": self_attn_total,
            "mlp_layers": mlp_total,
        }
        
        if print_details:
            print("\n" + "=" * 70)
            print("ParaCAT Action Head 参数量统计")
            print("=" * 70)
            
            # 配置信息
            print(f"\n📋 模型配置:")
            print(f"   • chunk_size: {self.chunk_size}")
            print(f"   • action_dim: {self.action_dim}")
            print(f"   • q_seq_len: {self.q_seq_len} (= chunk_size × action_dim = {self.chunk_size} × {self.action_dim})")
            print(f"   • hidden_dim: {self._hidden_dim}")
            print(f"   • num_heads: {self.num_heads}")
            print(f"   • num_transformer_blocks: {self.num_transformer_blocks}")
            print(f"   • num_mlp_layers: {self.num_mlp_layers}")
            print(f"   • mlp_expand_dim: {self.mlp_expand_dim}")
            
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
            
            # Cross-Attention Block
            print(f"\n2️⃣ Cross-Attention Block (× 1):")
            print(f"   • 总参数量:           {format_num(cross_attn_total):>15} params")
            print(f"     包含: MultiheadAttention + LayerNorm + FFN")
            
            # Self-Attention Blocks
            print(f"\n3️⃣ Self-Attention Blocks (× {self.num_transformer_blocks}):")
            if len(self.self_attention_blocks) > 0:
                single_sa_params, _ = count_params(self.self_attention_blocks[0])
                print(f"   • 单个 block:         {format_num(single_sa_params):>15} params")
                print(f"   • 总计 ({self.num_transformer_blocks} blocks):    {format_num(self_attn_total):>15} params")
                print(f"     计算: {format_num(single_sa_params)} × {self.num_transformer_blocks} = {format_num(self_attn_total)}")
            
            # MLP Layers
            print(f"\n4️⃣ MLP Layers (× {self.num_mlp_layers}):")
            for i, mlp_layer in enumerate(self.mlp_layers):
                mlp_params, _ = count_params(mlp_layer)
                print(f"   • MLP Layer {i+1}:        {format_num(mlp_params):>15} params")
            print(f"   ─────────────────────────────────────────")
            print(f"   小计:                 {format_num(mlp_total):>15} params")
            
            # 总计
            print(f"\n" + "=" * 70)
            print(f"📈 总计:")
            print(f"   • 总参数量:           {format_num(total_params):>15} params")
            print(f"   • 可训练参数量:       {format_num(trainable_params):>15} params")
            print(f"   • 参数量 (MB):        {total_params * 4 / 1024 / 1024:>15.2f} MB (float32)")
            print(f"   • 参数量 (MB):        {total_params * 2 / 1024 / 1024:>15.2f} MB (float16/bf16)")
            print("=" * 70)
            
            # 验证总和
            computed_total = query_subtotal + cross_attn_total + self_attn_total + mlp_total
            if computed_total == total_params:
                print(f"✓ 参数量验证通过: {format_num(query_subtotal)} + {format_num(cross_attn_total)} + {format_num(self_attn_total)} + {format_num(mlp_total)} = {format_num(total_params)}")
            else:
                print(f"⚠️ 参数量不匹配: 计算值 {format_num(computed_total)} ≠ 实际值 {format_num(total_params)}")
        
        return result


def create_paracat_action_head(
    chunk_size: int,
    action_dim: int,
    hidden_dim: Optional[int] = None,
    num_transformer_blocks: int = 2,
    num_mlp_layers: int = 2,
    mlp_expand_dim: int = 1024,
    num_heads: int = 8,
    **kwargs
) -> ParaCATActionHead:
    """
    Factory function to create a ParaCAT Action Head.
    
    Args:
        chunk_size: Size of action chunk (required)
        action_dim: Dimension of action space (required)
        hidden_dim: Hidden dimension. If None, auto-inferred from Pons output.
        num_transformer_blocks: Number of self-attention blocks (default: 2)
        num_mlp_layers: Number of MLP layers (default: 2)
        mlp_expand_dim: MLP intermediate expansion dimension (default: 1024)
        num_heads: Number of attention heads (default: 8)
        **kwargs: Additional arguments
    
    Returns:
        Configured ParaCATActionHead instance
    
    Example:
        >>> action_head = create_paracat_action_head(
        ...     chunk_size=16,
        ...     action_dim=7,
        ...     num_transformer_blocks=4,
        ...     num_mlp_layers=3,
        ...     mlp_expand_dim=2048
        ... )
        >>> # hidden_dim auto-inferred from first forward pass
        >>> output = action_head(pons_output)  # (B, 16, 7, 3)
    """
    return ParaCATActionHead(
        chunk_size=chunk_size,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_transformer_blocks=num_transformer_blocks,
        num_mlp_layers=num_mlp_layers,
        mlp_expand_dim=mlp_expand_dim,
        num_heads=num_heads,
        **kwargs
    )


# ============================================================================
# Testing and Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ParaCAT Action Head - Testing and Example Usage")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Test parameters
    batch_size = 4
    pons_seq_len = 64  # Pons output sequence length
    hidden_dim = 2560  # Pons output hidden dim
    chunk_size = 16
    action_dim = 7
    
    # -------------------------------------------------------------------------
    # Test 1: Auto-inferred hidden_dim
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 1: Auto-inferred hidden_dim")
    print("-" * 70)
    
    action_head = create_paracat_action_head(
        chunk_size=chunk_size,
        action_dim=action_dim,
        num_transformer_blocks=2,
        num_mlp_layers=3,
        mlp_expand_dim=1024
    ).to(device)
    
    # Simulate Pons output
    pons_output = torch.randn(batch_size, pons_seq_len, hidden_dim, device=device)
    
    print(f"Input Pons output shape: {pons_output.shape}")
    print(f"Expected Q tokens: chunk_size * action_dim = {chunk_size} * {action_dim} = {chunk_size * action_dim}")
    
    with torch.no_grad():
        output = action_head(pons_output)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: ({batch_size}, {chunk_size}, {action_dim}, 3)")
    assert output.shape == (batch_size, chunk_size, action_dim, 3), "Shape mismatch!"
    print("✓ Test 1 passed!")
    
    # -------------------------------------------------------------------------
    # Test 2: Explicit hidden_dim
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 2: Explicit hidden_dim")
    print("-" * 70)
    
    action_head_explicit = create_paracat_action_head(
        chunk_size=8,
        action_dim=6,
        hidden_dim=1024,
        num_transformer_blocks=3,
        num_mlp_layers=2,
        mlp_expand_dim=512
    ).to(device)
    
    pons_output_2 = torch.randn(batch_size, 32, 1024, device=device)
    
    print(f"Input Pons output shape: {pons_output_2.shape}")
    
    with torch.no_grad():
        output_2 = action_head_explicit(pons_output_2)
    
    print(f"Output shape: {output_2.shape}")
    print(f"Expected output shape: ({batch_size}, 8, 6, 3)")
    assert output_2.shape == (batch_size, 8, 6, 3), "Shape mismatch!"
    print("✓ Test 2 passed!")
    
    # -------------------------------------------------------------------------
    # Test 3: Different configurations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 3: Different configurations")
    print("-" * 70)
    
    configs = [
        {"chunk_size": 4, "action_dim": 3, "num_transformer_blocks": 1, "num_mlp_layers": 1},
        {"chunk_size": 32, "action_dim": 8, "num_transformer_blocks": 4, "num_mlp_layers": 4},
        {"chunk_size": 10, "action_dim": 5, "num_transformer_blocks": 2, "num_mlp_layers": 2},
    ]
    
    for config in configs:
        action_head_config = create_paracat_action_head(
            hidden_dim=512,
            mlp_expand_dim=256,
            **config
        ).to(device)
        
        pons_out = torch.randn(2, 16, 512, device=device)
        
        with torch.no_grad():
            out = action_head_config(pons_out)
        
        expected_shape = (2, config["chunk_size"], config["action_dim"], 3)
        print(f"  Config: chunk={config['chunk_size']}, action_dim={config['action_dim']}, "
              f"blocks={config['num_transformer_blocks']}, mlp_layers={config['num_mlp_layers']} "
              f"-> output: {out.shape}")
        assert out.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {out.shape}"
    
    print("✓ Test 3 passed!")
    
    # -------------------------------------------------------------------------
    # Test 4: predict_action alias
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 4: predict_action alias")
    print("-" * 70)
    
    action_head_alias = create_paracat_action_head(
        chunk_size=16,
        action_dim=7,
        hidden_dim=hidden_dim
    ).to(device)
    
    with torch.no_grad():
        output_alias = action_head_alias.predict_action(pons_output)
    
    print(f"Output shape from predict_action: {output_alias.shape}")
    assert output_alias.shape == (batch_size, 16, 7, 3), "Shape mismatch!"
    print("✓ Test 4 passed!")
    
    # -------------------------------------------------------------------------
    # Test 5: Gradient flow
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("Test 5: Gradient flow")
    print("-" * 70)
    
    action_head_grad = create_paracat_action_head(
        chunk_size=chunk_size,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        num_transformer_blocks=2,
        num_mlp_layers=2
    ).to(device)
    
    pons_input = torch.randn(batch_size, pons_seq_len, hidden_dim, device=device, requires_grad=True)
    
    output_grad = action_head_grad(pons_input)
    loss = output_grad.sum()
    loss.backward()
    
    print(f"Pons input gradient exists: {pons_input.grad is not None}")
    print(f"Pons input gradient shape: {pons_input.grad.shape}")
    assert pons_input.grad is not None, "No gradient!"
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
    action_head.print_model_summary()

