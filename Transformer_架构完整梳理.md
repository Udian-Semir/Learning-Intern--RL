# Transformer 架构完整梳理 (从原理到前沿, 2025版)

> 目标: 从零理解 Transformer 是什么、为什么有效、以及你在 sai_0 / π₀ / ACT 代码中遇到的各种变体。

---

## 目录

1. [Why Transformer? — 为什么需要 Transformer](#1)
2. [核心组件逐层拆解](#2)
3. [三大范式: Encoder-Only / Decoder-Only / Encoder-Decoder](#3)
4. [Vision Transformer (ViT) — Transformer 如何"看"图像](#4)
5. [DiT (Diffusion Transformer) — Transformer 如何"生成"连续信号](#5)
6. [VLA 中的 Transformer — sai_0 / π₀ / ACT 的具体架构](#6)
7. [VLA 三架构的显存和计算对比](#7)
8. [高效 Transformer 变体 (2023-2025)](#8)
9. [最新前沿 (2023-2025)](#9)

> 📖 阅读建议:
> - 第 1-3 章是基础, 建议按顺序阅读
> - 第 4-6 章可以跳读, 关注和你项目相关的部分
> - 每章末尾有 `【在你的项目中】` 小节, 连接到 sai_0 / π₀ / ACT 的实际代码

---

<a name="1"></a>
## 1. Why Transformer? — 为什么需要 Transformer

### 1.1 在 Transformer 之前

**RNN (循环神经网络)**: 逐个 token 处理序列, 每个 token 看前一个的"记忆"
- 问题 1: 无法并行 — token 1 算完才能算 token 2, GPU 闲置
- 问题 2: 长序列梯度消失 — 第 100 个 token 和第 1 个 token 之间传 100 层, 梯度趋近于 0

**CNN (卷积神经网络)**: 用固定大小的"窗口"扫过序列
- 问题: 窗口大小固定, 无法捕获远距离依赖 (窗口外看不到)

**Transformer 的解决方案**: 让每个 token **同时看到所有其他 token**, 用可学习的"注意力权重"决定该关注谁。

### 1.2 Transformer 的核心洞察

```
RNN:    token1 → token2 → token3 → ... → tokenN  (串行步骤 = N)
CNN:    [窗口3个]扫过序列                            (远距离需要 log_k(N) 层)
Transformer: 每个 token ↔ 所有其他 token  (并行, 1步)
```

**本质**: Transformer 把"序列建模"变成了"集合建模" — 每个 token 评估它和所有其他 token 的相关性, 然后加权聚合信息。

**代价**: O(N²) 的计算和显存, 如果 N=1000 tokens, 注意力矩阵为 1000×1000 = 1M 个元素。这就是为什么 VLA (视觉 token 多) 比纯 LLM 更吃显存的原因之一。

> **【在你的项目中】**
> sai_0 的 DiT: 150 个视觉 token + 16 个动作 token = 166 个 token, 注意力矩阵 = 166×166 ≈ 27k 元素 — 还好。
> π₀ 的 PaliGemma: ~200 个视觉 token + 50 个文本 token + 50 个动作 token = 300 个 token, 注意力矩阵 ≈ 90k 元素 — 更大但仍可管理。

---

<a name="2"></a>
## 2. 核心组件逐层拆解

### 2.1 Input Embedding — 把一切变成向量

Transformer 只能处理向量。所有输入必须先"嵌入"(embed)到一个固定维度的向量空间。

```python
# 文本: "pick up the apple" → token IDs → embedding matrix lookup
token_ids = [2456, 389, 5, 12034]  # 词汇表中的编号
embeddings = embedding_table[token_ids]  # (4, 512) 每个词 → 512维向量

# 图像: 224×224×3 → 切成 14×14 的 patch → 256 个 patch
# 每个 patch → 线性投影 → 512维向量
patches = image_to_patches(image)  # (256, 14*14*3) = (256, 588)
image_embeddings = linear_projection(patches)  # (256, 512)

# 机器人状态: [0.5, 0.0, 0.2, ...] → 线性层 → 512维向量
state_embedding = state_proj(state)  # (1, 512)
```

### 2.2 Positional Encoding — "第一个词"和"第二个词"有区别吗？

Transformer 没有内置的顺序概念。如果不加位置编码, "A 爱 B" 和 "B 爱 A" 看起来完全一样。

**解决方案**: 给每个位置的 embedding 加上一个独特的"位置签名"。

**Sinusoidal Positional Encoding (原始论文, 固定值)**:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**为什么用 sin/cos?** 因为 `PE(pos+k)` 可以表示为 `PE(pos)` 的线性函数 — 模型可以学到"相对位置" (token A 在 token B 前面 3 个位置)。

**Learned Positional Embedding (现代做法)**: 直接让模型学习位置嵌入 (nn.Embedding)。GPT 系列用这个。更灵活但无法外推到训练时没见过的长度。

**RoPE (Rotary Position Embedding, 2021, 当前主流)**:

```
RoPE 把位置信息直接"旋转"进 Q 和 K 向量:
Q_rotated = Q × rotation_matrix(pos_Q)
K_rotated = K × rotation_matrix(pos_K)

这使得 attention score (Q·K) 自然地依赖于 (pos_Q - pos_K) — 相对位置!
```

**优势**: 天然编码相对位置, 长度外推性好, 不需要额外参数。LLaMA、Gemma (π₀ 用的)、Qwen (sai_0 用的) 全都用 RoPE。

### 2.3 Self-Attention — "我应该关注谁?"

这是 Transformer 最核心的运算。每个 token 产生三个向量:

- **Query (Q)** — "我在找什么?" (当前 token 的需求)
- **Key (K)** — "我是什么?" (每个 token 的标签)
- **Value (V)** — "我有什么信息?" (每个 token 的内容)

**完整流程**:

```
Q = x · W_Q   # (N, d_k)
K = x · W_K   # (N, d_k)
V = x · W_V   # (N, d_k)

# Step 1: 计算注意力分数 (点积 + 缩放)
scores = Q @ K^T / sqrt(d_k)    # (N, N) 矩阵, 每个元素 = 第i个token对第j个token的"兴趣度"

# Step 2: Softmax 归一化 (把分数变成概率)
attention_weights = softmax(scores, dim=-1)  # 每行求和 = 1

# Step 3: 聚合 — 用注意力权重加权求和 V
output = attention_weights @ V   # (N, d_v)

# 一行代码总结:
Attention(Q,K,V) = softmax(Q·K^T / √d_k) · V
```

**直观理解**:
- tokens = ["我", "喜欢", "吃", "苹果"]
- 当处理"吃"时: Q("吃") 和 K("苹果") 的匹配度很高 → attention weight 大 → 输出包含很多 V("苹果") 的信息
- 当处理"吃"时: Q("吃") 和 K("我") 的匹配度低 → attention weight 小

### 2.4 Multi-Head Attention (MHA)

不是只做一次 Attention, 而是做 H 次 (H=8 或 16), 每次用不同的 W_Q/W_K/W_V:

```
head_i = Attention(x·W^Q_i, x·W^K_i, x·W^V_i)
MultiHead(x) = Concat(head_1, ..., head_H) · W_O
```

**为什么需要多头?** 不同 head 学习不同类型的"关注":
- Head 1: "动词和宾语之间的关系"
- Head 2: "形容词和被修饰名词之间的关系"
- Head 3: "代词和它指代对象之间的关系"
- ...

### 2.5 高效 Attention 变体 (本节关键! — 你在 π₀ 代码中看到的)

**MHA (原始)**: 每头都有独立的 Q、K、V, 显存占用 = H × (N × d_head × 3)

**Multi-Query Attention (MQA, 2019)**: 所有头共享 K 和 V, 只有 Q 是独立的。显存减半但可能损失质量。

**Grouped Query Attention (GQA, 2023)**: MHA 和 MQA 的折中 — K/V 在 G 组之间共享 (1 < G < H)。这是当前最主流方案。

| 变体 | Q 数量 | K/V 数量 | 参数量 | 在哪里用 |
|------|--------|---------|--------|---------|
| MHA | H | H | 最大 | ViT, BERT |
| MQA | H | 1 | 最小 | PaLM, Gemini 1.0 |
| **GQA** | H | G (如 2 或 4) | 中等 | **LLaMA 2/3, Gemma 2/3, π₀ 的 Gemma backbone** |

> **【在你的项目中】**
> π₀ 的 Gemma backbone 使用 MQA (num_kv_heads=1, num_heads=8), 这是 Multi-Query Attention。这意味着 KV Cache 只需要存 1 份而不是 8 份, 显存减为原来的 1/8。这是 π₀ 能在消费级 GPU 上做推理的关键设计之一。

### 2.6 Feed-Forward Network (FFN) — "每个 token 独立思考"

Attention 让 token 之间通信。FFN 让每个 token 独立处理信息:

```
FFN(x) = Linear_2(GELU(Linear_1(x)))
# 或 SwiGLU (现代标准):
FFN(x) = Linear_2(Swish(Linear_1(x)) ⊙ Linear_3(x))
```

**维度变化**: d_model → 4×d_model → d_model (典型扩张比 = 4)

**为什么需要 FFN?** Attention 只做线性加权。FFN 引入非线性变换, 让模型能学更复杂的模式。

### 2.7 Layer Normalization — "保持数值稳定"

深度网络训练时, 每层输出值的分布会漂移。LayerNorm 把每个 token 的向量归一化到均值 0 方差 1:

```
LayerNorm(x) = γ · (x - mean) / sqrt(var + ε) + β
```

**Pre-Norm vs Post-Norm**:
- Post-Norm (原始): Attention/FFN → Add → Norm (残差之后归一化)
- **Pre-Norm (现代标准)**: Norm → Attention/FFN → Add (归一化之后再计算)。训练更稳定, GPT-3/LLaMA/Gemma 都用 Pre-Norm

**RMSNorm (你会在 π₀ 代码中看到)**: LayerNorm 的简化版 — 只除以均方根, 不去减均值。更快。Gemma 系列用 RMSNorm。

**adaRMSNorm (π₀.₅ 的 Action Expert 用)**: 在 RMSNorm 基础上, 注入"时间步信息"(Flow Matching 的 τ)。每一层的归一化都会被 τ 调制:

```
adaRMSNorm(x, τ) = RMSNorm(x) × (1 + γ(τ)) + β(τ)
```

### 2.8 残差连接 — "信息高速公路"

```
output = LayerNorm(x + Attention(x))  # 保留输入, 加上 Attention 的贡献
output = LayerNorm(output + FFN(output))
```

**为什么需要?** 假设 Attention 层"啥都没学到", 输出为 0, 残差连接保证信息原样传过去 `x + 0 = x`。这让深层网络至少不比浅网络差。没有残差连接, 56 层的 Transformer 根本训练不了 (梯度消失)。

> **【在你的项目中】**
> sai_0 的 DiT 用的是 AdaLayerNorm (自适应 LayerNorm), π₀ 用的是 RMSNorm。两者都是归一化变体, 但 AdaLayerNorm 可以注入时间步信息 (类似 adaRMSNorm)。这就是为什么 sai_0 的 Action Head 能感知 "去噪到哪一步了"。

---

<a name="3"></a>
## 3. 三大范式: Encoder-Only / Decoder-Only / Encoder-Decoder

### 3.1 Encoder-Only (BERT 系列, ~2018)

```
Input → [Encoder × N] → Output (每个位置的表示)
         ↑ 双向注意力
```

- 每个 token 可以看左右两侧的所有 token
- **用途**: 理解任务 — 分类、序列标注、句子相似度
- **代表**: BERT, RoBERTa
- **在 VLA 中**: sai_0 的 DiT (action 和 VLM features 之间)、ACT 的 CVAE encoder

### 3.2 Decoder-Only (GPT 系列, ~2018-现在, 绝对主流)

```
Input → [Decoder × N] → Output (下一个 token 的预测)
         ↑ 因果注意力 (只能看左边的)
```

- 每个 token 只能看它自己和前面的 token
- **用途**: 生成任务 — 对话、写作、代码生成
- **代表**: GPT-3/4, LLaMA, Gemma, Qwen, Claude
- **在 VLA 中**: π₀ 的 PaliGemma backbone

### 3.3 Encoder-Decoder (原始 Transformer, ~2017)

```
Input → [Encoder × N] → [Decoder × N] → Output
         ↑ 双向注意力        ↑ 因果 + 交叉注意力
```

- Encoder 处理输入 (双向), Decoder 生成输出 (因果, Cross-Attention 看 Encoder 输出)
- **用途**: 翻译、摘要
- **代表**: T5, BART, 原始 Transformer
- **在 VLA 中**: ACT 的 Transformer Decoder (Image encoder → cross-attention → action decoder)、π₀-small 的非 VLM 基线

### 3.4 为什么 Decoder-Only 成了主流?

| | Encoder-Only | Decoder-Only |
|---|---|---|
| 训练效率 | 需要特殊预训练任务 (MLM) | 自然的下一个 token 预测 |
| 零样本能力 | 弱 | 强 — 下一个 token 预测就是"做题" |
| 规模扩展 | 一般 | 极好 — LLaMA 405B 仍然稳定训练 |
| VLA 适配 | 需改造 (如添加 Cross-Attn) | 自然 — 输入观测 → 预测动作 token |

**在 VLA 中**: π₀ 用的是 Decoder-Only (PaliGemma 本质是 Gemma, 一个 Decoder-Only LM), 只是对动作 token 用了双向注意力 (而非因果)。

---

<a name="4"></a>
## 4. Vision Transformer (ViT) — Transformer 如何"看"图像

### 4.1 核心思路: 把图像切成"patch"

```
224×224×3 的图像
→ 切成 16×16 的 patch (每个 patch = 16×16×3 = 768 个像素值)
→ 196 个 patch
→ 每个 patch 通过线性层变成 768 维向量 (patch embedding)
→ 加上位置编码
→ 送入标准 Transformer Encoder
```

**ViT 做的事**: 把"图像识别"变成"序列建模" — 每个 patch 就像一个"词"。

### 4.2 SigLIP (π₀ 用的视觉编码器)

SigLIP = Sigmoid Loss for Language-Image Pre-training。比 CLIP 更好的图像-文本对齐模型。

```
图像 → ViT → 图像特征 (256 个 patch embeddings, 各 1152 维)
文本 → Transformer → 文本特征

训练目标: 成对的 (图像, 文本) 的相似度 > 不成对的 — 用 Sigmoid loss
```

在 π₀ 中: SigLIP 输出 256 个 token (So400m/14 变体, 256 patch × 1152 维), 给 PaliGemma 做前缀。

### 4.3 为什么 VLA 中不微调 ViT?

和"为什么不微调 VLM"一样的原因:
- ViT 的预训练包含了大量通用视觉知识 (物体形状、纹理、空间关系)
- 微调可能导致过拟合到机器人训练集, 丢失泛化能力
- 冻结 ViT + 只训练下游层的策略在 VLA 中效果最好

---

<a name="5"></a>
## 5. DiT (Diffusion Transformer) — Transformer 如何"生成"连续信号

### 5.1 DiT 是什么?

DiT = Diffusion Transformer (Peebles & Xie, ICCV 2023)。把 UNet (传统 Diffusion 的 backbone) 替换成 **纯 Transformer**。

```
传统 Diffusion (Stable Diffusion 1/2):  UNet (CNN-based)
现代 Diffusion (SD3, Sora, sai_0):      DiT (Transformer-based)
```

### 5.2 DiT 架构

```
noisy_action (B, H, action_dim)  +  timestep t
        │                              │
        ▼                              ▼
   Linear Projection              Time Embedding (sin/cos → MLP)
        │                              │
        ├──────── concat ──────────────┘
        ▼
   [DiT Block × N]:
     ├─ AdaLayerNorm (用 timestep 做 scale + shift)
     ├─ Self-Attention (动作序列内部)
     ├─ Cross-Attention (动作 token 看 VLM features)  ← 关键!
     └─ FeedForward
        │
        ▼
   Linear Projection → pred_velocity (B, H, action_dim)
```

> **【数学推导】AdaLayerNorm 如何注入时间步**
>
> 原始 LayerNorm: `y = γ·(x - μ)/σ + β` (γ 和 β 是学习的参数, 固定值)
>
> AdaLayerNorm: `y = γ(t)·(x - μ)/σ + β(t)`
> ```
> 其中 γ(t) = 1 + Linear(t_embed)         # scale 调制
>       β(t) = Linear(t_embed)             # shift 调制
> ```
>
> 每一层 DiT Block 的归一化参数都由时间步 t 动态决定。当 t→1 (低噪声) 时, 模型输出的 velocity 应该趋于 0 (因为已经接近真实动作), AdaLayerNorm 帮助模型在各个噪声水平表现出不同行为。

### 5.3 Cross-Attention 在 DiT 中的关键作用

```
动作 token (Query):     "我想生成第 5 步的抓取动作"
VLM features (Key/Value): [桌子区域, 苹果区域, 夹爪区域, ...]
Cross-Attention:         动作 token 从 VLM features 中"查找"苹果在哪里
```

这和 UNet 用 Cross-Attention 看文本 embedding 是一个原理。在 sai_0 中, Cross-Attention 让动作 token "看到" VLM 对场景的理解。

---

<a name="6"></a>
## 6. VLA 中的 Transformer — sai_0 / π₀ / ACT 的具体架构

### 6.1 sai_0 (Flow_Matching_1)

```
输入:
  图像 → Qwen3-VL (冻结) → VLM hidden states (B, S, 1536)
  状态 → MLP (Embodiment-specific) → state token (B, 1, 256)
  噪声动作 → Action Encoder (+time embedding) → action tokens (B, 16, 256)
  Future tokens → learned embedding → (B, 32, 256)

序列: [state][future][action] → (B, 49, 256)
    ↓
DiT (Transformer Decoder, 双向注意力):
  ├─ Self-Attention (序列内部)
  ├─ Cross-Attention (序列 → VLM features)
  └─ AdaLayerNorm (时间步调制)
    ↓
Action Decoder → velocity prediction (B, 16, 7)
```

**参数**: ~200M (仅 Action Head, VLM 不计)

### 6.2 π₀

```
输入:
  图像 → SigLIP (冻结) → 256 image tokens
  文本 → tokenizer → ~50 text tokens
  状态 → linear → state token
  噪声动作 → Action Expert → action tokens (50个)
  时间步 τ → Action Expert 内 adaRMSNorm

序列: [image tokens] [text tokens] [state][action tokens]
       ← PaliGemma (2B) →  ← Action Expert (300M) →

PaliGemma 和 Action Expert 是两组权重,
只在 Self-Attention 层交互 (共享 QKV 投影)
MLP 层各自独立

输出: velocity field → Euler 10步 → action (B, 50, 32)
```

**参数**: ~3.3B (PaliGemma 2B + Action Expert 300M)

### 6.3 ACT (Action Chunking with Transformers)

```
输入:
  4×图像 → ResNet18 → flatten → 1200 image tokens
  关节状态 → linear → state token
  z (风格变量, 32维) → linear → z token
  动作序列 → linear → action tokens (100个)

CVAE Encoder (训练时用, 推理时丢弃):
  [CLS] + state + action → Transformer → z_mean, z_std

CVAE Decoder (Policy):
  image tokens + state + z → Transformer Encoder (4层 self-attn)
    → Transformer Decoder (7层 cross-attn, Q=position emb, KV=encoder output)
    → MLP → action (100 × 14)
```

**参数**: ~80M (全部可训练, 没有 VLM)

### 6.4 三架构对比

| | sai_0 | π₀ | ACT |
|---|---|---|---|
| VLM | Qwen3-VL (冻结) | PaliGemma (冻结) | ResNet18 (从头训) |
| Action Head | DiT | Action Expert (Gemma子集) | CVAE + Transformer |
| 训练参数量 | ~200M | ~300-860M | ~80M |
| 时间步注入 | DiT (层间) | π₀: 拼接, π₀.₅: adaRMSNorm (层内) | 不需要 (CVAE 不是 diffusion) |
| 位置编码 | Learned PE | RoPE (Gemma自带) | 2D Sinusoidal (图像) + Learned (动作) |
| 注意力 | 双向 (动作块内) | 块因果 (PaliGemma块只能看自己) | 双向 (Encoder) + 交叉 (Decoder) |
| 归一化 | AdaLayerNorm | RMSNorm / adaRMSNorm | LayerNorm |

---

<a name="7"></a>
## 7. VLA 三架构的显存和计算对比

### 7.1 推理时显存分解 (批大小=1, bfloat16)

| 组件 | sai_0 (Qwen2B) | π₀ (PaliGemma2B) | ACT |
|------|---------------|-----------------|-----|
| VLM/ViT 权重 | 4 GB | 4 GB | 0.16 GB (ResNet18) |
| VLM KV Cache | ~0.1 GB (150 tokens) | ~0.2 GB (300 tokens) | 无 (图像直接展平) |
| Action Head 权重 | ~0.4 GB | ~0.6 GB | ~0.16 GB |
| Action Head 中间激活 | ~0.5 GB | ~1.0 GB | ~0.3 GB |
| **总计 (推理)** | **~5 GB** | **~6 GB** | **~0.6 GB** |

### 7.2 训练时显存对比 (批大小=32, bfloat16)

| 组件 | sai_0 | π₀ (LoRA) | ACT |
|------|-------|----------|-----|
| 模型权重 | 2.2B × 2B = 4.4 GB | 3.3B × 2B = 6.6 GB | 80M × 2B = 0.16 GB |
| 梯度 | 0 (VLM冻结) + ~0.8 GB (AH) | ~0 GB (LoRA only) | 0.32 GB |
| 优化器状态 | ~1.6 GB | ~0 GB (LoRA only) | 0.64 GB |
| 中间激活 (B=32) | ~20 GB | ~15 GB | ~8 GB |
| **总计 (训练)** | **~27 GB** | **~22 GB** | **~9 GB** |
| 推荐 GPU | RTX 4090 (24GB, 刚好) | RTX 4090 (24GB, LoRA) | RTX 3060 (12GB) |

> **注**: π₀ 全量微调需要 ~70 GB (A100), 远超消费级 GPU。

---

<a name="8"></a>
## 8. 高效 Transformer 变体 (2023-2025)

### 8.1 FlashAttention (Dao et al., 2022, 2023, 2024)

**核心问题**: 标准 Attention 需要存整个 N×N 的注意力矩阵 (对 2048 tokens = 4M 元素 = 16 MB), 这是显存瓶颈。

**解决方案**: 把 Q、K、V 分块, 一块一块地算 Attention, 不存储中间结果。通过 IO-aware tiling + recomputation, 把原本 O(N²) 的显存降为 O(N)。

**影响**: 几乎所有 Transformer 实现都在用。v1 (2022) 支持纯 causal, v2 (2023) 支持训练中的任意 mask, v3 (2024) 支持 Hopper GPU 新特性。

### 8.2 Mixture of Experts (MoE)

不是所有参数都参与每次计算。每个 token 只激活部分"专家":

```
Input token → Router → 选择 top-2 experts (如 expert #3 和 #7)
                       ↓
          Expert #3(x) + Expert #7(x) → 加权合并
```

**实例**: Mixtral 8×7B (470亿参数, 但每次只用 130亿), GPT-4 (传言 8×220B)

**在 VLA 中**: π₀ 的 PaliGemma 和 Action Expert 本质上是 MoE 的 2-expert 特例 — 图像+文本 token 走 expert 0 (PaliGemma), 动作+状态 token 走 expert 1 (Action Expert)。

### 8.3 Speculative Decoding (2023)

用一个小模型"猜测"接下来 N 个 token, 然后大模型一次性验证 (而非逐个生成)。加速 2-3×, 完全不损失质量。

这不是模型架构变化, 而是推理策略。在 VLA 的离散 token 预测 (π₀.₅ 的 FAST 推理) 中可能有用。

### 8.4 RingAttention / Striped Attention (2023-2024)

把长序列拆分到多 GPU 上, 每个 GPU 算自己负责的 attention 块, 然后在 GPU 间通信。支持 1M+ token 上下文。

### 8.5 Linear Attention / Mamba / State Space Models (2023-2024)

用 O(N) 复杂度替代 Transformer 的 O(N²) Attention。

- **Mamba-1** (2023): 选择性 SSM, 在长序列上比 Transformer 更快
- **Mamba-2** (2024): 揭示 SSM 和 Attention 的深层数学联系
- **Jamba / Samba** (2024): Mamba + MoE 混合

**现状**: 对超长序列 (100k+ tokens) 有优势, 对短序列 (VLA 的 ~300 tokens) 无明显优势。VLA 领域仍以 Transformer 为主。

---

<a name="9"></a>
## 9. 最新前沿 (2023-2025)

### 9.1 Reasoning / Chain-of-Thought 训练 (o1, 2024)

OpenAI o1: 用强化学习训练模型"在内部做推理" — 产生不被用户看到的中间思考 token。这本质上仍是 Transformer, 但训练目标变了。

**在 VLA 中的对应**: π₀.₅ 的分层推理 — 先"想"子任务 ℓ̂, 再做动作 a。这是 CoT 在机器人领域的应用。

### 9.2 长上下文 (Gemini 1.5, 2024)

Gemini 1.5 Pro 支持 1M token 上下文。关键技术: RingAttention + 高质量长上下文训练数据。

**对 VLA 的启示**: 如果 VLA 能处理 10 分钟的观测历史, 而非当前的 1 帧, 可能实现更好的错误恢复。

### 9.3 多模态 Transformer (GPT-4V, Gemini, 2024)

统一的 Transformer 同时处理文本、图像、音频、视频。这和 VLA 的"文本+图像+动作"思路一脉相承。

### 9.4 Byte-Latent Transformer (2024)

不固定 tokenization, 而是让模型学习如何"动态地"将字节分组为 token。对 VLA 的含义: 可能消除"动作用离散还是连续表示"的纠结 — 让模型自己决定。

---

## 总结: VLA 工程师需要的 Transformer 知识地图

```
你的学习/复习顺序:

1. Attention 公式 (2.3) ← 必须理解
2. Positional Encoding + RoPE (2.2) ← 理解为什么 VLA 需要"知道位置"
3. Decoder-Only vs Encoder-Only (3.1/3.2) ← 理解 π₀ 和 sai_0 的架构差异
4. Multi-Query Attention (2.5) ← 理解 π₀ 的 KV Cache 优化
5. ViT (4) ← 理解图像怎么变成 token
6. DiT + AdaLayerNorm (5) ← 理解 sai_0 的 Action Head
7. MoE (8.2) ← 理解 π₀ 的 PaliGemma vs Action Expert 的关系
8. GQA / FlashAttention (8.1/2.5) ← 理解显存优化
```

**你不需要**:
- 手动实现 Attention (PyTorch/JAX 都有)
- 理解每个变体的所有细节 (先掌握基础, 细节随用随查)
- 记住 8.4-8.5 (长序列优化和 SSM, 对 VLA 的 ~300 token 序列不关键)