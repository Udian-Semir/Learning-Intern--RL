# VLM Hidden States 提取工具

## ⚠️ 注意混淆点：层号索引说明

### Qwen3-VL-2B 模型结构

```
(language_model): Qwen3VLTextModel(
  (embed_tokens): Embedding(151936, 2048)        # embedding 层
  (layers): ModuleList(
    (0-27): 28 x Qwen3VLTextDecoderLayer(...)   # transformer 层索引 0-27
  )
  (norm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
)
```

### `hidden_states` 的实际结构

当调用 `output_hidden_states=True` 时，返回的 `hidden_states` 是一个 tuple，包含 **29** 个元素：

| 索引 | 对应内容 |
|------|----------|
| `hidden_states[0]` | embedding 层输出 (`embed_tokens`) |
| `hidden_states[1]` | transformer layer 0 的输出 |
| `hidden_states[2]` | transformer layer 1 的输出 |
| ... | ... |
| `hidden_states[14]` | transformer layer 13 的输出 |
| `hidden_states[15]` | transformer layer 14 的输出 |
| ... | ... |
| `hidden_states[28]` | transformer layer 27 的输出 (最后一层) |

### 脚本参数 `--layers` 的映射

代码中 `--layers` 参数的范围是 **1-28**，直接作为 `hidden_states` 的索引：

```python
hidden_state = outputs.hidden_states[layer_idx]  # layer_idx 就是用户输入
```

| 用户输入 | hidden_states 索引 | 实际 transformer 层 |
|----------|-------------------|---------------------|
| `--layers 1` | `[1]` | layer 0 (第1层) |
| `--layers 14` | `[14]` | layer 13 (第14层) |
| `--layers 28` | `[28]` | layer 27 (最后一层) |

**简单理解**：用户输入的是"从1开始计数的层号"，范围 1-28 对应 transformer 的 layer 0-27。

---

## 输出 Shape 说明

### Embedding 层输出示例

```
hidden_states[0].shape = (batch_size, seq_len, hidden_dim)
                       = (batch_size, seq_len, 2048)
```

### 具体例子

假设输入 2 张 128×128 图像 + 一段文本指令：

```python
# 输入示例
# - 2张图像 (agentview + wrist)
# - 文本: "What action should the robot take to pick up the red cup?"

hidden_states[0].shape = (1, 592, 2048)
#                         ↑   ↑    ↑
#                      batch seq  hidden_dim

# seq_len = 592 的组成（大致）:
# - 系统 prompt tokens: ~20
# - 图像1 tokens: ~256 (取决于分辨率和patch大小)
# - 图像2 tokens: ~256
# - 用户文本 tokens: ~50
# - generation prompt tokens: ~10
```

### 各层输出 Shape 对比

```python
# Embedding 层输出
hidden_states[0].shape  = (1, 592, 2048)

# Transformer layer 0-27 输出 (shape 相同)
hidden_states[1].shape  = (1, 592, 2048)
hidden_states[14].shape = (1, 592, 2048)
hidden_states[28].shape = (1, 592, 2048)
```

**所有层的输出 shape 都是 `(batch, seq_len, 2048)`**，只是不同层的特征表示不同：
- 浅层（如 layer 1-5）：更接近词汇/局部特征
- 中层（如 layer 12-16）：语义特征
- 深层（如 layer 24-27）：更抽象的任务相关特征

---

## 使用方法

### 单层提取
```bash
python qwen/extract_vlm_hidden_states.py --dataset_path /path/to/dataset --layers 14
```

### 多层提取（在序列维度拼接）
```bash
python qwen/extract_vlm_hidden_states.py --dataset_path /path/to/dataset --layers 14,15,16
```

### 完整参数示例
```bash
python qwen/extract_vlm_hidden_states.py \
    --dataset_path /path/to/dataset \
    --layers 14,15,16 \
    --gpu 0 \
    --add_action_prompt \
    --flip_images
```

### 输出格式

| 模式 | 输出形状 |
|------|----------|
| 单层 (`--layers 14`) | `(1, seq_len, hidden_dim)` |
| 多层 (`--layers 14,15,16`) | `(3, seq_len, hidden_dim)` |

