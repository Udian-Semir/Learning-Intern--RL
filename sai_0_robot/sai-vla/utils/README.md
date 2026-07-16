# Utils 工具模块

## LeRobot 通用数据加载器 (lerobot_dataset_loader.py)

### 功能说明

这是一个**通用的 LeRobot 格式数据集加载器**，用于加载完整的机器人轨迹数据，供各个模块（VLM、Action Heads 等）使用。

**核心功能：**
1. 加载 LeRobot 标准格式的数据集（parquet + videos + vlm_hidden_states）
2. 支持多相机图像、VLM hidden states、机器人状态、动作等数据
3. 支持 Action Chunking（预测未来多步动作）
4. 自动适配单层和多层 VLM hidden states 格式
5. 提供高效的批处理和缓存机制

---

## 数据加载器输出格式

### 单个样本 (Dataset.__getitem__)

调用 `dataset[idx]` 返回一个字典，包含以下字段：

```python
{
    'images': {
        'observation.images.top': np.ndarray,         # (height, width, 3), uint8
        'observation.images.left_wrist': np.ndarray,  # (height, width, 3), uint8
        # ... 其他相机
    },
    'vlm_hidden_states': np.ndarray,  # (num_layers, seq_len, hidden_dim)
                                      # 例如: (3, 512, 2048) 表示 3 层 VLM，每层 512 tokens，每个 token 2048 维
    'observation_state': np.ndarray,  # (state_dim,)
                                      # 例如: (16,) 表示 16 维机器人状态（关节位置、速度等）
    'actions': np.ndarray,            # 根据 enable_chunking 决定形状：
                                      # - Chunking 启用: (num_chunks, action_dim)，例如 (25, 16)
                                      # - Chunking 禁用: (action_dim,)，例如 (16,)
    'task_description': str or None,  # 任务描述文本，例如 "Pick up an apple."
    'episode_index': int,             # Episode 索引
    'frame_index': int,               # 当前帧在 episode 中的索引
    'vlm_index': int,                 # VLM hidden state 文件索引
}
```

### 批处理数据 (DataLoader)

使用 `create_lerobot_dataloader()` 创建的 DataLoader 返回批处理后的数据：

```python
batch = next(iter(dataloader))
# batch 结构:
{
    'images': {
        'observation.images.top': torch.Tensor,        # (batch_size, height, width, 3), float32, 范围 [0, 255] (uint8 转 float32，未归一化)
        'observation.images.left_wrist': torch.Tensor, # (batch_size, height, width, 3), float32, 范围 [0, 255] (uint8 转 float32，未归一化)
        # ... 其他相机
        # 注意: VLM 模块会根据各自需求进行归一化（如 Eagle 归一化到 [-1, 1]）
    },
    'vlm_hidden_states': torch.Tensor,  # (batch_size, num_layers, seq_len, hidden_dim)
                                        # 例如: (4, 3, 512, 2048) 表示 batch_size=4
    'observation_state': torch.Tensor,  # (batch_size, state_dim)
                                        # 例如: (4, 16)
    'actions': torch.Tensor,            # (batch_size, num_chunks, action_dim) 或 (batch_size, action_dim)
                                        # 例如: (4, 25, 16)
    'task_description': List[str],      # 批次中每个样本的任务描述
                                        # 例如: ["Pick up an apple.", "Pick up an apple.", ...]
    'episode_index': torch.Tensor,      # (batch_size,), int64
    'frame_index': torch.Tensor,        # (batch_size,), int64
    'vlm_index': torch.Tensor,          # (batch_size,), int64
}
```

---

## 详细字段说明

### 1. `images` (多相机图像)

- **类型**: `dict[str, np.ndarray]` (单样本) 或 `dict[str, torch.Tensor]` (批处理)
- **格式**: 
  - 单样本: `(height, width, 3)`, uint8, RGB 格式，范围 [0, 255]
  - 批处理: `(batch_size, height, width, 3)`, float32, 范围 [0, 255] (uint8 转 float32，**未归一化**)
- **说明**: 
  - 包含所有相机视角的图像，key 名称由数据集决定（如 `observation.images.top`）
  - **重要**: 数据加载器不进行归一化，VLM 模块会根据各自需求进行预处理
  - 例如 Eagle 会归一化到 [-1, 1]，Qwen 有自己的归一化方式

**示例：**
```python
# 单样本
images = sample['images']
top_camera = images['observation.images.top']  # (800, 1280, 3), uint8

# 批处理
images = batch['images']
top_camera_batch = images['observation.images.top']  # (4, 800, 1280, 3), float32
```

### 2. `vlm_hidden_states` (VLM 隐藏层状态)

- **类型**: `np.ndarray` (单样本) 或 `torch.Tensor` (批处理)
- **形状**: 
  - 单样本: `(num_layers, seq_len, hidden_dim)`
  - 批处理: `(batch_size, num_layers, seq_len, hidden_dim)`
- **说明**: 
  - `num_layers`: VLM 提取的隐藏层数量（例如: 3）
  - `seq_len`: 序列长度，即 token 数量（例如: 512）
  - `hidden_dim`: 每个 token 的特征维度（例如: 2048）
  - **自动适配**: 数据集可能存储单层 `(seq_len, hidden_dim)` 或多层格式，加载器会自动扩展为统一的多层格式

**示例：**
```python
# 单样本: (3, 512, 2048)
vlm = sample['vlm_hidden_states']
layer_0 = vlm[0]  # 第 0 层: (512, 2048)
layer_1 = vlm[1]  # 第 1 层: (512, 2048)
layer_2 = vlm[2]  # 第 2 层: (512, 2048)

# 批处理: (4, 3, 512, 2048)
vlm_batch = batch['vlm_hidden_states']
batch_layer_0 = vlm_batch[:, 0, :, :]  # (4, 512, 2048)
```

### 3. `observation_state` (机器人本体感知状态)

- **类型**: `np.ndarray` (单样本) 或 `torch.Tensor` (批处理)
- **形状**: 
  - 单样本: `(state_dim,)`
  - 批处理: `(batch_size, state_dim)`
- **说明**: 机器人的当前状态，通常包括关节位置、速度、力矩等
  - `state_dim`: 状态维度（例如: 16 表示 16 自由度机械臂）

**示例：**
```python
# 单样本: (16,)
state = sample['observation_state']  # [q1, q2, ..., q16]

# 批处理: (4, 16)
state_batch = batch['observation_state']
```

### 4. `actions` (机器人动作)

- **类型**: `np.ndarray` (单样本) 或 `torch.Tensor` (批处理)
- **形状**: **取决于是否启用 Action Chunking**
  - **Chunking 启用** (`enable_chunking=True`):
    - 单样本: `(num_chunks, action_dim)`
    - 批处理: `(batch_size, num_chunks, action_dim)`
  - **Chunking 禁用** (`enable_chunking=False`):
    - 单样本: `(action_dim,)`
    - 批处理: `(batch_size, action_dim)`
- **说明**: 
  - `num_chunks`: 预测的未来动作步数（例如: 25）
  - `action_dim`: 动作维度（例如: 16）

**示例：**
```python
# Chunking 启用
actions = sample['actions']  # (25, 16) - 预测未来 25 步，每步 16 维
actions_batch = batch['actions']  # (4, 25, 16)

# Chunking 禁用
actions = sample['actions']  # (16,) - 只预测当前步
actions_batch = batch['actions']  # (4, 16)
```

### 5. `task_description` (任务描述)

- **类型**: `str` 或 `None` (单样本) / `List[str]` (批处理)
- **说明**: 任务的自然语言描述，用于提示 VLM

**示例：**
```python
# 单样本
task = sample['task_description']  # "Pick up an apple."

# 批处理
tasks = batch['task_description']  # ["Pick up an apple.", "Pick up an apple.", ...]
```

### 6. 元数据字段

- `episode_index`: Episode 索引 (int / torch.Tensor)
- `frame_index`: 当前帧索引 (int / torch.Tensor)
- `vlm_index`: VLM hidden state 文件索引 (int / torch.Tensor)

---

## 使用示例

### 基本用法

```python
from lerobot_dataset_loader import create_lerobot_dataloader

# 创建 DataLoader
dataloader = create_lerobot_dataloader(
    dataset_path="/path/to/lerobot/dataset",
    batch_size=4,
    num_workers=4,
    shuffle=True,
    num_action_chunks=25,
    enable_chunking=True,
)

# 迭代数据
for batch in dataloader:
    # 提取数据
    images = batch['images']  # dict of tensors
    vlm_hidden_states = batch['vlm_hidden_states']  # (batch_size, num_layers, seq_len, hidden_dim)
    observation_state = batch['observation_state']  # (batch_size, state_dim)
    actions = batch['actions']  # (batch_size, num_chunks, action_dim)
    
    # 训练模型
    # model(vlm_hidden_states, observation_state) -> predicted_actions
    # loss = criterion(predicted_actions, actions)
```

### 训练时的数据处理

```python
for batch in dataloader:
    # VLM hidden states: (batch_size, num_layers, seq_len, hidden_dim)
    vlm_tensor = batch['vlm_hidden_states'].to(device)
    
    # 拆分为列表（用于某些模型）
    vlm_hidden_states = [vlm_tensor[:, i, :, :] for i in range(vlm_tensor.size(1))]
    # 结果: List[Tensor]，每个 tensor shape = (batch_size, seq_len, hidden_dim)
    
    # Observation state
    proprioception = batch['observation_state'].to(device)  # (batch_size, state_dim)
    
    # Ground truth actions
    ground_truth_actions = batch['actions'].to(device)  # (batch_size, num_chunks, action_dim)
    
    # 训练...
```

---

## 配置参数说明

### `create_lerobot_dataloader()` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_path` | str | (必需) | 数据集根目录路径 |
| `batch_size` | int | 8 | 批次大小 |
| `num_workers` | int | 4 | 数据加载线程数 |
| `shuffle` | bool | True | 是否打乱数据 |
| `split` | str | "train" | 数据集划分 |
| `num_action_chunks` | int | 25 | 动作序列长度 |
| `enable_chunking` | bool | True | 是否启用 action chunking |
| `episode_indices` | List[int] | None | 指定 episode 列表，None 加载全部 |
| `cache_vlm_states` | bool | False | 是否缓存 VLM states 到内存 |
| `verbose` | bool | True | 是否打印详细信息 |

---

## 数据格式要求

数据集必须遵循 LeRobot 标准格式：

```
dataset_folder/
├── data/
│   └── chunk-000/
│       └── episode_*.parquet       # 包含 observation.state, action 等
├── meta/
│   ├── info.json                   # 数据集元信息
│   ├── episodes.jsonl              # Episode 索引
│   ├── stats.json                  # 统计信息
│   └── tasks.jsonl                 # 任务描述（可选）
├── videos/
│   └── chunk-000/
│       └── observation.images.*/   # 视频文件
│           └── episode_*.mp4
└── vlm_hidden_states/
    └── hidden_state_*.npy          # VLM hidden states
```

详细格式要求请参考各 Action Head 的 README.md。
