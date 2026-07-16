# OFT1_0 Action Head - 训练与推理指南

## ⚠️ 重要配置说明

### Constants 配置规则

**训练时必须在 `constants.py` 中手动修改的参数：**

以下参数直接影响模型的输入输出维度，**必须在训练前**在 `constants.py` 文件中手动设置，无法通过命令行参数覆盖：

```python
# Action-related
ACTION_DIM = 16                      # 机器人动作维度（例如：16自由度机械臂）
NUM_ACTIONS_CHUNK = 25               # 动作序列长度（预测未来25步）

# Proprio
PROPRIO_DIM = 16                     # 本体感知维度（关节位置、速度等状态）
USE_PROPRIO_PROJECTOR = True         # 是否使用本体感知投影器
USE_NOISY_ACTION_PROJECTOR = False   # 是否使用噪声动作投影器

# Diffusion or L1
USE_DIFFUSION = False                # 是否使用扩散模型（False=使用L1回归）
NUM_DIFFUSION_STEPS = 50             # 扩散步数（仅在USE_DIFFUSION=True时有效）
```

**为什么这些参数必须在 constants.py 中设置？**
- 这些参数决定了模型的输入输出维度和网络结构
- 修改这些值会改变模型架构，无法通过加载检查点兼容
- 训练时必须与数据集的实际维度匹配

**可通过命令行覆盖的参数：**
- `LLM_OUTPUT_DIM_MLP_INPUT_DIM`（VLM 输出维度）
- `NUM_VLM_HIDDEN_LAYERS`（VLM 隐藏层数量）

这两个参数可以在 `train.py` 中通过 `--vlm_output_dim` 和 `--num_vlm_layers` 参数动态指定。

---

## 输出格式说明

### 模型输出结构

模型的最终输出维度为 `(batch_size, 1, NUM_ACTIONS_CHUNK × ACTION_DIM)`

**为什么要拉平为 `NUM_ACTIONS_CHUNK × ACTION_DIM`？**

为了适配 **MLP (Multi-Layer Perceptron) 结构**的 Action Head：
- 原始动作维度：`(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)` = `(batch_size, 25, 16)`
- MLP 输入要求：**单一的向量维度**
- 拉平后维度：`(batch_size, 1, NUM_ACTIONS_CHUNK × ACTION_DIM)` = `(batch_size, 1, 400)`

**处理流程：**
```python
# 训练时：Ground Truth Actions 拉平
ground_truth_actions = batch['actions']  # shape: (batch_size, 25, 16)
gt_actions_reshaped = ground_truth_actions.view(
    ground_truth_actions.size(0), 1, NUM_ACTIONS_CHUNK * ACTION_DIM
)  # shape: (batch_size, 1, 400)

# 推理后：可以 reshape 回多步动作格式
action_predictions = pipeline(vlm_hidden_states, proprioception)  # (batch_size, 1, 400)
actions_reshaped = action_predictions.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)  # (batch_size, 25, 16)
```

这样设计使得 MLP 可以将 Transformer 输出的高维特征直接映射到完整的动作序列。

---

# 数据格式（LeRobot Dataset）

### 数据集格式要求

数据必须遵循 **LeRobot 标准格式**。`--data_path` 参数指向的文件夹必须包含以下结构：

```
dataset_folder/
├── data/
│   └── chunk-000/
│       └── *.parquet        # 包含 observation.state, action 等数据
├── meta/
│   ├── info.json            # 数据集元信息（必需）
│   ├── episodes.jsonl       # Episode 索引
│   ├── stats.json           # 统计信息
│   ├── tasks.jsonl          # 任务描述
│   └── modality.json        # 模态配置
├── videos/
│   └── chunk-000/
│       └── *.mp4            # 视频文件（可选）
└── vlm_hidden_states/
    └── hidden_state_*.npy   # VLM hidden states（必需）
```

**必需文件**：
- `meta/info.json`: 包含 `total_episodes`、`features` 等元信息
- `data/chunk-*/`: Parquet 格式的观测和动作数据
- `vlm_hidden_states/`: 预计算的 VLM hidden states（.npy 文件）

**示例数据集路径**：
```bash
/home/sythoid_01/文档/Huangwenlong/n1.5-split/raw_data_timestamp_align_to_lerobot_data/demo_data_task_description_pickupanapple_images_2/pickupanapple_v1_hidden_dim_1_512_2048
```

# 1. 推理

```bash
# 基本用法
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1

# 指定设备
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1 --device cuda:7

# 指定 VLM 配置
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1 \
  --vlm_output_dim 2048 --num_vlm_layers 3
```

输入：
action_predictions = pipeline(vlm_hidden_states, proprioception) # (bsz, 1, NUM_ACTIONS_CHUNK*ACTION_DIM)

参数细节：
    # Dummy VLM hidden states (3 layers as per NUM_VLM_HIDDEN_LAYERS)
    vlm_hidden_states = [
        torch.randn(batch_size, seq_len, hidden_dim, device=device)
        for _ in range(NUM_VLM_HIDDEN_LAYERS)
    ]

    # Dummy proprioception data
    proprioception = torch.randn(batch_size, PROPRIO_DIM, device=device)

# 2. 训练

```bash
python train.py --data_path /path/to/your/data \
                --batch_size 16 \
                --learning_rate 1e-4 \
                --num_epochs 100 \
                --num_transformer_blocks 2 \
                --num_attention_heads 8

python train.py --data_path /home/sythoid_01/文档/Huangwenlong/n1.5-split/raw_data_timestamp_align_to_lerobot_data/demo_data_task_description_pickupanapple_images_2/pickupanapple_v1 \
                --batch_size 4 \
                --num_epochs 10
```

其余的参数，例如 chunk size 在 `constants.py` 中设置

**Checkpoint 文件夹结构**

训练过程中会生成以下目录结构：

```
experiments/
└── pickupanapple_v1_hidden_dim_3_512_2048/  （或根据数据集自动命名）
    ├── checkpoints/
    │   ├── epoch_1/
    │   │   ├── action_head.pt    # 第1轮的模型权重
    │   │   └── config.json       # 模型配置（包含 transformer blocks、attention heads 等）
    │   ├── epoch_2/
    │   │   ├── action_head.pt
    │   │   └── config.json
    │   └── .../
    └── logs/
        ├── events.out.tfevents.*  # TensorBoard 日志
        └── wandb/                 # Weights & Biases 日志（如果启用）
```

每个 epoch 文件夹包含：
- `action_head.pt`: PyTorch state_dict 格式的完整 pipeline 权重
- `config.json`: 模型配置，包含 `num_transformer_blocks`, `num_attention_heads`, `vlm_output_dim`, `num_vlm_layers` 等参数