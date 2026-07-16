Flow_Matching_0 (Standalone Action Head with Pretrained Weights Support)

独立的动作头训练/推理原型，支持从预训练权重微调。使用 LeRobot Dataset Loader 加载真实数据。保留与原项目一致的核心结构（Flow Matching + DiT + 分类特定MLP），用于"VLM预计算hidden states → 动作头"分离式流程。

## 新特性

- ✨ **LeRobot 数据集支持**：直接从 LeRobot 格式数据集加载图像、VLM states、actions 等完整数据
- 🔧 **预训练权重微调**：支持加载预训练权重进行迁移学习
- 📊 **训练/验证划分**：支持自动划分训练集和验证集，保存最佳模型
- 📈 **进度条显示**：使用 tqdm 显示训练进度

目录

- models/
  - action_head/
    - flow_matching_action_head.py: FlowMatching动作头实现
    - cross_attention_dit.py: DiT与交叉注意力模块
    - action_encoder.py: 正弦位置编码与Action编码器
    - category_specific_mlp.py: 多机体特定的MLP解码器
- config.py: 配置文件（包含完整的预训练模型配置）
- train_with_pretrained.py: 预训练权重微调脚本（使用 LeRobot 数据集）
- extract_pretrained_weights.py: 从完整模型提取action head权重
- infer.py: 推理脚本（加载权重，给定hidden states与state输出动作）

依赖

- Python 3.10+
- PyTorch
- transformers
- diffusers
- numpy
- tqdm

数据格式（LeRobot Dataset）

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

### 数据内容

使用 LeRobot 格式数据集，通过 `lerobot_dataset_loader.py` 加载：
- **VLM hidden states**: (num_layers, seq_len, hidden_dim)，如 (3, 512, 2048)，取最后一层
- **Images**: 多相机图像，如 top camera、left_wrist camera
- **Observation state**: (state_dim,)，如 (16,) 机器人状态
- **Actions**: (num_chunks, action_dim)，如 (16, 16) 支持 action chunking
- **Task descriptions**: 任务描述文本

训练时自动处理：
- VLM states 取最后一层作为 backbone_features
- 自动 padding state 到 max_state_dim=64
- 自动 padding action 到 max_action_dim=32
- Embodiment ID 固定为 31

训练

### 使用预训练权重微调（推荐）

**步骤 1：提取预训练权重**

```bash
cd gr00t_split/Action_Heads/Flow_Matching_0

python extract_pretrained_weights.py \
  --pretrained_dir /home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e \
  --output ./pretrained_action_head.pt
```

这将生成 `pretrained_action_head.pt` 文件（包含314个权重参数），同时生成 `structure.txt` 显示权重结构。

**步骤 2：使用 LeRobot 数据集训练**

```bash
conda activate gr00t

# 基本用法
python train_with_pretrained.py \
  --data_path /path/to/lerobot/dataset \
  --pretrained_weights ./pretrained_action_head.pt \
  --steps 1000 \
  --out_dir ./fm0_ckpts_finetuned

# 完整参数示例
python train_with_pretrained.py \
  --data_path /home/sythoid_01/文档/Huangwenlong/n1.5-split/raw_data_timestamp_align_to_lerobot_data/demo_data_task_description_pickupanapple_images_2/pickupanapple_v1 \
  --pretrained_weights ./pretrained_action_head.pt \
  --batch_size 4 \
  --num_workers 0 \
  --epochs 10 \
  --steps 1000 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --num_action_chunks 16 \
  --val_split 0.1 \
  --device cuda:7 \
  --out_dir ./fm0_ckpts_finetuned
```

**输出文件**

训练完成后会生成：
- `fm0_ckpts_finetuned/action_head.pt` - 最终模型权重
- `fm0_ckpts_finetuned/action_head_best.pt` - 验证集上最佳模型权重（如果有验证集）
- `fm0_ckpts_finetuned/config.json` - 模型配置文件

**Checkpoint 文件夹结构**

训练过程中会生成以下目录结构：

```
fm0_ckpts_finetuned/  （或你指定的 --out_dir）
├── epoch_0/
│   ├── action_head.pt    # 第0轮的模型权重
│   └── config.json       # 模型配置（包含维度、超参数等）
├── epoch_1/
│   ├── action_head.pt
│   └── config.json
├── .../
└── best/                 # 验证集最佳模型（如果启用验证集）
    ├── action_head.pt    # 最佳模型权重
    └── config.json
```

每个 epoch 文件夹包含：
- `action_head.pt`: PyTorch state_dict 格式的模型权重
- `config.json`: 模型配置，包含 `backbone_dim`, `action_dim`, `action_horizon`, `max_state_dim`, `max_action_dim` 等参数

**重要参数说明**

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--data_path` | 必需 | LeRobot 数据集路径 |
| `--pretrained_weights` | pretrained_action_head.pt | 预训练权重文件路径 |
| `--batch_size` | 4 | 批次大小 |
| `--num_workers` | 0 | 数据加载线程数 |
| `--epochs` | 1 | 训练轮数 |
| `--steps` | 1000 | 最大训练步数 |
| `--lr` | 1e-4 | 学习率 |
| `--weight_decay` | 1e-5 | 权重衰减 |
| `--num_action_chunks` | 16 | Action chunk 数量（对应 action_horizon） |
| `--val_split` | 0.1 | 验证集比例（0.0-1.0） |
| `--device` | cuda:7 | 训练设备 |
| `--out_dir` | ./fm0_ckpts_finetuned | 输出目录 |

推理

```bash
# 使用随机输入
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1/checkpoints/epoch_0

# 使用最佳模型
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1/checkpoints/best

# 使用NPZ样本
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1/checkpoints/epoch_0 --npz /path/to/sample.npz

# 指定设备
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1/checkpoints/epoch_0 --device cuda:0
```

重要配置说明

### 预训练模型参数（使用 config.py 中的配置）

```python
# 必须与预训练模型一致的参数
backbone_dim = 1536          # Action backbone 输入维度（注意：实际使用2048）
max_action_dim = 32          # 最大动作维度（padding目标）
action_horizon = 16          # 动作预测时间步
max_state_dim = 64           # 最大状态维度（padding目标）
```

### LeRobot 数据集处理流程

训练时自动从 LeRobot 数据集读取实际维度：
- **actual_action_dim**: 从 `info.json` 读取实际动作维度
- **actual_state_dim**: 从 `info.json` 读取实际状态维度
- **自动 padding**: LeRobot dataloader 自动处理维度对齐

数据转换过程（`lerobot_collate_fn`）：
1. **VLM states**: (batch, num_layers, seq_len, hidden_dim) → 取最后一层 → (batch, seq_len, hidden_dim)
2. **State**: (batch, state_dim) → padding 到 max_state_dim → 添加时间维度 → (batch, 1, max_state_dim)
3. **Actions**: (batch, num_chunks, action_dim) → padding 到 max_action_dim → (batch, num_chunks, max_action_dim)
4. **Masks**: 自动生成 attention_mask 和 action_mask（真实维度为1，padding维度为0）

### 与原项目一致性注意事项

- **维度与时序**：action_horizon=16，backbone_embedding_dim=2048，与主仓一致
- **Embodiment路由**：本项目默认单机体 embodiment_id=31
- **权重兼容性**：
  - ✅ 预训练权重只会加载形状匹配的参数（314/314）
  - ✅ 支持不同 actual_action_dim（自动从数据集读取并 padding）
  - ✅ 支持训练/验证集划分，自动保存最佳模型
  - ❌ 不支持修改 max_action_dim、action_horizon 等核心配置（会导致权重不匹配）

模型架构说明

### State 和 Action 的处理流程

```python
# 1. 数据预处理（ConcatTransform）
state = torch.cat([state.position, state.velocity, ...], dim=-1)  # [T, D_state]
action = torch.cat([action.position, action.velocity, ...], dim=-1)  # [T, D_action]

# 2. Padding（GR00TTransform）
state_padded = pad_or_truncate(state, max_state_dim=64)  # [T_state=1, 64]
action_padded = pad(action, max_action_dim=32)  # [T_action=16, 32]

# 3. 模型编码（FlowMatchingActionHead.forward）
vl_embs = backbone_output.backbone_features  # [B, S, 2048]
state_features = state_encoder(state_padded)  # [B, 1, hidden_dim]
action_features = action_encoder(noisy_action)  # [B, 16, hidden_dim]

# 4. 序列拼接
sa_embs = torch.cat([
    state_features,      # 1 token: 整个state作为1个token
    future_tokens,       # n tokens: 可学习的future tokens
    action_features      # 16 tokens: 每个时间步1个token
], dim=1)
```

### 完整配置（来自预训练模型）

```python
FlowmatchingActionHeadConfig(
    # 基础维度
    backbone_embedding_dim=2048,
    input_embedding_dim=1536,  # 注意：实际使用2048
    hidden_size=1024,
    action_dim=32,
    action_horizon=16,
    max_state_dim=64,
    
    # DiT配置
    diffusion_model_cfg={
        'num_layers': 16,
        'num_attention_heads': 32,
        'attention_head_dim': 48,
        'cross_attention_dim': 2048,
        'dropout': 0.2,
        'interleave_self_attention': True,
    },
    
    # 训练参数
    num_timestep_buckets=1000,
    num_inference_timesteps=4,
    noise_beta_alpha=1.5,
    noise_beta_beta=1.0,
    
    # 机体配置
    max_num_embodiments=32,
    
    # VL自注意力配置
    use_vlln=True,
    vl_self_attention_cfg={
        'num_layers': 4,
        'num_attention_heads': 32,
        'attention_head_dim': 64,
    }
)
```

## 快速开始

```bash
# 1. 激活环境
conda activate gr00t

# 2. 进入目录
cd gr00t_split/Action_Heads/Flow_Matching_0

# 3. 提取预训练权重
python extract_pretrained_weights.py \
  --pretrained_dir ~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/xxx \
  --output ./pretrained_action_head.pt

# 4. 训练（使用 LeRobot 数据集）
python train_with_pretrained.py \
  --data_path /path/to/lerobot/dataset \
  --pretrained_weights ./pretrained_action_head.pt \
  --steps 1000

# 5. 推理
python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1/checkpoints/epoch_0
```

备注

- 使用 LeRobot Dataset Loader 统一数据加载流程，与其他模块保持一致
- 训练时自动读取数据集的 action_dim 和 state_dim，无需手动配置
- 支持训练/验证集自动划分，训练过程中保存验证集上的最佳模型
- 提取的预训练权重包含314个参数，覆盖 action_encoder、action_decoder、DiT blocks 等所有核心组件
- 若后续需要导入为库使用，可将 models/ 作为纯模块引用，或将本目录整体迁移至独立仓库，保持完全可移植
