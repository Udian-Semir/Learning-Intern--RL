# OFT 遥操作推理服务

基于 FastAPI 的 VLM + OFT Pipeline 实时推理服务。

## 架构

```
Client (图像 + State)
       │
       ▼
┌──────────────────────────────────────┐
│           FastAPI Server             │
│                                      │
│  ┌─────────────────────────────────┐ │
│  │        VLM Backbone             │ │
│  │  (Eagle / Qwen)                 │ │
│  └──────────────┬──────────────────┘ │
│                 │ hidden states      │
│                 ▼                    │
│  ┌─────────────────────────────────┐ │
│  │      VLM2OFT Pipeline           │ │
│  │  (TransformerBlocks +           │ │
│  │   ProprioProjector +            │ │
│  │   L1RegressionActionHead)       │ │
│  └──────────────┬──────────────────┘ │
│                 │                    │
└─────────────────┼────────────────────┘
                  │
                  ▼
         Action Chunk [50, 14]
         (连续动作)
```

## 与 ParaCAT (tele) 的区别

| 组件 | ParaCAT (tele) | OFT (tele_parad) |
|------|---------------|------------------|
| Adapter | Pons Adapter | TransformerBlocks (内置) |
| Action Head | ParaCAT Action Head | L1RegressionActionHead |
| 输出类型 | 离散动作 {-1, 0, 1} | 连续动作 (L1 回归) |
| 后处理 | 反离散化 | 无需处理 |
| State 处理 | State Mapper | ProprioProjector_Changed |

## 快速开始

### 1. 配置

编辑 `config.yaml`，设置模型路径：

```yaml
vlm:
  type: "eagle2_5_vl"
  model_path: "/path/to/GR00T-N1.5-3B"

oft:
  checkpoint: "/path/to/action_head.pt"
  num_transformer_blocks: 2
  num_attention_heads: 8
  num_vlm_layers: 1
```

### 2. 启动服务器

```bash
# 使用默认配置
python server.py --config config.yaml --port 8000
python server.py --config config.yaml --offline --port 8000
# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python server.py --config config.yaml --port 8000
```

### 3. 使用客户端

```python
from client import OFTClient

# 连接服务器
client = OFTClient("http://localhost:8000")

# 等待服务就绪
client.wait_until_ready()

# 执行预测
result = client.predict_from_files(
    image_paths=["camera.jpg"],
    state=[0.1, 0.2, ...],  # 14 维状态
    instruction="pick up the bottle"
)

# 获取动作 (连续值)
actions = result['actions']  # [chunk_size, action_dim]
print(f"预测 {len(actions)} 步动作")
```

## API 端点

### POST /predict

执行预测。

**请求:**
```json
{
    "images": ["<base64_encoded_image>"],
    "state": [0.1, 0.2, ...],
    "instruction": "pick up the bottle"
}
```

**响应:**
```json
{
    "actions": [[0.1, 0.2, ...], ...],
    "timing": {
        "vlm_time": 0.15,
        "oft_time": 0.02,
        "total_time": 0.17
    },
    "chunk_size": 50,
    "action_dim": 14
}
```

### GET /health

健康检查。

**响应:**
```json
{
    "status": "ready",
    "models_loaded": {
        "vlm": true,
        "oft_pipeline": true
    },
    "device": "cuda:0"
}
```

### GET /info

获取模型信息。

## 客户端命令行

```bash
# 测试连接
python client.py --server http://localhost:8000 --image test.jpg

# 带状态
python client.py --server http://localhost:8000 --image test.jpg --state 0.1 0.2 ...
```

## 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `device` | 计算设备 | `cuda:0` |
| `vlm.type` | VLM 类型 | `eagle2_5_vl` |
| `vlm.model_path` | VLM 模型路径 | - |
| `vlm.layers` | 提取层 | `[-1]` |
| `oft.checkpoint` | OFT 权重路径 | - |
| `oft.num_transformer_blocks` | Transformer 块数 | `2` |
| `oft.num_attention_heads` | 注意力头数 | `8` |
| `oft.num_vlm_layers` | VLM 层数 | `1` |
| `chunk_size` | 动作块大小 | `50` |
| `action_dim` | 动作维度 | `14` |
| `proprio_dim` | Proprio 维度 | `14` |
| `action_postprocess.gripper_binarize.enabled` | 启用输出动作 Gripper 二值化 | `false` |
| `action_postprocess.gripper_binarize.columns` | 需要二值化的动作索引列表 | `[]` |

## 输出动作后处理

### Gripper 二值化

启用后，预测输出的动作中指定列会被二值化处理：
- `> 0` 变为 `1`
- `<= 0` 变为 `0`

配置示例：
```yaml
action_postprocess:
  gripper_binarize:
    enabled: true
    columns: [12, 13]  # 第 12、13 列（通常是 gripper）
```

## 性能

典型推理时间 (RTX 4090):
- VLM (Eagle): ~150ms
- OFT Pipeline: ~20ms
- **总计: ~170ms**

## 注意事项

1. 确保 checkpoint 目录下有 `config.json` 文件，会自动加载配置
2. OFT 输出的是连续动作，不需要反离散化处理
3. 图像需要 Base64 编码传输
4. State 预处理配置需要与训练时一致
5. `proprio_dim` 需要与训练时的 state 维度一致
