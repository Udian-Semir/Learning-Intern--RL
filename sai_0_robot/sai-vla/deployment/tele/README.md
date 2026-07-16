# ParaCAT 遥操作推理服务

基于 FastAPI 的 VLM + Pons + ParaCAT 实时推理服务。

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
│  │       Pons Adapter              │ │
│  │  (+ State Mapper)               │ │
│  └──────────────┬──────────────────┘ │
│                 │                    │
│                 ▼                    │
│  ┌─────────────────────────────────┐ │
│  │    ParaCAT Action Head          │ │
│  │  (离散动作预测)                  │ │
│  └──────────────┬──────────────────┘ │
│                 │                    │
└─────────────────┼────────────────────┘
                  │
                  ▼
         Action Chunk [25, 14]
```

## 快速开始

### 1. 配置

编辑 `config.yaml`，设置模型路径：

```yaml
vlm:
  type: "eagle2_5_vl"
  model_path: "/path/to/GR00T-N1.5-3B"

pons:
  checkpoint: "/path/to/pons.pt"

paracat:
  checkpoint: "/path/to/paracat.pt"
```

### 2. 启动服务器

```bash
# 使用默认配置
python server.py --config config.yaml --port 8000

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python server.py --config config.yaml --port 8000
```

### 3. 使用客户端

```python
from client import ParaCATClient

# 连接服务器
client = ParaCATClient("http://localhost:8000")

# 等待服务就绪
client.wait_until_ready()

# 执行预测
result = client.predict_from_files(
    image_paths=["camera.jpg"],
    state=[0.1, 0.2, ...],  # 14 维状态
    instruction="pick up the bottle"
)

# 获取动作
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
    "discrete_actions": [[-1, 0, 1, ...], ...],
    "timing": {
        "vlm_time": 0.15,
        "pons_time": 0.02,
        "paracat_time": 0.01,
        "total_time": 0.18
    },
    "chunk_size": 25,
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
        "pons": true,
        "paracat": true,
        "state_mapper": true
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
| `pons.checkpoint` | Pons 权重路径 | - |
| `pons.q_seq_len` | Query 序列长度 | `128` |
| `paracat.checkpoint` | ParaCAT 权重路径 | - |
| `chunk_size` | 动作块大小 | `25` |
| `action_dim` | 动作维度 | `14` |
| `undiscrete_columns` | 反离散化列 | `[0-11]` |
| `undiscrete_deltas` | Delta 值 | - |
| `gripper_columns` | Gripper 列 | `[12, 13]` |

## 性能

典型推理时间 (RTX 4090):
- VLM (Eagle): ~150ms
- Pons + State Mapper: ~20ms
- ParaCAT: ~10ms
- **总计: ~180ms**

## 注意事项

1. 确保 checkpoint 目录下有 `config.json` 文件，会自动加载配置
2. State Mapper 会自动从 checkpoint 目录查找 `state_mapper.pt`
3. 图像需要 Base64 编码传输
4. State 预处理配置需要与训练时一致
