# Sai0_1 VLA 推理服务器

基于 FastAPI 的高性能视觉-语言-动作 (VLA) 推理服务器。

## 功能特性

### VLM Backbone 支持
- **Qwen3-VL**: 2B, 4B, 7B 版本
- **Eagle 2.5 VL**: GR00T-N1.5-3B

### Action Head 支持
- **Flow Matching 0**: GR00T N1.5 原始架构
- **Flow Matching 1**: 自定义配置，支持多层 VLM
- **OFT 1.0**: L1 Regression + Transformer

### 数据预处理
- **图像预处理**:
  - Resize (自定义尺寸)
  - 水平/垂直翻转
  - 180度旋转

- **状态预处理**:
  - 零值转换为 -1 (适用于夹爪状态)
  - 最大最小值归一化 (按索引配置)

## 快速开始
<!-- python server.py --config config.yaml -->
### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config.yaml`:

```yaml
# 服务器配置
server:
  host: "0.0.0.0"
  port: 5000

# Pipeline 配置
pipeline:
  action_head_ckpt: "/path/to/checkpoint.pt"
  action_head_type: "flow_matching_1"
  vlm_type: "qwen3_vl"
  vlm_model_path: "Qwen/Qwen3-VL-2B-Instruct"
  device: "cuda:0"

# 预处理配置
preprocess:
  image:
    resize: [256, 256]
    flip_horizontal: true
  state:
    index_configs:
      "6":
        zero_to_minus_one: true
      "0":
        enable_normalization: true
        min_val: -0.5
        max_val: 0.5
```

### 3. 启动服务器

```bash
# 使用配置文件
python server.py --config config.yaml

# 或使用命令行参数
python server.py \
    --action_head_ckpt /path/to/checkpoint.pt \
    --vlm_type qwen3_vl \
    --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
    --device cuda:0 \
    --port 5000
```

### 4. 使用客户端

```python
from client import Sai0Client
from PIL import Image
import numpy as np

# 创建客户端
client = Sai0Client("http://localhost:5000")

# 加载图像
images = [
    Image.open("agentview.jpg"),
    Image.open("wrist.jpg")
]

# 当前状态
state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]  # 索引 6 是夹爪状态 (0 会被转换为 -1)

# 预测
result = client.predict(
    images=images,
    state=state,
    prompt="pick up the red apple"
)

print(f"预测动作: {result['actions']}")
print(f"动作形状: {result['metadata']['action_shape']}")
print(f"推理时间: {result['timing']['inference_time']:.3f}s")
```

## API 文档

启动服务器后，访问 http://localhost:5000/docs 查看交互式 API 文档。

### 主要接口

#### POST /predict
预测单次动作。

**请求体**:
```json
{
    "images": ["base64_encoded_image_1", "base64_encoded_image_2"],
    "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
    "prompt": "pick up the object",
    "image_format": "base64",
    "image_resize": [256, 256],
    "image_flip_horizontal": true
}
```

**响应**:
```json
{
    "actions": [[0.01, 0.02, ...], ...],
    "timing": {
        "preprocess_time": 0.001,
        "inference_time": 0.05,
        "total_time": 0.051
    },
    "metadata": {
        "num_images": 2,
        "state_dim": 7,
        "action_shape": [16, 7],
        "preprocessed_state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1.0]
    }
}
```

#### POST /predict_batch
批量预测动作。

#### POST /config/update
动态更新预处理配置。

```python
# 更新图像预处理
client.update_preprocess_config(
    image_preprocess={
        "resize": [320, 240],
        "flip_horizontal": False
    }
)

# 更新状态预处理
client.update_preprocess_config(
    state_preprocess={
        6: {"zero_to_minus_one": True},
        0: {"enable_normalization": True, "min_val": -1.0, "max_val": 1.0}
    }
)
```

#### GET /config/preprocess
获取当前预处理配置。

#### GET /health
健康检查。

#### GET /info
获取服务器信息。

#### GET /latency_stats
获取延迟统计。

## 状态预处理详解

### 零值转换为 -1

适用于夹爪状态，将 0 (关闭) 转换为 -1:

```yaml
state:
  index_configs:
    "6":  # 夹爪状态索引
      zero_to_minus_one: true
```

### 最大最小值归一化

将指定索引的值归一化到 [-1, 1] 范围:

```yaml
state:
  index_configs:
    "0":  # 位置 x
      enable_normalization: true
      min_val: -0.5
      max_val: 0.5
    "1":  # 位置 y
      enable_normalization: true
      min_val: -0.5
      max_val: 0.5
```

归一化公式:
```
normalized = ((value - min_val) / (max_val - min_val)) * 2 - 1
```

### 组合使用

可以同时启用零值转换和归一化:

```yaml
state:
  index_configs:
    "7":
      zero_to_minus_one: true
      enable_normalization: true
      min_val: 0.0
      max_val: 0.085
```

## 图像格式支持

### Base64 格式 (推荐)

```python
import base64
from PIL import Image
from io import BytesIO

# 编码
img = Image.open("image.jpg")
buffered = BytesIO()
img.save(buffered, format="JPEG")
encoded = base64.b64encode(buffered.getvalue()).decode()

# 使用
result = client.predict(
    images=[encoded],
    state=state,
    image_format="base64"
)
```

### Numpy 格式

```python
import numpy as np

# 图像转为 numpy array
img_array = np.array(img)  # shape: (H, W, 3)

# 使用
result = client.predict(
    images=[img_array.tolist()],
    state=state,
    image_format="numpy"
)
```

## 监控仪表盘 (Dashboard)

服务器内置了一个实时监控仪表盘，启动服务后浏览器访问：

```
http://<host>:<port>/dashboard
```

例如本地启动时访问 http://localhost:5000/dashboard 。

### 仪表盘功能

| 区域 | 内容 |
|---|---|
| **KPI 卡片** | 总请求数、成功率、错误数、平均推理耗时、队列深度 |
| **Per-User 表格** | 每个 API Key 的调用次数及占比（Key 已脱敏） |
| **Per-Suite 表格** | 每个 task suite 的调用次数及占比 |
| **GPU 状态** | GPU 利用率、显存使用量进度条 |

- 页面右上角可切换自动刷新间隔（3s / 5s / 10s / 30s / 关闭）
- 数据来源于 `GET /v1/metrics` 接口
- 统计数据会持久化到 `logs/usage_stats.json`，服务重启后自动恢复

### Metrics API

也可以直接调用接口获取 JSON 格式的统计数据：

```bash
curl http://localhost:5000/v1/metrics -H "Authorization: Bearer <API_KEY>"
```

返回示例：

```json
{
  "version": "2.0.0",
  "queue_depth": 0,
  "v1_act": {
    "total_requests": 1234,
    "total_errors": 5,
    "avg_inference_ms": 85.32,
    "per_user": {"sk-abc***ef01": 800, "sk-xyz***gh23": 434},
    "per_suite": {"libero_spatial": 1000, "libero_object": 234}
  },
  "gpu": {
    "gpu_utilization_pct": 45,
    "gpu_memory_used_mb": 8192,
    "gpu_memory_total_mb": 24576
  }
}
```

### 日志文件

服务器日志同时输出到终端和文件：

```
deployment/Sai0_1_server/logs/
├── server.log         # 完整服务日志
└── usage_stats.json   # API 使用统计（持久化）
```

## 性能监控

```python
# 获取延迟统计
stats = client.get_latency_stats()
print(f"平均延迟: {stats['mean']:.2f}ms")
print(f"最小延迟: {stats['min']:.2f}ms")
print(f"最大延迟: {stats['max']:.2f}ms")

# 重置统计
client.reset_latency_stats()
```

## 常见问题

### 1. CUDA 内存不足

减小 batch size 或使用较小的模型:
```yaml
pipeline:
  vlm_model_path: "Qwen/Qwen3-VL-2B-Instruct"  # 使用 2B 而非 4B
```

### 2. 推理速度慢

- 确保使用 GPU
- 增加预热步数
- 减小图像尺寸

```yaml
pipeline:
  warmup_steps: 5
preprocess:
  image:
    resize: [224, 224]
```

### 3. 动作不准确

- 检查状态归一化配置
- 确保图像预处理与训练时一致
- 验证 action head checkpoint 版本

## 文件结构

```
deployment/Sai0_1_server/
├── server.py          # 服务器主程序
├── client.py          # Python 客户端
├── config.yaml        # 配置文件
├── dashboard.html     # 监控仪表盘前端
├── auth.py            # API Key 鉴权
├── queue_worker.py    # 推理队列
├── requirements.txt   # 依赖列表
├── README.md          # 本文档
└── logs/              # 运行时自动生成
    ├── server.log         # 服务日志
    └── usage_stats.json   # API 使用统计
```
