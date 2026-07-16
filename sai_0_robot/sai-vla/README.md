# Sai0-VLA: Vision-Language-Action Model

基于视觉-语言-动作（VLA）的机器人控制模型，支持多种 VLM 后端和动作头架构。

## ⚠️ 训练前必读：共享内存配置

**训练 Action Head 时，需要增加共享内存 `/dev/shm` 大小，否则可能导致训练失败！**

### 临时增加（重启后失效）

```bash
sudo mount -o remount,size=2T /dev/shm
df -h /dev/shm
```

### 永久增加（编辑 /etc/fstab）

```bash
# 添加或修改以下行:
tmpfs /dev/shm tmpfs defaults,size=2T 0 0
```

> 💡 **提示：** 如果物理内存不足以支撑训练，还需要修改 swap 空间大小。可以通过以下命令查看和调整：
> ```bash
> # 查看当前 swap
> swapon --show
> 
> # 创建新的 swap 文件（例如 64GB）
> sudo fallocate -l 64G /swapfile
> sudo chmod 600 /swapfile
> sudo mkswap /swapfile
> sudo swapon /swapfile
> 
> # 永久生效：在 /etc/fstab 添加
> # /swapfile none swap sw 0 0
> ```

> 完整迁移步骤如下（建议在低负载时操作）：
>
> ```bash
> # 1. 关闭当前所有 swap
> sudo swapoff -a
>
> # 2. 在 /data 下创建新的 swapfile（推荐大小根据需求调整，这里示例 32G）
> sudo fallocate -l 2T /data/swapfile
> # 如果 fallocate 失败，可改用 dd（更慢但更兼容）
> # sudo dd if=/dev/zero of=/data/swapfile bs=1G count=32 oflag=direct
>
> # 3. 设置正确权限（必须 600）
> sudo chmod 600 /data/swapfile
>
> # 4. 格式化为 swap
> sudo mkswap /data/swapfile
>
> # 5. 立即启用新 swap
> sudo swapon /data/swapfile
>
> # 6. 验证是否生效
> swapon --show
> free -h
> df -h /data   # 已用空间应增加约 32G
> ```

---

## 🔧 环境安装

### 快速安装

```bash
# 1. 运行环境安装脚本
bash install_env.sh

# 2. 激活环境
conda activate qwen_eagle_hwl

# 3. 安装额外依赖 (必须)
pip install timm
pip install ninja
pip install decord
pip install pynvml

# 4. 安装 Flash Attention (从本地 whl 文件)
pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

pip install --upgrade "transformers==5.0.0rc1"
pip install lmdb
```

### 环境配置说明

| 配置项 | 版本 |
|-------|------|
| Python | 3.10.0 |
| PyTorch | 2.8.0+cu128 |
| CUDA | 12.8 |
| Conda 环境名 | qwen_eagle_hwl |

### 安装脚本参数

```bash
# 查看帮助
bash install_env.sh --help

# 指定 CUDA 版本
bash install_env.sh --cuda 11.8

# 跳过 LIBERO 仿真环境
bash install_env.sh --skip-libero

# 完整安装 (包含所有可选依赖)
bash install_env.sh --full
```

---

## ⚠️ 重要提示

**如果在提取 Qwen VLM 隐藏层时遇到问题（如维度不匹配、层索引错误等），请在向 AI 寻求帮助时提供以下文件作为知识补充：**

- 📄 `VLMs/S0/backbone/factory.py` - VLM 后端工厂类，包含隐藏层提取的完整实现逻辑

该文件包含了所有 VLM 后端（Qwen3-VL、Eagle 等）的初始化和隐藏层提取方法，能帮助 AI 更准确地诊断和解决问题。

## 📁 项目结构

```
sai0-vla/
├── Action_Heads/              # 动作头模型实现
│   ├── Flow_Matching_0/       # Flow Matching 架构（使用预训练权重）
│   ├── Flow_Matching_1/       # Flow Matching 架构（完整训练）
│   └── OFT1_0/                # Optimal Flow Transport 架构
├── VLMs/                      # 视觉-语言模型后端
│   └── S0/                    # S0 VLM 实现（Qwen3-VL, Eagle 等）
├── VLAs/                      # 端到端 VLA 管道
│   └── Sai0/                  # Sai0 VLA 实现
├── utils/                     # 工具函数
│   └── lerobot_dataset_loader.py  # LeRobot 数据集加载器
├── deployment/                # 🚀 部署相关
│   └── sai0_server/          # HTTP 推理服务器
│       ├── server.py          # FastAPI 服务器（主程序）
│       ├── client.py          # Python 客户端库
│       ├── client_test.py     # 命令行测试工具
│       ├── example_client_test.py  # 完整使用示例
│       ├── config.yaml        # 服务器配置文件
│       ├── requirements.txt   # Python 依赖
│       └── README.md          # 详细文档
├── start_server.sh            # 🚀 快速启动脚本
└── .gitignore                 # Git 忽略规则
```

## 🔑 核心文件说明

### 1. `start_server.sh` - 快速启动脚本
一键启动 Sai0 推理服务器的便捷脚本，位于项目根目录。

**功能：**
- 自动加载 VLA 模型到 GPU
- 启动 FastAPI HTTP 服务器
- 支持自定义端口、VLM 后端、动作头路径

**用法：**
```bash
# 使用默认配置启动
./start_server.sh

# 自定义配置
./start_server.sh --port 8000 --vlm-backend Qwen3-VL-2B --action-head /path/to/checkpoint
```

### 2. `deployment/sai0_server/` - HTTP 推理服务
完整的 HTTP API 部署方案，支持本地/远程推理。

#### 主要文件：

- **`server.py`** - FastAPI 服务器主程序
  - 提供 `/predict`（单次推理）和 `/predict_batch`（批量推理）接口
  - 支持 base64 和 numpy 两种图像格式
  - 自动处理 RGB/RGBA/灰度图像转换
  - GPU 模型单次加载，多次推理

- **`client.py`** - Python 客户端库
  - 封装 HTTP 请求逻辑
  - 自动处理图像编码（base64 或 numpy）
  - 提供简洁的 API 接口

- **`client_test.py`** - 命令行测试工具
  - 快速测试服务器功能
  - 显示详细的性能统计
  - 支持单图/多图推理

- **`example_client_test.py`** - 完整使用示例
  - 演示 4 种使用方式：
    1. ~~Base64 格式（有损压缩）~~
    2. Numpy 格式（无损传输，推荐）
    3. 直接传递 numpy array
    4. 多图像推理（双目相机/多视角）

- **`config.yaml`** - 服务器配置文件
  - VLM 后端选择
  - 动作头路径
  - 推理参数（温度、采样步数等）

### 3. `VLAs/Sai0/` - Sai0 VLA 实现
端到端的视觉-语言-动作管道。

- **`end2end_pipeline.py`** - 核心推理管道
- **`run_inference.py`** - 独立推理脚本（无需服务器）

### 4. `Action_Heads/` - 动作头模型
支持多种动作预测架构：

- **Flow_Matching_0/** - 使用预训练权重快速训练
- **Flow_Matching_1/** - 完整 Flow Matching 训练
- **OFT1_0/** - Optimal Flow Transport 架构

### 5. `VLMs/S0/` - VLM 后端
支持的视觉-语言模型：

- Qwen3-VL-2B（默认，轻量高效）
- Qwen3-VL-4B / 8B / 32B
- Eagle-2.5（高精度）

## 🚀 快速开始：部署推理服务器

### 步骤 1: 安装依赖

```bash
# 安装服务器依赖
cd deployment/sai0_server
pip install -r requirements.txt

# 或者手动安装
pip install fastapi uvicorn pydantic requests pillow numpy torch transformers
```

### 步骤 2: 配置服务器

编辑 `deployment/sai0_server/config.yaml`：

```yaml
model:
  vlm_backend: "Qwen3-VL-2B"  # VLM 后端选择
  action_head_ckpt: "Action_Heads/Flow_Matching_1/experiments/pickupanapple_v1_hidden_dim_1_512_2048/best_model.pth"

inference:
  temperature: 1.0
  num_steps: 50
```

### 步骤 3: 启动服务器

**方式 1: 使用快速启动脚本（推荐）**

```bash
# 在项目根目录执行
./start_server.sh

# 自定义参数
./start_server.sh --port 8000 --vlm-backend Qwen3-VL-2B
```

**方式 2: 手动启动**

```bash
cd deployment/sai0_server
python server.py
```

服务器启动后会显示：

```
🚀 Sai0 VLA 推理服务器启动成功！
📍 访问地址: http://localhost:5000
📖 API 文档: http://localhost:5000/docs
```

### 步骤 4: 测试推理

**方式 1: 使用示例脚本**

```bash
# 运行完整示例（包含 4 种使用方式）
python deployment/sai0_server/example_client_test.py
```

**方式 2: 使用命令行工具**

```bash
python deployment/sai0_server/client_test.py \
  --image /path/to/image.png \
  --state "[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]" \
  --prompt "Pick up an apple." \
  --numpy  # 使用 numpy 格式（推荐）
```

**方式 3: Python 代码调用**

```python
from client import Sai0Client
import numpy as np
from PIL import Image

# 创建客户端
client = Sai0Client("http://localhost:5000")

# 准备数据
image = Image.open("test.png")
state = np.zeros(16, dtype=np.float32)

# 推理
result = client.predict(
    images=[image],
    state=state,
    prompt="Pick up an apple.",
    use_numpy_format=True  # 使用无损传输
)

print(f"Actions: {result['actions']}")
print(f"Shape: {np.array(result['actions']).shape}")
```

**方式 4: curl 测试**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "images": [[[[255,0,0],[0,255,0]],[[0,0,255],[255,255,255]]]],
    "state": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    "image_format": "numpy",
    "prompt": "test"
  }'
```

## 📊 图像格式对比

服务器支持两种图像传输格式：

| 格式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **Numpy** | 无损传输，保证数据完整性 | JSON 数据量大（~2MB） | 局域网/本地推理（推荐） |
| **Base64** | JPEG 压缩，数据量小（~50KB） | 有损压缩，可能影响精度 | 广域网/云端推理 |

**推荐配置：**
- 本地/局域网：使用 `use_numpy_format=True`（无损）
- 远程/云端：使用 `use_numpy_format=False`（压缩）

## 🔧 高级用法

### 多图像推理（双目相机/多视角）

```python
# 加载多张图像
image1 = Image.open("left_camera.png")
image2 = Image.open("right_camera.png")

# 一次性推理
result = client.predict(
    images=[image1, image2],  # 传递多张图像
    state=state,
    prompt="Pick up an apple.",
    use_numpy_format=True
)
```

### 批量推理（提高吞吐量）

```python
result = client.predict_batch(
    batch_images=[[image1], [image2], [image3]],
    batch_states=[state1, state2, state3],
    batch_prompts=["prompt1", "prompt2", "prompt3"],
    use_numpy_format=True
)
```

### 自定义推理参数

```python
result = client.predict(
    images=[image],
    state=state,
    prompt="Pick up an apple.",
    use_numpy_format=True,
    temperature=1.5,      # 调整采样温度
    num_steps=100         # 增加采样步数
)
```

## 📝 API 接口

### `POST /predict` - 单次推理

**请求体：**
```json
{
  "images": ["base64_string"] 或 [[[...]]], // base64 或 numpy
  "state": [0, 0, ...],
  "prompt": "Pick up an apple.",
  "image_format": "numpy",  // 可选："base64" 或 "numpy"
  "temperature": 1.0,       // 可选
  "num_steps": 50          // 可选
}
```

**响应：**
```json
{
  "actions": [[...], [...], ...],
  "timing": {
    "total_time": 0.15,
    "preprocess_time": 0.02,
    "inference_time": 0.13
  }
}
```

### `POST /predict_batch` - 批量推理

支持一次处理多个请求，提高吞吐量。

### `GET /health` - 健康检查

检查服务器状态和模型是否已加载。

### `GET /info` - 模型信息

返回当前加载的 VLM 后端和动作头信息。

## 🛠️ 训练新模型

### 训练动作头

```bash
cd Action_Heads/Flow_Matching_1
python train.py
```

### 使用预训练权重

```bash
cd Action_Heads/Flow_Matching_0
python train_with_pretrained_action_head_weight.py
```

## 📚 更多文档

- **部署详细文档**: `deployment/sai0_server/README.md`
- **图像格式说明**: `deployment/sai0_server/IMAGE_FORMATS.md`
- **动作头文档**: `Action_Heads/*/README.md`
- **VLM 文档**: `VLMs/S0/README.md`

## 🐛 常见问题

### Q: 服务器启动失败？
A: 检查 GPU 是否可用，以及动作头路径是否正确。

### Q: RGBA 图像报错？
A: 最新版本已自动处理 RGBA → RGB 转换，无需手动处理。

### Q: 推理速度慢？
A: 使用批量推理（`/predict_batch`）或减少 `num_steps`。

### Q: 如何切换 VLM 后端？
A: 修改 `config.yaml` 中的 `vlm_backend` 或使用 `start_server.sh --vlm-backend <name>`。

## 📄 许可证

[待补充]

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**快速开始命令汇总：**

```bash
# 1. 启动服务器
./start_server.sh

# 2. 运行完整示例
python deployment/sai0_server/example_client_test.py

# 3. 命令行测试
python deployment/sai0_server/client_test.py --image test.png --prompt "Pick up an apple." --numpy

# 4. 查看 API 文档
# 浏览器访问: http://localhost:5000/docs
```
