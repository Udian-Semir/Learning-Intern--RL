# 图像格式使用说明

## 支持的两种图像格式

### 1. Base64 格式（默认）

**特点：**
- ✅ 数据量小（JPEG 压缩）
- ✅ 网络传输快
- ❌ 有损压缩，可能影响精度

**使用场景：**
- 广域网/云端部署
- 带宽受限的场景
- 对图像质量要求不高的任务

**示例：**
```python
from client import Sai0Client
import numpy as np
from PIL import Image

client = Sai0Client("http://localhost:5000")

image = Image.open("frame.jpg")
state = np.zeros(16)

# 使用 base64 格式（默认）
result = client.predict(
    images=[image],
    state=state,
    use_numpy_format=False  # 或者省略（默认 False）
)
```

**API 请求格式：**
```json
{
  "images": [
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBg...",  // base64 字符串
    "..."
  ],
  "state": [0.1, 0.2, ..., 0.16],
  "image_format": "base64"
}
```

---

### 2. Numpy 格式（推荐，无损）

**特点：**
- ✅ 无损传输，完整保留图像数据
- ✅ 保证推理精度
- ❌ JSON 数据量大，网络传输较慢

**使用场景：**
- 局域网部署（推荐）
- 本地推理
- 对精度要求高的任务
- 输入已经是 numpy array 的场景

**示例：**
```python
from client import Sai0Client
import numpy as np
from PIL import Image

client = Sai0Client("http://localhost:5000")

# 方式 1: 从 PIL Image
image = Image.open("frame.jpg")
result = client.predict(
    images=[image],
    state=state,
    use_numpy_format=True  # 使用 numpy 格式
)

# 方式 2: 直接传递 numpy array
img_array = np.array(image, dtype=np.uint8)  # shape: (H, W, 3)
result = client.predict(
    images=[img_array],  # 直接传递
    state=state,
    use_numpy_format=True
)
```

**API 请求格式：**
```json
{
  "images": [
    [
      [[255, 128, 64], [255, 130, 65], ...],  // 第一行像素
      [[254, 127, 63], [253, 129, 64], ...],  // 第二行像素
      ...
    ],
    ...
  ],
  "state": [0.1, 0.2, ..., 0.16],
  "image_format": "numpy"
}
```

---

## 性能对比

### 局域网环境 (1 Gbps)

| 格式 | 数据量 (单图) | 传输时间 | 推理时间 | 总时间 |
|------|--------------|---------|---------|--------|
| Base64 | ~50 KB | ~5 ms | ~150 ms | ~155 ms |
| Numpy | ~2 MB | ~20 ms | ~150 ms | ~170 ms |

**结论：局域网推荐 Numpy 格式**（时间差异小，但精度更高）

### 广域网环境 (10 Mbps)

| 格式 | 数据量 (单图) | 传输时间 | 推理时间 | 总时间 |
|------|--------------|---------|---------|--------|
| Base64 | ~50 KB | ~40 ms | ~150 ms | ~190 ms |
| Numpy | ~2 MB | ~1600 ms | ~150 ms | ~1750 ms |

**结论：广域网推荐 Base64 格式**（传输时间占主导）

---

## 完整示例

### 机器人实时控制（局域网）

```python
from client import Sai0Client
import numpy as np

# 初始化客户端
client = Sai0Client("http://192.168.1.100:5000")

# 机器人控制循环
while True:
    # 从相机获取图像（已经是 numpy array）
    images = [
        camera1.read(),  # numpy array (480, 640, 3)
        camera2.read()
    ]
    
    # 从机器人获取状态
    state = robot.get_state()  # numpy array (16,)
    
    # 预测动作（使用 numpy 格式，无损）
    result = client.predict(
        images=images,
        state=state,
        use_numpy_format=True  # 推荐
    )
    
    # 获取动作
    actions = np.array(result['actions'])
    
    # 发送动作到机器人
    robot.execute(actions)
```

### 云端推理（广域网）

```python
from client import Sai0Client
from PIL import Image
import numpy as np

# 连接云端服务器
client = Sai0Client("http://cloud-server.com:5000")

# 加载图像
image = Image.open("frame.jpg")
state = np.zeros(16)

# 使用 base64 格式（压缩，节省带宽）
result = client.predict(
    images=[image],
    state=state,
    use_numpy_format=False  # 使用 base64
)

actions = result['actions']
```

---

## 最佳实践

1. **局域网/本地部署**：使用 `use_numpy_format=True`
   - 无损传输，保证精度
   - 网络延迟小，传输时间可接受

2. **广域网/云端部署**：使用 `use_numpy_format=False`（默认）
   - 压缩传输，节省带宽
   - 网络延迟大，传输时间占主导

3. **输入已是 numpy array**：直接传递，使用 `use_numpy_format=True`
   - 避免 PIL 转换
   - 保持原始数据格式

4. **对比测试**：使用 `example_image_formats.py` 测试两种格式
   ```bash
   python example_image_formats.py
   ```

---

## curl 测试示例

### Base64 格式

```bash
# 1. 将图像转换为 base64
IMAGE_BASE64=$(base64 -w 0 frame.jpg)

# 2. 发送请求
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"images\": [\"$IMAGE_BASE64\"],
    \"state\": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    \"image_format\": \"base64\"
  }"
```

### Numpy 格式

```bash
# 1. 将图像转换为 numpy array JSON（使用 Python）
python -c "
import numpy as np
from PIL import Image
import json

img = Image.open('frame.jpg')
img_array = np.array(img).tolist()
print(json.dumps(img_array))
" > image_numpy.json

# 2. 发送请求
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d "{
    \"images\": [$(cat image_numpy.json)],
    \"state\": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    \"image_format\": \"numpy\"
  }"
```

---

## 故障排除

### 问题 1: Numpy 格式返回 "Invalid image shape"

**原因：** 图像数组 shape 不正确

**解决：**
```python
# 确保是 (H, W, 3) 格式，dtype=uint8
img_array = np.array(image, dtype=np.uint8)
assert img_array.ndim == 3
assert img_array.shape[2] == 3
```

### 问题 2: Base64 格式图像失真

**原因：** JPEG 压缩导致

**解决：** 使用 numpy 格式，或提高 JPEG 质量

### 问题 3: Numpy 格式传输太慢

**原因：** JSON 数据量大

**解决：** 
- 使用 base64 格式
- 或升级网络带宽
- 或考虑本地部署
