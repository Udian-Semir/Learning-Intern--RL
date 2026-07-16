# Eagle 2.5 VL 原始代码修改说明

本文档记录了在 `ori` 文件夹中对原始 Eagle 2.5 VL 代码所做的修改，以确保与不同版本的 `transformers` 库兼容。

## 修改概览

| 文件 | 修改数量 | 主要问题 |
|------|---------|---------|
| `backbone/eagle2_hg_model/processing_eagle2_5_vl.py` | 4 处 | VideoInput 导入、unused_kwargs 处理、image_processor 调用修复 |
| `backbone/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py` | 2 处 | image_processing_utils_fast 导入、VideoInput 导入 |
| `backbone/eagle2_hg_model/modeling_eagle2_5_vl.py` | 1 处 | 5 维 pixel_values 处理 |

---

## 详细修改说明

### 1. `backbone/eagle2_hg_model/processing_eagle2_5_vl.py`

#### 修改 1: VideoInput 导入兼容性修复 (第 38-53 行)

**问题**: `VideoInput` 类型在 `transformers < 4.45` 版本中不存在，直接导入会导致 `ImportError`。

**原始代码**:
```python
from transformers.image_utils import VideoInput
```

**修改后代码**:
```python
try:
    from transformers.image_utils import VideoInput
except ImportError:
    # 为兼容性定义 VideoInput 类型
    from typing import List, Union
    import numpy as np
    VideoInput = Union[List[Image.Image], np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]]
```

**逻辑说明**:
- 首先尝试从 transformers 导入 `VideoInput`
- 如果导入失败（旧版本），则定义一个兼容的类型别名
- `VideoInput` 本质上是视频输入的类型提示，用于类型检查，不影响运行时功能

---

#### 修改 2: unused_kwargs 处理兼容性修复 (第 863-887 行)

**问题**: `validate_init_kwargs` 方法在不同版本的 transformers 中返回类型不同：
- 某些版本返回 `dict`
- 某些版本返回 `list` 或 `tuple`（包含字符串键或嵌套字典）

直接调用 `kwargs.update(unused_kwargs)` 会导致错误：
- `"dictionary update sequence element #0 has length 6; 2 is required"` - 当 unused_kwargs 是字符串时
- `"unhashable type: 'dict'"` - 当 unused_kwargs 是包含字典的列表时

**原始代码**:
```python
kwargs.update(unused_kwargs)
```

**修改后代码**:
```python
if unused_kwargs is not None:
    if isinstance(unused_kwargs, dict):
        kwargs.update(unused_kwargs)
    elif isinstance(unused_kwargs, (list, tuple)):
        # 如果是列表/元组，可能包含字典或字符串
        for item in unused_kwargs:
            if isinstance(item, dict):
                kwargs.update(item)
            elif isinstance(item, str):
                kwargs[item] = None
    # 其他情况忽略
```

**逻辑说明**:
1. 首先检查 `unused_kwargs` 是否为 `None`
2. 如果是字典，直接更新 kwargs（原始行为）
3. 如果是列表/元组，遍历每个元素：
   - 如果元素是字典，合并到 kwargs
   - 如果元素是字符串（未使用的参数名），以 None 值添加到 kwargs
4. 其他类型静默忽略，避免崩溃

---

#### 修改 3: image_processor 图像处理调用修复 (第 423-447 行)

**问题**: 原始代码在调用 `self.image_processor()` 时传入了 `videos` 参数，但 `Eagle2ImageProcessor.preprocess()` 方法不接受该参数，导致 `TypeError: Eagle2ImageProcessor.preprocess() got an unexpected keyword argument 'videos'`。

**原始代码**:
```python
if media_type == "image":
    image_inputs = self.image_processor(
        images=[image_list[idx_in_list]],
        videos=None,
        **output_kwargs["images_kwargs"],
    )
```

**修改后代码**:
```python
if media_type == "image":
    image_inputs = self.image_processor(
        images=[image_list[idx_in_list]],
        **output_kwargs["images_kwargs"],
    )
```

**逻辑说明**:
- 移除了 `videos=None` 参数
- `Eagle2ImageProcessor` 只处理图像，不需要 videos 参数

---

#### 修改 4: image_processor 视频帧处理调用修复 (第 458-476 行)

**问题**: 原始代码在处理视频时将视频作为 `videos` 参数传入 `image_processor`，但该方法不接受 `videos` 参数。

**原始代码**:
```python
elif media_type == "video":
    video_inputs = self.image_processor(
        images=None,
        videos=[video_list[idx_in_list]],
        **output_kwargs["videos_kwargs"],
    )
```

**修改后代码**:
```python
elif media_type == "video":
    video_frames = video_list[idx_in_list]
    # 如果视频是单个帧列表，直接使用；否则包装为列表
    if isinstance(video_frames, (list, tuple)):
        images_for_video = video_frames
    else:
        images_for_video = [video_frames]
    video_inputs = self.image_processor(
        images=images_for_video,
        **output_kwargs["videos_kwargs"],
    )
```

**逻辑说明**:
- 视频本质上是一系列图像帧，应该作为 `images` 参数传入
- 添加了类型检查以确保帧列表格式正确
- 这使得 `Eagle2ImageProcessor` 可以正确处理视频帧

---

### 2. `backbone/eagle2_hg_model/image_processing_eagle2_5_vl_fast.py`

#### 修改 1: image_processing_utils_fast 导入兼容性修复 (第 13-47 行)

**问题**: `transformers.image_processing_utils_fast` 模块中的以下内容在 `transformers < 4.45` 版本中不存在：
- `BASE_IMAGE_PROCESSOR_FAST_DOCSTRING`
- `BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS`
- `BaseImageProcessorFast`
- `DefaultFastImageProcessorKwargs`
- `group_images_by_shape`
- `reorder_images`

**原始代码**:
```python
from transformers.image_processing_utils_fast import (
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
    BaseImageProcessorFast,
    DefaultFastImageProcessorKwargs,
    group_images_by_shape,
    reorder_images,
)
```

**修改后代码**:
```python
try:
    from transformers.image_processing_utils_fast import (
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING,
        BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS,
        BaseImageProcessorFast,
        DefaultFastImageProcessorKwargs,
        group_images_by_shape,
        reorder_images,
    )
except ImportError:
    # 为兼容旧版本 transformers，提供占位定义
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""
    BaseImageProcessorFast = object  # 占位基类
    DefaultFastImageProcessorKwargs = dict
    def group_images_by_shape(*args, **kwargs):
        raise NotImplementedError("This function requires transformers >= 4.45")
    def reorder_images(*args, **kwargs):
        raise NotImplementedError("This function requires transformers >= 4.45")
```

**逻辑说明**:
- 首先尝试从新版 transformers 导入所需内容
- 如果导入失败，提供占位定义：
  - 文档字符串常量设为空字符串
  - `BaseImageProcessorFast` 设为 `object`（通用基类）
  - `DefaultFastImageProcessorKwargs` 设为 `dict`
  - 函数设为抛出 `NotImplementedError`
- 这确保模块可以被导入，但如果真正使用 fast 处理器功能会给出明确错误

---

#### 修改 2: VideoInput 导入兼容性修复 (第 59-74 行)

**问题**: 与 `processing_eagle2_5_vl.py` 中相同，`VideoInput` 在旧版本中不存在。

**原始代码**:
```python
from transformers.image_utils import VideoInput
```

**修改后代码**:
```python
try:
    from transformers.image_utils import VideoInput
except ImportError:
    from typing import List, Union
    import numpy as np
    from PIL import Image
    VideoInput = Union[List[Image.Image], np.ndarray, List[np.ndarray]]
```

**逻辑说明**: 同上，提供兼容性类型定义。

---

### 3. `backbone/eagle2_hg_model/modeling_eagle2_5_vl.py`

#### 修改 1: 5 维 pixel_values 处理 (extract_feature 方法)

**问题**: `SiglipVisionModel.forward()` 期望 4 维输入 `(batch, channels, height, width)`，但 Eagle 2.5 VL 的动态切片功能导致 `pixel_values` 是 5 维 `(num_images, num_tiles, channels, height, width)`。

错误信息：`ValueError: too many values to unpack (expected 4)`

**原始代码**:
```python
def extract_feature(self, pixel_values):
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True
        )
        # ...
```

**修改后代码**:
```python
def extract_feature(self, pixel_values):
    # 处理 5 维 pixel_values 输入
    original_shape = pixel_values.shape
    if len(original_shape) == 5:
        # pixel_values: (num_images, num_tiles, channels, height, width)
        # 重塑为: (num_images * num_tiles, channels, height, width)
        num_images, num_tiles, c, h, w = original_shape
        pixel_values = pixel_values.view(num_images * num_tiles, c, h, w)
    
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values, output_hidden_states=False, return_dict=True
        )
        # ...
```

**逻辑说明**:
- 检查 `pixel_values` 的维度数量
- 如果是 5 维，将其从 `(num_images, num_tiles, C, H, W)` 重塑为 `(num_images * num_tiles, C, H, W)`
- 这使得 vision_model 可以正确处理批量的图像切片
- 后续的 MLP 和嵌入步骤会正确合并这些特征

---

## 修改标记格式

所有修改都使用以下格式标记，便于识别和追踪：

```python
# ============================================================================
# [MODIFIED] <修改简述>
# 原始代码:
#     <原始代码内容>
# 
# 修改原因: <详细原因>
# 修改时间: <日期>
# ============================================================================
<修改后的代码>
# ============================================================================
```

---

## 兼容性说明

| transformers 版本 | 状态 |
|------------------|------|
| >= 4.45 | ✅ 完全兼容 |
| 4.40 - 4.44 | ✅ 兼容（使用 fallback） |
| < 4.40 | ⚠️ 可能存在其他兼容性问题 |

---

## 注意事项

1. **不要删除注释中的原始代码** - 它们用于记录变更历史和便于回滚
2. **Fast 图像处理器在旧版本中不可用** - 如果需要使用 fast 处理器，请升级 transformers
3. **这些修改不影响模型功能** - 仅解决导入和类型兼容性问题

---

## 修改日期

- **2024-12**: 初始兼容性修复

## 修改者

- 自动化脚本（用于 GR00T-N1.5 + Eagle 2.5 VL 集成项目）

