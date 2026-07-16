# Cosmos Reason 2B VL (GR00T N1.6) 原始代码修改说明

本文档记录了在 `ori` 文件夹中对原始 GR00T N1.6 代码所做的修改，以确保不依赖外部 `gr00t` 包。

## 修改概览

| 文件 | 修改数量 | 主要问题 |
|------|---------|---------|
| `gr00t_n1d6/gr00t_n1d6.py` | 1 处 | gr00t 包导入替换为相对导入和本地定义 |
| `gr00t_n1d6/processing_gr00t_n1d6.py` | 1 处 | gr00t 包导入替换为本地定义 |
| `gr00t_n1d6/setup.py` | 1 处 | gr00t 包导入替换为相对导入和本地定义 |
| `base/model_pipeline.py` | 1 处 | gr00t 包导入替换为本地定义 |

---

## 详细修改说明

### 1. `gr00t_n1d6/gr00t_n1d6.py`

#### 修改: gr00t 导入替换

**问题**: 原始代码依赖外部 `gr00t` 包的导入。

**原始代码**:
```python
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT
from gr00t.model.modules.eagle_backbone import EagleBackbone
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
```

**修改后代码**:
```python
# 使用相对导入从本地 modules 目录
from ..modules.dit import AlternateVLDiT, DiT
from ..modules.eagle_backbone import EagleBackbone
from ..modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)

# Gr00tN1d6Config 本地定义
@dataclass
class Gr00tN1d6Config(PretrainedConfig):
    """GR00T N1.6 模型配置类"""
    model_type = "gr00t_n1d6"
    # ... 完整配置定义
```

---

### 2. `gr00t_n1d6/processing_gr00t_n1d6.py`

#### 修改: gr00t 导入替换为本地定义

**问题**: 原始代码依赖 gr00t 包的数据处理类。

**原始代码**:
```python
from gr00t.configs.data.embodiment_configs import ModalityConfig
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.interfaces import BaseProcessor
from gr00t.data.state_action.state_action_processor import StateActionProcessor
from gr00t.data.utils import parse_modality_configs, to_json_serializable
```

**修改后代码**:
```python
# 本地定义 ModalityConfig
@dataclass
class ModalityConfig:
    """模态配置类"""
    modality_keys: List[str] = field(default_factory=list)
    delta_indices: List[int] = field(default_factory=list)

# 本地定义 EmbodimentTag
class EmbodimentTag(str, Enum):
    """Embodiment 标签枚举"""
    LIBERO_PANDA = "libero_panda"
    # ...

# 本地定义 BaseProcessor
class BaseProcessor:
    """基础处理器类"""
    # ...

# 本地定义辅助函数
def parse_modality_configs(configs: dict) -> dict:
    return configs

def to_json_serializable(obj):
    # ...
```

---

### 3. `gr00t_n1d6/setup.py`

#### 修改: gr00t 导入替换为相对导入

**问题**: 原始代码依赖 gr00t 包的配置和模型类。

**原始代码**:
```python
from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.experiment.dist_utils import get_rank
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.model.gr00t_n1d6.processing_gr00t_n1d6 import Gr00tN1d6Processor
from gr00t.model.registry import register_model
```

**修改后代码**:
```python
# 使用相对导入
from .gr00t_n1d6 import Gr00tN1d6, Gr00tN1d6Config
from .processing_gr00t_n1d6 import Gr00tN1d6Processor
from ..base.model_pipeline import ModelPipeline
from ..registry import register_model

# 本地定义 Config 和其他占位符
@dataclass
class Config:
    model: Any = None
    data: Any = None
    training: Any = None

def get_rank() -> int:
    # ...

class DatasetFactory:
    # ...
```

---

### 4. `base/model_pipeline.py`

#### 修改: gr00t 导入替换为本地定义

**问题**: 原始代码依赖 gr00t 包的基础类。

**原始代码**:
```python
from gr00t.configs.base_config import Config
from gr00t.data.collator import BasicDataCollator
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.interfaces import BaseProcessor
```

**修改后代码**:
```python
# 本地定义占位符类
@dataclass
class Config:
    model: Any = None
    data: Any = None
    training: Any = None

class BasicDataCollator:
    def __call__(self, features):
        return features

class DatasetFactory:
    # ...

class BaseProcessor:
    pass
```

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

## 重要说明

### modules/ 目录

`modules/` 目录下的文件（`eagle_backbone.py`, `dit.py`, `embodiment_conditioned_mlp.py` 等）**没有**依赖 `gr00t` 包，因此无需修改。

### nvidia/Eagle-Block2A-2B-v2/ 目录

`modules/nvidia/Eagle-Block2A-2B-v2/` 目录包含 Eagle3_VL 模型的完整实现，这些文件**没有**依赖 `gr00t` 包，可以直接被 `CosmosReason2BVLBackbone` 使用。

---

## 兼容性说明

本地定义的类和函数是**占位符**，提供基本功能以确保代码可以被导入和使用。如果需要完整的 GR00T N1.6 训练功能，可能需要：

1. 安装完整的 `gr00t` 包
2. 或者完善本地定义的占位符类

对于 `CosmosReason2BVLBackbone` 的 hidden states 提取功能，这些占位符已经足够。

---

## 注意事项

1. **不要删除注释中的原始代码** - 它们用于记录变更历史和便于回滚
2. **占位符类可能功能不完整** - 仅用于确保导入不报错
3. **这些修改不影响 backbone 核心功能** - `CosmosReason2BVLBackbone` 直接使用 Eagle3_VL 模型

---

## 修改日期

- **2025-01**: 初始兼容性修复，移除 gr00t 包依赖

## 修改者

- 自动化脚本（用于 Cosmos Reason 2B VL 集成项目）

