"""
Cosmos Reason 2B VL Hidden States 提取模块

提供从 LeRobot 数据集中使用 Cosmos Reason 2B VL 模型提取 hidden states 的功能。

使用示例:
    # 命令行使用
    python -m utils.extract_vlm_hidden_state.S0_1.cosmos.cosmos_extract_vlm_hidden_states \\
        --dataset_path /path/to/dataset
    
    # Python API
    from utils.extract_vlm_hidden_state.S0_1.cosmos import extract_cosmos_hidden_states
    
    processed, errors = extract_cosmos_hidden_states(
        dataset_path="/path/to/dataset",
        layers=[-1],
        image_keys=["top", "left_wrist"]
    )
"""

from .cosmos_extract_vlm_hidden_states import (
    extract_cosmos_hidden_states,
    DEFAULT_COSMOS_MODEL_PATH,
    DEFAULT_LAYERS,
    DEFAULT_IMAGE_KEYS,
)

__all__ = [
    "extract_cosmos_hidden_states",
    "DEFAULT_COSMOS_MODEL_PATH",
    "DEFAULT_LAYERS",
    "DEFAULT_IMAGE_KEYS",
]

