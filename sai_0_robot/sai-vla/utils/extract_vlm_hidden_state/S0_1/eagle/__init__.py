"""
Eagle 2.5 VL Hidden States 提取模块

提供从 LeRobot 数据集中使用 Eagle 2.5 VL 模型提取 hidden states 的功能。

使用示例:
    # 命令行使用
    python -m utils.extract_vlm_hidden_state.S0_1.eagle.eagle_extract_vlm_hidden_states \\
        --dataset_path /path/to/dataset
    
    # Python API
    from utils.extract_vlm_hidden_state.S0_1.eagle import extract_eagle_hidden_states
    
    processed, errors = extract_eagle_hidden_states(
        dataset_path="/path/to/dataset",
        layers=[-4, -3, -2],
        image_keys=["top", "left_wrist"]
    )
"""

from .eagle_extract_vlm_hidden_states import (
    extract_eagle_hidden_states,
    DEFAULT_EAGLE_MODEL_PATH,
    DEFAULT_LAYERS,
    DEFAULT_IMAGE_KEYS,
)

__all__ = [
    "extract_eagle_hidden_states",
    "DEFAULT_EAGLE_MODEL_PATH",
    "DEFAULT_LAYERS",
    "DEFAULT_IMAGE_KEYS",
]

