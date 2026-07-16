"""
Sai0_1 VLA 模块

整合 VLM Backbone (S0_1) 和 Action Heads 的统一接口。

================================================================================
                              模块结构
================================================================================

VLAs/Sai0_1/
├── __init__.py          # 模块入口 (当前文件)
├── config.py            # 配置类
├── data_utils.py        # 数据加载工具
├── sai0_model.py        # 主模型类
└── inference.py         # 推理工具

================================================================================
                              使用示例
================================================================================

1. 训练模式 (使用预提取的 VLM hidden states):

    from VLAs.Sai0_1 import Sai0Config, Sai0Model, create_sai0_dataloader
    
    # 创建配置
    config = Sai0Config.for_qwen3_libero(
        dataset_path="/path/to/lerobot_dataset",
        pretrained_weights="./pretrained_action_head.pt",
    )
    
    # 创建数据加载器
    dataloader, normalizers = create_sai0_dataloader(
        dataset_path=config.data.dataset_path,
        batch_size=32,
        num_action_chunks=16,
    )
    
    # 创建模型 (只需要 Action Head)
    model = Sai0Model(config)
    
    # 训练循环
    for backbone_output, action_head_inputs in dataloader:
        loss = model.compute_loss(backbone_output, action_head_inputs)
        loss.backward()
        optimizer.step()

2. 推理模式:

    from VLAs.Sai0_1 import Sai0Inference
    from PIL import Image
    import numpy as np
    
    # 从 checkpoint 加载
    inference = Sai0Inference.from_checkpoint(
        checkpoint_path="./checkpoints/best/action_head.pt",
        vlm_type="qwen3_vl",
        vlm_model_path="Qwen/Qwen3-VL-2B-Instruct",
        dataset_path="/path/to/dataset",  # 用于加载归一化统计
    )
    
    # 准备输入
    images = [Image.open("agentview.jpg"), Image.open("wrist.jpg")]
    instruction = "pick up the red apple"
    state = np.array([...])  # 当前机器人状态
    
    # 预测动作
    actions = inference.predict(images, instruction, state)
    # actions: (num_chunks, action_dim)

3. 数据集评估:

    from VLAs.Sai0_1 import evaluate_checkpoint
    
    results = evaluate_checkpoint(
        checkpoint_path="./checkpoints/best/action_head.pt",
        dataset_path="/path/to/eval_dataset",
        vlm_type="qwen3_vl",
        num_samples=100,
    )
    print(f"Mean Loss: {results['mean_loss']:.6f}")

4. 快速推理 (一次性使用):

    from VLAs.Sai0_1 import quick_inference
    
    actions = quick_inference(
        images=[img1, img2],
        instruction="pick up the apple",
        state=current_state,
        checkpoint_path="./checkpoint.pt",
        vlm_type="qwen3_vl",
    )

================================================================================
                              配置预设
================================================================================

- Sai0Config.for_qwen3_libero():  Qwen3-VL-2B + LIBERO 数据集
- Sai0Config.for_eagle_libero():  Eagle 2.5 VL + LIBERO 数据集

================================================================================
                              组件说明
================================================================================

VLM Backbone (S0_1):
    - 支持 Qwen3-VL (2B, 4B, 7B)
    - 支持 Eagle 2.5 VL (GR00T-N1.5-3B)
    - 提取视觉-语言特征

Action Heads:
    ┌────────────────────┬──────────────────────────────────────────────────┐
    │ 类型                │ 说明                                              │
    ├────────────────────┼──────────────────────────────────────────────────┤
    │ flow_matching_0    │ GR00T N1.5 原始 Flow Matching 架构                │
    │ (别名: fm0)        │ - 支持预训练权重                                   │
    │                    │ - max_action_dim=32, max_state_dim=64            │
    ├────────────────────┼──────────────────────────────────────────────────┤
    │ flow_matching_1    │ 自定义 Flow Matching 架构                         │
    │ (别名: fm1)        │ - 支持多层 VLM hidden states                      │
    │                    │ - 可配置 vlm_output_dim, action_backbone_dim     │
    ├────────────────────┼──────────────────────────────────────────────────┤
    │ oft_1_0            │ OFT (L1 Regression + Transformer)                │
    │ (别名: oft)        │ - 轻量级架构                                      │
    │                    │ - 支持 Diffusion 或 L1 回归                       │
    └────────────────────┴──────────────────────────────────────────────────┘

Data Utils:
    - 基于 LeRobot 格式
    - 支持 VLM hidden states 加载
    - 自动归一化处理
    - 四元数到轴角转换

================================================================================
                           Action Head 使用示例
================================================================================

# Flow Matching 0 (默认，GR00T N1.5)
config = Sai0Config(
    action_head=ActionHeadConfig.for_flow_matching_0(
        pretrained_weights="./pretrained.pt"
    )
)

# Flow Matching 1 (自定义，支持多层 VLM)
config = Sai0Config(
    action_head=ActionHeadConfig.for_flow_matching_1(
        vlm_output_dim=2048,  # VLM 隐藏维度
        action_backbone_dim=1536,
    )
)

# OFT 1.0
config = Sai0Config(
    action_head=ActionHeadConfig.for_oft(
        llm_output_dim=4096,
        num_vlm_layers=3,
    )
)
"""

__version__ = "0.1.0"

# ============================================================================
# 配置类
# ============================================================================

from .config import (
    Sai0Config,
    VLMConfig,
    ActionHeadConfig,
    DataConfig,
    TrainingConfig,
    get_vlm_hidden_dim,
)

# ============================================================================
# 数据工具
# ============================================================================

from .data_utils import (
    # DataLoader 创建
    create_sai0_dataloader,
    create_dataloader_from_config,
    
    # 归一化
    MinMaxNormalizer,
    load_normalization_stats,
    
    # Collate 函数
    sai0_collate_fn,
    
    # 数据集信息
    get_dataset_info,
    print_dataset_info,
    
    # 数据转换
    quat2axisangle_torch,
    convert_state_quat_to_axisangle,
)

# ============================================================================
# 模型类
# ============================================================================

from .sai0_model import (
    Sai0Model,
    create_sai0_model,
    create_model_from_config,
)

# ============================================================================
# 推理工具
# ============================================================================

from .inference import (
    Sai0Inference,
    RealtimeInference,
    quick_inference,
    evaluate_checkpoint,
)

# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 版本
    "__version__",
    
    # 配置类
    "Sai0Config",
    "VLMConfig",
    "ActionHeadConfig",
    "DataConfig",
    "TrainingConfig",
    "get_vlm_hidden_dim",
    
    # 数据工具
    "create_sai0_dataloader",
    "create_dataloader_from_config",
    "MinMaxNormalizer",
    "load_normalization_stats",
    "sai0_collate_fn",
    "get_dataset_info",
    "print_dataset_info",
    "quat2axisangle_torch",
    "convert_state_quat_to_axisangle",
    
    # 模型类
    "Sai0Model",
    "create_sai0_model",
    "create_model_from_config",
    
    # 推理工具
    "Sai0Inference",
    "RealtimeInference",
    "quick_inference",
    "evaluate_checkpoint",
]


# ============================================================================
# 便捷函数
# ============================================================================

def create_model(
    vlm_type: str = "qwen3_vl",
    vlm_model_path: str = None,
    pretrained_weights: str = None,
    device: str = "cuda:0",
    **kwargs
) -> Sai0Model:
    """
    快速创建 Sai0 模型
    
    Args:
        vlm_type: VLM 类型 (qwen3_vl, eagle2_5_vl)
        vlm_model_path: VLM 模型路径
        pretrained_weights: Action Head 预训练权重
        device: 设备
        **kwargs: 其他配置参数
    
    Returns:
        Sai0Model 实例
        
    Example:
        model = create_model(
            vlm_type="qwen3_vl",
            pretrained_weights="./checkpoint.pt"
        )
    """
    return create_sai0_model(
        vlm_type=vlm_type,
        vlm_model_path=vlm_model_path,
        pretrained_weights=pretrained_weights,
        device=device,
        **kwargs
    )


def create_dataloader(
    dataset_path: str,
    batch_size: int = 32,
    num_action_chunks: int = 16,
    **kwargs
):
    """
    快速创建数据加载器
    
    Args:
        dataset_path: 数据集路径
        batch_size: 批次大小
        num_action_chunks: Action chunk 数量
        **kwargs: 其他参数
    
    Returns:
        (DataLoader, normalizers) 元组
        
    Example:
        dataloader, normalizers = create_dataloader(
            dataset_path="/path/to/dataset",
            batch_size=32
        )
    """
    return create_sai0_dataloader(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_action_chunks=num_action_chunks,
        **kwargs
    )

