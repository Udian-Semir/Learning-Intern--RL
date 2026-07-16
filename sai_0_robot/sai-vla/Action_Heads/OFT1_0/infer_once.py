"""
OFT Action Head 单次推理脚本
Single Inference Script for OFT Action Head

使用方法 Usage:
  python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1
  python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1 --device cuda:7

============================================================================
NPZ 文件格式要求 Required NPZ File Format:
============================================================================

如果要使用真实数据进行推理，需要提供包含以下字段的 .npz 文件：
If using real data for inference, provide a .npz file with the following fields:

必需字段 Required fields:
  - backbone_features: np.ndarray, shape (seq_len, hidden_dim)
    例如 Example: (1536, 2048) 表示 1536 个 token，每个 2048 维
    说明: VLM 输出的特征向量，通常是多层拼接后的结果
    Note: Feature vectors from VLM, usually concatenated from multiple layers
  
  - backbone_attention_mask: np.ndarray, shape (seq_len,), dtype=int64
    例如 Example: (1536,) 全1表示所有token都有效
    说明: 注意力掩码，1表示有效token，0表示padding
    Note: Attention mask, 1=valid token, 0=padding
  
  - state: np.ndarray, shape (1, state_dim)
    例如 Example: (1, 64) 表示 1 个时间步，64 维状态
    说明: 机器人当前状态（关节位置、速度等），需要padding到max_state_dim
    Note: Robot current state (joint positions, velocities, etc.), padded to max_state_dim

创建 NPZ 示例 Example of creating NPZ:
  ```python
  import numpy as np
  
  # 假设从 VLM 获取了 3 层隐藏状态，每层 512 tokens，每个 token 2048 维
  # Assume we got 3 layers of hidden states from VLM, 512 tokens per layer, 2048 dim each
  vlm_hidden = ...  # shape: (3, 512, 2048)
  
  # 拼接所有层: (3, 512, 2048) -> (1536, 2048)
  # Concatenate all layers
  backbone_features = vlm_hidden.reshape(-1, 2048)  # (1536, 2048)
  
  # 创建注意力掩码（全1）
  # Create attention mask (all ones)
  backbone_attention_mask = np.ones(1536, dtype=np.int64)
  
  # 机器人状态（假设16维，需要padding到64维）
  # Robot state (assume 16-dim, need padding to 64-dim)
  state = np.zeros((1, 64), dtype=np.float32)
  state[0, :16] = [0.1, 0.2, ...]  # 实际状态值 actual state values
  
  # 保存为 NPZ
  # Save as NPZ
  np.savez('sample.npz',
           backbone_features=backbone_features,
           backbone_attention_mask=backbone_attention_mask,
           state=state)
  ```

注意事项 Notes:
  - backbone_features 的 seq_len 应该是 num_layers * tokens_per_layer
  - state 需要 padding 到 max_state_dim（通常是64）
  - 所有数组的 dtype 应该是 float32 或 int64
  - NPZ file should match the dimensions expected by the model
============================================================================
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vlm2oft_pipeline import create_vlm2oft_pipeline
from constants import (
    NUM_VLM_HIDDEN_LAYERS,
    LLM_OUTPUT_DIM_MLP_INPUT_DIM,
    PROPRIO_DIM,
    ACTION_DIM,
    NUM_ACTIONS_CHUNK
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="检查点目录路径")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备")
    parser.add_argument("--num_transformer_blocks", type=int, default=2, help="Transformer blocks 数量")
    parser.add_argument("--num_attention_heads", type=int, default=8, help="注意力头数量")
    parser.add_argument("--vlm_output_dim", type=int, default=None, help="VLM输出维度（默认从constants读取）")
    parser.add_argument("--num_vlm_layers", type=int, default=None, help="VLM层数（默认从constants读取）")
    parser.add_argument("--npz", type=str, default="", help="NPZ数据文件路径（可选，不提供则使用随机输入）")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    
    # 加载配置（如果存在）
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        print(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            cfg_dict = json.load(f)
        num_transformer_blocks = cfg_dict.get("num_transformer_blocks", args.num_transformer_blocks)
        num_attention_heads = cfg_dict.get("num_attention_heads", args.num_attention_heads)
        vlm_output_dim = cfg_dict.get("vlm_output_dim", args.vlm_output_dim)
        num_vlm_layers = cfg_dict.get("num_vlm_layers", args.num_vlm_layers)
    else:
        print(f"No config.json found, using command line arguments")
        num_transformer_blocks = args.num_transformer_blocks
        num_attention_heads = args.num_attention_heads
        vlm_output_dim = args.vlm_output_dim
        num_vlm_layers = args.num_vlm_layers
    
    # 使用默认值（如果未提供）
    if vlm_output_dim is None:
        vlm_output_dim = LLM_OUTPUT_DIM_MLP_INPUT_DIM
    if num_vlm_layers is None:
        num_vlm_layers = NUM_VLM_HIDDEN_LAYERS
    
    print(f"\nModel Configuration:")
    print(f"  num_transformer_blocks: {num_transformer_blocks}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  vlm_output_dim: {vlm_output_dim}")
    print(f"  num_vlm_layers: {num_vlm_layers}")
    print(f"  proprio_dim: {PROPRIO_DIM}")
    print(f"  action_dim: {ACTION_DIM}")
    print(f"  num_actions_chunk: {NUM_ACTIONS_CHUNK}")

    # 创建模型
    print(f"\nCreating model...")
    pipeline = create_vlm2oft_pipeline(
        num_transformer_blocks=num_transformer_blocks,
        num_attention_heads=num_attention_heads,
        vlm_output_dim=vlm_output_dim,
        num_vlm_layers=num_vlm_layers
    ).to(args.device).eval()
    
    # 加载权重
    model_path = ckpt_dir / "action_head.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=args.device)
    pipeline.load_state_dict(state_dict, strict=False)
    print(f"Model loaded successfully")
    print(f"Total parameters: {sum(p.numel() for p in pipeline.parameters()):,}")

    # 准备输入数据
    batch_size = 1
    
    if args.npz:
        # 从 NPZ 文件加载真实数据
        print(f"\nLoading data from {args.npz}...")
        npz_data = np.load(args.npz)
        
        # 检查必需字段
        required_fields = ['backbone_features', 'backbone_attention_mask', 'state']
        for field in required_fields:
            if field not in npz_data:
                raise ValueError(f"Missing required field '{field}' in NPZ file")
        
        # 加载数据
        backbone_features = npz_data['backbone_features']  # (seq_len_total, hidden_dim)
        backbone_attention_mask = npz_data['backbone_attention_mask']  # (seq_len_total,)
        state = npz_data['state']  # (1, state_dim)
        
        print(f"Loaded data shapes:")
        print(f"  backbone_features: {backbone_features.shape}")
        print(f"  backbone_attention_mask: {backbone_attention_mask.shape}")
        print(f"  state: {state.shape}")
        
        # 验证维度
        seq_len_total = backbone_features.shape[0]
        if seq_len_total % num_vlm_layers != 0:
            raise ValueError(
                f"backbone_features seq_len ({seq_len_total}) is not divisible by "
                f"num_vlm_layers ({num_vlm_layers}). Expected seq_len to be num_layers × tokens_per_layer."
            )
        
        seq_len_per_layer = seq_len_total // num_vlm_layers
        
        # 将 backbone_features 拆分为多层
        # 从 (seq_len_total, hidden_dim) 拆分为 num_vlm_layers 个 (seq_len_per_layer, hidden_dim)
        vlm_hidden_states = []
        for i in range(num_vlm_layers):
            start_idx = i * seq_len_per_layer
            end_idx = (i + 1) * seq_len_per_layer
            layer_features = backbone_features[start_idx:end_idx, :]  # (seq_len_per_layer, hidden_dim)
            # 添加 batch 维度: (seq_len_per_layer, hidden_dim) -> (1, seq_len_per_layer, hidden_dim)
            layer_tensor = torch.from_numpy(layer_features).unsqueeze(0).to(args.device)
            vlm_hidden_states.append(layer_tensor)
        
        # Proprioception: 取 state 的前 PROPRIO_DIM 维
        proprioception = torch.from_numpy(state[:, :PROPRIO_DIM]).to(args.device)  # (1, proprio_dim)
        
        print(f"\nPrepared input data:")
        print(f"  VLM hidden states: {num_vlm_layers} layers × {vlm_hidden_states[0].shape}")
        print(f"  Proprioception shape: {proprioception.shape}")
        print(f"  Using REAL data from NPZ file")
        
    else:
        # 生成随机输入数据
        print(f"\nGenerating random input data...")
        seq_len = 512
        
        # VLM hidden states: List of tensors, each (batch_size, seq_len, vlm_output_dim)
        vlm_hidden_states = [
            torch.randn(batch_size, seq_len, vlm_output_dim, device=args.device)
            for _ in range(num_vlm_layers)
        ]
        
        # Proprioception: (batch_size, proprio_dim)
        proprioception = torch.randn(batch_size, PROPRIO_DIM, device=args.device)
        
        print(f"VLM hidden states: {num_vlm_layers} layers × {vlm_hidden_states[0].shape}")
        print(f"Proprioception shape: {proprioception.shape}")
        print(f"Using RANDOM data")

    # 推理
    print(f"\nRunning inference...")
    with torch.no_grad():
        action_predictions = pipeline(vlm_hidden_states, proprioception)
    
    # 转换为 numpy
    action_predictions = action_predictions.detach().cpu().numpy()
    
    # 打印结果
    print(f"\nInference completed!")
    print(f"action_predictions shape: {action_predictions.shape}")
    print(f"Expected shape: ({batch_size}, 1, {NUM_ACTIONS_CHUNK * ACTION_DIM})")
    print(f"action_predictions range: [{action_predictions.min():.4f}, {action_predictions.max():.4f}]")
    print(f"action_predictions mean: {action_predictions.mean():.4f}, std: {action_predictions.std():.4f}")
    
    # 重塑为 (batch_size, num_chunks, action_dim) 以便更好地可视化
    action_predictions_reshaped = action_predictions.reshape(batch_size, NUM_ACTIONS_CHUNK, ACTION_DIM)
    
    print(f"\nFirst 3 action chunks (out of {NUM_ACTIONS_CHUNK}):")
    for i in range(min(3, NUM_ACTIONS_CHUNK)):
        print(f"  Chunk {i}: {action_predictions_reshaped[0, i, :8]}...")  # 只显示前8个维度


if __name__ == "__main__":
    # 使用方法示例：
    # 
    # 方法 1: 使用命令行参数（推荐）
    # python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1
    # python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1 --device cuda:7
    # python infer_once.py --ckpt_dir ./experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1 --vlm_output_dim 2048
    # 
    # 方法 2: 直接设置参数运行（用于测试）
    import sys
    sys.argv = [
        'infer_once.py',
        '--ckpt_dir', '/home/sythoid_01/文档/Huangwenlong/n1.5-split/gr00t_split/Action_Heads/OFT1_0/experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_1',
        '--device', 'cuda:7',
        '--num_transformer_blocks', '2',
        '--num_attention_heads', '8',
        '--num_vlm_layers', '3',
        '--vlm_output_dim', '2048',
        # '--npz', '/path/to/sample.npz',  # 可选：使用真实数据
    ]
    
    main()
