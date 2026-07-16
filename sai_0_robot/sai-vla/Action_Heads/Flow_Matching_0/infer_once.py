"""
Flow Matching Action Head 单次推理脚本
Single Inference Script for Flow Matching Action Head

使用方法 Usage:
  python infer_once.py --ckpt_dir ./checkpoints/epoch_0
  python infer_once.py --ckpt_dir ./checkpoints/best --npz /path/to/sample.npz

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
from transformers.feature_extraction_utils import BatchFeature

from models.action_head.flow_matching_action_head import FlowmatchingActionHead, FlowmatchingActionHeadConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    # 若未提供NPZ，则用随机输入
    parser.add_argument("--npz", type=str, default="")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    cfg_dict = json.load(open(ckpt_dir / "config.json"))
    # transformers.PretrainedConfig不存储额外字段到__dict__之外，这里直接用dict构造
    cfg = FlowmatchingActionHeadConfig(**{k: v for k, v in cfg_dict.items() if not k.startswith("_")})
    # 额外字段
    if "max_state_dim" in cfg_dict:
        cfg.max_state_dim = cfg_dict["max_state_dim"]

    model = FlowmatchingActionHead(cfg).to(args.device).eval()
    state_dict = torch.load(ckpt_dir / "action_head.pt", map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    if args.npz:
        d = np.load(args.npz)
        bf = torch.from_numpy(d["backbone_features"]).unsqueeze(0).to(args.device)
        bmask = torch.from_numpy(d["backbone_attention_mask"]).unsqueeze(0).to(args.device)
        state = torch.from_numpy(d["state"]).unsqueeze(0).to(args.device)
    else:
        S = 256
        backbone_dim = cfg.backbone_embedding_dim
        bf = torch.randn(1, S, backbone_dim, device=args.device)
        bmask = torch.ones(1, S, dtype=torch.long, device=args.device)
        state = torch.zeros(1, 1, getattr(cfg, "max_state_dim", 64), device=args.device)

    # ! 推理 -> 进入get_action方法的数据，bf三维，bmask二维，state三维，然后会转换为BatchFeature
    bb = BatchFeature(data={"backbone_features": bf, "backbone_attention_mask": bmask})
    ah = BatchFeature(data={"state": state, "embodiment_id": torch.tensor([31], device=args.device)})
    
    print("\nRunning inference...")
    with torch.no_grad():
        out = model.get_action(bb, ah)["action_pred"].detach().cpu().numpy()
    
    print(f"\nInference completed!")
    print(f"action_pred shape: {out.shape}")
    print(f"action_pred range: [{out.min():.4f}, {out.max():.4f}]")
    print(f"action_pred mean: {out.mean():.4f}, std: {out.std():.4f}")
    
    # 打印前几个 action
    print(f"\nFirst 3 actions (if available):")
    for i in range(min(3, out.shape[1])):
        print(f"  Action {i}: {out[0, i, :8]}...")  # 只显示前8个维度


if __name__ == "__main__":
    # 使用方法示例：
    # 
    # 方法 1: 使用命令行参数（推荐）
    # python infer.py --ckpt_dir ./experiments/pickupanapple_v1/checkpoints/epoch_0
    # python infer.py --ckpt_dir ./experiments/pickupanapple_v1/checkpoints/best --device cuda:7
    # python infer.py --ckpt_dir ./fm0_ckpts_finetuned --npz /path/to/sample.npz
    # 
    # 方法 2: 直接设置参数运行（用于测试）
    # import sys
    # sys.argv = [
    #     'infer.py',
    #     '--ckpt_dir', '/home/sythoid_01/文档/Huangwenlong/n1.5-split/gr00t_split/Action_Heads/Flow_Matching_0/experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_0',
    #     '--device', 'cuda:7',
    #     # '--npz', '/path/to/sample.npz',  # 可选：使用真实数据
    # ]
    
    main()



