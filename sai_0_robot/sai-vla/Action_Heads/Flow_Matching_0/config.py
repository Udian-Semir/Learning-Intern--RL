"""
Flow Matching Action Head Configuration
"""

# 支持多种导入方式，提高兼容性
try:
    # 方式1: 绝对导入（当作为模块直接导入时）
    from models.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig
except ImportError:
    try:
        # 方式2: 相对导入（当作为包的一部分导入时）
        from .models.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig
    except ImportError:
        try:
            # 方式3: 从 Action_Heads.Flow_Matching_0 包导入
            from Action_Heads.Flow_Matching_0.models.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig
        except ImportError:
            # 方式4: 添加路径后导入
            import sys
            from pathlib import Path
            # 添加当前目录到路径
            current_dir = Path(__file__).parent
            if str(current_dir) not in sys.path:
                sys.path.insert(0, str(current_dir))
            from models.action_head.flow_matching_action_head import FlowmatchingActionHeadConfig


def get_flowmatching_action_head_config_original(
    action_backbone_dim: int = 1536,
    vlm_output_dim: int = 2048,
    vl_self_attention_head_dim: int = 64,
    vl_self_attention_num_attention_heads: int = 32,
    action_dim: int = 16,
    action_horizon: int = 16,
    max_state_dim: int = 64,
    max_action_dim: int = 32,
) -> FlowmatchingActionHeadConfig:
    """
    创建 FlowMatching ActionHead 的配置（原始完整版）
    与原项目保持完全一致的配置参数
    
    Args:
        action_backbone_dim: Action backbone 特征维度 (默认 1536)
        vlm_output_dim: VLM 输出维度 (默认 2048)
        vl_self_attention_head_dim: VL self-attention head 维度 (默认 64)
        vl_self_attention_num_attention_heads: VL self-attention head 数量 (默认 32)
        action_dim: 动作空间维度
        action_horizon: 动作预测时间步长
        max_state_dim: 最大状态维度
        max_action_dim: 最大动作维度
    
    Returns:
        FlowmatchingActionHeadConfig 实例
    """
    cfg = FlowmatchingActionHeadConfig(
        add_pos_embed=True,
        model_dtype='float32',
        diffusion_model_cfg={
            'attention_head_dim': 48,
            'cross_attention_dim': vlm_output_dim,
            'dropout': 0.2,
            'final_dropout': True,
            'interleave_self_attention': True,
            'norm_type': 'ada_norm',
            'num_attention_heads': 32,
            'num_layers': 16,
            'output_dim': 1024,
            'positional_embeddings': None,
        },
        input_embedding_dim=action_backbone_dim,
        backbone_embedding_dim=vlm_output_dim,
        hidden_size=1024,
        max_seq_len=1024,
        action_dim=action_dim,
        action_horizon=action_horizon,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        num_inference_timesteps=4,
        max_num_embodiments=32,
        tune_projector=True,
        tune_diffusion_model=True,
        load_pretrained_det_decode_layer_path=None,
        detection_coeff=1.0,
        freeze_decode_layer=False,
        expand_batch=None,
        use_vlln=True,
        vl_self_attention_cfg={
            'attention_head_dim': vl_self_attention_head_dim,
            'dropout': 0.2,
            'final_dropout': True,
            'num_attention_heads': vl_self_attention_num_attention_heads,
            'num_layers': 4,
            'positional_embeddings': None,
        },
        num_target_vision_tokens=32,
    )
    
    # 额外字段（源码通过kwargs动态挂载）
    cfg.max_state_dim = max_state_dim
    cfg.max_action_dim = max_action_dim
    
    return cfg
