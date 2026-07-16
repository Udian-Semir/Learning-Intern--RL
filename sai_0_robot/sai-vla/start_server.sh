#!/bin/bash
# Sai0 VLA 推理服务器快速启动脚本

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/deployment/Sai0_1_server"

# 配置参数（根据实际情况修改）
ACTION_HEAD_CKPT="$SCRIPT_DIR/Action_Heads/Flow_Matching_1/experiments/pickupanapple_v1_hidden_dim_3_512_2048/checkpoints/epoch_0"
VLM_BACKEND="qwen3-vl"
VLM_MODEL_NAME="Qwen/Qwen3-VL-2B-Instruct"
VLM_LAYER_INDICES="2,6,10"
DEVICE="cuda:0"
HOST="0.0.0.0"
PORT=5000

# 启动服务器
echo "======================================================"
echo "Sai0 VLA 推理服务器"
echo "======================================================"
echo "Action Head: $ACTION_HEAD_CKPT"
echo "VLM Backend: $VLM_BACKEND"
echo "VLM Model: $VLM_MODEL_NAME"
echo "Device: $DEVICE"
echo "Server: http://$HOST:$PORT"
echo "API Docs: http://$HOST:$PORT/docs"
echo "======================================================"

python server.py \
    --action_head_ckpt "$ACTION_HEAD_CKPT" \
    --vlm_backend "$VLM_BACKEND" \
    --vlm_model_name "$VLM_MODEL_NAME" \
    --vlm_layer_indices "$VLM_LAYER_INDICES" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT"
