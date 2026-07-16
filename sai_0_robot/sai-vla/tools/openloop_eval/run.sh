#!/bin/bash
# ============================================================================
# Dataset196 Open-Loop Evaluation
# ----------------------------------------------------------------------------
# 在 dataset196 (rightarm state_joint action_eefdelta) 上,
# 用 unitree_v2 step_300000 checkpoint 做 5 条随机 episode 的整段开环预测.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CHECKPOINT="${CHECKPOINT:-/data_disk2/hwl/checkpoints/unitree_v2_pretrain/action7_chunk16/p6000_bsz500*1*8_tb4_400000steps_layer14_20260511_wpunitree_v2_pretrain_unitree_v2_pretrain_CBC_true_dtypefloat32_USE_AMP_true_MD_true/checkpoints/step_300000}"
DATASET="${DATASET:-/data_disk2/hwl/datasets/dataset196/lerobot_dataset196_rightarm_state_joint_action_eefdelta_filter_by_velocity}"
# 默认: 用预提取的 npy hidden states (与 _then_tele 训练脚本数据来源一致)
VLM_HIDDEN_STATES_DIR="${VLM_HIDDEN_STATES_DIR:-${DATASET}/vlm_hidden_states}"
# USE_PRE_EXTRACTED_VLM=false 时切回在线 VLM 模式
USE_PRE_EXTRACTED_VLM="${USE_PRE_EXTRACTED_VLM:-true}"

DEVICE="${DEVICE:-cuda:0}"
NUM_EPISODES="${NUM_EPISODES:-5}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../outputs/openloop_eval}"

# 离线模式: 不再向 HF 发请求, 使用本地缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_ROOT}"

EXTRA_ARGS=""
if [ -n "${EPISODES:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --episodes ${EPISODES}"
fi
if [ "${USE_PRE_EXTRACTED_VLM}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --use_pre_extracted_vlm --vlm_hidden_states_dir ${VLM_HIDDEN_STATES_DIR}"
else
    EXTRA_ARGS="${EXTRA_ARGS} --no_use_pre_extracted_vlm"
fi

python tools/openloop_eval/openloop_eval.py \
    --checkpoint "${CHECKPOINT}" \
    --dataset_path "${DATASET}" \
    --device "${DEVICE}" \
    --num_episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --output_dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}
