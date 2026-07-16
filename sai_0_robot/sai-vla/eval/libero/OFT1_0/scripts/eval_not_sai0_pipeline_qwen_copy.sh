#!/bin/bash
# ============================================================================
# OFT1_0 LIBERO 评估脚本 (非 Sai0 pipeline, Qwen - 单层版本)
# ============================================================================

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ===== 基本配置 =====
CHECKPOINT_PATH="/data/HuangWenlong/datasets/qwen_extract_middle_layer_hidden_state/libero_github_convert_for_qwen2b-only-libero_spatial_weight/OFT1_0/1layer-14/bsz56*4_tb4/checkpoints/step_40000/action_head.pt"
DATA_PATH="/data/HuangWenlong/datasets/qwen_extract_middle_layer_hidden_state/libero_github_convert_for_qwen2b-only-libero_spatial"
VLM_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
VLM_LAYERS="14"
VLM_OUTPUT_DIM=2048

# ===== 多次评估配置 =====
# 评估次数 (默认为1)
NUM_EVAL_RUNS=1

# ===== 结果保存目录配置 =====
# 自定义路径 (取消注释并修改以使用自定义路径)
# VIDEO_DIR=""

# ===== Action Head 配置 =====
ADD_ACTION_PROMPT="true"
NUM_TRANSFORMER_BLOCKS=4
NUM_ATTENTION_HEADS=8
DROPOUT=0.1
ACTION_HEAD_HIDDEN_DIM=4096

# ===== LIBERO 配置 =====
TASK_SUITE_NAME="libero_spatial"
NUM_TRIALS_PER_TASK=5
ACTION_CHUNK_SIZE=16
EXECUTE_ALL_CHUNKS="true"

# ===== 系统配置 =====
DEVICE="cuda:1"
HEADLESS="true"
FLIP_IMAGES="true"
VERBOSE="true"

# ===== 获取脚本所在目录 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== 自动生成 VIDEO_DIR =====
# 从 VLM_MODEL_PATH 提取模型大小 (如 2B, 4B, 8B)
extract_model_size() {
    local path="$1"
    if [[ "$path" =~ ([0-9]+)[Bb] ]]; then
        echo "${BASH_REMATCH[1]}B"
    else
        echo "unknown"
    fi
}

# 从 CHECKPOINT_PATH 提取 step 数字
extract_step_number() {
    local path="$1"
    if [[ "$path" =~ step_([0-9]+) ]]; then
        echo "step_${BASH_REMATCH[1]}"
    else
        echo "step_unknown"
    fi
}

# 获取模型大小和 step 数字
MODEL_SIZE=$(extract_model_size "${VLM_MODEL_PATH}")
STEP_NUM=$(extract_step_number "${CHECKPOINT_PATH}")

# Action Prompt 标识
if [ "${ADD_ACTION_PROMPT}" = "true" ]; then
    ACTION_PROMPT_TAG="act"
else
    ACTION_PROMPT_TAG="noact"
fi

# 构建实验目录名
EXPERIMENT_NAME="qwen_${MODEL_SIZE}_${VLM_LAYERS}_${VLM_OUTPUT_DIM}_${ACTION_PROMPT_TAG}_${NUM_TRIALS_PER_TASK}trials_${TASK_SUITE_NAME}_chunk${ACTION_CHUNK_SIZE}_${STEP_NUM}"

# ===== 打印配置 =====
echo "============================================================================"
echo "🤖 OFT1_0 LIBERO 评估 (非 Sai0 pipeline, Qwen - 单层版本)"
echo "============================================================================"
echo ""
echo "📁 路径配置:"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Data: ${DATA_PATH}"
echo "  Experiment Dir: ${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}"
echo ""
echo "🔮 VLM 配置:"
echo "  Model: ${VLM_MODEL_PATH}"
echo "  Model Size: ${MODEL_SIZE}"
echo "  Layers: ${VLM_LAYERS}"
echo "  Output Dim: ${VLM_OUTPUT_DIM}"
echo ""
echo "🎮 LIBERO 配置:"
echo "  Task Suite: ${TASK_SUITE_NAME}"
echo "  Trials per task: ${NUM_TRIALS_PER_TASK}"
echo "  Action Chunk Size: ${ACTION_CHUNK_SIZE}"
echo ""
echo "🔄 多次评估配置:"
echo "  Num Eval Runs: ${NUM_EVAL_RUNS}"
echo ""
echo "============================================================================"

# ===== 多次评估循环 =====
declare -a SUCCESS_RATES
TOTAL_SUCCESS_RATE=0

for ((RUN_IDX=1; RUN_IDX<=NUM_EVAL_RUNS; RUN_IDX++)); do
    echo ""
    echo "============================================================================"
    echo "🔄 开始第 ${RUN_IDX}/${NUM_EVAL_RUNS} 次评估"
    echo "============================================================================"
    
    # 构建当前运行的 VIDEO_DIR
    if [ -n "${VIDEO_DIR}" ]; then
        CURRENT_VIDEO_DIR="${VIDEO_DIR}_run${RUN_IDX}"
    else
        CURRENT_VIDEO_DIR="${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}/eval_rollouts_${RUN_IDX}_not_sai0"
    fi
    
    mkdir -p "${CURRENT_VIDEO_DIR}"
    echo "  Video Dir: ${CURRENT_VIDEO_DIR}"
    
    # 构建命令
    CMD="python /home/sythoid_01/文档/Huangwenlong/LIBERO/custom_hwl/sai0-vla/eval/libero/OFT1_0/eval_not_sai0_pipeline_qwen.py"
    CMD="${CMD} --checkpoint_path ${CHECKPOINT_PATH}"
    CMD="${CMD} --data_path ${DATA_PATH}"
    CMD="${CMD} --vlm_model_path ${VLM_MODEL_PATH}"
    CMD="${CMD} --vlm_layers ${VLM_LAYERS}"
    CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"
    CMD="${CMD} --num_transformer_blocks ${NUM_TRANSFORMER_BLOCKS}"
    CMD="${CMD} --num_attention_heads ${NUM_ATTENTION_HEADS}"
    CMD="${CMD} --dropout ${DROPOUT}"
    CMD="${CMD} --action_head_hidden_dim ${ACTION_HEAD_HIDDEN_DIM}"
    CMD="${CMD} --task_suite_name ${TASK_SUITE_NAME}"
    CMD="${CMD} --num_trials_per_task ${NUM_TRIALS_PER_TASK}"
    CMD="${CMD} --action_chunk_size ${ACTION_CHUNK_SIZE}"
    CMD="${CMD} --device ${DEVICE}"
    CMD="${CMD} --video_dir ${CURRENT_VIDEO_DIR}"
    CMD="${CMD} --log_dir ${CURRENT_VIDEO_DIR}"
    
    if [ "${ADD_ACTION_PROMPT}" = "true" ]; then
        CMD="${CMD} --add_action_prompt"
    fi
    
    if [ "${EXECUTE_ALL_CHUNKS}" = "true" ]; then
        CMD="${CMD} --execute_all_chunks"
    fi
    
    if [ "${HEADLESS}" = "true" ]; then
        CMD="${CMD} --headless"
    fi
    
    if [ "${FLIP_IMAGES}" = "true" ]; then
        CMD="${CMD} --flip_images"
    fi
    
    if [ "${VERBOSE}" = "true" ]; then
        CMD="${CMD} --verbose"
    fi
    
    echo ""
    echo "执行命令:"
    echo "${CMD}"
    echo ""
    
    # 执行评估并捕获输出
    OUTPUT_FILE="${CURRENT_VIDEO_DIR}/eval_output_run${RUN_IDX}.log"
    
    # 在日志顶部写入完整参数配置
    {
        echo "============================================================================"
        echo "📋 评估参数配置记录"
        echo "============================================================================"
        echo "记录时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "评估次数: 第 ${RUN_IDX}/${NUM_EVAL_RUNS} 次"
        echo ""
        echo "===== 基本配置 ====="
        echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
        echo "DATA_PATH=${DATA_PATH}"
        echo "VIDEO_DIR=${CURRENT_VIDEO_DIR}"
        echo ""
        echo "===== VLM 配置 ====="
        echo "VLM_MODEL_PATH=${VLM_MODEL_PATH}"
        echo "MODEL_SIZE=${MODEL_SIZE}"
        echo "VLM_LAYERS=${VLM_LAYERS}"
        echo "VLM_OUTPUT_DIM=${VLM_OUTPUT_DIM}"
        echo ""
        echo "===== Action Head 配置 ====="
        echo "ADD_ACTION_PROMPT=${ADD_ACTION_PROMPT}"
        echo "NUM_TRANSFORMER_BLOCKS=${NUM_TRANSFORMER_BLOCKS}"
        echo "NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS}"
        echo "DROPOUT=${DROPOUT}"
        echo "ACTION_HEAD_HIDDEN_DIM=${ACTION_HEAD_HIDDEN_DIM}"
        echo ""
        echo "===== LIBERO 配置 ====="
        echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
        echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
        echo "ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE}"
        echo "EXECUTE_ALL_CHUNKS=${EXECUTE_ALL_CHUNKS}"
        echo ""
        echo "===== 系统配置 ====="
        echo "DEVICE=${DEVICE}"
        echo "HEADLESS=${HEADLESS}"
        echo "FLIP_IMAGES=${FLIP_IMAGES}"
        echo "VERBOSE=${VERBOSE}"
        echo ""
        echo "===== 多次评估配置 ====="
        echo "NUM_EVAL_RUNS=${NUM_EVAL_RUNS}"
        echo ""
        echo "===== 执行命令 ====="
        echo "${CMD}"
        echo ""
        echo "============================================================================"
        echo "📋 评估输出开始"
        echo "============================================================================"
        echo ""
    } > "${OUTPUT_FILE}"
    
    eval ${CMD} 2>&1 | tee -a "${OUTPUT_FILE}"
    
    # 从输出中提取成功率
    CURRENT_SUCCESS_RATE=$(grep -oP '(?i)(success\s*rate|avg.*success|average.*success|overall.*success)[:\s]*(\d+\.?\d*)' "${OUTPUT_FILE}" | tail -1 | grep -oP '\d+\.?\d*$' || echo "0")
    
    if [ -z "${CURRENT_SUCCESS_RATE}" ] || [ "${CURRENT_SUCCESS_RATE}" = "0" ]; then
        CURRENT_SUCCESS_RATE=$(grep -oP '\d+\.?\d*(?=\s*%?\s*(success|完成))' "${OUTPUT_FILE}" | tail -1 || echo "0")
    fi
    
    if [ -z "${CURRENT_SUCCESS_RATE}" ]; then
        CURRENT_SUCCESS_RATE="0"
    fi
    
    echo ""
    echo "============================================================================"
    echo "📊 第 ${RUN_IDX} 次评估完成"
    echo "   成功率: ${CURRENT_SUCCESS_RATE}"
    echo "============================================================================"
    
    SUCCESS_RATES+=("${CURRENT_SUCCESS_RATE}")
    TOTAL_SUCCESS_RATE=$(echo "${TOTAL_SUCCESS_RATE} + ${CURRENT_SUCCESS_RATE}" | bc -l 2>/dev/null || echo "${TOTAL_SUCCESS_RATE}")
done

# ===== 打印汇总结果 =====
echo ""
echo "============================================================================"
echo "📈 所有评估完成 - 汇总结果"
echo "============================================================================"
echo ""
echo "各次评估成功率:"
for ((i=0; i<${#SUCCESS_RATES[@]}; i++)); do
    echo "  第 $((i+1)) 次: ${SUCCESS_RATES[$i]}"
done
echo ""

if [ ${NUM_EVAL_RUNS} -gt 0 ]; then
    AVG_SUCCESS_RATE=$(echo "scale=4; ${TOTAL_SUCCESS_RATE} / ${NUM_EVAL_RUNS}" | bc -l 2>/dev/null || echo "N/A")
    echo "📊 平均成功率 (${NUM_EVAL_RUNS} 次评估): ${AVG_SUCCESS_RATE}"
else
    echo "📊 平均成功率: N/A (无有效评估)"
fi
echo ""
echo "============================================================================"
