#!/bin/bash
# ============================================================================
# Sai0_1 LIBERO 评估脚本 - 支持任务指令替换
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数
#   2. 在 TASK_INSTRUCTION_OVERRIDES 中配置要替换的任务指令
#   3. 运行: bash run_eval_Sai0_1_change_task_name.sh
#
# 任务指令替换功能:
#   - 可以替换指定任务ID的指令
#   - 格式: TASK_INSTRUCTION_OVERRIDES["任务ID"]="新指令"
#   - 例如: TASK_INSTRUCTION_OVERRIDES["1"]="put the bowl on the white square plate"
#
# ============================================================================

# ===== 基本配置 =====

# Checkpoint 路径 (必填)
CHECKPOINT_PATH="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_goal/libero_lerobot_goal_sys1_qwen_weight/OFT1_0/1layer_14/5090_bsz230*8_tb4_20000steps_layer14_20251231_new2/checkpoints/step_20000/action_head.pt"

# 数据集路径 (用于归一化统计量)
DATASET_PATH="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_goal/libero_lerobot_goal_sys1_qwen"

# ===== 🔄 任务指令替换配置 =====
# 在这里配置要替换的任务指令
# 注意: 任务ID从0开始
#
# ⚠️ 优先级说明:
#   - 如果 BATCH_OVERRIDES 非空，则使用 BATCH_OVERRIDES (方式2)
#   - 如果 BATCH_OVERRIDES 为空，则使用 OVERRIDE_TASK_ID + OVERRIDE_INSTRUCTION (方式1)
#   - 如果都为空，则不进行任何替换
#
# 方式1: 单个替换 (简单) - 优先级低
# 只替换一个任务，当 BATCH_OVERRIDES 为空时生效
OVERRIDE_TASK_ID="1"
OVERRIDE_INSTRUCTION="put the bowl on the white square plate that is on the top left corner of the table"

# 方式2: 批量替换 (高级) - 优先级高
# 格式: "task_id1:instruction1|task_id2:instruction2|..."
# 非空时优先使用此配置，忽略方式1
# 例如: "0:pick up the red apple|1:put the bowl on the white square plate"
BATCH_OVERRIDES=""

# 是否启用任务指令替换 (总开关)
# 设为 "false" 时，上面的配置都不生效
ENABLE_INSTRUCTION_OVERRIDE="true"

# ===== 多次评估配置 =====
NUM_EVAL_RUNS=5

# ===== VLM 配置 =====
GPU_NAME="5090"
VLM_TYPE="qwen3_vl"
VLM_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
VLM_LAYERS="14"
VLM_OUTPUT_DIM=2048

# ===== Prompt 配置 =====
CONTENT_ORDER="images_first"
LOWERCASE_INSTRUCTION="true"
ADD_GENERATION_PROMPT="true"
ADD_ACTION_PROMPT="true"

# ===== LIBERO 环境配置 =====
TASK_SUITE_NAME="libero_goal"
NUM_TRIALS_PER_TASK=10
TASK_IDS=""  # 留空评估全部，或指定如 "0,1,2"
MAX_TASKS=-1
NUM_STEPS_WAIT=10
MAX_STEPS=600
ENV_SEED=""

# ===== 推理配置 =====
ACTION_CHUNK_SIZE=16
EXECUTE_ALL_CHUNKS="true"

# ===== OFT 模型配置 =====
NUM_TRANSFORMER_BLOCKS=4
NUM_ATTENTION_HEADS=8
DROPOUT=0.1
ACTION_HEAD_HIDDEN_DIM=4096
NUM_ACTION_CHUNKS=16
ACTION_DIM=7

# ===== 系统配置 =====
GPU_ID=1
DEVICE="cuda:0"
FLIP_IMAGES="true"
HEADLESS="true"
VERBOSE="true"

# ===== 训练数据测试模式 =====
USE_TRAINING_DATA="false"
NUM_TEST_SAMPLES=100

# ===== 运行脚本 =====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# ===== 自动生成 VIDEO_DIR =====
extract_model_size() {
    local path="$1"
    if [[ "$path" =~ ([0-9]+)[Bb] ]]; then
        echo "${BASH_REMATCH[1]}B"
    else
        echo "unknown"
    fi
}

extract_step_number() {
    local path="$1"
    if [[ "$path" =~ step_([0-9]+) ]]; then
        echo "step_${BASH_REMATCH[1]}"
    else
        echo "step_unknown"
    fi
}

MODEL_SIZE=$(extract_model_size "${VLM_MODEL_PATH}")
STEP_NUM=$(extract_step_number "${CHECKPOINT_PATH}")

if [ "${ADD_ACTION_PROMPT}" = "true" ]; then
    ACTION_PROMPT_TAG="act"
else
    ACTION_PROMPT_TAG="noact"
fi

# 添加 override 标识
if [ "${ENABLE_INSTRUCTION_OVERRIDE}" = "true" ]; then
    OVERRIDE_TAG="_override"
else
    OVERRIDE_TAG=""
fi

EXPERIMENT_NAME="${VLM_TYPE}_${MODEL_SIZE}_${VLM_LAYERS}_${VLM_OUTPUT_DIM}_${ACTION_PROMPT_TAG}_${NUM_TRIALS_PER_TASK}trials_${TASK_SUITE_NAME}_chunk${ACTION_CHUNK_SIZE}_${STEP_NUM}_${GPU_NAME}${OVERRIDE_TAG}_$(date +%Y%m%d)"

cd "${PROJECT_ROOT}"

# 环境设置
export ROBOSUITE_LOG="${HOME}/.robosuite/robosuite.log"
mkdir -p "${HOME}/.robosuite"
export TRANSFORMERS_VERBOSITY=info
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ===== 构建任务指令替换字符串 =====
# 注意: 这里只构建参数值，在CMD构建时会正确添加引号
OVERRIDE_VALUE=""
if [ "${ENABLE_INSTRUCTION_OVERRIDE}" = "true" ]; then
    # 方式2: 批量替换 (优先级高)
    if [ -n "${BATCH_OVERRIDES}" ]; then
        OVERRIDE_VALUE="${BATCH_OVERRIDES}"
    # 方式1: 单个替换
    elif [ -n "${OVERRIDE_TASK_ID}" ] && [ -n "${OVERRIDE_INSTRUCTION}" ]; then
        OVERRIDE_VALUE="${OVERRIDE_TASK_ID}:${OVERRIDE_INSTRUCTION}"
    fi
fi

# 打印配置
echo "============================================================================"
echo "🤖 Sai0_1 LIBERO 评估 (支持任务指令替换)"
echo "============================================================================"
echo ""
echo "📁 路径配置:"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Experiment Dir: ${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}"
echo ""
echo "🔮 VLM 配置:"
echo "  Type: ${VLM_TYPE}"
echo "  Model: ${VLM_MODEL_PATH}"
echo "  Model Size: ${MODEL_SIZE}"
echo "  Layers: ${VLM_LAYERS}"
echo "  Output Dim: ${VLM_OUTPUT_DIM}"
echo ""
echo "📝 Prompt 配置:"
echo "  Content Order: ${CONTENT_ORDER}"
echo "  Lowercase: ${LOWERCASE_INSTRUCTION}"
echo "  Generation Prompt: ${ADD_GENERATION_PROMPT}"
echo "  Action Prompt: ${ADD_ACTION_PROMPT}"
echo ""
echo "🎮 LIBERO 配置:"
echo "  Task Suite: ${TASK_SUITE_NAME}"
echo "  Trials per task: ${NUM_TRIALS_PER_TASK}"
echo "  Max tasks: ${MAX_TASKS}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Env Seed: ${ENV_SEED:-随机}"
echo ""
echo "🔧 推理配置:"
echo "  Action Chunk Size: ${ACTION_CHUNK_SIZE}"
echo "  Execute All Chunks: ${EXECUTE_ALL_CHUNKS}"
echo ""
echo "💻 系统配置:"
echo "  GPU: ${GPU_ID}"
echo "  Flip Images: ${FLIP_IMAGES}"
echo "  Headless: ${HEADLESS}"
echo "  Verbose: ${VERBOSE}"
echo ""

# ===== 🔄 打印任务指令替换配置 =====
echo "🔄 任务指令替换配置:"
echo "  启用替换: ${ENABLE_INSTRUCTION_OVERRIDE}"
if [ "${ENABLE_INSTRUCTION_OVERRIDE}" = "true" ]; then
    if [ -n "${BATCH_OVERRIDES}" ]; then
        echo "  批量替换: ${BATCH_OVERRIDES}"
    elif [ -n "${OVERRIDE_TASK_ID}" ] && [ -n "${OVERRIDE_INSTRUCTION}" ]; then
        echo "  任务ID: ${OVERRIDE_TASK_ID}"
        echo "  新指令: ${OVERRIDE_INSTRUCTION}"
    else
        echo "  ⚠️ 未配置替换内容"
    fi
fi
echo ""

echo "🧪 测试模式:"
echo "  Use Training Data: ${USE_TRAINING_DATA}"
echo "  Num Test Samples: ${NUM_TEST_SAMPLES}"
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
    
    if [ -n "${VIDEO_DIR}" ]; then
        CURRENT_VIDEO_DIR="${VIDEO_DIR}_run${RUN_IDX}"
    else
        CURRENT_VIDEO_DIR="${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}/eval_rollouts_${RUN_IDX}_sai0_1"
    fi
    
    mkdir -p "${CURRENT_VIDEO_DIR}"
    
    echo "  Video Dir: ${CURRENT_VIDEO_DIR}"
    
    # 构建命令
    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python -m eval.Sai0_1.libero.OFT1_0.eval_Sai0_1"
    CMD="${CMD} --checkpoint_path ${CHECKPOINT_PATH}"
    CMD="${CMD} --vlm_model_path ${VLM_MODEL_PATH}"
    CMD="${CMD} --vlm_type ${VLM_TYPE}"
    CMD="${CMD} --dataset_path ${DATASET_PATH}"
    CMD="${CMD} --vlm_layers ${VLM_LAYERS}"
    CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"
    CMD="${CMD} --content_order ${CONTENT_ORDER}"
    CMD="${CMD} --task_suite_name ${TASK_SUITE_NAME}"
    CMD="${CMD} --num_trials_per_task ${NUM_TRIALS_PER_TASK}"
    CMD="${CMD} --max_tasks ${MAX_TASKS}"
    CMD="${CMD} --num_steps_wait ${NUM_STEPS_WAIT}"
    CMD="${CMD} --max_steps ${MAX_STEPS}"
    
    if [ -n "${ENV_SEED}" ]; then
        CMD="${CMD} --env_seed ${ENV_SEED}"
    fi
    
    CMD="${CMD} --action_chunk_size ${ACTION_CHUNK_SIZE}"
    CMD="${CMD} --num_transformer_blocks ${NUM_TRANSFORMER_BLOCKS}"
    CMD="${CMD} --num_attention_heads ${NUM_ATTENTION_HEADS}"
    CMD="${CMD} --dropout ${DROPOUT}"
    CMD="${CMD} --action_head_hidden_dim ${ACTION_HEAD_HIDDEN_DIM}"
    CMD="${CMD} --num_action_chunks ${NUM_ACTION_CHUNKS}"
    CMD="${CMD} --action_dim ${ACTION_DIM}"
    CMD="${CMD} --device ${DEVICE}"
    CMD="${CMD} --video_dir ${CURRENT_VIDEO_DIR}"
    CMD="${CMD} --num_test_samples ${NUM_TEST_SAMPLES}"

    if [ -n "${TASK_IDS}" ]; then
        CMD="${CMD} --task_ids ${TASK_IDS}"
    fi

    if [ "${FLIP_IMAGES}" = "true" ]; then
        CMD="${CMD} --flip_images"
    else
        CMD="${CMD} --no_flip_images"
    fi

    if [ "${ADD_ACTION_PROMPT}" = "true" ]; then
        CMD="${CMD} --add_action_prompt"
    else
        CMD="${CMD} --no_action_prompt"
    fi

    if [ "${LOWERCASE_INSTRUCTION}" = "true" ]; then
        CMD="${CMD} --lowercase_instruction"
    else
        CMD="${CMD} --no_lowercase_instruction"
    fi

    if [ "${ADD_GENERATION_PROMPT}" = "true" ]; then
        CMD="${CMD} --add_generation_prompt"
    else
        CMD="${CMD} --no_generation_prompt"
    fi

    if [ "${EXECUTE_ALL_CHUNKS}" = "true" ]; then
        CMD="${CMD} --execute_all_chunks"
    fi

    if [ "${HEADLESS}" = "true" ]; then
        CMD="${CMD} --headless"
    fi

    if [ "${VERBOSE}" = "true" ]; then
        CMD="${CMD} --verbose"
    fi

    if [ "${USE_TRAINING_DATA}" = "true" ]; then
        CMD="${CMD} --use_training_data"
    fi
    
    # ===== 添加任务指令替换参数 =====
    # 注意: 带空格的指令需要正确处理引号
    if [ -n "${OVERRIDE_VALUE}" ]; then
        CMD="${CMD} --task_instruction_override '${OVERRIDE_VALUE}'"
    fi
    
    echo ""
    echo "执行命令:"
    echo "${CMD}"
    echo ""
    
    OUTPUT_FILE="${CURRENT_VIDEO_DIR}/eval_output_run${RUN_IDX}.log"
    
    # 写入日志头
    {
        echo "============================================================================"
        echo "📋 评估参数配置记录"
        echo "============================================================================"
        echo "记录时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "评估次数: 第 ${RUN_IDX}/${NUM_EVAL_RUNS} 次"
        echo ""
        echo "===== 基本配置 ====="
        echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
        echo "DATASET_PATH=${DATASET_PATH}"
        echo "VIDEO_DIR=${CURRENT_VIDEO_DIR}"
        echo ""
        echo "===== VLM 配置 ====="
        echo "VLM_TYPE=${VLM_TYPE}"
        echo "VLM_MODEL_PATH=${VLM_MODEL_PATH}"
        echo "MODEL_SIZE=${MODEL_SIZE}"
        echo "VLM_LAYERS=${VLM_LAYERS}"
        echo "VLM_OUTPUT_DIM=${VLM_OUTPUT_DIM}"
        echo ""
        echo "===== Prompt 配置 ====="
        echo "CONTENT_ORDER=${CONTENT_ORDER}"
        echo "LOWERCASE_INSTRUCTION=${LOWERCASE_INSTRUCTION}"
        echo "ADD_GENERATION_PROMPT=${ADD_GENERATION_PROMPT}"
        echo "ADD_ACTION_PROMPT=${ADD_ACTION_PROMPT}"
        echo ""
        echo "===== LIBERO 环境配置 ====="
        echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
        echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
        echo "TASK_IDS=${TASK_IDS}"
        echo "MAX_TASKS=${MAX_TASKS}"
        echo "NUM_STEPS_WAIT=${NUM_STEPS_WAIT}"
        echo "MAX_STEPS=${MAX_STEPS}"
        echo "ENV_SEED=${ENV_SEED:-随机}"
        echo ""
        echo "===== 🔄 任务指令替换配置 ====="
        echo "ENABLE_INSTRUCTION_OVERRIDE=${ENABLE_INSTRUCTION_OVERRIDE}"
        echo "OVERRIDE_TASK_ID=${OVERRIDE_TASK_ID}"
        echo "OVERRIDE_INSTRUCTION=${OVERRIDE_INSTRUCTION}"
        echo "BATCH_OVERRIDES=${BATCH_OVERRIDES}"
        echo "OVERRIDE_VALUE=${OVERRIDE_VALUE}"
        echo ""
        echo "===== 推理配置 ====="
        echo "ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE}"
        echo "EXECUTE_ALL_CHUNKS=${EXECUTE_ALL_CHUNKS}"
        echo ""
        echo "===== OFT 模型配置 ====="
        echo "NUM_TRANSFORMER_BLOCKS=${NUM_TRANSFORMER_BLOCKS}"
        echo "NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS}"
        echo "DROPOUT=${DROPOUT}"
        echo "ACTION_HEAD_HIDDEN_DIM=${ACTION_HEAD_HIDDEN_DIM}"
        echo "NUM_ACTION_CHUNKS=${NUM_ACTION_CHUNKS}"
        echo "ACTION_DIM=${ACTION_DIM}"
        echo ""
        echo "===== 系统配置 ====="
        echo "GPU_ID=${GPU_ID}"
        echo "DEVICE=${DEVICE}"
        echo "FLIP_IMAGES=${FLIP_IMAGES}"
        echo "HEADLESS=${HEADLESS}"
        echo "VERBOSE=${VERBOSE}"
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

# ===== 保存汇总日志 =====
EXPERIMENT_DIR="${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}"
SUMMARY_LOG="${EXPERIMENT_DIR}/eval_summary.log"

{
    echo "============================================================================"
    echo "📈 评估汇总报告"
    echo "============================================================================"
    echo "生成时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "实验名称: ${EXPERIMENT_NAME}"
    echo ""
    echo "===== 基本配置 ====="
    echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
    echo "DATASET_PATH=${DATASET_PATH}"
    echo "VLM_TYPE=${VLM_TYPE}"
    echo "VLM_MODEL_PATH=${VLM_MODEL_PATH}"
    echo "MODEL_SIZE=${MODEL_SIZE}"
    echo "VLM_LAYERS=${VLM_LAYERS}"
    echo "VLM_OUTPUT_DIM=${VLM_OUTPUT_DIM}"
    echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
    echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
    echo "ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE}"
    echo "NUM_EVAL_RUNS=${NUM_EVAL_RUNS}"
    echo ""
    echo "===== 🔄 任务指令替换配置 ====="
    echo "ENABLE_INSTRUCTION_OVERRIDE=${ENABLE_INSTRUCTION_OVERRIDE}"
    if [ "${ENABLE_INSTRUCTION_OVERRIDE}" = "true" ]; then
        if [ -n "${BATCH_OVERRIDES}" ]; then
            echo "BATCH_OVERRIDES=${BATCH_OVERRIDES}"
        elif [ -n "${OVERRIDE_TASK_ID}" ] && [ -n "${OVERRIDE_INSTRUCTION}" ]; then
            echo "OVERRIDE_TASK_ID=${OVERRIDE_TASK_ID}"
            echo "OVERRIDE_INSTRUCTION=${OVERRIDE_INSTRUCTION}"
        fi
    fi
    echo ""
    echo "============================================================================"
    echo "📊 各次评估成功率"
    echo "============================================================================"
    for ((i=0; i<${#SUCCESS_RATES[@]}; i++)); do
        echo "  第 $((i+1)) 次: ${SUCCESS_RATES[$i]}"
    done
    echo ""
    echo "📊 平均成功率 (${NUM_EVAL_RUNS} 次评估): ${AVG_SUCCESS_RATE}"
    echo ""
    echo "============================================================================"
    echo "📋 各次评估 EVALUATION SUMMARY"
    echo "============================================================================"
    
    for ((RUN_IDX=1; RUN_IDX<=NUM_EVAL_RUNS; RUN_IDX++)); do
        if [ -n "${VIDEO_DIR}" ]; then
            RUN_VIDEO_DIR="${VIDEO_DIR}_run${RUN_IDX}"
        else
            RUN_VIDEO_DIR="${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}/eval_rollouts_${RUN_IDX}_sai0_1"
        fi
        RUN_OUTPUT_FILE="${RUN_VIDEO_DIR}/eval_output_run${RUN_IDX}.log"
        
        echo ""
        echo "===== 第 ${RUN_IDX} 次评估 ====="
        if [ -f "${RUN_OUTPUT_FILE}" ]; then
            awk '
            /EVALUATION SUMMARY/ { found=1 }
            found {
                print
                if (/^=+$/) delim_count++
                if (delim_count >= 2) exit
            }
            ' "${RUN_OUTPUT_FILE}"
        else
            echo "日志文件不存在: ${RUN_OUTPUT_FILE}"
        fi
    done
    
    echo ""
    echo "============================================================================"
    echo "📈 汇总报告生成完成"
    echo "============================================================================"
} > "${SUMMARY_LOG}"

echo ""
echo "📝 汇总日志已保存到: ${SUMMARY_LOG}"
echo "============================================================================"
