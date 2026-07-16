#!/bin/bash
# ============================================================================
# Flow Matching Action Head (Flow_Matching_1) LIBERO 评估脚本
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数
#   2. 运行: bash run_eval_Sai0_1.sh
#
# 支持两种评估模式:
#   - LIBERO 仿真环境评估 (默认)
#   - 训练数据测试模式 (设置 USE_TRAINING_DATA="true")
#
# Flow_Matching_1 版本支持多层 VLM hidden states
#
# 任务指令替换功能:
#   - 可以替换指定任务ID的指令
#   - 格式: TASK_INSTRUCTION_OVERRIDES["任务ID"]="新指令"
#   - 例如: OVERRIDE_TASK_ID="1", OVERRIDE_INSTRUCTION="put the bowl on the white square plate"
#
# ============================================================================

# ===== 基本配置 =====

# Checkpoint 路径 (必填)
# 训练好的 action_head.pt 文件路径
CHECKPOINT_PATH="/data/HuangWenlong/datasets/qwen_extract_three_layers_hidden_state/libero_github_convert_for_qwen2b-only-libero_spatial_weight/Flow_Matching_1/3layer-1_14_28/bsz32*4/checkpoints/step_10000/action_head.pt"

# 数据集路径 (用于归一化统计量)
# 需要包含 meta/stats.json 文件
DATASET_PATH="/data/HuangWenlong/datasets/qwen_extract_three_layers_hidden_state/libero_github_convert_for_qwen2b-only-libero_spatial"

# ===== 多次评估配置 =====
# 评估次数 (默认为1)
NUM_EVAL_RUNS=1

# ===== 结果保存目录配置 =====
# 自定义路径 (取消注释并修改以使用自定义路径)
# VIDEO_DIR=""

# 自动生成的 VIDEO_DIR 格式说明:
# experiments/VLM_TYPE_ModelSize_VLM_LAYERS_VLM_OUTPUT_DIM_ActionPrompt_NumTrials_TaskSuite_ChunkSize_StepXXXXX/eval_rollouts_N_sai0_1
# 其中 N 为评估次数序号

# ===== VLM 配置 =====

GPU_NAME="A6000pro"
# VLM 类型
# 可选: "qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"
VLM_TYPE="qwen3_vl"

# VLM 模型路径
# Qwen: "Qwen/Qwen3-VL-2B-Instruct" 或 "Qwen/Qwen3-VL-4B-Instruct"
# Eagle (GR00T-N1.5-3B): "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
# Cosmos (GR00T-N1.6-3B): "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"
VLM_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"

# VLM 提取层 (多层，逗号分隔)
# Qwen2B: "1,14,28" 或其他层组合
# Qwen4B: "16,17,18"
# Eagle: "-1", Cosmos: "-1" 或 "-5" (使用负数索引)
VLM_LAYERS="1,14,28"

# VLM 输出维度
# Qwen2B: 2048 = 64 * 32
# Qwen4B: 2560 = 64 * 40
# Eagle: 2048, Cosmos: 2048
# 注意: 实际维度会从 checkpoint 自动检测
VLM_OUTPUT_DIM=2048

# Action backbone 维度
# Qwen2B: 1536
# Qwen4B: 2560
# Eagle: 1536, Cosmos: 1536
ACTION_BACKBONE_DIM=1536

# VL self-attention head 维度 (默认 64)
VL_SELF_ATTENTION_HEAD_DIM=64

# VL self-attention head 数量
# Qwen2B: 32 (2048 / 64 = 32)
# Qwen4B: 40 (2560 / 64 = 40)
# Eagle: 32, Cosmos: 32
VL_SELF_ATTENTION_NUM_HEADS=32

# ===== Prompt 配置 =====

# 内容顺序
# 可选: "images_first", "text_first", "interleaved", "single_image"
CONTENT_ORDER="images_first"

# 是否将指令转为小写 (true/false)
LOWERCASE_INSTRUCTION="true"

# 是否添加 generation prompt (true/false)
ADD_GENERATION_PROMPT="true"

# 是否添加 action prompt (true/false)
ADD_ACTION_PROMPT="true"

# ===== LIBERO 环境配置 =====

# 任务套件
# 可选: "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"
TASK_SUITE_NAME="libero_spatial"

# 每个任务的评估次数
NUM_TRIALS_PER_TASK=5

# 指定评估的任务 ID (留空则评估全部)
# 例如: "0,1,2" 表示只评估前三个任务
TASK_IDS=""

# 最大评估任务数 (-1 表示全部)
MAX_TASKS=-1

# 等待物体稳定的步数
NUM_STEPS_WAIT=10

# 每个 episode 的最大步数
MAX_STEPS=600

# 环境随机种子 (留空则使用随机数)
# 设置后每次评估使用相同的种子，便于结果复现
# 例如: ENV_SEED=42
ENV_SEED=""

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
OVERRIDE_INSTRUCTION="put the bowl on the white square plate"

# 方式2: 批量替换 (高级) - 优先级高
# 格式: "task_id1:instruction1|task_id2:instruction2|..."
# 非空时优先使用此配置，忽略方式1
# 例如: "0:pick up the red apple|1:put the bowl on the white square plate"
BATCH_OVERRIDES=""

# 是否启用任务指令替换 (总开关)
# 设为 "false" 时，上面的配置都不生效
ENABLE_INSTRUCTION_OVERRIDE="true"

# ===== 推理配置 =====

# 每次执行的动作数
# 设置为 1 表示每步重新预测
ACTION_CHUNK_SIZE=16

# 是否执行所有预测的动作 (true/false)
EXECUTE_ALL_CHUNKS="true"

# ===== Flow Matching 模型配置 =====

# 最大状态维度 (必须与预训练模型一致)
MAX_STATE_DIM=64

# 最大动作维度 (必须与预训练模型一致)
MAX_ACTION_DIM=32

# 预测的动作块数量 (action chunking)
NUM_ACTION_CHUNKS=16

# 实际动作维度
# LIBERO: 7 (x, y, z, ax, ay, az, gripper)
ACTION_DIM=7

# 推理时的去噪步数 (默认 4)
NUM_INFERENCE_TIMESTEPS=4

# ===== 系统配置 =====

# GPU 设备 ID
GPU_ID=0

# PyTorch 设备
DEVICE="cuda:0"

# 是否翻转图像 (true/false)
FLIP_IMAGES="true"

# 无头模式 (true/false)
HEADLESS="true"

# 详细输出 (true/false)
VERBOSE="true"

# ===== 训练数据测试模式 =====

# 是否使用训练数据进行测试 (true/false)
USE_TRAINING_DATA="false"

# 测试样本数量
NUM_TEST_SAMPLES=100

# ===== 运行脚本 =====

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# ===== 自动生成 VIDEO_DIR =====
# 从 VLM_MODEL_PATH 提取模型大小 (如 2B, 4B, 8B)
extract_model_size() {
    local path="$1"
    # 匹配 xB 格式 (如 2B, 4B, 8B, 3B)
    if [[ "$path" =~ ([0-9]+)[Bb] ]]; then
        echo "${BASH_REMATCH[1]}B"
    else
        echo "unknown"
    fi
}

# 从 CHECKPOINT_PATH 提取 step 数字
extract_step_number() {
    local path="$1"
    # 匹配 step_XXXXX 格式
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

# 添加 override 标识
if [ "${ENABLE_INSTRUCTION_OVERRIDE}" = "true" ]; then
    OVERRIDE_TAG="_override"
else
    OVERRIDE_TAG=""
fi

# 构建实验目录名
EXPERIMENT_NAME="${VLM_TYPE}_${MODEL_SIZE}_${VLM_LAYERS}_${VLM_OUTPUT_DIM}_${ACTION_PROMPT_TAG}_${NUM_TRIALS_PER_TASK}trials_${TASK_SUITE_NAME}_chunk${ACTION_CHUNK_SIZE}_${STEP_NUM}_${GPU_NAME}${OVERRIDE_TAG}_$(date +%Y%m%d)"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 设置 robosuite 日志目录
export ROBOSUITE_LOG="${HOME}/.robosuite/robosuite.log"
mkdir -p "${HOME}/.robosuite"

# 显示模型加载详细日志
export TRANSFORMERS_VERBOSITY=info

# ===== 加速模型加载 =====
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true

# 离线模式 - 跳过网络检查，直接使用本地缓存
# 如果模型已下载，这会大幅加速加载
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
echo "🚀 Flow Matching Action Head (Flow_Matching_1 - 多层) LIBERO 评估"
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
echo "  Layers: ${VLM_LAYERS} (多层)"
echo "  Output Dim: ${VLM_OUTPUT_DIM}"
echo "  Action Backbone Dim: ${ACTION_BACKBONE_DIM}"
echo "  VL Attention Heads: ${VL_SELF_ATTENTION_NUM_HEADS} × ${VL_SELF_ATTENTION_HEAD_DIM}"
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
echo "🔧 Flow Matching 配置:"
echo "  Max State Dim: ${MAX_STATE_DIM}"
echo "  Max Action Dim: ${MAX_ACTION_DIM}"
echo "  Action Chunks: ${NUM_ACTION_CHUNKS}"
echo "  Action Dim: ${ACTION_DIM}"
echo "  Inference Timesteps: ${NUM_INFERENCE_TIMESTEPS}"
echo ""
echo "⚡ 推理配置:"
echo "  Action Chunk Size: ${ACTION_CHUNK_SIZE}"
echo "  Execute All Chunks: ${EXECUTE_ALL_CHUNKS}"
echo ""
echo "💻 系统配置:"
echo "  GPU: ${GPU_ID}"
echo "  Flip Images: ${FLIP_IMAGES}"
echo "  Headless: ${HEADLESS}"
echo "  Verbose: ${VERBOSE}"
echo ""
echo "🧪 测试模式:"
echo "  Use Training Data: ${USE_TRAINING_DATA}"
echo "  Num Test Samples: ${NUM_TEST_SAMPLES}"
echo ""
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
echo "🔄 多次评估配置:"
echo "  Num Eval Runs: ${NUM_EVAL_RUNS}"
echo ""
echo "============================================================================"

# ===== 多次评估循环 =====
# 存储每次评估的成功率
declare -a SUCCESS_RATES
TOTAL_SUCCESS_RATE=0

for ((RUN_IDX=1; RUN_IDX<=NUM_EVAL_RUNS; RUN_IDX++)); do
    echo ""
    echo "============================================================================"
    echo "🔄 开始第 ${RUN_IDX}/${NUM_EVAL_RUNS} 次评估"
    echo "============================================================================"
    
    # 构建当前运行的 VIDEO_DIR
    # 如果用户自定义了 VIDEO_DIR (非空)，则使用自定义路径
    if [ -n "${VIDEO_DIR}" ]; then
        CURRENT_VIDEO_DIR="${VIDEO_DIR}_run${RUN_IDX}"
    else
        # 使用自动生成的路径
        CURRENT_VIDEO_DIR="${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}/eval_rollouts_${RUN_IDX}_sai0_1"
    fi
    
    # 创建目录
    mkdir -p "${CURRENT_VIDEO_DIR}"
    
    echo "  Video Dir: ${CURRENT_VIDEO_DIR}"
    
    # 构建命令 (使用当前的 VIDEO_DIR)
    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python -m eval.Sai0_1.libero.Flow_Matching_1.eval_Sai0_1"
    CMD="${CMD} --checkpoint_path ${CHECKPOINT_PATH}"
    CMD="${CMD} --vlm_model_path ${VLM_MODEL_PATH}"
    CMD="${CMD} --vlm_type ${VLM_TYPE}"
    CMD="${CMD} --dataset_path ${DATASET_PATH}"
    CMD="${CMD} --vlm_layers ${VLM_LAYERS}"
    CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"
    CMD="${CMD} --action_backbone_dim ${ACTION_BACKBONE_DIM}"
    CMD="${CMD} --vl_self_attention_head_dim ${VL_SELF_ATTENTION_HEAD_DIM}"
    CMD="${CMD} --vl_self_attention_num_attention_heads ${VL_SELF_ATTENTION_NUM_HEADS}"
    CMD="${CMD} --content_order ${CONTENT_ORDER}"
    CMD="${CMD} --task_suite_name ${TASK_SUITE_NAME}"
    CMD="${CMD} --num_trials_per_task ${NUM_TRIALS_PER_TASK}"
    CMD="${CMD} --max_tasks ${MAX_TASKS}"
    CMD="${CMD} --num_steps_wait ${NUM_STEPS_WAIT}"
    CMD="${CMD} --max_steps ${MAX_STEPS}"
    
    # 可选参数 - 环境随机种子
    if [ -n "${ENV_SEED}" ]; then
        CMD="${CMD} --env_seed ${ENV_SEED}"
    fi
    
    CMD="${CMD} --action_chunk_size ${ACTION_CHUNK_SIZE}"
    CMD="${CMD} --max_state_dim ${MAX_STATE_DIM}"
    CMD="${CMD} --max_action_dim ${MAX_ACTION_DIM}"
    CMD="${CMD} --num_action_chunks ${NUM_ACTION_CHUNKS}"
    CMD="${CMD} --action_dim ${ACTION_DIM}"
    CMD="${CMD} --num_inference_timesteps ${NUM_INFERENCE_TIMESTEPS}"
    CMD="${CMD} --device ${DEVICE}"
    CMD="${CMD} --video_dir ${CURRENT_VIDEO_DIR}"
    CMD="${CMD} --num_test_samples ${NUM_TEST_SAMPLES}"

    # 可选参数 - 任务 ID
    if [ -n "${TASK_IDS}" ]; then
        CMD="${CMD} --task_ids ${TASK_IDS}"
    fi

    # 可选参数 - 图像翻转
    if [ "${FLIP_IMAGES}" = "true" ]; then
        CMD="${CMD} --flip_images"
    else
        CMD="${CMD} --no_flip_images"
    fi

    # 可选参数 - Action Prompt
    if [ "${ADD_ACTION_PROMPT}" = "true" ]; then
        CMD="${CMD} --add_action_prompt"
    else
        CMD="${CMD} --no_action_prompt"
    fi

    # 可选参数 - 指令小写
    if [ "${LOWERCASE_INSTRUCTION}" = "true" ]; then
        CMD="${CMD} --lowercase_instruction"
    else
        CMD="${CMD} --no_lowercase_instruction"
    fi

    # 可选参数 - Generation Prompt
    if [ "${ADD_GENERATION_PROMPT}" = "true" ]; then
        CMD="${CMD} --add_generation_prompt"
    else
        CMD="${CMD} --no_generation_prompt"
    fi

    # 可选参数 - 执行所有动作块
    if [ "${EXECUTE_ALL_CHUNKS}" = "true" ]; then
        CMD="${CMD} --execute_all_chunks"
    fi

    # 可选参数 - 无头模式
    if [ "${HEADLESS}" = "true" ]; then
        CMD="${CMD} --headless"
    fi

    # 可选参数 - 详细输出
    if [ "${VERBOSE}" = "true" ]; then
        CMD="${CMD} --verbose"
    fi

    # 可选参数 - 训练数据测试模式
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
        echo "DATASET_PATH=${DATASET_PATH}"
        echo "VIDEO_DIR=${CURRENT_VIDEO_DIR}"
        echo ""
        echo "===== VLM 配置 ====="
        echo "VLM_TYPE=${VLM_TYPE}"
        echo "VLM_MODEL_PATH=${VLM_MODEL_PATH}"
        echo "MODEL_SIZE=${MODEL_SIZE}"
        echo "VLM_LAYERS=${VLM_LAYERS}"
        echo "VLM_OUTPUT_DIM=${VLM_OUTPUT_DIM}"
        echo "ACTION_BACKBONE_DIM=${ACTION_BACKBONE_DIM}"
        echo "VL_SELF_ATTENTION_HEAD_DIM=${VL_SELF_ATTENTION_HEAD_DIM}"
        echo "VL_SELF_ATTENTION_NUM_HEADS=${VL_SELF_ATTENTION_NUM_HEADS}"
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
        echo "===== 推理配置 ====="
        echo "ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE}"
        echo "EXECUTE_ALL_CHUNKS=${EXECUTE_ALL_CHUNKS}"
        echo ""
        echo "===== Flow Matching 模型配置 ====="
        echo "MAX_STATE_DIM=${MAX_STATE_DIM}"
        echo "MAX_ACTION_DIM=${MAX_ACTION_DIM}"
        echo "NUM_ACTION_CHUNKS=${NUM_ACTION_CHUNKS}"
        echo "ACTION_DIM=${ACTION_DIM}"
        echo "NUM_INFERENCE_TIMESTEPS=${NUM_INFERENCE_TIMESTEPS}"
        echo ""
        echo "===== 系统配置 ====="
        echo "GPU_ID=${GPU_ID}"
        echo "DEVICE=${DEVICE}"
        echo "FLIP_IMAGES=${FLIP_IMAGES}"
        echo "HEADLESS=${HEADLESS}"
        echo "VERBOSE=${VERBOSE}"
        echo ""
        echo "===== 训练数据测试模式 ====="
        echo "USE_TRAINING_DATA=${USE_TRAINING_DATA}"
        echo "NUM_TEST_SAMPLES=${NUM_TEST_SAMPLES}"
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
    
    # 从输出中提取成功率 (假设输出格式包含 "Success Rate:" 或类似格式)
    # 尝试匹配常见的成功率输出格式
    CURRENT_SUCCESS_RATE=$(grep -oP '(?i)(success\s*rate|avg.*success|average.*success|overall.*success)[:\s]*(\d+\.?\d*)' "${OUTPUT_FILE}" | tail -1 | grep -oP '\d+\.?\d*$' || echo "0")
    
    if [ -z "${CURRENT_SUCCESS_RATE}" ] || [ "${CURRENT_SUCCESS_RATE}" = "0" ]; then
        # 尝试其他格式
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
    
    # 记录成功率
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

# 计算平均值
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
    
    # 提取每次评估的 EVALUATION SUMMARY
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
            # 提取 EVALUATION SUMMARY 部分 (从 "EVALUATION SUMMARY" 到结束分隔线)
            # 使用 awk 捕获完整的评估摘要块
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

