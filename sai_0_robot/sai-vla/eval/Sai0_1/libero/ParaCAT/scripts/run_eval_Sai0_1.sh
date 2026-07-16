#!/bin/bash
# ============================================================================
# ParaCAT Action Head LIBERO 评估脚本
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
# ParaCAT 使用离散化动作预测 (3分类: 后退/不动/前进)
# 可选使用 Pons Adapter 进行特征聚合
#
# ============================================================================

# ===== 基本配置 =====

# ParaCAT Checkpoint 路径 (必填)
# 训练好的 paracat.pt 文件路径
PARACAT_CHECKPOINT="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys1_qwen_2b_14_weight/ParaCAT/1layer_14/libero_object/5090_bsz80*1*8_pons_q128_chunk16_tb2_20000steps_20260117/checkpoints/step_20000/paracat.pt"

# Pons Checkpoint 路径 (可选)
# 预训练的 pons.pt 文件路径，留空则不使用 Pons
PONS_CHECKPOINT="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys1_qwen_2b_14_weight/ParaCAT/1layer_14/libero_object/5090_bsz80*1*8_pons_q128_chunk16_tb2_20000steps_20260117/checkpoints/step_20000/pons.pt"

# 数据集路径 (用于归一化统计量)
# 需要包含 meta/stats.json 文件
DATASET_PATH="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys1_qwen_2b_14"

# ===== 多次评估配置 =====
# 评估次数 (默认为1)
NUM_EVAL_RUNS=5

# ===== 结果保存目录配置 =====
# 自定义路径 (取消注释并修改以使用自定义路径)
# VIDEO_DIR=""

# ===== VLM 配置 =====

GPU_NAME="5090"
# VLM 类型
# 可选: "qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"
VLM_TYPE="qwen3_vl"

# VLM 模型路径
# Qwen: "Qwen/Qwen3-VL-2B-Instruct" 或 "Qwen/Qwen3-VL-4B-Instruct"
# Eagle (GR00T-N1.5-3B): "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"
# Cosmos (GR00T-N1.6-3B): "/home/sythoid_01/.cache/huggingface/hub/models--nvidia--GR00T-N1.6-3B/snapshots/d0814e7ecb19202e7c8468b46098b0b7ef3a6d61"
VLM_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"

# VLM 提取层
# Qwen2B: "14", Qwen4B: "16", Eagle: "-1", Cosmos: "-1" 或 "-5"
VLM_LAYERS="14"

# VLM 输出维度
# Qwen2B: 2048, Qwen4B: 2560, Eagle: 2048, Cosmos: 2048
VLM_OUTPUT_DIM=2048

# ===== ParaCAT 模型配置 =====

# 动作块大小 (action chunking)
CHUNK_SIZE=16

# 动作维度
# LIBERO: 7 (x, y, z, ax, ay, az, gripper)
ACTION_DIM=7

# Transformer 块数量
NUM_TRANSFORMER_BLOCKS=2

# MLP 层数量
NUM_MLP_LAYERS=2

# MLP 扩展维度
MLP_EXPAND_DIM=1024

# 注意力头数量
NUM_HEADS=8

# ===== Pons Adapter 配置 =====

# Pons query 序列长度
PONS_Q_SEQ_LEN=128

# Pons 块数量
PONS_NUM_BLOCKS=2

# Pons 注意力头数量
PONS_NUM_HEADS=8

# ===== 离散化配置 =====
# 反离散化列索引 (空格分隔)
# 例如: "0 1 2 3 4 5" 表示前6列需要反离散化
# 这些列会乘以 delta 还原为连续值
UNDISCRETE_COLUMNS="0 1 2 3 4 5"

# 对应列的 delta 值 (空格分隔)
# 例如: "0.01 0.01 0.01 0.05 0.05 0.05"
UNDISCRETE_DELTAS="0.9 0.9 0.9 0.15 0.3 0.3"

# ===== Gripper 列参数 (LIBERO 专用) =====
# Gripper 列索引 (空格分隔)
# 这些列不做反离散化（不乘 delta），直接保持 {-1, 0, 1} 传给 LIBERO 环境
# 示例:
#   单个 gripper: GRIPPER_COLUMNS="6"
#   多个 gripper: GRIPPER_COLUMNS="6 7"
#
# 注意: LIBERO 环境可以接受 {-1, 0, 1} 的 gripper 值:
#   -1: 关闭夹爪 (close)
#    0: 保持当前状态 (maintain)
#   +1: 打开夹爪 (open)
GRIPPER_COLUMNS="6"

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
TASK_SUITE_NAME="libero_object"

# 每个任务的评估次数
NUM_TRIALS_PER_TASK=10

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
ENV_SEED=""

# ===== 推理配置 =====

# 每次执行的动作数
# 设置为 1 表示每步重新预测
ACTION_CHUNK_SIZE=16

# 是否执行所有预测的动作 (true/false)
EXECUTE_ALL_CHUNKS="true"

# ===== 系统配置 =====

# GPU 设备 ID
GPU_ID=1

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
    if [[ "$path" =~ ([0-9]+)[Bb] ]]; then
        echo "${BASH_REMATCH[1]}B"
    else
        echo "unknown"
    fi
}

# 从 PARACAT_CHECKPOINT 提取 step 数字
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
STEP_NUM=$(extract_step_number "${PARACAT_CHECKPOINT}")

# Action Prompt 标识
if [ "${ADD_ACTION_PROMPT}" = "true" ]; then
    ACTION_PROMPT_TAG="act"
else
    ACTION_PROMPT_TAG="noact"
fi

# Pons 标识
if [ -n "${PONS_CHECKPOINT}" ]; then
    PONS_TAG="pons"
else
    PONS_TAG="nopons"
fi

# 构建实验目录名
EXPERIMENT_NAME="ParaCAT_${VLM_TYPE}_${MODEL_SIZE}_${VLM_LAYERS}_${VLM_OUTPUT_DIM}_${ACTION_PROMPT_TAG}_${PONS_TAG}_${NUM_TRIALS_PER_TASK}trials_${TASK_SUITE_NAME}_chunk${ACTION_CHUNK_SIZE}_${STEP_NUM}_${GPU_NAME}_$(date +%Y%m%d)"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 设置 robosuite 日志目录
export ROBOSUITE_LOG="${HOME}/.robosuite/robosuite.log"
mkdir -p "${HOME}/.robosuite"

# ===== 加速模型加载 =====
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=true

# 离线模式 - 跳过网络检查，直接使用本地缓存
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 打印配置
echo "============================================================================"
echo "🚀 ParaCAT Action Head LIBERO 评估"
echo "============================================================================"
echo ""
echo "📁 路径配置:"
echo "  ParaCAT Checkpoint: ${PARACAT_CHECKPOINT}"
echo "  Pons Checkpoint: ${PONS_CHECKPOINT:-未使用}"
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
echo "🎯 ParaCAT 配置:"
echo "  Chunk Size: ${CHUNK_SIZE}"
echo "  Action Dim: ${ACTION_DIM}"
echo "  Transformer Blocks: ${NUM_TRANSFORMER_BLOCKS}"
echo "  MLP Layers: ${NUM_MLP_LAYERS}"
echo "  MLP Expand Dim: ${MLP_EXPAND_DIM}"
echo "  Num Heads: ${NUM_HEADS}"
echo ""
echo "🔗 Pons 配置:"
echo "  使用 Pons: ${PONS_CHECKPOINT:+是}${PONS_CHECKPOINT:-否}"
echo "  Q Seq Len: ${PONS_Q_SEQ_LEN}"
echo "  Num Blocks: ${PONS_NUM_BLOCKS}"
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
echo "⚡ 推理配置:"
echo "  Action Chunk Size: ${ACTION_CHUNK_SIZE}"
echo "  Execute All Chunks: ${EXECUTE_ALL_CHUNKS}"
echo ""
echo "🔢 离散化配置:"
echo "  Undiscrete Columns: ${UNDISCRETE_COLUMNS:-未设置}"
echo "  Undiscrete Deltas: ${UNDISCRETE_DELTAS:-未设置}"
echo "  Gripper Columns (不反离散化): ${GRIPPER_COLUMNS:-未设置}"
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
    if [ -n "${VIDEO_DIR}" ]; then
        CURRENT_VIDEO_DIR="${VIDEO_DIR}_run${RUN_IDX}"
    else
        CURRENT_VIDEO_DIR="${SCRIPT_DIR}/../experiments/${EXPERIMENT_NAME}/eval_rollouts_${RUN_IDX}_sai0_1"
    fi
    
    # 创建目录
    mkdir -p "${CURRENT_VIDEO_DIR}"
    
    echo "  Video Dir: ${CURRENT_VIDEO_DIR}"
    
    # 构建命令
    CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python -m eval.Sai0_1.libero.ParaCAT.eval_Sai0_1"
    CMD="${CMD} --paracat_checkpoint ${PARACAT_CHECKPOINT}"
    CMD="${CMD} --vlm_model_path ${VLM_MODEL_PATH}"
    CMD="${CMD} --vlm_type ${VLM_TYPE}"
    CMD="${CMD} --dataset_path ${DATASET_PATH}"
    CMD="${CMD} --vlm_layers ${VLM_LAYERS}"
    CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"
    
    # ParaCAT 参数
    CMD="${CMD} --chunk_size ${CHUNK_SIZE}"
    CMD="${CMD} --action_dim ${ACTION_DIM}"
    CMD="${CMD} --num_transformer_blocks ${NUM_TRANSFORMER_BLOCKS}"
    CMD="${CMD} --num_mlp_layers ${NUM_MLP_LAYERS}"
    CMD="${CMD} --mlp_expand_dim ${MLP_EXPAND_DIM}"
    CMD="${CMD} --num_heads ${NUM_HEADS}"
    
    # Pons 参数 (可选)
    if [ -n "${PONS_CHECKPOINT}" ]; then
        CMD="${CMD} --pons_checkpoint ${PONS_CHECKPOINT}"
    fi
    CMD="${CMD} --pons_q_seq_len ${PONS_Q_SEQ_LEN}"
    CMD="${CMD} --pons_num_blocks ${PONS_NUM_BLOCKS}"
    
    # 离散化参数 (可选)
    if [ -n "${UNDISCRETE_COLUMNS}" ]; then
        CMD="${CMD} --undiscrete_columns ${UNDISCRETE_COLUMNS}"
    fi
    if [ -n "${UNDISCRETE_DELTAS}" ]; then
        CMD="${CMD} --undiscrete_deltas ${UNDISCRETE_DELTAS}"
    fi
    
    # Gripper 列参数 (可选)
    if [ -n "${GRIPPER_COLUMNS}" ]; then
        CMD="${CMD} --gripper_columns ${GRIPPER_COLUMNS}"
    fi
    
    # Prompt 参数
    CMD="${CMD} --content_order ${CONTENT_ORDER}"
    
    # LIBERO 参数
    CMD="${CMD} --task_suite_name ${TASK_SUITE_NAME}"
    CMD="${CMD} --num_trials_per_task ${NUM_TRIALS_PER_TASK}"
    CMD="${CMD} --max_tasks ${MAX_TASKS}"
    CMD="${CMD} --num_steps_wait ${NUM_STEPS_WAIT}"
    CMD="${CMD} --max_steps ${MAX_STEPS}"
    
    # 可选参数 - 环境随机种子
    if [ -n "${ENV_SEED}" ]; then
        CMD="${CMD} --env_seed ${ENV_SEED}"
    fi
    
    # 推理参数
    CMD="${CMD} --action_chunk_size ${ACTION_CHUNK_SIZE}"
    
    # 系统参数
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
        echo "PARACAT_CHECKPOINT=${PARACAT_CHECKPOINT}"
        echo "PONS_CHECKPOINT=${PONS_CHECKPOINT}"
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
        echo "===== ParaCAT 配置 ====="
        echo "CHUNK_SIZE=${CHUNK_SIZE}"
        echo "ACTION_DIM=${ACTION_DIM}"
        echo "NUM_TRANSFORMER_BLOCKS=${NUM_TRANSFORMER_BLOCKS}"
        echo "NUM_MLP_LAYERS=${NUM_MLP_LAYERS}"
        echo "MLP_EXPAND_DIM=${MLP_EXPAND_DIM}"
        echo "NUM_HEADS=${NUM_HEADS}"
        echo ""
        echo "===== Pons 配置 ====="
        echo "PONS_Q_SEQ_LEN=${PONS_Q_SEQ_LEN}"
        echo "PONS_NUM_BLOCKS=${PONS_NUM_BLOCKS}"
        echo ""
        echo "===== 离散化配置 ====="
        echo "UNDISCRETE_COLUMNS=${UNDISCRETE_COLUMNS}"
        echo "UNDISCRETE_DELTAS=${UNDISCRETE_DELTAS}"
        echo "GRIPPER_COLUMNS=${GRIPPER_COLUMNS}"
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
    echo "PARACAT_CHECKPOINT=${PARACAT_CHECKPOINT}"
    echo "PONS_CHECKPOINT=${PONS_CHECKPOINT}"
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

