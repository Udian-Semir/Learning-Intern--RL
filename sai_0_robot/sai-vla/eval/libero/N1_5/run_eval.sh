#!/bin/bash
# =============================================================================
# GR00T N1.5 LIBERO Evaluation Script
# =============================================================================
# 使用方法: 
#   1. 修改下面的 MODEL_PATH 为你的 checkpoint 路径
#   2. 运行: bash run_eval.sh
# =============================================================================

# ==========================
# 【必须修改】模型路径
# ==========================
MODEL_PATH="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_eagle25-only-libero_spatial_weight/checkpoints/step_4200"  # <-- 修改这里！

# ==========================
# 评估配置
# ==========================
BENCHMARK="libero_spatial"           # 可选: libero_spatial, libero_object, libero_goal, libero_10, libero_90, libero_100
TASK_ID=""                      # 留空评估所有任务，或设置为具体数字如 "0" 评估单个任务
NUM_ROLLOUTS=50                 # 每个任务的 rollout 数量
MAX_STEPS="600"                    # 留空自动设置，或手动指定如 "600"

# ==========================
# 多次评估配置
# ==========================
NUM_EVAL_RUNS=1                 # 评估次数 (默认为1)

# ==========================
# 模型配置
# ==========================
EMBODIMENT_TAG="new_embodiment" # 训练时使用的 embodiment tag
ACTION_NORM="min_max"           # 可选: min_max, mean_std
DENOISING_STEPS=""              # 留空使用默认值，或设置如 "4"
BASE_MODEL_PATH="nvidia/GR00T-N1.5-3B"  # 基础预训练模型路径（HuggingFace 或本地路径）

# ==========================
# 环境配置
# ==========================
RESOLUTION=256                  # 相机分辨率
NUM_STEPS_WAIT=10               # 环境稳定等待步数
SEED=42                         # 随机种子
DEVICE="cuda:0"                 # 推理设备

# ==========================
# 输出配置
# ==========================
SAVE_VIDEO=true                # 是否保存视频: true/false
RESULTS_DIR="./n1_5_eval_results"

# 自定义路径 (取消注释并修改以使用自定义路径)
# VIDEO_DIR=""

# 自动生成的 VIDEO_DIR 格式说明:
# experiments/N1_5_ModelSize_Benchmark_NumRollouts_StepXXXXX/eval_rollouts_N_n1_5
# 其中 N 为评估次数序号

# =============================================================================
# 以下为执行代码，一般不需要修改
# =============================================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_n1_5.py"

# ===== 自动生成 VIDEO_DIR =====
# 从 BASE_MODEL_PATH 提取模型大小 (如 3B)
extract_model_size() {
    local path="$1"
    if [[ "$path" =~ ([0-9]+)[Bb] ]]; then
        echo "${BASH_REMATCH[1]}B"
    else
        echo "unknown"
    fi
}

# 从 MODEL_PATH 提取 step 数字
extract_step_number() {
    local path="$1"
    if [[ "$path" =~ step_([0-9]+) ]]; then
        echo "step_${BASH_REMATCH[1]}"
    else
        echo "step_unknown"
    fi
}

# 获取模型大小和 step 数字
MODEL_SIZE=$(extract_model_size "${BASE_MODEL_PATH}")
STEP_NUM=$(extract_step_number "${MODEL_PATH}")

# 构建实验目录名
EXPERIMENT_NAME="N1_5_${MODEL_SIZE}_${BENCHMARK}_${NUM_ROLLOUTS}rollouts_${STEP_NUM}"

# 检查模型路径
if [[ "$MODEL_PATH" == "/path/to/your/finetuned/checkpoint" ]]; then
    echo "❌ 错误: 请先修改 MODEL_PATH 为你的 checkpoint 路径！"
    echo "   打开 run_eval.sh 并修改第 15 行的 MODEL_PATH"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

# 打印配置
echo "=============================================="
echo "GR00T N1.5 LIBERO Evaluation"
echo "=============================================="
echo "Model Path:      ${MODEL_PATH}"
echo "Base Model:      ${BASE_MODEL_PATH}"
echo "Model Size:      ${MODEL_SIZE}"
echo "Benchmark:       ${BENCHMARK}"
echo "Task ID:         ${TASK_ID:-all}"
echo "Num Rollouts:    ${NUM_ROLLOUTS}"
echo "Max Steps:       ${MAX_STEPS:-auto}"
echo "Embodiment Tag:  ${EMBODIMENT_TAG}"
echo "Action Norm:     ${ACTION_NORM}"
echo "Denoising Steps: ${DENOISING_STEPS:-default}"
echo "Resolution:      ${RESOLUTION}"
echo "Device:          ${DEVICE}"
echo "Save Video:      ${SAVE_VIDEO}"
echo "Results Dir:     ${RESULTS_DIR}"
echo "Experiment Dir:  ${SCRIPT_DIR}/experiments/${EXPERIMENT_NAME}"
echo ""
echo "Num Eval Runs:   ${NUM_EVAL_RUNS}"
echo "=============================================="

# ===== 多次评估循环 =====
declare -a SUCCESS_RATES
TOTAL_SUCCESS_RATE=0

for ((RUN_IDX=1; RUN_IDX<=NUM_EVAL_RUNS; RUN_IDX++)); do
    echo ""
    echo "=============================================="
    echo "🔄 开始第 ${RUN_IDX}/${NUM_EVAL_RUNS} 次评估"
    echo "=============================================="
    
    # 构建当前运行的 VIDEO_DIR
    if [ -n "${VIDEO_DIR}" ]; then
        CURRENT_VIDEO_DIR="${VIDEO_DIR}_run${RUN_IDX}"
    else
        CURRENT_VIDEO_DIR="${SCRIPT_DIR}/experiments/${EXPERIMENT_NAME}/eval_rollouts_${RUN_IDX}_n1_5"
    fi
    
    mkdir -p "${CURRENT_VIDEO_DIR}"
    echo "Video Dir: ${CURRENT_VIDEO_DIR}"
    
    # 构建命令
    CMD="python ${EVAL_SCRIPT}"
    CMD="${CMD} --model-path ${MODEL_PATH}"
    CMD="${CMD} --base-model-path ${BASE_MODEL_PATH}"
    CMD="${CMD} --benchmark ${BENCHMARK}"
    CMD="${CMD} --embodiment-tag ${EMBODIMENT_TAG}"
    CMD="${CMD} --action-norm ${ACTION_NORM}"
    CMD="${CMD} --num-rollouts ${NUM_ROLLOUTS}"
    CMD="${CMD} --resolution ${RESOLUTION}"
    CMD="${CMD} --num-steps-wait ${NUM_STEPS_WAIT}"
    CMD="${CMD} --seed ${SEED}"
    CMD="${CMD} --device ${DEVICE}"
    CMD="${CMD} --results-dir ${RESULTS_DIR}"
    CMD="${CMD} --video-dir ${CURRENT_VIDEO_DIR}"

    # 可选参数
    if [[ -n "$TASK_ID" ]]; then
        CMD="${CMD} --task-id ${TASK_ID}"
    fi

    if [[ -n "$MAX_STEPS" ]]; then
        CMD="${CMD} --max-steps ${MAX_STEPS}"
    fi

    if [[ -n "$DENOISING_STEPS" ]]; then
        CMD="${CMD} --denoising-steps ${DENOISING_STEPS}"
    fi

    if [[ "$SAVE_VIDEO" == "true" ]]; then
        CMD="${CMD} --save-video"
    fi

    echo ""
    echo "Running command:"
    echo "${CMD}"
    echo ""

    # 执行评估并捕获输出
    OUTPUT_FILE="${CURRENT_VIDEO_DIR}/eval_output_run${RUN_IDX}.log"
    
    # 在日志顶部写入完整参数配置
    {
        echo "=============================================="
        echo "📋 评估参数配置记录"
        echo "=============================================="
        echo "记录时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "评估次数: 第 ${RUN_IDX}/${NUM_EVAL_RUNS} 次"
        echo ""
        echo "===== 模型配置 ====="
        echo "MODEL_PATH=${MODEL_PATH}"
        echo "BASE_MODEL_PATH=${BASE_MODEL_PATH}"
        echo "MODEL_SIZE=${MODEL_SIZE}"
        echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
        echo "ACTION_NORM=${ACTION_NORM}"
        echo "DENOISING_STEPS=${DENOISING_STEPS:-default}"
        echo ""
        echo "===== 评估配置 ====="
        echo "BENCHMARK=${BENCHMARK}"
        echo "TASK_ID=${TASK_ID:-all}"
        echo "NUM_ROLLOUTS=${NUM_ROLLOUTS}"
        echo "MAX_STEPS=${MAX_STEPS:-auto}"
        echo ""
        echo "===== 环境配置 ====="
        echo "RESOLUTION=${RESOLUTION}"
        echo "NUM_STEPS_WAIT=${NUM_STEPS_WAIT}"
        echo "SEED=${SEED}"
        echo "DEVICE=${DEVICE}"
        echo ""
        echo "===== 输出配置 ====="
        echo "SAVE_VIDEO=${SAVE_VIDEO}"
        echo "RESULTS_DIR=${RESULTS_DIR}"
        echo "VIDEO_DIR=${CURRENT_VIDEO_DIR}"
        echo ""
        echo "===== 多次评估配置 ====="
        echo "NUM_EVAL_RUNS=${NUM_EVAL_RUNS}"
        echo ""
        echo "===== 执行命令 ====="
        echo "${CMD}"
        echo ""
        echo "=============================================="
        echo "📋 评估输出开始"
        echo "=============================================="
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
    echo "=============================================="
    echo "📊 第 ${RUN_IDX} 次评估完成"
    echo "   成功率: ${CURRENT_SUCCESS_RATE}"
    echo "=============================================="
    
    SUCCESS_RATES+=("${CURRENT_SUCCESS_RATE}")
    TOTAL_SUCCESS_RATE=$(echo "${TOTAL_SUCCESS_RATE} + ${CURRENT_SUCCESS_RATE}" | bc -l 2>/dev/null || echo "${TOTAL_SUCCESS_RATE}")
done

# ===== 打印汇总结果 =====
echo ""
echo "=============================================="
echo "📈 所有评估完成 - 汇总结果"
echo "=============================================="
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
echo "=============================================="
