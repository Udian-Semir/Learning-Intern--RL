#!/usr/bin/env bash
# ============================================================================
# Sai0_1 OFT1_0 - LIBERO-plus 评估启动脚本 (qwen_eagle_hwl 环境)
# ============================================================================
#
# 使用前置条件:
#   1. conda env 名为 qwen_eagle_hwl, 已经安装 sai0-vla 项目所需依赖
#   2. /data_disk1/hwl/LIBERO-plus 已经 git clone 完成
#   3. (首次) 运行: python -m eval.Sai0_1.libero_plus.OFT1_0.setup_libero_plus
#      会做: touch __init__.py + symlink assets + 写独立 config.yaml
#
# 用法:
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus.sh
#
# 单独评估某一类扰动 (示例):
#   CATEGORIES="Camera Viewpoints" \\
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus.sh
#
# 快速调试 (前 20 个任务):
#   MAX_TASKS=20 bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus.sh
# ============================================================================

set -e

# ===== Conda 环境 =====
CONDA_ENV="${CONDA_ENV:-qwen_eagle_hwl}"

# ===== 模型 / 数据集 路径 (按需修改) =====

# Action Head 训练好的 checkpoint (libero_plus_lerobot 上训的 OFT1_0)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data_disk2/hwl/checkpoints/libero_plus_lerobot/action7_chunk16_pretrain/p6000_bsz500*1*8_tb4_200000steps_layer14_20260526_wplibero_plus_libero_plus_CBC_true_dtypefloat32_USE_AMP_true/checkpoints/step_200000/action_head.pt}"

# 训练时使用的 lerobot 数据集 (用于读 stats.json)
DATASET_PATH="${DATASET_PATH:-/data_disk2/hwl/datasets/libero_plus_lerobot}"

# ===== VLM 配置 =====
# Qwen3-VL-2B (训练时使用): VLM_OUTPUT_DIM=2048, layer 14
VLM_TYPE="${VLM_TYPE:-qwen3_vl}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
VLM_LAYERS="${VLM_LAYERS:-14}"
VLM_OUTPUT_DIM="${VLM_OUTPUT_DIM:-2048}"

# ===== Prompt 配置 (与训练保持一致) =====
CONTENT_ORDER="${CONTENT_ORDER:-images_first}"
LOWERCASE_INSTRUCTION="${LOWERCASE_INSTRUCTION:-true}"
ADD_GENERATION_PROMPT="${ADD_GENERATION_PROMPT:-true}"
ADD_ACTION_PROMPT="${ADD_ACTION_PROMPT:-false}"

# ===== OFT 模型超参 =====
NUM_TRANSFORMER_BLOCKS="${NUM_TRANSFORMER_BLOCKS:-4}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
DROPOUT="${DROPOUT:-0.1}"
ACTION_HEAD_HIDDEN_DIM="${ACTION_HEAD_HIDDEN_DIM:-4096}"
NUM_ACTION_CHUNKS="${NUM_ACTION_CHUNKS:-16}"
ACTION_DIM="${ACTION_DIM:-7}"

# ===== LIBERO-plus 配置 =====
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"  # libero_spatial / libero_object / libero_goal / libero_10
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"        # 官方约定 1
TASK_IDS="${TASK_IDS:-}"                                # 留空 -> 全部
MAX_TASKS="${MAX_TASKS:--1}"                            # -1 -> 全部
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
ENV_SEED="${ENV_SEED:-}"
CATEGORIES="${CATEGORIES:-}"                            # e.g. "Camera Viewpoints,Robot Initial States"
DIFFICULTY_LEVELS="${DIFFICULTY_LEVELS:-}"              # e.g. "1,2,3"

# ===== 推理配置 =====
EXECUTE_ALL_CHUNKS="${EXECUTE_ALL_CHUNKS:-true}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-16}"

# ===== 系统配置 =====
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda:0}"
# FLIP_IMAGES: 把 LIBERO env 直出的颠倒图像 (OpenGL 渲染) 翻 180° 后再喂给 VLM.
# 训练时 LeRobot 数据集是"正向"方向, 必须翻转才能跟训练对齐.
FLIP_IMAGES="${FLIP_IMAGES:-true}"
# VIDEO_FLIP: 视频保存时是否翻转 (仅影响 mp4 展示, 不影响给模型的输入).
VIDEO_FLIP="${VIDEO_FLIP:-true}"
# 默认保存每个 task 的视频 (libero_spatial 2402 task 会存 ~2402 个 mp4)
# 想抽样减磁盘:  SAVE_VIDEO_EVERY=50
# 不想存:        SAVE_VIDEOS=false
SAVE_VIDEOS="${SAVE_VIDEOS:-true}"
SAVE_VIDEO_EVERY="${SAVE_VIDEO_EVERY:-1}"
VERBOSE="${VERBOSE:-false}"
RESUME="${RESUME:-true}"

# ===== 实时进度显示 =====
# PRINT_PER_TASK=true:  每完成 1 个 task 在 stdout 单独 print 一行带 ✓/✗ 的成功率
# SUMMARY_EVERY=100:    每 100 个 ep 打印一次完整 per-category / per-difficulty 表格 (设 0 关掉)
PRINT_PER_TASK="${PRINT_PER_TASK:-true}"
SUMMARY_EVERY="${SUMMARY_EVERY:-100}"

# ===== LIBERO-plus 环境引导 (eval_libero_plus.py 内部也会做, 这里只是显式 export) =====
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data_disk1/hwl/LIBERO-plus}"
export LIBERO_PLUS_CONFIG_DIR="${LIBERO_PLUS_CONFIG_DIR:-$HOME/.libero_plus_sai0}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_DIR}"

# OFT1_0 在 constants.py 里通过 _env_int 读 ACTION_DIM / NUM_ACTIONS_CHUNK / PROPRIO_DIM,
# 默认值 (8 / 1000 / 8) 与 libero_plus_lerobot (action7_chunk16) 不一致,必须显式 export
export ACTION_DIM="${ACTION_DIM}"
export NUM_ACTIONS_CHUNK="${NUM_ACTION_CHUNKS}"
export PROPRIO_DIM="${PROPRIO_DIM:-8}"

# 修复 robosuite log 权限
export ROBOSUITE_LOG="${HOME}/.robosuite/robosuite.log"
mkdir -p "${HOME}/.robosuite"

# 禁用 HuggingFace 网络请求(模型已经下载到本地缓存,在线模式会卡 1-3 分钟)
# 想强制走在线: HF_HUB_OFFLINE=0 HF_HUB_OFFLINE=0 bash run_eval_libero_plus.sh
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"

# ===== 输出路径 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# 自动从 CHECKPOINT_PATH 提取 step 编号
extract_step_number() {
    local path="$1"
    if [[ "$path" =~ step_([0-9]+) ]]; then
        echo "step_${BASH_REMATCH[1]}"
    else
        echo "step_unknown"
    fi
}
STEP_TAG=$(extract_step_number "${CHECKPOINT_PATH}")

ACT_TAG="noact"
[[ "${ADD_ACTION_PROMPT}" == "true" ]] && ACT_TAG="act"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${VLM_TYPE}_${VLM_LAYERS}_${VLM_OUTPUT_DIM}_${ACT_TAG}_${TASK_SUITE_NAME}_${STEP_TAG}_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/experiments}"
VIDEO_DIR="${VIDEO_DIR:-${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${VIDEO_DIR}}"
mkdir -p "${VIDEO_DIR}"

# ===== 打印配置 =====
echo "============================================================================"
echo "🤖 Sai0_1 OFT1_0 - LIBERO-plus 评估"
echo "============================================================================"
echo "Conda env       : ${CONDA_ENV}"
echo "LIBERO_PLUS_ROOT: ${LIBERO_PLUS_ROOT}"
echo "Config dir      : ${LIBERO_PLUS_CONFIG_DIR}"
echo "Checkpoint      : ${CHECKPOINT_PATH}"
echo "Dataset         : ${DATASET_PATH}"
echo "VLM model       : ${VLM_MODEL_PATH}"
echo "Task suite      : ${TASK_SUITE_NAME}"
echo "Trials per task : ${NUM_TRIALS_PER_TASK}"
echo "Max tasks       : ${MAX_TASKS}"
echo "Categories      : ${CATEGORIES:-(all)}"
echo "Diff levels     : ${DIFFICULTY_LEVELS:-(all)}"
echo "Video dir       : ${VIDEO_DIR}"
echo "Save videos     : ${SAVE_VIDEOS} (每 ${SAVE_VIDEO_EVERY} 个 task)"
echo "Resume          : ${RESUME}"
echo "============================================================================"

# ===== Setup (确保 LIBERO-plus 环境就绪) =====
echo ""
echo "🛠️  Step 1/2: 确保 LIBERO-plus 环境配置好"

cd "${PROJECT_ROOT}"
"$(conda info --base)/envs/${CONDA_ENV}/bin/python" -m eval.Sai0_1.libero_plus.OFT1_0.setup_libero_plus \
    --libero_plus_root "${LIBERO_PLUS_ROOT}" \
    --config_dir "${LIBERO_PLUS_CONFIG_DIR}"

# ===== 构建评估命令 =====
echo ""
echo "🚀 Step 2/2: 启动 LIBERO-plus 评估"

CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} \"$(conda info --base)/envs/${CONDA_ENV}/bin/python\" -m eval.Sai0_1.libero_plus.OFT1_0.eval_libero_plus"
CMD="${CMD} --checkpoint_path \"${CHECKPOINT_PATH}\""
CMD="${CMD} --vlm_model_path \"${VLM_MODEL_PATH}\""
CMD="${CMD} --vlm_type ${VLM_TYPE}"
CMD="${CMD} --dataset_path \"${DATASET_PATH}\""
CMD="${CMD} --vlm_layers ${VLM_LAYERS}"
CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"
CMD="${CMD} --content_order ${CONTENT_ORDER}"
CMD="${CMD} --task_suite_name ${TASK_SUITE_NAME}"
CMD="${CMD} --num_trials_per_task ${NUM_TRIALS_PER_TASK}"
CMD="${CMD} --max_tasks ${MAX_TASKS}"
CMD="${CMD} --num_steps_wait ${NUM_STEPS_WAIT}"
CMD="${CMD} --max_steps ${MAX_STEPS}"
CMD="${CMD} --action_chunk_size ${ACTION_CHUNK_SIZE}"
CMD="${CMD} --num_transformer_blocks ${NUM_TRANSFORMER_BLOCKS}"
CMD="${CMD} --num_attention_heads ${NUM_ATTENTION_HEADS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --action_head_hidden_dim ${ACTION_HEAD_HIDDEN_DIM}"
CMD="${CMD} --num_action_chunks ${NUM_ACTION_CHUNKS}"
CMD="${CMD} --action_dim ${ACTION_DIM}"
CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --video_dir \"${VIDEO_DIR}\""
CMD="${CMD} --log_dir \"${LOG_DIR}\""
CMD="${CMD} --save_video_every ${SAVE_VIDEO_EVERY}"

# 可选参数
[[ -n "${TASK_IDS}" ]] && CMD="${CMD} --task_ids \"${TASK_IDS}\""
[[ -n "${CATEGORIES}" ]] && CMD="${CMD} --categories \"${CATEGORIES}\""
[[ -n "${DIFFICULTY_LEVELS}" ]] && CMD="${CMD} --difficulty_levels \"${DIFFICULTY_LEVELS}\""
[[ -n "${ENV_SEED}" ]] && CMD="${CMD} --env_seed ${ENV_SEED}"

if [[ "${FLIP_IMAGES}" == "true" ]]; then
    CMD="${CMD} --flip_images"
else
    CMD="${CMD} --no_flip_images"
fi

if [[ "${VIDEO_FLIP}" == "true" ]]; then
    CMD="${CMD} --video_flip"
else
    CMD="${CMD} --no_video_flip"
fi

if [[ "${LOWERCASE_INSTRUCTION}" == "true" ]]; then
    CMD="${CMD} --lowercase_instruction"
else
    CMD="${CMD} --no_lowercase_instruction"
fi

if [[ "${ADD_GENERATION_PROMPT}" == "true" ]]; then
    CMD="${CMD} --add_generation_prompt"
else
    CMD="${CMD} --no_generation_prompt"
fi

if [[ "${ADD_ACTION_PROMPT}" == "true" ]]; then
    CMD="${CMD} --add_action_prompt"
else
    CMD="${CMD} --no_action_prompt"
fi

if [[ "${EXECUTE_ALL_CHUNKS}" == "true" ]]; then
    CMD="${CMD} --execute_all_chunks"
else
    CMD="${CMD} --no_execute_all_chunks"
fi

[[ "${SAVE_VIDEOS}" == "true" ]] && CMD="${CMD} --save_videos"
[[ "${VERBOSE}" == "true" ]] && CMD="${CMD} --verbose"

if [[ "${PRINT_PER_TASK}" == "true" ]]; then
    CMD="${CMD} --print_per_task"
else
    CMD="${CMD} --no_print_per_task"
fi
CMD="${CMD} --summary_every ${SUMMARY_EVERY}"

if [[ "${RESUME}" == "true" ]]; then
    CMD="${CMD} --resume"
else
    CMD="${CMD} --no_resume"
fi

# ===== 把完整配置 dump 到日志 =====
OUTPUT_LOG="${VIDEO_DIR}/eval_output.log"
{
    echo "================================ EVAL CONFIG ================================"
    echo "Time         : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "EXPERIMENT   : ${EXPERIMENT_NAME}"
    echo "VIDEO_DIR    : ${VIDEO_DIR}"
    echo ""
    echo "CHECKPOINT_PATH      = ${CHECKPOINT_PATH}"
    echo "DATASET_PATH         = ${DATASET_PATH}"
    echo "VLM_TYPE             = ${VLM_TYPE}"
    echo "VLM_MODEL_PATH       = ${VLM_MODEL_PATH}"
    echo "VLM_LAYERS           = ${VLM_LAYERS}"
    echo "VLM_OUTPUT_DIM       = ${VLM_OUTPUT_DIM}"
    echo "CONTENT_ORDER        = ${CONTENT_ORDER}"
    echo "LOWERCASE_INSTRUCTION= ${LOWERCASE_INSTRUCTION}"
    echo "ADD_GENERATION_PROMPT= ${ADD_GENERATION_PROMPT}"
    echo "ADD_ACTION_PROMPT    = ${ADD_ACTION_PROMPT}"
    echo "TASK_SUITE_NAME      = ${TASK_SUITE_NAME}"
    echo "NUM_TRIALS_PER_TASK  = ${NUM_TRIALS_PER_TASK}"
    echo "TASK_IDS             = ${TASK_IDS}"
    echo "MAX_TASKS            = ${MAX_TASKS}"
    echo "MAX_STEPS            = ${MAX_STEPS}"
    echo "CATEGORIES           = ${CATEGORIES}"
    echo "DIFFICULTY_LEVELS    = ${DIFFICULTY_LEVELS}"
    echo "ACTION_CHUNK_SIZE    = ${ACTION_CHUNK_SIZE}"
    echo "EXECUTE_ALL_CHUNKS   = ${EXECUTE_ALL_CHUNKS}"
    echo "NUM_TRANSFORMER_BLOCKS=${NUM_TRANSFORMER_BLOCKS}"
    echo "NUM_ATTENTION_HEADS  = ${NUM_ATTENTION_HEADS}"
    echo "ACTION_HEAD_HIDDEN_DIM=${ACTION_HEAD_HIDDEN_DIM}"
    echo "NUM_ACTION_CHUNKS    = ${NUM_ACTION_CHUNKS}"
    echo "ACTION_DIM           = ${ACTION_DIM}"
    echo "GPU_ID               = ${GPU_ID}"
    echo "FLIP_IMAGES          = ${FLIP_IMAGES}"
    echo "SAVE_VIDEOS          = ${SAVE_VIDEOS}"
    echo "RESUME               = ${RESUME}"
    echo ""
    echo "CMD: ${CMD}"
    echo "============================================================================"
} > "${OUTPUT_LOG}"

echo ""
echo "执行命令:"
echo "${CMD}"
echo ""

# ===== 执行 =====
eval ${CMD} 2>&1 | tee -a "${OUTPUT_LOG}"

echo ""
echo "============================================================================"
echo "✅ 评估完成"
echo "  - 结果 JSON  : ${VIDEO_DIR}/eval_results_${TASK_SUITE_NAME}.json"
echo "  - 日志       : ${OUTPUT_LOG}"
echo "============================================================================"
