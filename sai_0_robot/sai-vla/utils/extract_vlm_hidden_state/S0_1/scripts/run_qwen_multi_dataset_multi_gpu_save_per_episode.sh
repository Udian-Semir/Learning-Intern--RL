#!/bin/bash
# ============================================================================
# Qwen3-VL 多数据集多 GPU Hidden States 提取脚本
# ============================================================================

#python utils/migrate_vlm_npy_to_chunk_npz.py \
#  --dataset_root /data_disk1/hwl/unitree_train_v2_recipe_lerobot --all

# ===== 配置参数 =====

# 多个 LeRobot 子数据集所在根目录
DATASET_ROOT="/data_disk1/hwl/unitree_train_v2_recipe_lerobot"

# 只处理指定数据集时填写，逗号分隔；留空则按 recipe.json 或目录自动枚举
DATASET_NAMES=""

# 统一输出根目录；留空则输出到各自数据集目录下的 vlm_hidden_states
OUTPUT_ROOT=""

# 使用哪些 GPU，每张卡会常驻加载一个完整模型
GPU_IDS="0,1,2,3,4,5,6,7"

# 模型路径
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"

# 提取层
LAYERS="14"

# 图像键名 (可选)
# 留空时每个数据集自动检测自己的 observation.images.* 键名
IMAGE_KEYS=""

# 模型推理数据类型
DTYPE="bfloat16"

# 保存 hidden states 的数据类型
SAVE_DTYPE="float32"

# 是否翻转图像 (true/false)
FLIP_IMAGES="false"

# 断点续传 (可选)
START_IDX=""
END_IDX=""

# 数据加载配置
NUM_WORKERS=4
PREFETCH_SIZE=8

# Prompt 配置
PROMPT_TEMPLATE="simple"
CONTENT_ORDER="images_first"
LOWERCASE_INSTRUCTION="true"
ADD_GENERATION_PROMPT="true"

# 详细输出模式 (true/false)
VERBOSE="true"

# 仅做数据集和 image keys 检查，不真正执行
DRY_RUN="false"

# 仅处理前 N 个数据集，调试时可用；留空表示全部
MAX_DATASETS=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# 日志文件路径，记录已完成/失败的数据集，支持断点续跑
LOG_FILE="${SCRIPT_DIR}/extraction_log.jsonl"

# ===== 运行脚本 =====

# 离线模式: 不再向 HuggingFace 发网络请求, 直接读本地缓存
# (已下载过模型后推荐开启, 避免 Network is unreachable 失败)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${PROJECT_ROOT}"

CMD="python -m utils.extract_vlm_hidden_state.S0_1.qwen.qwen_multi_dataset_multi_gpu_extraction"
CMD="${CMD} --dataset_root ${DATASET_ROOT}"
CMD="${CMD} --gpu_ids ${GPU_IDS}"
CMD="${CMD} --model_path ${MODEL_PATH}"
CMD="${CMD} --layers ${LAYERS}"
CMD="${CMD} --dtype ${DTYPE}"
CMD="${CMD} --save_dtype ${SAVE_DTYPE}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --prefetch_size ${PREFETCH_SIZE}"
CMD="${CMD} --prompt_template ${PROMPT_TEMPLATE}"
CMD="${CMD} --content_order ${CONTENT_ORDER}"

if [ -n "${DATASET_NAMES}" ]; then
    CMD="${CMD} --dataset_names ${DATASET_NAMES}"
fi

if [ -n "${OUTPUT_ROOT}" ]; then
    CMD="${CMD} --output_root ${OUTPUT_ROOT}"
fi

if [ -n "${IMAGE_KEYS}" ]; then
    CMD="${CMD} --image_keys ${IMAGE_KEYS}"
fi

if [ -n "${START_IDX}" ]; then
    CMD="${CMD} --start_idx ${START_IDX}"
fi

if [ -n "${END_IDX}" ]; then
    CMD="${CMD} --end_idx ${END_IDX}"
fi

if [ -n "${MAX_DATASETS}" ]; then
    CMD="${CMD} --max_datasets ${MAX_DATASETS}"
fi

if [ -n "${LOG_FILE}" ]; then
    CMD="${CMD} --log_file ${LOG_FILE}"
fi

if [ "${FLIP_IMAGES}" = "true" ]; then
    CMD="${CMD} --flip_images"
else
    CMD="${CMD} --no_flip_images"
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

if [ "${VERBOSE}" = "true" ]; then
    CMD="${CMD} --verbose"
fi

if [ "${DRY_RUN}" = "true" ]; then
    CMD="${CMD} --dry_run"
fi

echo "============================================================================"
echo "🔮 Qwen3-VL 多数据集多 GPU Hidden States 提取"
echo "============================================================================"
echo "数据集根目录: ${DATASET_ROOT}"
echo "GPU IDs: ${GPU_IDS}"
echo "模型路径: ${MODEL_PATH}"
echo "提取层: ${LAYERS}"
echo "图像视角: ${IMAGE_KEYS:-自动检测}"
echo "Prompt 模板: ${PROMPT_TEMPLATE}"
echo "内容顺序: ${CONTENT_ORDER}"
echo "详细输出: ${VERBOSE}"
echo "Dry Run: ${DRY_RUN}"
echo "日志文件: ${LOG_FILE}"
echo "============================================================================"
echo ""
echo "执行命令:"
echo "${CMD}"
echo ""

eval ${CMD}
