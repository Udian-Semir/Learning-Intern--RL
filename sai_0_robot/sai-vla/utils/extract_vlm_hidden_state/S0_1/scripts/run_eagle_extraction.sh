#!/bin/bash
# ============================================================================
# Eagle 2.5 VL Hidden States 提取脚本
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数
#   2. 运行: bash run_eagle_extraction.sh
#
# 注意事项:
# !    - DTYPE 参数 (bfloat16/float16/float32) 仅影响模型加载和推理精度，
#     可减少 GPU 显存占用并加速推理。
# !    - 保存的 .npy 文件始终为 float32 格式 (代码中SAVE_DTYPE参数强制转换)。
#     如需修改保存格式，请编辑 VLMs/S0_1/backbone/model_selector.py 中的:
#     arr = stacked[:, 0, :, :].cpu().float().numpy()
#
# ============================================================================

# ===== 配置参数 =====

# 数据集路径 (必填)
DATASET_PATH="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset197_transformed_add1roi_wusun/"

# 模型路径 (可选，默认使用 GR00T-N1.5-3B)
MODEL_PATH="/home/dev/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"

# 输出目录 (可选，默认: {DATASET_PATH}/vlm_hidden_states)
OUTPUT_DIR=""

# 提取的层号 (Eagle 使用负数索引)
# 例如: "-1" 或 "-4,-3,-2,-1"
LAYERS="-1"

# 图像视角键名 (根据数据集调整)
# 例如: "agentview,wrist" 或 "top,left_wrist" 或 "main"
IMAGE_KEYS="main,right_wrist"

# 设备
DEVICE="cuda:0"

# 模型推理数据类型
DTYPE="bfloat16"

# 保存 VLM hidden states 的数据类型
# float32: 保持 bfloat16 推理的完整精度 (bfloat16→float32→bfloat16 无损)（默认值）
# float16: 文件减半，但有精度损失 (bfloat16→float16 会损失精度)
SAVE_DTYPE="float32"

# 是否翻转IMAGE_KEYS中的图像 (true/false)
FLIP_IMAGES="false"

# 断点续传 (可选)
START_IDX=""
END_IDX=""

# 数据加载配置
NUM_WORKERS=4
PREFETCH_SIZE=8

# Prompt 配置
# 预设模板: "action", "simple", "detailed", "step_by_step"
# 或自定义模板: "Robot needs to: {instruction}"
PROMPT_TEMPLATE="simple"

# 内容顺序: "images_first", "text_first", "interleaved", "single_image"
CONTENT_ORDER="images_first"

# 是否将指令转为小写 (true/false)，只将任务描述（{instruction}）转为小写，模板部分保持不变
LOWERCASE_INSTRUCTION="true"

# 是否添加 generation prompt (true/false)
# 对于聊天模型，通常设为 true
ADD_GENERATION_PROMPT="true"

# 详细输出模式 (true/false)
VERBOSE="true"

# ===== 运行脚本 =====

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 构建命令
CMD="python -m utils.extract_vlm_hidden_state.S0_1.eagle.eagle_extract_vlm_hidden_states"
CMD="${CMD} --dataset_path ${DATASET_PATH}"
CMD="${CMD} --model_path ${MODEL_PATH}"
# 使用 = 连接，避免负数被误解为参数标志
CMD="${CMD} --layers=${LAYERS}"
CMD="${CMD} --image_keys ${IMAGE_KEYS}"
CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --dtype ${DTYPE}"
CMD="${CMD} --save_dtype ${SAVE_DTYPE}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --prefetch_size ${PREFETCH_SIZE}"
CMD="${CMD} --prompt_template ${PROMPT_TEMPLATE}"
CMD="${CMD} --content_order ${CONTENT_ORDER}"

# 可选参数
if [ -n "${OUTPUT_DIR}" ]; then
    CMD="${CMD} --output_dir ${OUTPUT_DIR}"
fi

if [ "${FLIP_IMAGES}" = "true" ]; then
    CMD="${CMD} --flip_images"
else
    CMD="${CMD} --no_flip_images"
fi

if [ -n "${START_IDX}" ]; then
    CMD="${CMD} --start_idx ${START_IDX}"
fi

if [ -n "${END_IDX}" ]; then
    CMD="${CMD} --end_idx ${END_IDX}"
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

# 打印配置
echo "============================================================================"
echo "🦅 Eagle 2.5 VL Hidden States 提取"
echo "============================================================================"
echo "数据集路径: ${DATASET_PATH}"
echo "模型路径: ${MODEL_PATH}"
echo "提取层: ${LAYERS}"
echo "图像视角: ${IMAGE_KEYS}"
echo "设备: ${DEVICE}"
echo "Prompt 模板: ${PROMPT_TEMPLATE}"
echo "内容顺序: ${CONTENT_ORDER}"
echo "小写指令: ${LOWERCASE_INSTRUCTION}"
echo "Generation Prompt: ${ADD_GENERATION_PROMPT}"
echo "详细输出: ${VERBOSE}"
echo "============================================================================"
echo ""
echo "执行命令:"
echo "${CMD}"
echo ""

# 执行
eval ${CMD}

