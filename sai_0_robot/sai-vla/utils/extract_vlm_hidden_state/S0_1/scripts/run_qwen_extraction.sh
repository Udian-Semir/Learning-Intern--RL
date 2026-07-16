#!/bin/bash
# ============================================================================
# Qwen3-VL Hidden States 提取脚本 (单卡 / 单数据集 / chunk-npz 输出)
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数
#   2. 运行: bash run_qwen_extraction.sh
#
# 保存格式 (由 SAVE_PER_EPISODE 控制):
#   - SAVE_PER_EPISODE="true"  (默认, 推荐):
#         {OUTPUT_DIR}/chunk-XXX.npz
#         每个 npz 内部 key 形如 "episode_XXXXXX" -> (num_frames, num_layers, seq_len, hidden_dim)
#         单数据集内一个 chunk 打包 chunks_size 个 episode (读取 meta/info.json),
#         inode 占用小, 与多 GPU 脚本输出格式一致,
#         训练端 utils/lerobot_dataset_loader.py 会自动识别 "chunk_npz"。
#   - SAVE_PER_EPISODE="false" (旧格式, 兼容):
#         {OUTPUT_DIR}/hidden_state_XXXXXX.npy
#         每帧一个文件, shape: (num_layers, seq_len, hidden_dim),
#         训练端会识别为 "per_frame"。
#
# 注意事项:
# !    - DTYPE 参数 (bfloat16/float16/float32) 仅影响模型加载和推理精度，
#     可减少 GPU 显存占用并加速推理。
# !    - SAVE_DTYPE 控制落盘精度 (float32 / float16), 与 DTYPE 独立。
# !    - 断点续跑: chunk-npz 模式会检查 chunk 内 episode_XXXXXX key 是否已存在,
#     已提取的 episode 会自动跳过 (同时兼容旧的 episode_XXXXXX.npy)。
#
# ============================================================================

# ===== 配置参数 =====

# 数据集路径 (必填)
DATASET_PATH="/data_disk2/hwl/datasets/dataset196/lerobot_dataset196_rightarm_state_joint_action_eefdelta_filter_by_velocity"

# 模型路径 (可选，默认使用 Qwen/Qwen3-VL-2B-Instruct)
# 可选模型:
#   - Qwen/Qwen3-VL-2B-Instruct (28层, hidden_dim=1536)
#   - Qwen/Qwen3-VL-4B-Instruct (36层, hidden_dim=2560)
#   - Qwen/Qwen3-VL-7B-Instruct (28层, hidden_dim=3584)
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"

# 输出目录 (可选，默认: {DATASET_PATH}/vlm_hidden_states)
OUTPUT_DIR=""

# 提取的层号 (2B: 1-28, 4B: 1-36, 7B: 1-28)
# 例如: "14" 或 "14,15,16"
# 推荐: 中间层或后几层
LAYERS="14"

# 图像视角键名 (可选)
# 留空时会从当前数据集自动检测 observation.images.* 键名
# 例如: "agentview,wrist" 或 "top,left_wrist"
IMAGE_KEYS=""

# 设备
DEVICE="cuda:0"

# 模型推理数据类型
DTYPE="bfloat16"

# 保存 VLM hidden states 的数据类型
# float32: 保持 bfloat16 推理的完整精度 (bfloat16→float32→bfloat16 无损)（默认值）
# float16: 文件减半，但有精度损失 (bfloat16→float16 会损失精度)
SAVE_DTYPE="float32"

# 保存模式 (true/false)
# true  : 按 episode 分组, 打包为 chunk-XXX.npz (默认, 推荐)
# false : 按帧保存 hidden_state_XXXXXX.npy (旧格式)
SAVE_PER_EPISODE="false"

# 是否翻转图像 (true/false)
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

# 是否将指令转为小写 (true/false)
LOWERCASE_INSTRUCTION="true"

# 是否添加 generation prompt (true/false)
# 对于聊天模型，通常设为 true
ADD_GENERATION_PROMPT="true"

# 详细输出模式 (true/false)
# 设为 true 会打印 token 信息
VERBOSE="true"

# ===== 运行脚本 =====

# 离线模式: 不再向 HuggingFace 发网络请求, 直接读本地缓存
# (已下载过模型后推荐开启, 避免 Network is unreachable 失败)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 可选: 显式指定 HF 缓存路径 (默认就是 ~/.cache/huggingface)
# export HF_HOME="${HOME}/.cache/huggingface"

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 构建命令
CMD="python -m utils.extract_vlm_hidden_state.S0_1.qwen.qwen_extract_vlm_hidden_states"
CMD="${CMD} --dataset_path ${DATASET_PATH}"
CMD="${CMD} --model_path ${MODEL_PATH}"
CMD="${CMD} --layers ${LAYERS}"
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

if [ -n "${IMAGE_KEYS}" ]; then
    CMD="${CMD} --image_keys ${IMAGE_KEYS}"
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

if [ "${SAVE_PER_EPISODE}" = "true" ]; then
    CMD="${CMD} --save_per_episode"
else
    CMD="${CMD} --no_save_per_episode"
fi

if [ "${VERBOSE}" = "true" ]; then
    CMD="${CMD} --verbose"
fi

# 打印配置
echo "============================================================================"
echo "🔮 Qwen3-VL Hidden States 提取"
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
echo "保存模式: $([ "${SAVE_PER_EPISODE}" = "true" ] && echo "per-episode chunk-npz" || echo "per-frame npy")"
echo "保存数据类型: ${SAVE_DTYPE}"
echo "详细输出: ${VERBOSE}"
echo "============================================================================"
echo ""
echo "执行命令:"
echo "${CMD}"
echo ""

# 执行
eval ${CMD}

