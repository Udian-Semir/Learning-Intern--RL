#!/bin/bash
# ============================================================================
# ParaCAT Action Head 遥操作数据评估脚本 - 累计误差绘图
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数
#   2. 运行: bash run_eval_cumulative_error.sh
#
# 功能:
#   - 加载指定 episode 的遥操作数据
#   - 实时通过 VLM backbone 提取 hidden states
#   - 使用 ParaCAT action head 预测动作 delta
#   - 绘制 GT 和预测的累计误差对比图
#
# ============================================================================

# ===== 基本配置 =====

# ParaCAT Checkpoint 路径 (必填)
PARACAT_CHECKPOINT="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset256_weight_eagle/p6000_bsz80*1*8_pons_q128_chunk25_tb2_8000steps_layer-1_20260130_wpflow_matching_0_eagle_libero_test_newest_mydataset256_USE_SHARED_CACHE_true_CACHE_VLM_STATES_false_USE_AMP_true_webdataset_false/checkpoints/step_8000/paracat.pt"

# Pons Checkpoint 路径 (可选，留空则不使用 Pons)
PONS_CHECKPOINT="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset256_weight_eagle/p6000_bsz80*1*8_pons_q128_chunk25_tb2_8000steps_layer-1_20260130_wpflow_matching_0_eagle_libero_test_newest_mydataset256_USE_SHARED_CACHE_true_CACHE_VLM_STATES_false_USE_AMP_true_webdataset_false/checkpoints/step_8000/pons.pt"

# State Mapper Checkpoint 路径 (可选，留空则从 checkpoint 目录自动查找)
# 如果训练时启用了 state_mapper，会自动从 config.json 检测
STATE_MAPPER_CHECKPOINT="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset256_weight_eagle/p6000_bsz80*1*8_pons_q128_chunk25_tb2_8000steps_layer-1_20260130_wpflow_matching_0_eagle_libero_test_newest_mydataset256_USE_SHARED_CACHE_true_CACHE_VLM_STATES_false_USE_AMP_true_webdataset_false/checkpoints/step_8000/state_mapper.pt"

# 数据集路径 (必填)
DATASET_PATH="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset256"

# 要评估的 Episode 索引
EPISODE_IDX=0

# ===== VLM 配置 =====

# VLM 类型
# 可选: "qwen3_vl", "eagle2_5_vl", "cosmos_reason_2b_vl"
VLM_TYPE="eagle2_5_vl"

# VLM 模型路径
# Qwen: "Qwen/Qwen3-VL-2B-Instruct" 或 "Qwen/Qwen3-VL-4B-Instruct"
# Eagle (GR00T-N1.5-3B): 见下方路径
VLM_MODEL_PATH="/home/dev/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e"

# VLM 提取层
# Qwen2B: "14", Qwen4B: "16", Eagle: "-1"
VLM_LAYERS="-1"

# VLM 输出维度
# Qwen2B: 2048, Qwen4B: 2560, Eagle: 2048
VLM_OUTPUT_DIM=2048

# ===== ParaCAT 模型配置 =====

# 动作块大小 (action chunking)
CHUNK_SIZE=25

# 动作维度 (遥操作数据)
# 示例: 14 = 6关节*2手臂 + 2手部
ACTION_DIM=14

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

# ===== 图像配置 =====

# 图像视角键名 (逗号分隔)
# 根据数据集中的视角名称配置
IMAGE_KEYS="main"

# 是否翻转图像 (true/false)
FLIP_IMAGES="false"

# ===== 离散化配置 =====

# 需要反离散化的列索引 (空格分隔)
# 示例: "0 1 2 3 4 5 6 7 8 9 10 11" 表示前 12 列需要反离散化
UNDISCRETE_COLUMNS="0 1 2 3 4 5 6 7 8 9 10 11"

# 对应列的 delta 值 (空格分隔)
# 示例: 关节角度的 delta 值
UNDISCRETE_DELTAS="6 6 6 2 2 2 6 6 6 2 2 2"

# Gripper 列索引 (空格分隔)
# Gripper 原始值已是 {-1, 0, 1}，不需要 delta 离散化
# 示例: 单 gripper "6"，双 gripper "12 13"
# 注意: UNDISCRETE_COLUMNS + GRIPPER_COLUMNS 应覆盖所有 action 列
GRIPPER_COLUMNS=""

# ===== State 预处理配置 =====
# 预处理执行顺序 (空格分隔，按顺序执行)
# 可选值: hand_binary, euler_to_axisangle
# 示例: "hand_binary euler_to_axisangle" 先手部二值化，再欧拉角转轴角
# 留空表示不进行预处理
STATE_PROCESS_ORDER="hand_binary"

# --- 手部二值化配置 (hand_binary) ---
# 手部数据在【原始 state】中的列索引范围 (每组: 起始索引 结束索引，左闭右开)
# 支持多组手部数据，按顺序处理
# 示例: 单手 "0 6"，双手 "6 12 18 24" (左手6-12，右手18-24)
# 每组6维->1维，多组按顺序处理时自动计算索引偏移
HAND_BINARY_COLUMNS="12 18 18 24"
# 二值化阈值 (平均值 > threshold -> 1, 否则 -> -1)
HAND_BINARY_THRESHOLD=442

# --- 欧拉角转轴角配置 (euler_to_axisangle) ---
# 欧拉角在【原始 state】中的列索引 (每组3列，可配置多组)
# 支持多组欧拉角 (如左臂、右臂)，每3列为一组
# 示例: 单臂 "9 10 11"，双臂 "13 14 15 16 17 18"
# 注意: 索引基于原始 state，处理时会自动根据之前的处理调整偏移
STATE_EULER_TO_AXISANGLE_COLUMNS=""

# State 维度 (设置为【处理后】的维度)
STATE_DIM=14

# ===== State Mapper 配置 (可选) =====
# 注意: State Mapper 会从 checkpoint 目录的 config.json 自动检测是否启用
# 如果训练时使用了 state_mapper，评估时也需要提供正确的 state 预处理配置

# 是否强制启用 State Mapper (true/false)
# 通常不需要设置，程序会从 config.json 自动检测
ENABLE_STATE_MAPPER="true"

# State Mapper 使用 minmax 归一化的列索引 (空格分隔)
# 与训练时的配置保持一致
# 示例: "0 1 2 3 4 5 6 7 8 9 10 11" (关节位置)
STATE_NORM_COLUMNS_MINMAX="0 1 2 3 4 5 6 7 8 9 10 11"

# State Mapper 使用 axisangle 归一化的列索引 (空格分隔)
# 对于已转换为轴角的列 (范围 [-pi, pi])
# 示例: "12 13" (手部开合)
STATE_NORM_COLUMNS_AXISANGLE=""

# ===== Prompt 配置 =====

# 内容顺序
# 可选: "images_first", "text_first", "interleaved", "single_image"
CONTENT_ORDER="images_first"

# 是否将指令转为小写 (true/false)
LOWERCASE_INSTRUCTION="true"

# 是否添加 action prompt (true/false)
ADD_ACTION_PROMPT="false"

# ===== 系统配置 =====

# GPU 设备 ID
GPU_ID=0

# PyTorch 设备
DEVICE="cuda:0"

# 输出目录
OUTPUT_DIR="./eval_plots/tele_paracat_${VLM_TYPE}_episode_${EPISODE_IDX}_chunk_${CHUNK_SIZE}_$(date +%Y%m%d_%H%M%S)"

# 详细输出 (true/false)
VERBOSE="true"

# ===== 运行脚本 =====

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# 切换到项目根目录
cd "${PROJECT_ROOT}"

# 打印配置
echo "============================================================================"
echo "🚀 ParaCAT 遥操作数据评估 - 累计误差绘图"
echo "============================================================================"
echo ""
echo "📁 路径配置:"
echo "  ParaCAT Checkpoint: ${PARACAT_CHECKPOINT}"
echo "  Pons Checkpoint: ${PONS_CHECKPOINT:-未使用}"
echo "  Dataset: ${DATASET_PATH}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo ""
echo "🔮 VLM 配置:"
echo "  Type: ${VLM_TYPE}"
echo "  Model: ${VLM_MODEL_PATH}"
echo "  Layers: ${VLM_LAYERS}"
echo "  Output Dim: ${VLM_OUTPUT_DIM}"
echo ""
echo "🎯 ParaCAT 配置:"
echo "  Chunk Size: ${CHUNK_SIZE}"
echo "  Action Dim: ${ACTION_DIM}"
echo "  Transformer Blocks: ${NUM_TRANSFORMER_BLOCKS}"
echo "  MLP Layers: ${NUM_MLP_LAYERS}"
echo ""
echo "🔗 Pons 配置:"
echo "  使用 Pons: ${PONS_CHECKPOINT:+是}${PONS_CHECKPOINT:-否}"
echo "  Q Seq Len: ${PONS_Q_SEQ_LEN}"
echo "  Num Blocks: ${PONS_NUM_BLOCKS}"
echo ""
echo "📝 评估配置:"
echo "  Episode Index: ${EPISODE_IDX}"
echo "  Image Keys: ${IMAGE_KEYS}"
echo "  Flip Images: ${FLIP_IMAGES}"
echo ""
echo "🔢 离散化配置:"
echo "  Undiscrete Columns: ${UNDISCRETE_COLUMNS:-未设置}"
echo "  Undiscrete Deltas: ${UNDISCRETE_DELTAS:-未设置}"
echo "  Gripper Columns: ${GRIPPER_COLUMNS:-未设置}"
echo ""
echo "🔧 State 预处理配置:"
echo "  Process Order: ${STATE_PROCESS_ORDER:-未设置}"
echo "  Hand Binary Columns: ${HAND_BINARY_COLUMNS:-未设置}"
echo "  Hand Binary Threshold: ${HAND_BINARY_THRESHOLD}"
echo "  Euler to Axisangle Columns: ${STATE_EULER_TO_AXISANGLE_COLUMNS:-未设置}"
echo "  State Dim: ${STATE_DIM}"
echo ""
echo "📊 State Mapper 配置:"
echo "  State Mapper Checkpoint: ${STATE_MAPPER_CHECKPOINT:-自动检测}"
echo "  强制启用: ${ENABLE_STATE_MAPPER}"
echo "  MinMax Columns: ${STATE_NORM_COLUMNS_MINMAX:-未设置}"
echo "  AxisAngle Columns: ${STATE_NORM_COLUMNS_AXISANGLE:-未设置}"
echo ""
echo "============================================================================"

# 构建命令
CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} python -m eval.Sai0_1.tele.ParaCAT.eval_cumulative_error"
CMD="${CMD} --paracat_checkpoint ${PARACAT_CHECKPOINT}"
CMD="${CMD} --dataset_path ${DATASET_PATH}"
CMD="${CMD} --episode_idx ${EPISODE_IDX}"
CMD="${CMD} --vlm_model_path ${VLM_MODEL_PATH}"
CMD="${CMD} --vlm_type ${VLM_TYPE}"
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

# 图像参数
CMD="${CMD} --image_keys ${IMAGE_KEYS}"

# 离散化参数 (可选)
if [ -n "${UNDISCRETE_COLUMNS}" ]; then
    CMD="${CMD} --undiscrete_columns ${UNDISCRETE_COLUMNS}"
fi
if [ -n "${UNDISCRETE_DELTAS}" ]; then
    CMD="${CMD} --undiscrete_deltas ${UNDISCRETE_DELTAS}"
fi
if [ -n "${GRIPPER_COLUMNS}" ]; then
    CMD="${CMD} --gripper_columns ${GRIPPER_COLUMNS}"
fi

# State 预处理参数 (可选)
if [ -n "${STATE_PROCESS_ORDER}" ]; then
    CMD="${CMD} --state_process_order ${STATE_PROCESS_ORDER}"
fi
if [ -n "${HAND_BINARY_COLUMNS}" ]; then
    CMD="${CMD} --hand_binary_columns ${HAND_BINARY_COLUMNS}"
fi
CMD="${CMD} --hand_binary_threshold ${HAND_BINARY_THRESHOLD}"
if [ -n "${STATE_EULER_TO_AXISANGLE_COLUMNS}" ]; then
    CMD="${CMD} --state_euler_to_axisangle_columns ${STATE_EULER_TO_AXISANGLE_COLUMNS}"
fi
CMD="${CMD} --state_dim ${STATE_DIM}"

# State Mapper 参数 (可选)
if [ -n "${STATE_MAPPER_CHECKPOINT}" ]; then
    CMD="${CMD} --state_mapper_checkpoint ${STATE_MAPPER_CHECKPOINT}"
fi
if [ "${ENABLE_STATE_MAPPER}" = "true" ]; then
    CMD="${CMD} --enable_state_mapper"
fi
if [ -n "${STATE_NORM_COLUMNS_MINMAX}" ]; then
    CMD="${CMD} --state_norm_columns_minmax ${STATE_NORM_COLUMNS_MINMAX}"
fi
if [ -n "${STATE_NORM_COLUMNS_AXISANGLE}" ]; then
    CMD="${CMD} --state_norm_columns_axisangle ${STATE_NORM_COLUMNS_AXISANGLE}"
fi

# Prompt 参数
CMD="${CMD} --content_order ${CONTENT_ORDER}"

# 系统参数
CMD="${CMD} --device ${DEVICE}"
CMD="${CMD} --output_dir ${OUTPUT_DIR}"

# 可选参数 - 图像翻转
if [ "${FLIP_IMAGES}" = "true" ]; then
    CMD="${CMD} --flip_images"
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

# 可选参数 - 详细输出
if [ "${VERBOSE}" = "true" ]; then
    CMD="${CMD} --verbose"
fi

echo ""
echo "执行命令:"
echo "${CMD}"
echo ""

# 执行
eval ${CMD}

echo ""
echo "============================================================================"
echo "📊 评估完成!"
echo "   输出目录: ${OUTPUT_DIR}"
echo "============================================================================"
