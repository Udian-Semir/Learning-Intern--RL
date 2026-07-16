#!/bin/bash

# ============================================================================
# 训练前清理残留的共享内存（防止之前异常退出导致的内存泄漏）
# ============================================================================
echo "🧹 清理残留的共享内存..."

echo "  清理前 /dev/shm 使用情况:"
df -h /dev/shm | tail -1

SHM_COUNT=$(ls /dev/shm/psm_* 2>/dev/null | wc -l)
if [ "$SHM_COUNT" -gt 0 ]; then
    echo "  发现 ${SHM_COUNT} 个残留的共享内存文件，正在清理..."
    rm -f /dev/shm/psm_*
    echo "  ✓ 清理完成"
else
    echo "  ✓ 没有发现残留的共享内存"
fi

echo "  清理后 /dev/shm 使用情况:"
df -h /dev/shm | tail -1
echo ""

# ============================================================================

START_TIME=$(date +%s)
START_TIME_FORMATTED=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================================
# Pons + ParaCAT 联合训练脚本
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数（数据集路径、GPU、训练参数等）
#   2. 运行: bash train_pons_paracat.sh
#
# 支持:
#   - 多GPU分布式训练 (使用 torchrun)
#   - Pons 和 ParaCAT 端到端联合训练
#   - 支持 action 离散化 (每列不同 delta)
#   - 可选加载预训练权重
#   - 混合精度训练 (AMP)
#
# 数据流:
#   VLM Hidden States -> Pons -> ParaCAT -> Actions
#
# ============================================================================

# ===== 基本配置 =====

# 数据集路径 (必填)
DATA_PATH="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/dataset73_256_filter"

# GPU 设备 ID
GPU_IDS="0,1,2,3,4,5,6,7"

# GPU 数量
NUM_GPUS=8

# Master 端口
MASTER_PORT=29512

# ===== 训练超参数 =====

BATCH_SIZE=128
NUM_WORKERS=8
PREFETCH_FACTOR=4
STEPS=40000
LR=1e-4
PONS_LR_SCALE=1.0  # Pons 学习率缩放因子 (pons_lr = lr * scale)
WEIGHT_DECAY=1e-5
WARMUP_RATIO=0.05

# ===== ParaCAT 模型参数 =====

# Action chunk 大小
CHUNK_SIZE=50

# Action 维度 (会自动从数据集检测)
ACTION_DIM=14

# Transformer 块数量 (self-attention)
NUM_TRANSFORMER_BLOCKS=2

# MLP 层数量
NUM_MLP_LAYERS=2

# MLP 中间维度
MLP_EXPAND_DIM=1024

# 注意力头数量
NUM_HEADS=8

# ===== Pons 模型参数 =====

# Query 序列长度
PONS_Q_SEQ_LEN=128

# Cross-Attention 块数量
PONS_NUM_BLOCKS=2

# Pons 注意力头数量
PONS_NUM_HEADS=8

# Pons Dropout
PONS_DROPOUT=0.1

# ===== VLM 参数 =====

NUM_VLM_LAYERS=1
VLM_OUTPUT_DIM=2048

# ===== 离散化参数 =====

# 是否启用离散化
DISCRETE_ACTIONS="true"

# 离散化方法选择
# - "constrain_delta": 简单累积误差方法 (默认)
# - "chunk_calculus": 基于微积分的方法，带趋势预测
DISCRETE_METHOD="chunk_calculus"

# 离散化列索引 (空格分隔，例如: "0 1 2 3 4 5")
# 这些列会经过离散化函数处理
DISCRETE_COLUMNS="0 1 2 3 4 5 6 7 8 9 10 11"

# 对应列的 delta 值 (空格分隔，例如: "0.01 0.01 0.01 0.02 0.02 0.02")
DISCRETE_DELTAS="2 2 2 0.5 0.5 0.5 2 2 2 0.5 0.5 0.5"

# 是否启用反离散化配置 (用于推理)
UNDISCRETE_ACTIONS="false"
UNDISCRETE_COLUMNS=""
UNDISCRETE_DELTAS=""

# ===== Gripper 列参数 (LIBERO 专用) =====
# Gripper 列索引 (空格分隔，与 DISCRETE_COLUMNS 格式一致)
# 示例:
#   单个 gripper: GRIPPER_COLUMNS="6"
#   多个 gripper: GRIPPER_COLUMNS="6 7"
#
# 注意:
#   1. len(DISCRETE_COLUMNS) + len(GRIPPER_COLUMNS) 必须等于 ACTION_DIM
#   2. LIBERO 数据集的 gripper 原始值已是 {-1, 0, 1}，无需 delta 离散化
#   3. 训练时只需 +1 转为类别索引 {0, 1, 2}
GRIPPER_COLUMNS="12 13"

# Gripper 处理方法 (函数名) # ! 作为预留，目前没有被调用
# 默认使用 libero_gripper_to_class_idx (在 utils/discrete.py 中定义)
GRIPPER_METHOD="libero_gripper_to_class_idx"

# ===== 预训练权重 (可选) =====

# 预训练 Pons checkpoint 路径 (留空则从头训练)
PONS_CHECKPOINT=""

# 预训练 ParaCAT checkpoint 路径 (留空则从头训练)
PARACAT_CHECKPOINT=""

# ===== 保存配置 =====

GPU_NAME="p6000_202623"
WHICH_LAYER="-1" # 只在文件命名起作用，不影响实际训练
MODEL_NAME="pons_paracat"
DATASET_NAME="mydataset256" # libero_spatial

SAVE_EVERY_STEPS=4000

# ===== 系统配置 =====

GRADIENT_ACCUMULATION_STEPS=1
USE_AMP="true"
AMP_DTYPE="bfloat16"

# ===== 数据加载优化 =====

VLM_DTYPE="bfloat16"

# VLM 缓存策略选择:
#   1. USE_WEBDATASET="true"    : WebDataset 分片模式（超大规模数据，推荐）
#   2. USE_SHARED_CACHE="true"  : 共享内存缓存（推荐，多 GPU 共享，速度最快）
#   3. CACHE_VLM_STATES="true"  : mmap 缓存（备选，自动共享）
#   4. 都为 false: 无缓存（最慢，每次从磁盘读取）

# ===== WebDataset 配置 (大规模数据训练优化) =====
# 使用 WebDataset tar 分片格式可以显著减少大量小文件的 I/O 开销
# 
# 转换命令 (在项目根目录执行):
#   python -m utils.webdataset_utils convert \
#       --input_path /data/.../libero_lerobot_spatial_sys1_qwen_2b_14 \
#       --samples_per_shard 5000 \
#       --num_action_chunks 25
#
USE_WEBDATASET="false"

# WebDataset 分片路径模式
WEBDATASET_SHARD_PATTERN="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_spatial/libero_lerobot_spatial_sys1_qwen_2b_14/webdataset_shard5000_ac16/shard-{000000..000010}.tar"

# WebDataset shuffle 缓冲区大小
WEBDATASET_SHUFFLE_BUFFER=1000

# ===== WebDataset 分批缓存模式 =====
USE_WEBDATASET_CACHED="false"

# 每批缓存的分片数量
WEBDATASET_CACHE_SHARDS=8

# 分批缓存的数据类型
WEBDATASET_CACHE_DTYPE="float32"

USE_SHARED_CACHE="true"

# 共享内存缓存的数据类型
CACHE_DTYPE="float32"

CACHE_VLM_STATES="false"
SKIP_IMAGES="true"

# ===== State 四元数转轴角配置 ===== # ! 用于libero数据集
# 是否将 9 维 state (四元数: gripper1, gripper2, x, y, z, qx, qy, qz, qw) 
# 转换为 8 维 state (轴角: gripper1, gripper2, x, y, z, ax, ay, az)
#
# 使用方法:
#   "true"  - 启用转换 (9维 -> 8维，适用于四元数格式的数据集)
#   "false" - 禁用转换 (保持原始维度，适用于已经是轴角格式的数据集)
CONVERT_QUAT_TO_AXISANGLE="false"

# ===== Observation State 映射器配置 =====
# 将 observation_state 归一化后通过 MLP 映射到 VLM 隐藏空间，
# 然后拼接到 Pons 输出末尾，作为额外的条件信息参与 ParaCAT 训练
#
# 支持两种归一化方式:
#   1. minmax: 从 stats.json 读取 observation.state 的 min/max 进行归一化
#   2. axisangle: 欧拉角 -> 四元数 -> 轴角，然后用 [-pi, pi] 归一化
#
# 使用示例 (LIBERO 数据集, state 维度 8):
#   - 列 0,1: gripper 状态 (minmax)
#   - 列 2,3,4: xyz 位置 (minmax)
#   - 列 5,6,7: 欧拉角旋转 (axisangle)

# 是否启用 state 映射器
ENABLE_STATE_MAPPER="true"

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
# 示例: 单臂 "9 10 11"，双臂 "13 14 15 16 17 18" (左臂13-15，右臂16-18)
# 注意: 索引基于原始 state，处理时会自动根据之前的处理调整偏移
STATE_EULER_TO_AXISANGLE_COLUMNS=""

# State 维度 (设置为【处理后】的维度)
STATE_DIM=14

# ! 这里要注意修改，parquet改成6+6+1+1方便索引stat对应索引
# 使用 minmax 归一化的列索引 (空格分隔)
# 从 stats.json 读取 observation.state 的 min/max
# 示例: gripper (0,1) + xyz (2,3,4) = "0 1 2 3 4"
STATE_NORM_COLUMNS_MINMAX="0 1 2 3 4 5 6 7 8 9 10 11"

# 使用 axisangle 归一化的列索引 (对已转换为轴角的列做 [-pi, pi] -> [0,1] 归一化)
# 示例: 旋转列 (5,6,7) = "5 6 7"
STATE_NORM_COLUMNS_AXISANGLE=""

# ===== Weights & Biases 配置 =====

USE_WANDB="true"
WANDB_PROJECT="flow_matching_0_eagle_libero_test_newest"

# Checkpoint 输出目录
OUT_DIR="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset256_weight_eagle/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_pons_q${PONS_Q_SEQ_LEN}_chunk${CHUNK_SIZE}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_layer${WHICH_LAYER}_$(date +%Y%m%d)_wp${WANDB_PROJECT}_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}/checkpoints"

# 日志输出目录
LOG_DIR="/home/dev/文档/huangwenlong/vla-data-pipeline/lerobot_output/my_dataset256_weight_eagle/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_pons_q${PONS_Q_SEQ_LEN}_chunk${CHUNK_SIZE}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_layer${WHICH_LAYER}_$(date +%Y%m%d)_wp${WANDB_PROJECT}_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}/logs"

# Wandb 运行名称
WANDB_RUN_NAME="${MODEL_NAME}_layer${WHICH_LAYER}_${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_pons_q${PONS_Q_SEQ_LEN}_chunk${CHUNK_SIZE}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_paracat_$(date +%Y%m%d)_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}"

WANDB_LOG_FREQ=1

# ===== 分布式训练环境变量 =====

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400
export NCCL_TIMEOUT=14400000
export TORCH_NCCL_BLOCKING_WAIT=0
export NCCL_BLOCKING_WAIT=0

export TORCH_DISTRIBUTED_TIMEOUT_SEC=14400
export TORCH_CPP_LOG_LEVEL=INFO

export TORCH_NCCL_TRACE_BUFFER_SIZE=1000
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

export TORCH_NCCL_ENABLE_MONITORING=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4

# ===== 构建训练命令 =====

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TRAIN_SCRIPT="${PROJECT_ROOT}/Action_Heads/ParaCAT/train_multigpu_pons_paracat.py"

CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} --rdzv_backend=c10d --rdzv_conf timeout=7200 ${TRAIN_SCRIPT}"

# 数据相关参数
CMD="${CMD} --data_path ${DATA_PATH}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --prefetch_factor ${PREFETCH_FACTOR}"

# 训练超参数
CMD="${CMD} --steps ${STEPS}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --pons_lr_scale ${PONS_LR_SCALE}"
CMD="${CMD} --weight_decay ${WEIGHT_DECAY}"
CMD="${CMD} --warmup_ratio ${WARMUP_RATIO}"

# ParaCAT 模型参数
CMD="${CMD} --chunk_size ${CHUNK_SIZE}"
CMD="${CMD} --action_dim ${ACTION_DIM}"
CMD="${CMD} --num_transformer_blocks ${NUM_TRANSFORMER_BLOCKS}"
CMD="${CMD} --num_mlp_layers ${NUM_MLP_LAYERS}"
CMD="${CMD} --mlp_expand_dim ${MLP_EXPAND_DIM}"
CMD="${CMD} --num_heads ${NUM_HEADS}"

# Pons 模型参数
CMD="${CMD} --pons_q_seq_len ${PONS_Q_SEQ_LEN}"
CMD="${CMD} --pons_num_blocks ${PONS_NUM_BLOCKS}"
CMD="${CMD} --pons_num_heads ${PONS_NUM_HEADS}"
CMD="${CMD} --pons_dropout ${PONS_DROPOUT}"

# VLM 参数
CMD="${CMD} --num_vlm_layers ${NUM_VLM_LAYERS}"
CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"

# 离散化参数
if [ "${DISCRETE_ACTIONS}" = "true" ]; then
    CMD="${CMD} --discrete_actions"
    if [ -n "${DISCRETE_COLUMNS}" ]; then
        CMD="${CMD} --discrete_columns ${DISCRETE_COLUMNS}"
    fi
    if [ -n "${DISCRETE_DELTAS}" ]; then
        CMD="${CMD} --discrete_deltas ${DISCRETE_DELTAS}"
    fi
    if [ -n "${DISCRETE_METHOD}" ]; then
        CMD="${CMD} --discrete_method ${DISCRETE_METHOD}"
    fi
fi

if [ "${UNDISCRETE_ACTIONS}" = "true" ]; then
    CMD="${CMD} --undiscrete_actions"
    if [ -n "${UNDISCRETE_COLUMNS}" ]; then
        CMD="${CMD} --undiscrete_columns ${UNDISCRETE_COLUMNS}"
    fi
    if [ -n "${UNDISCRETE_DELTAS}" ]; then
        CMD="${CMD} --undiscrete_deltas ${UNDISCRETE_DELTAS}"
    fi
fi

# Gripper 参数 (LIBERO 专用)
if [ -n "${GRIPPER_COLUMNS}" ]; then
    CMD="${CMD} --gripper_columns ${GRIPPER_COLUMNS}"
fi
if [ -n "${GRIPPER_METHOD}" ]; then
    CMD="${CMD} --gripper_method ${GRIPPER_METHOD}"
fi

# 预训练权重
if [ -n "${PONS_CHECKPOINT}" ]; then
    CMD="${CMD} --pons_checkpoint ${PONS_CHECKPOINT}"
fi

if [ -n "${PARACAT_CHECKPOINT}" ]; then
    CMD="${CMD} --paracat_checkpoint ${PARACAT_CHECKPOINT}"
fi

# 保存配置
CMD="${CMD} --out_dir ${OUT_DIR}"
CMD="${CMD} --log_dir ${LOG_DIR}"
CMD="${CMD} --save_every_steps ${SAVE_EVERY_STEPS}"

# 系统配置
CMD="${CMD} --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS}"

# 混合精度训练
if [ "${USE_AMP}" = "true" ]; then
    CMD="${CMD} --use_amp"
    CMD="${CMD} --amp_dtype ${AMP_DTYPE}"
else
    CMD="${CMD} --no_amp"
fi

# 数据加载优化
CMD="${CMD} --vlm_dtype ${VLM_DTYPE}"

# 数据加载模式优先级: WebDataset 分批缓存 > WebDataset > 共享内存缓存 > mmap 模式
if [ "${USE_WEBDATASET_CACHED}" = "true" ] && [ -n "${WEBDATASET_SHARD_PATTERN}" ]; then
    # WebDataset 分批缓存模式 (优先级最高)
    CMD="${CMD} --use_webdataset_cached"
    CMD="${CMD} --webdataset_shard_pattern \"${WEBDATASET_SHARD_PATTERN}\""
    CMD="${CMD} --webdataset_cache_shards ${WEBDATASET_CACHE_SHARDS}"
    CMD="${CMD} --webdataset_cache_dtype ${WEBDATASET_CACHE_DTYPE}"
elif [ "${USE_WEBDATASET}" = "true" ] && [ -n "${WEBDATASET_SHARD_PATTERN}" ]; then
    # WebDataset 模式 (次优先级)
    CMD="${CMD} --use_webdataset"
    CMD="${CMD} --webdataset_shard_pattern \"${WEBDATASET_SHARD_PATTERN}\""
    CMD="${CMD} --webdataset_shuffle_buffer ${WEBDATASET_SHUFFLE_BUFFER}"
# 共享内存缓存 (第三优先级，自动检测形状)
elif [ "${USE_SHARED_CACHE}" = "true" ]; then
    CMD="${CMD} --use_shared_cache"
    CMD="${CMD} --cache_dtype ${CACHE_DTYPE}"
elif [ "${CACHE_VLM_STATES}" = "true" ]; then
    CMD="${CMD} --cache_vlm_states"
fi

if [ "${SKIP_IMAGES}" = "true" ]; then
    CMD="${CMD} --skip_images"
fi

# 四元数转轴角
if [ "${CONVERT_QUAT_TO_AXISANGLE}" = "true" ]; then
    CMD="${CMD} --convert_quat_to_axisangle"
else
    CMD="${CMD} --no_convert_quat_to_axisangle"
fi

# State Mapper 配置
if [ "${ENABLE_STATE_MAPPER}" = "true" ]; then
    CMD="${CMD} --enable_state_mapper"
    CMD="${CMD} --state_dim ${STATE_DIM}"
    if [ -n "${STATE_NORM_COLUMNS_MINMAX}" ]; then
        CMD="${CMD} --state_norm_columns_minmax ${STATE_NORM_COLUMNS_MINMAX}"
    fi
    if [ -n "${STATE_NORM_COLUMNS_AXISANGLE}" ]; then
        CMD="${CMD} --state_norm_columns_axisangle ${STATE_NORM_COLUMNS_AXISANGLE}"
    fi
    # State 预处理参数
    if [ -n "${STATE_PROCESS_ORDER}" ]; then
        CMD="${CMD} --state_process_order ${STATE_PROCESS_ORDER}"
    fi
    if [ -n "${HAND_BINARY_COLUMNS}" ]; then
        CMD="${CMD} --hand_binary_columns ${HAND_BINARY_COLUMNS}"
        CMD="${CMD} --hand_binary_threshold ${HAND_BINARY_THRESHOLD}"
    fi
    if [ -n "${STATE_EULER_TO_AXISANGLE_COLUMNS}" ]; then
        CMD="${CMD} --state_euler_to_axisangle_columns ${STATE_EULER_TO_AXISANGLE_COLUMNS}"
    fi
fi

# Wandb 配置
if [ "${USE_WANDB}" = "true" ]; then
    CMD="${CMD} --use_wandb"
    CMD="${CMD} --wandb_project ${WANDB_PROJECT}"
    if [ -n "${WANDB_RUN_NAME}" ]; then
        CMD="${CMD} --wandb_run_name ${WANDB_RUN_NAME}"
    fi
    CMD="${CMD} --wandb_log_freq ${WANDB_LOG_FREQ}"
else
    CMD="${CMD} --no_wandb"
fi

# ===== 打印配置信息 =====

echo "============================================================================"
echo "🔗🐱 Pons + ParaCAT 联合训练"
echo "============================================================================"
echo ""
echo "📁 路径配置:"
echo "  数据集: ${DATA_PATH}"
echo "  输出目录: ${OUT_DIR}"
echo "  日志目录: ${LOG_DIR}"
echo ""
echo "💻 系统配置:"
echo "  GPU: ${GPU_IDS} (${NUM_GPUS} GPUs)"
echo "  Master Port: ${MASTER_PORT}"
echo "  有效 Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))"
echo ""
echo "🔧 训练参数:"
echo "  Batch Size (per GPU): ${BATCH_SIZE}"
echo "  Max Steps: ${STEPS}"
echo "  Learning Rate: ${LR}"
echo "  Pons LR Scale: ${PONS_LR_SCALE}"
echo "  Warmup Ratio: ${WARMUP_RATIO}"
echo "  Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo ""
echo "🔗 Pons 模型配置:"
echo "  Query Seq Len: ${PONS_Q_SEQ_LEN}"
echo "  Num Blocks: ${PONS_NUM_BLOCKS}"
echo "  Num Heads: ${PONS_NUM_HEADS}"
echo "  Dropout: ${PONS_DROPOUT}"
echo ""
echo "🐱 ParaCAT 模型配置:"
echo "  Chunk Size: ${CHUNK_SIZE}"
echo "  Action Dim: ${ACTION_DIM}"
echo "  Transformer Blocks: ${NUM_TRANSFORMER_BLOCKS}"
echo "  MLP Layers: ${NUM_MLP_LAYERS}"
echo "  MLP Expand Dim: ${MLP_EXPAND_DIM}"
echo "  Num Heads: ${NUM_HEADS}"
echo ""
echo "📊 离散化配置:"
echo "  Discrete Actions: ${DISCRETE_ACTIONS}"
if [ "${DISCRETE_ACTIONS}" = "true" ]; then
    echo "  Discrete Method: ${DISCRETE_METHOD}"
    echo "  Discrete Columns: ${DISCRETE_COLUMNS}"
    echo "  Discrete Deltas: ${DISCRETE_DELTAS}"
    echo "  Gripper Columns: ${GRIPPER_COLUMNS:-未设置}"
    echo "  Gripper Method: ${GRIPPER_METHOD:-未设置}"
fi
echo ""
if [ -n "${PONS_CHECKPOINT}" ] || [ -n "${PARACAT_CHECKPOINT}" ]; then
    echo "🎯 预训练权重:"
    if [ -n "${PONS_CHECKPOINT}" ]; then
        echo "  Pons: ${PONS_CHECKPOINT}"
    fi
    if [ -n "${PARACAT_CHECKPOINT}" ]; then
        echo "  ParaCAT: ${PARACAT_CHECKPOINT}"
    fi
    echo ""
fi
echo "💾 保存配置:"
echo "  Save Every Steps: ${SAVE_EVERY_STEPS}"
echo ""
echo "⚡ 性能配置:"
echo "  AMP: ${USE_AMP} (${AMP_DTYPE})"
echo "  VLM Dtype: ${VLM_DTYPE}"
echo "  Skip Images: ${SKIP_IMAGES}"
echo "  四元数转轴角: ${CONVERT_QUAT_TO_AXISANGLE}"
echo ""
if [ "${ENABLE_STATE_MAPPER}" = "true" ]; then
    echo "🔄 State Mapper 配置:"
    echo "  Enable State Mapper: ${ENABLE_STATE_MAPPER}"
    echo "  State Dim: ${STATE_DIM}"
    echo "  MinMax Norm Columns: ${STATE_NORM_COLUMNS_MINMAX}"
    echo "  AxisAngle Norm Columns: ${STATE_NORM_COLUMNS_AXISANGLE}"
    if [ -n "${STATE_PROCESS_ORDER}" ]; then
        echo "  State Preprocessing Order: ${STATE_PROCESS_ORDER}"
        if [ -n "${HAND_BINARY_COLUMNS}" ]; then
            echo "    hand_binary: columns=${HAND_BINARY_COLUMNS}, threshold=${HAND_BINARY_THRESHOLD}"
        fi
        if [ -n "${STATE_EULER_TO_AXISANGLE_COLUMNS}" ]; then
            echo "    euler_to_axisangle: columns=${STATE_EULER_TO_AXISANGLE_COLUMNS}"
        fi
    fi
    echo ""
fi
if [ "${USE_WANDB}" = "true" ]; then
    echo "📊 Wandb 配置:"
    echo "  Project: ${WANDB_PROJECT}"
    echo "  Run Name: ${WANDB_RUN_NAME}"
    echo ""
fi
echo "============================================================================"
echo ""
echo "执行命令:"
echo "${CMD}"
echo ""
echo "============================================================================"

# ===== 执行训练 =====

echo "⏱️ 训练开始时间: ${START_TIME_FORMATTED}"
echo ""

eval ${CMD}

# 记录结束时间
END_TIME=$(date +%s)
END_TIME_FORMATTED=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED_TIME=$((END_TIME - START_TIME))

HOURS=$((ELAPSED_TIME / 3600))
MINUTES=$(((ELAPSED_TIME % 3600) / 60))
SECONDS=$((ELAPSED_TIME % 60))

echo ""
echo "============================================================================"
echo "⏱️ 训练完成!"
echo "  开始时间: ${START_TIME_FORMATTED}"
echo "  结束时间: ${END_TIME_FORMATTED}"
echo "  执行时间: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒 (共 ${ELAPSED_TIME} 秒)"
echo "============================================================================"

