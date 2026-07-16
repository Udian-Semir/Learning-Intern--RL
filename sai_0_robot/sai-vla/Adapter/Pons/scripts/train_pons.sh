#!/bin/bash

# ============================================================================
# 训练前清理残留的共享内存（防止之前异常退出导致的内存泄漏）
# ============================================================================
echo "🧹 清理残留的共享内存..."

# 显示清理前的 /dev/shm 使用情况
echo "  清理前 /dev/shm 使用情况:"
df -h /dev/shm | tail -1

# 清理 Python SharedMemory 残留（以 psm_ 开头）
SHM_COUNT=$(ls /dev/shm/psm_* 2>/dev/null | wc -l)
if [ "$SHM_COUNT" -gt 0 ]; then
    echo "  发现 ${SHM_COUNT} 个残留的共享内存文件，正在清理..."
    rm -f /dev/shm/psm_*
    echo "  ✓ 清理完成"
else
    echo "  ✓ 没有发现残留的共享内存"
fi

# 显示清理后的 /dev/shm 使用情况
echo "  清理后 /dev/shm 使用情况:"
df -h /dev/shm | tail -1
echo ""

# ============================================================================

# 记录开始时间
START_TIME=$(date +%s)
START_TIME_FORMATTED=$(date '+%Y-%m-%d %H:%M:%S')

# ============================================================================
# Pons Adapter 训练脚本
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数（数据集路径、GPU、训练参数等）
#   2. 运行: bash train_pons.sh
#
# 支持:
#   - 多GPU分布式训练 (使用 torchrun)
#   - 自监督预训练 (重建目标)
#   - 混合精度训练 (AMP)
#   - 梯度累积
#   - Wandb 日志记录
#
# ============================================================================

# ===== 基本配置 =====

# 数据集路径 (必填)
# 需要包含 VLM hidden states，格式: (batch, num_layers, seq_len, hidden_dim)
DATA_PATH="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_10/libero_lerobot_10_sys0_eagle"

# GPU 设备 ID (逗号分隔，如 "0,1,2,3")
GPU_IDS="0,1,2,3,4,5,6,7"

# GPU 数量 (必须与 GPU_IDS 数量一致)
NUM_GPUS=8

# Master 端口 (分布式训练通信端口)
MASTER_PORT=29510

# ===== 训练超参数 =====

# 批次大小 (每个 GPU 的 batch size)
BATCH_SIZE=64

# 数据加载线程数
NUM_WORKERS=8

# 每个 worker 预取的 batch 数量
PREFETCH_FACTOR=4

# 最大训练步数
STEPS=10000

# 学习率
LR=1e-4

# 权重衰减
WEIGHT_DECAY=1e-5

# 预热比例
WARMUP_RATIO=0.05

# ===== Pons 模型参数 =====

# Query 序列长度 (Pons 压缩后的 token 数量)
PONS_Q_SEQ_LEN=64

# Cross-Attention 块数量
PONS_NUM_BLOCKS=2

# 注意力头数量
PONS_NUM_HEADS=8

# Dropout 比率
PONS_DROPOUT=0.1

# VLM 隐藏层数量
NUM_VLM_LAYERS=1

# VLM 输出维度
# Eagle 2.5 VL: 2048, Qwen2-VL-2B: 1536, Qwen2-VL-7B: 3584
VLM_OUTPUT_DIM=2048

# Action chunks (用于数据加载)
NUM_ACTION_CHUNKS=16

# ===== 保存配置 =====
GPU_NAME="5090"
MODEL_NAME="pons"
DATASET_NAME="libero_10"

# 每 N 个 step 保存一次 checkpoint
SAVE_EVERY_STEPS=2000

# ===== 系统配置 =====

# 梯度累积步数
GRADIENT_ACCUMULATION_STEPS=1

# 是否使用混合精度训练 (AMP)
USE_AMP="true"

# AMP 数据类型
AMP_DTYPE="bfloat16"

# ===== 数据加载优化参数 =====

# VLM hidden states 的数据类型
VLM_DTYPE="bfloat16"

# 是否使用共享内存缓存
USE_SHARED_CACHE="false"

# 是否使用 mmap 模式缓存
CACHE_VLM_STATES="false"

# 是否跳过加载图像
SKIP_IMAGES="true"

# ===== Weights & Biases 配置 =====

# 是否使用 Wandb 记录训练
USE_WANDB="true"

# Wandb 项目名称
WANDB_PROJECT="pons_training"

# Checkpoint 输出目录
OUT_DIR="/data/HuangWenlong/datasets/pons_weights/${DATASET_NAME}/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_q${PONS_Q_SEQ_LEN}_blocks${PONS_NUM_BLOCKS}_${STEPS}steps_$(date +%Y%m%d)/checkpoints"

# 日志输出目录
LOG_DIR="/data/HuangWenlong/datasets/pons_weights/${DATASET_NAME}/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_q${PONS_Q_SEQ_LEN}_blocks${PONS_NUM_BLOCKS}_${STEPS}steps_$(date +%Y%m%d)/logs"

# Wandb 运行名称
WANDB_RUN_NAME="${MODEL_NAME}_q${PONS_Q_SEQ_LEN}_blocks${PONS_NUM_BLOCKS}_${GPU_NAME}_bsz${BATCH_SIZE}*${NUM_GPUS}_${STEPS}steps_$(date +%Y%m%d)_${DATASET_NAME}"

# Wandb 日志记录频率
WANDB_LOG_FREQ=10

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

TRAIN_SCRIPT="${PROJECT_ROOT}/Adapter/Pons/train_multigpu.py"

CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} --rdzv_backend=c10d --rdzv_conf timeout=7200 ${TRAIN_SCRIPT}"

# 数据相关参数
CMD="${CMD} --data_path ${DATA_PATH}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --prefetch_factor ${PREFETCH_FACTOR}"

# 训练超参数
CMD="${CMD} --steps ${STEPS}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --weight_decay ${WEIGHT_DECAY}"
CMD="${CMD} --warmup_ratio ${WARMUP_RATIO}"

# Pons 模型参数
CMD="${CMD} --pons_q_seq_len ${PONS_Q_SEQ_LEN}"
CMD="${CMD} --pons_num_blocks ${PONS_NUM_BLOCKS}"
CMD="${CMD} --pons_num_heads ${PONS_NUM_HEADS}"
CMD="${CMD} --pons_dropout ${PONS_DROPOUT}"

# VLM 参数
CMD="${CMD} --num_vlm_layers ${NUM_VLM_LAYERS}"
CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"
CMD="${CMD} --num_action_chunks ${NUM_ACTION_CHUNKS}"

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

if [ "${USE_SHARED_CACHE}" = "true" ]; then
    CMD="${CMD} --use_shared_cache"
elif [ "${CACHE_VLM_STATES}" = "true" ]; then
    CMD="${CMD} --cache_vlm_states"
fi

if [ "${SKIP_IMAGES}" = "true" ]; then
    CMD="${CMD} --skip_images"
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
echo "🔗 Pons Adapter 训练"
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
echo "  Warmup Ratio: ${WARMUP_RATIO}"
echo "  Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo ""
echo "🔗 Pons 模型配置:"
echo "  Query Seq Len: ${PONS_Q_SEQ_LEN}"
echo "  Num Blocks: ${PONS_NUM_BLOCKS}"
echo "  Num Heads: ${PONS_NUM_HEADS}"
echo "  Dropout: ${PONS_DROPOUT}"
echo "  VLM Layers: ${NUM_VLM_LAYERS}"
echo "  VLM Output Dim: ${VLM_OUTPUT_DIM}"
echo ""
echo "💾 保存配置:"
echo "  Save Every Steps: ${SAVE_EVERY_STEPS}"
echo ""
echo "⚡ 性能配置:"
echo "  AMP: ${USE_AMP} (${AMP_DTYPE})"
echo "  VLM Dtype: ${VLM_DTYPE}"
echo "  Skip Images: ${SKIP_IMAGES}"
echo ""
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

# 记录结束时间并计算执行时间
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

