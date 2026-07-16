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
# Flow Matching 0 Action Head - Qwen 训练脚本
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数（数据集路径、GPU、训练参数等）
#   2. 运行: bash train_qwen.sh
#
# 支持:
#   - 多GPU分布式训练 (使用 torchrun)
#   - Qwen2B/Qwen4B VLM hidden states
#   - 混合精度训练 (AMP)
#   - 梯度累积
#   - Wandb 日志记录
#
# ============================================================================

# ===== 基本配置 =====

# 数据集路径 (必填)
# 需要包含 Qwen VLM hidden states，格式: (batch, num_layers, seq_len, hidden_dim)
DATA_PATH="/home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_goal/libero_lerobot_goal_sys1_qwen_2b_14"

# GPU 设备 ID (逗号分隔，如 "0,1,2,3")
GPU_IDS="0,1,2,3,4,5,6,7"

# GPU 数量 (必须与 GPU_IDS 数量一致)
NUM_GPUS=8

# Master 端口 (分布式训练通信端口)
MASTER_PORT=29503

# ===== 训练超参数 =====

# 批次大小 (每个 GPU 的 batch size)
# 有效 batch_size = batch_size * gradient_accumulation_steps * num_gpus
BATCH_SIZE=80

# 数据加载线程数
NUM_WORKERS=8

# 每个 worker 预取的 batch 数量 (仅在 NUM_WORKERS>0 时生效)
# 预取队列大小 = NUM_WORKERS × PREFETCH_FACTOR
PREFETCH_FACTOR=4

# 验证集比例 (0.0 表示不使用验证集)
VAL_SPLIT=0.0

# 训练轮数
EPOCHS=10000

# 最大训练步数 (达到此步数后自动停止)
STEPS=20000

# 学习率 (原始 N1.5 使用 1e-4)
LR=1e-4

# 权重衰减 (原始 N1.5 使用 1e-5)
WEIGHT_DECAY=1e-5

# 预热比例 (原始 N1.5 使用 0.05)
WARMUP_RATIO=0.05

# Adam 优化器参数 (原始 N1.5 使用)
ADAM_BETA1=0.95
ADAM_BETA2=0.999

# ===== 模型维度参数 =====

# 最大动作维度 (必须与预训练模型一致，不要修改)
MAX_ACTION_DIM=32

# 最大状态维度 (必须与预训练模型一致，不要修改)
MAX_STATE_DIM=64

# 动作预测时间步数 (原始 N1.5 LIBERO 使用 16)
NUM_ACTION_CHUNKS=16

# ===== 保存配置 =====
GPU_NAME="p6000"
# WHICH_LAYER="14_16_18" # 只在文件命名起作用，不影响实际训练（多层用下划线隔开）
WHICH_LAYER="14" # 只在文件命名起作用，不影响实际训练
MODEL_NAME="qwen2b" # 只在文件命名起作用，不影响实际训练 # qwen3-2b, qwen3-4b, qwen3-8b, qwen3-32b, eagle2-5
DATASET_NAME="libero_goal"

# ===== 预训练权重 (可选) =====

# 预训练权重文件路径 (留空则从头训练)
PRETRAINED_WEIGHTS=""

# 是否禁用预训练权重加载 (true: 从头训练, false: 加载预训练权重)
NO_PRETRAIN="true"

# 每 N 个 epoch 保存一次 checkpoint (设置很大的值=禁用)
SAVE_EVERY=999999

# 每 N 个 step 保存一次 checkpoint (0=禁用)
SAVE_EVERY_STEPS=4000

# 是否禁用保存最佳模型 (true: 禁用, false: 启用)
NO_SAVE_BEST="true"

# ===== 系统配置 =====

# 梯度累积步数
# 有效 batch_size = batch_size * gradient_accumulation_steps * num_gpus
GRADIENT_ACCUMULATION_STEPS=1

# 是否使用混合精度训练 (AMP)
# true: 使用 FP16/BF16 混合精度，显著减少显存占用 (推荐)
# false: 使用 FP32 全精度训练
USE_AMP="true"

# AMP 数据类型 (float16 或 bfloat16)
# float16: 适用于大多数 GPU
# bfloat16: 适用于 Ampere 及以上架构 (A100, RTX 3090+)
AMP_DTYPE="bfloat16"

# 清理 CUDA 缓存的频率 (每 N 步，0=禁用)
EMPTY_CACHE_FREQ=100

# ===== 数据加载优化参数 =====

# VLM hidden states 的数据类型 (float32, float16, bfloat16)
# 同时控制 RAM 缓存精度和 GPU tensor 精度
# float32: 保持原始精度，内存占用最大
# bfloat16: RAM 和 GPU 都使用 bfloat16 (减半内存，无损)
# float16: RAM 和 GPU 都使用 float16 (减半内存，有精度损失)
VLM_DTYPE="bfloat16"

# VLM 缓存策略选择:
#   1. USE_WEBDATASET="true"    : WebDataset 分片模式（超大规模数据，推荐）
#   2. USE_SHARED_CACHE="true"  : 共享内存缓存（推荐，多 GPU 共享，速度最快）
#      - 预加载所有轨迹的 VLM hidden states 到共享内存
#      - 多 GPU 共享同一份缓存，零拷贝读取
#      - 训练集和验证集都可以使用
#   3. CACHE_VLM_STATES="true"  : mmap 缓存（备选，自动共享）
#      - 使用 numpy mmap 模式，OS 页缓存自动管理
#   4. 都为 false: 无缓存（最慢，每次从磁盘读取）

# ===== WebDataset 配置 (大规模数据训练优化) =====
# 使用 WebDataset tar 分片格式可以显著减少大量小文件的 I/O 开销
# 
# 转换命令 (在项目根目录执行):
#   # 保留原始精度 (默认)
#   python -m utils.webdataset_utils convert \
#       --input_path /data/.../libero_lerobot_object_sys1_qwen_2b_14 \
#       --samples_per_shard 5000 \
#       --num_action_chunks 25
#
#   # 转换为 float16 (减半存储空间)
#   python -m utils.webdataset_utils convert \
#       --input_path /data/.../libero_lerobot_object_sys1_qwen_2b_14 \
#       --samples_per_shard 5000 \
#       --num_action_chunks 25 \
#       --convert_vlm_dtype \
#       --vlm_dtype float16
#
# 输出目录会自动创建在 input_path 下:
#   /data/.../libero_lerobot_object_sys1_qwen_2b_14/webdataset_shard5000_ac25/
#   ├── shard-000000.tar
#   ├── shard-000001.tar
#   ├── ...
#   └── meta.json
#
USE_WEBDATASET="false"

# WebDataset 分片路径模式 (仅在 USE_WEBDATASET="true" 时生效)
# 使用 brace expansion 语法匹配多个分片文件:
#   {000000..000012} 会展开为 000000, 000001, ..., 000012
#
# 示例 (假设有 13 个分片，编号 000000 到 000012):
#   WEBDATASET_SHARD_PATTERN="/data/.../webdataset_shard5000_ac25/shard-{000000..000012}.tar"
#
# 分布式训练时，各 GPU 会自动分配不同的分片读取
WEBDATASET_SHARD_PATTERN="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys1_qwen_2b_14/webdataset_shard5000_ac16/shard-{000000..000010}.tar"

# WebDataset shuffle 缓冲区大小 (越大随机性越好，但内存占用越多)
WEBDATASET_SHUFFLE_BUFFER=1000

# ===== WebDataset 分批缓存模式 =====
# 将分片分批加载到内存，每批训练完后加载下一批
# 结合 WebDataset 格式和共享内存的优点：分片管理 + 极速读取
# 
# 注意: USE_WEBDATASET_CACHED="true" 会自动禁用 USE_WEBDATASET
#       两种模式互斥，分批缓存模式优先级更高
USE_WEBDATASET_CACHED="false"

# 每批缓存的分片数量
# 内存估算: shards × 5000样本 × (~0.5MB/样本) = 总内存
# 例如: 4 分片 × 5000 × 0.5MB ≈ 10GB
WEBDATASET_CACHE_SHARDS=8

# 分批缓存的数据类型
# float16: 内存减半
# float32: 保持精度
WEBDATASET_CACHE_DTYPE="float32"

USE_SHARED_CACHE="true"

# 共享内存缓存的数据类型 (float32, float16)
# float16: 内存减半，推荐用于大规模数据集 (100GB -> 50GB)
# float32: 完整精度，内存占用大
# 注意: 训练时会转换为 VLM_DTYPE 指定的类型
CACHE_DTYPE="float32"

# 是否缓存 VLM hidden states 到 RAM
# true: 首个 epoch 后数据全部在内存中，后续 epoch 加速
# false: 每次都从磁盘读取 (默认)
# 注意: 需要足够的 RAM (约 28MB * 样本数 / 缓存精度系数)
CACHE_VLM_STATES="false"

# 是否跳过加载图像 (仅使用预保存的 VLM hidden states 训练时使用)
# true: 跳过图像加载，减少 I/O，加快数据加载速度
# false: 正常加载图像
SKIP_IMAGES="true"

# ===== State 四元数转轴角配置 =====
# 是否将 9 维 state (四元数: gripper1, gripper2, x, y, z, qx, qy, qz, qw) 
# 转换为 8 维 state (轴角: gripper1, gripper2, x, y, z, ax, ay, az)
#
# 使用方法:
#   "true"  - 启用转换 (9维 -> 8维，适用于四元数格式的数据集)
#   "false" - 禁用转换 (保持原始维度，适用于已经是轴角格式的数据集)
CONVERT_QUAT_TO_AXISANGLE="true"

# ===== Weights & Biases 配置 =====

# 是否使用 Wandb 记录训练
USE_WANDB="true"

# Wandb 项目名称(WANDB_PROJECT)
WANDB_PROJECT="flow_matching_0_eagle_libero_test_newest"

# Checkpoint 输出目录
OUT_DIR="/home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_goal/libero_lerobot_goal_sys1_qwen_2b_14_weight/FM0/1layer_14/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_${STEPS}steps_${NUM_ACTION_CHUNKS}chunks_NO_PRETRAIN-${NO_PRETRAIN}_layer${WHICH_LAYER}_$(date +%Y%m%d)_wp${WANDB_PROJECT}_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}/checkpoints"

# 日志输出目录
LOG_DIR="/home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_goal/libero_lerobot_goal_sys1_qwen_2b_14_weight/FM0/1layer_14/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_${STEPS}steps_${NUM_ACTION_CHUNKS}chunks_NO_PRETRAIN-${NO_PRETRAIN}_layer${WHICH_LAYER}_$(date +%Y%m%d)_wp${WANDB_PROJECT}_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}/logs"

# Wandb 运行名称 (留空则自动生成)
WANDB_RUN_NAME="${MODEL_NAME}_layer${WHICH_LAYER}_${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_${STEPS}steps_fm0_NO_PRETRAIN-${NO_PRETRAIN}_$(date +%Y%m%d)_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}"

# Wandb 基本日志记录频率 (每 N 步记录 loss, lr, batch_time 等)
WANDB_LOG_FREQ=1

# ===== 分布式训练环境变量 =====

# ===== 分布式训练环境变量 =====

# 解决 GPU 硬件问题导致的 NVML 错误
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # 按 PCI 总线顺序排列 GPU
export NCCL_NVLS_ENABLE=0                    # 禁用 NVLS (NVLink SHARP) 功能
export NCCL_P2P_DISABLE=1                    # 禁用 P2P，避免 NVML 拓扑检测访问坏的 GPU
export NCCL_SHM_DISABLE=0                    # 启用共享内存通信 (替代 P2P)
export NCCL_IB_DISABLE=1                     # 禁用 InfiniBand

# NCCL 超时设置 (支持长时间的 VLM states 预加载)
# 注意: NCCL_TIMEOUT 单位是毫秒，TORCH_NCCL_* 单位是秒
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1          # 启用异步错误处理
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=14400     # Watchdog 心跳超时: 4小时（增加）
export NCCL_TIMEOUT=14400000                      # NCCL 操作超时（毫秒）: 4小时
export TORCH_NCCL_BLOCKING_WAIT=0                 # 非阻塞等待
export NCCL_BLOCKING_WAIT=0                       # 非阻塞等待

# PyTorch 分布式超时设置
export TORCH_CPP_LOG_LEVEL=INFO                   # 日志级别

# 启用调试追踪，帮助定位问题
export TORCH_NCCL_TRACE_BUFFER_SIZE=1000          # 启用 FlightRecorder 追踪

# NCCL 调试设置 (帮助定位通信问题)
export NCCL_DEBUG=WARN                            # 调试级别: WARN/INFO/TRACE (INFO 会产生大量日志)
export NCCL_DEBUG_SUBSYS=ALL                      # 调试子系统: ALL/INIT/COLL/P2P/SHM/NET

# 额外的容错设置
export TORCH_NCCL_ENABLE_MONITORING=1             # 启用 NCCL 监控
export NCCL_ASYNC_ERROR_HANDLING=1                # NCCL 异步错误处理
export NCCL_SOCKET_NTHREADS=4                     # Socket 线程数 (提高通信稳定性)
export NCCL_NSOCKS_PERTHREAD=4                    # 每线程 Socket 数量

# ===== 构建训练命令 =====

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

# 训练脚本路径
TRAIN_SCRIPT="${PROJECT_ROOT}/Action_Heads/Flow_Matching_0/train_with_pretrained_action_head_weight_multigpu.py"

# 构建基础命令
# --rdzv_conf timeout=7200: 设置 rendezvous 超时为 2 小时
CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} --rdzv_backend=c10d --rdzv_conf timeout=7200 ${TRAIN_SCRIPT}"

# 数据相关参数
CMD="${CMD} --data_path ${DATA_PATH}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --prefetch_factor ${PREFETCH_FACTOR}"
CMD="${CMD} --val_split ${VAL_SPLIT}"

# 训练超参数
CMD="${CMD} --epochs ${EPOCHS}"
CMD="${CMD} --steps ${STEPS}"
CMD="${CMD} --lr ${LR}"
CMD="${CMD} --weight_decay ${WEIGHT_DECAY}"
CMD="${CMD} --warmup_ratio ${WARMUP_RATIO}"
CMD="${CMD} --adam_beta1 ${ADAM_BETA1}"
CMD="${CMD} --adam_beta2 ${ADAM_BETA2}"

# 模型维度参数
CMD="${CMD} --max_action_dim ${MAX_ACTION_DIM}"
CMD="${CMD} --max_state_dim ${MAX_STATE_DIM}"
CMD="${CMD} --num_action_chunks ${NUM_ACTION_CHUNKS}"

# 保存配置
CMD="${CMD} --out_dir ${OUT_DIR}"
CMD="${CMD} --log_dir ${LOG_DIR}"
CMD="${CMD} --save_every ${SAVE_EVERY}"
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

# CUDA 缓存清理
if [ "${EMPTY_CACHE_FREQ}" -gt 0 ]; then
    CMD="${CMD} --empty_cache_freq ${EMPTY_CACHE_FREQ}"
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

# 预训练权重 (可选)
if [ "${NO_PRETRAIN}" = "true" ]; then
    CMD="${CMD} --no_pretrain"
elif [ -n "${PRETRAINED_WEIGHTS}" ]; then
    CMD="${CMD} --pretrained_weights ${PRETRAINED_WEIGHTS}"
fi

# 禁用保存最佳模型
if [ "${NO_SAVE_BEST}" = "true" ]; then
    CMD="${CMD} --no_save_best"
fi

# ===== 打印配置信息 =====

echo "============================================================================"
echo "🚀 Flow Matching 0 Action Head - Qwen 训练"
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
echo "  Epochs: ${EPOCHS}"
echo "  Max Steps: ${STEPS}"
echo "  Learning Rate: ${LR}"
echo "  Weight Decay: ${WEIGHT_DECAY}"
echo "  Warmup Ratio: ${WARMUP_RATIO}"
echo "  Adam Beta1: ${ADAM_BETA1}"
echo "  Adam Beta2: ${ADAM_BETA2}"
echo "  Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo ""
echo "🤖 模型配置:"
echo "  Max Action Dim: ${MAX_ACTION_DIM}"
echo "  Max State Dim: ${MAX_STATE_DIM}"
echo "  Action Chunks: ${NUM_ACTION_CHUNKS}"
echo ""
echo "💾 保存配置:"
echo "  Save Every: ${SAVE_EVERY} epochs"
echo "  Save Every Steps: ${SAVE_EVERY_STEPS}"
echo "  No Save Best: ${NO_SAVE_BEST}"
echo ""
echo "⚡ 性能配置:"
echo "  AMP: ${USE_AMP}"
if [ "${USE_AMP}" = "true" ]; then
    echo "  AMP Dtype: ${AMP_DTYPE}"
fi
echo "  Empty Cache Freq: ${EMPTY_CACHE_FREQ}"
echo "  VLM Dtype: ${VLM_DTYPE}"
echo "  Skip Images: ${SKIP_IMAGES}"
echo "  四元数转轴角: ${CONVERT_QUAT_TO_AXISANGLE}"
echo "  Use Shared Cache: ${USE_SHARED_CACHE}"
echo ""
if [ "${USE_WANDB}" = "true" ]; then
    echo "📊 Wandb 配置:"
    echo "  Project: ${WANDB_PROJECT}"
    echo "  Run Name: ${WANDB_RUN_NAME}"
    echo "  Log Freq: ${WANDB_LOG_FREQ} steps"
    echo ""
fi
if [ -n "${PRETRAINED_WEIGHTS}" ]; then
    echo "🎯 预训练权重:"
    echo "  ${PRETRAINED_WEIGHTS}"
    echo ""
elif [ "${NO_PRETRAIN}" = "true" ]; then
    echo "🎯 从头训练 (无预训练权重)"
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

# 转换为小时:分钟:秒格式
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

