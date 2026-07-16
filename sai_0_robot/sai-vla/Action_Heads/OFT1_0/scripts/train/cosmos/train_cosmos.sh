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
# OFT1_0 Action Head - Cosmos Reason 2B VL (GR00T-N1.6-3B) 训练脚本
# ============================================================================
#
# 使用说明:
#   1. 修改下面的配置参数（数据集路径、GPU、训练参数等）
#   2. 运行: bash train_cosmos.sh
#
# 支持:
#   - 多GPU分布式训练 (使用 torchrun)
#   - Cosmos Reason 2B VL (GR00T-N1.6-3B) VLM hidden states
#   - 混合精度训练 (AMP)
#   - 梯度累积
#   - Wandb 日志记录
#
# Cosmos Reason 2B VL 参数说明:
#   - VLM 输出维度: 2048 (text_config.hidden_size)
#   - 默认提取层: -1 (最后一层) 或其他负数索引
#
# ============================================================================

# ===== 基本配置 =====

# 数据集路径 (必填)
# 需要包含 Cosmos VLM hidden states，格式: (batch, num_layers, seq_len, hidden_dim)
DATA_PATH="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys0_cosmos_-1"

# GPU 设备 ID (逗号分隔，如 "0,1,2,3")
GPU_IDS="0,1,2,3,4,5,6,7"

# GPU 数量 (必须与 GPU_IDS 数量一致)
NUM_GPUS=8

# Master 端口 (分布式训练通信端口)
MASTER_PORT=29501

# ===== 训练超参数 =====

# 批次大小 (每个 GPU 的 batch size)
# 有效 batch_size = batch_size * gradient_accumulation_steps * num_gpus
BATCH_SIZE=80

# 数据加载线程数
# 使用共享内存缓存时，可以设置 num_workers > 0
# worker 进程会共享同一块内存，不会复制数据
NUM_WORKERS=8

# 每个 worker 预取的 batch 数量 (仅在 NUM_WORKERS>0 时生效)
# 预取队列大小 = NUM_WORKERS × PREFETCH_FACTOR
PREFETCH_FACTOR=4

# 验证集比例 (0.0 表示不使用验证集)
VAL_SPLIT=0.0

# 训练轮数
EPOCHS=9999999

# 最大训练步数 (达到此步数后自动停止)
STEPS=20000

# 学习率 (原始 N1.5 使用 1e-4)
LR=1e-4

# 权重衰减 (原始 N1.5 使用 1e-5)
# AdamW 优化器的 L2 正则化参数，整个训练过程中生效
WEIGHT_DECAY=1e-5

# 预热比例 (原始 N1.5 使用 0.05)
# warmup_steps = total_steps * warmup_ratio
# 所有学习率调度模式（cosine/constant/warmup_step_decay）的预热阶段都生效
# 预热阶段：学习率从 0 线性增加到 LR
WARMUP_RATIO=0.05

# 学习率调度类型 ("cosine", "constant", "warmup_step_decay")
# cosine: warmup + cosine decay (默认，原始 N1.5 使用)
# constant: warmup + 保持不变
# warmup_step_decay: warmup + 保持 + 指定step后切换到新学习率
LR_SCHEDULER_TYPE="cosine"

# Step Decay 参数 (仅在 LR_SCHEDULER_TYPE="warmup_step_decay" 时生效)
# 从第几个 step 开始改变学习率
LR_DECAY_STEP=10000
# 改变后的学习率值
LR_DECAY_VALUE=5e-5

# Adam 优化器参数 (原始 N1.5 使用)
ADAM_BETA1=0.95
ADAM_BETA2=0.999

# ===== OFT 模型参数 =====

# Transformer 块数量
NUM_TRANSFORMER_BLOCKS=4

# 注意力头数量
NUM_ATTENTION_HEADS=8

# Dropout 比率
DROPOUT=0.1

# Action head 隐藏层维度
ACTION_HEAD_HIDDEN_DIM=4096

# VLM 隐藏层数量 (Cosmos 通常使用 1 层)
NUM_VLM_LAYERS=1

# VLM 输出维度
# Cosmos Reason 2B VL (GR00T-N1.6-3B): 2048
VLM_OUTPUT_DIM=2048

# ===== 保存配置 =====
GPU_NAME="5090"
# WHICH_LAYER="14_16_18" # 只在文件命名起作用，不影响实际训练（多层用下划线隔开）
WHICH_LAYER="-1" # 只在文件命名起作用，不影响实际训练
MODEL_NAME="cosmos" # 只在文件命名起作用，不影响实际训练 # qwen3-2b, qwen3-4b, qwen3-8b, qwen3-32b, eagle2-5, cosmos
DATASET_NAME="libero_object"

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
# 仅控制 GPU tensor 精度（RAM 缓存固定为 float32）
# float32: 保持原始精度，GPU 显存占用最大
# bfloat16: GPU 使用 bfloat16 (减半显存，无损，推荐)
# float16: GPU 使用 float16 (减半显存，有精度损失)
# 注意: 共享内存缓存始终使用 float32，在 collate_fn 中转换为指定 dtype
VLM_DTYPE="bfloat16"

# ===== VLM 缓存配置 =====
# 缓存模式选择（三选一）:
#   1. USE_WEBDATASET="true"    : WebDataset 分片模式（超大规模数据，推荐）
#   2. USE_SHARED_CACHE="true"  : 共享内存缓存（多 GPU 共享，速度最快）
#   3. CACHE_VLM_STATES="true"  : mmap 模式（OS 页缓存，自动管理）
#   4. 都设为 false            : 无缓存（每次从磁盘读取）

# ===== WebDataset 配置 (大规模数据训练优化) =====
# 使用 WebDataset tar 分片格式可以显著减少大量小文件的 I/O 开销
# 
# 转换命令 (在项目根目录执行):
#   # 保留原始精度 (默认)
#   python -m utils.webdataset_utils convert \
#       --input_path /data/.../libero_lerobot_spatial_sys0_cosmos_-1 \
#       --samples_per_shard 5000 \
#       --num_action_chunks 25
#
#   # 转换为 float16 (减半存储空间)
#   python -m utils.webdataset_utils convert \
#       --input_path /data/.../libero_lerobot_spatial_sys0_cosmos_-1 \
#       --samples_per_shard 5000 \
#       --num_action_chunks 25 \
#       --convert_vlm_dtype \
#       --vlm_dtype float16
#
# 输出目录会自动创建在 input_path 下:
#   /data/.../libero_lerobot_spatial_sys0_cosmos_-1/webdataset_shard5000_ac25/
#   ├── shard-000000.tar
#   ├── shard-000001.tar
#   ├── ...
#   └── meta.json
#
USE_WEBDATASET="true"

# WebDataset 分片路径模式 (仅在 USE_WEBDATASET="true" 时生效)
# 使用 brace expansion 语法匹配多个分片文件:
#   {000000..000012} 会展开为 000000, 000001, ..., 000012
#
# 示例 (假设有 13 个分片，编号 000000 到 000012):
#   WEBDATASET_SHARD_PATTERN="/data/.../webdataset_shard5000_ac25/shard-{000000..000012}.tar"
#
# 分布式训练时，各 GPU 会自动分配不同的分片读取
WEBDATASET_SHARD_PATTERN="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys0_cosmos_-1/webdataset_shard5000_ac16/shard-{000000..000010}.tar"

# WebDataset shuffle 缓冲区大小 (越大随机性越好，但内存占用越多)
WEBDATASET_SHUFFLE_BUFFER=1000

# ===== WebDataset 分批缓存模式 =====
# 将分片分批加载到内存，每批训练完后加载下一批
# 结合 WebDataset 格式和共享内存的优点：分片管理 + 极速读取
# 
# 注意: USE_WEBDATASET_CACHED="true" 会自动禁用 USE_WEBDATASET
#       两种模式互斥，分批缓存模式优先级更高
USE_WEBDATASET_CACHED="true"

# 每批缓存的分片数量
# 内存估算: shards × 5000样本 × (~0.5MB/样本) = 总内存
# 例如: 4 分片 × 5000 × 0.5MB ≈ 10GB
WEBDATASET_CACHE_SHARDS=8

# 分批缓存的数据类型
# float16: 内存减半
# float32: 保持精度
WEBDATASET_CACHE_DTYPE="float32"

# 是否使用共享内存缓存 (推荐用于多 GPU 训练)
# true: rank 0 预加载所有数据到共享内存，其他 GPU 零拷贝访问
# 自动检测 VLM hidden states 的形状（支持可变 seq_len）
# false: 使用 mmap 模式或不缓存
USE_SHARED_CACHE="false"

# 共享内存缓存的数据类型 (float32, float16)
# float16: 内存减半，推荐用于大规模数据集 (100GB -> 50GB)
# float32: 完整精度，内存占用大
# 注意: 训练时会转换为 VLM_DTYPE 指定的类型
CACHE_DTYPE="float32"

# 是否使用 mmap 模式缓存 (仅在 USE_SHARED_CACHE="false" 时生效)
# mmap 模式：操作系统自动管理页缓存，多进程共享
CACHE_VLM_STATES="false"

# 是否跳过加载图像 (仅使用预保存的 VLM hidden states 训练时使用)
# true: 跳过图像加载，减少 I/O，加快数据加载速度
# false: 正常加载图像
SKIP_IMAGES="true"

# ===== State 四元数转轴角配置 ===== # ! 用于libero数据集
# 是否将 9 维 state (四元数: gripper1, gripper2, x, y, z, qx, qy, qz, qw) 
# 转换为 8 维 state (轴角: gripper1, gripper2, x, y, z, ax, ay, az)
#
# 使用方法:
#   "true"  - 启用转换 (9维 -> 8维，适用于四元数格式的数据集)
#   "false" - 禁用转换 (保持原始维度，适用于已经是轴角格式的数据集)
CONVERT_QUAT_TO_AXISANGLE="true"

# ===== Observation State 映射器配置 =====
# 将 observation_state 归一化后通过 MLP 映射到 VLM 隐藏空间，
# 然后拼接到输出末尾，作为额外的条件信息参与训练
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
ENABLE_STATE_MAPPER="false"

# ===== State 预处理配置 =====
# 预处理执行顺序 (空格分隔，按顺序执行)
# 可选值: hand_binary
# 留空表示不进行预处理
STATE_PROCESS_ORDER=""

# --- 手部二值化配置 (hand_binary) ---
# 手部数据在【原始 state】中的列索引范围 (每组: 起始索引 结束索引，左闭右开)
# 支持多组手部数据，按顺序处理
# 示例: 单手 "0 6"，双手 "6 12 18 24" (左手6-12，右手18-24)
# 每组6维->1维，多组按顺序处理时自动计算索引偏移
HAND_BINARY_COLUMNS=""
# 二值化阈值 (平均值 > threshold -> 1, 否则 -> 0)
HAND_BINARY_THRESHOLD=442

# State 维度 (设置为【处理后】的维度)
STATE_DIM=8

# 使用 minmax 归一化的列索引 (空格分隔)
# 从 stats.json 读取 observation.state 的 min/max
# 示例: gripper (0,1) + xyz (2,3,4) = "0 1 2 3 4"
STATE_NORM_COLUMNS_MINMAX="0 1 2 3 4"

# 使用 axisangle 归一化的列索引 (对已转换为轴角的列做 [-pi, pi] -> [0,1] 归一化)
# 示例: 旋转列 (5,6,7) = "5 6 7"
STATE_NORM_COLUMNS_AXISANGLE="5 6 7"

# ===== Action Polyfit 配置 =====
# 是否对 action chunk 进行多项式拟合
# true: 对每个 action chunk 进行多项式拟合后再训练
# false: 使用原始 action chunk 训练
USE_ACTION_POLYFIT="false"

# 多项式拟合阶数 (仅在 USE_ACTION_POLYFIT="true" 时生效)
# 常用值: 3 (三阶多项式), 4 (四阶多项式)
ACTION_POLYFIT_DEGREE=3

# ===== Action from State Diff 配置 =====
# 是否使用 state 差分替代 action
# true: 对 state chunk 进行多项式拟合后计算 state[t+1] - state[t] 替代原始 action
# false: 使用原始 action
USE_ACTION_FROM_STATE_DIFF="false"

# state 差分多项式拟合阶数 (仅在 USE_ACTION_FROM_STATE_DIFF="true" 时生效)
ACTION_FROM_STATE_DIFF_DEGREE=3

# 参与差分计算的 state 列索引 (与 ACTION_DIFF_TARGET_COLUMNS 一一对应)
# 例: state[0,1,2,3,4,5,6,7] -> action[0,1,2,3,4,5,6,7]
STATE_DIFF_COLUMNS="0 1 2 3 4 5 6 7"

# 差分结果赋值的 action 列索引 (与 STATE_DIFF_COLUMNS 一一对应)
ACTION_DIFF_TARGET_COLUMNS="0 1 2 3 4 5 6 7"

# 保持原始 action 值的列索引 (不被 state 差分替换)
ACTION_KEEP_ORIGINAL_COLUMNS="8 9 10 11 12 13"

# ===== Weights & Biases 配置 =====

# 是否使用 Wandb 记录训练
USE_WANDB="true"

# Wandb 项目名称(WANDB_PROJECT)
WANDB_PROJECT="flow_matching_0_eagle_libero_test_newest"

# Checkpoint 输出目录
OUT_DIR="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys0_cosmos_-1_weight/OFT1_0/1layer_-1/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_layer${WHICH_LAYER}_$(date +%Y%m%d)_wp${WANDB_PROJECT}_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}/checkpoints"

# 日志输出目录
LOG_DIR="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_object/libero_lerobot_object_sys0_cosmos_-1_weight/OFT1_0/1layer_-1/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_layer${WHICH_LAYER}_$(date +%Y%m%d)_wp${WANDB_PROJECT}_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}/logs"

# Wandb 运行名称 (留空则自动生成)
WANDB_RUN_NAME="${MODEL_NAME}_layer${WHICH_LAYER}_${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_oft1_$(date +%Y%m%d)_${DATASET_NAME}_USE_SHARED_CACHE_${USE_SHARED_CACHE}_CACHE_VLM_STATES_${CACHE_VLM_STATES}_USE_AMP_${USE_AMP}_webdataset_${USE_WEBDATASET}"

# Wandb 基本日志记录频率 (每 N 步记录 loss, lr, batch_time 等)
WANDB_LOG_FREQ=1

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
export TORCH_DISTRIBUTED_TIMEOUT_SEC=14400        # OFT1_0 Python 代码读取此变量
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
TRAIN_SCRIPT="${PROJECT_ROOT}/Action_Heads/OFT1_0/train_multigpu.py"

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
CMD="${CMD} --lr_scheduler_type ${LR_SCHEDULER_TYPE}"
if [ "${LR_SCHEDULER_TYPE}" = "warmup_step_decay" ]; then
    CMD="${CMD} --lr_decay_step ${LR_DECAY_STEP}"
    CMD="${CMD} --lr_decay_value ${LR_DECAY_VALUE}"
fi
CMD="${CMD} --adam_beta1 ${ADAM_BETA1}"
CMD="${CMD} --adam_beta2 ${ADAM_BETA2}"

# OFT 模型参数
CMD="${CMD} --num_transformer_blocks ${NUM_TRANSFORMER_BLOCKS}"
CMD="${CMD} --num_attention_heads ${NUM_ATTENTION_HEADS}"
CMD="${CMD} --dropout ${DROPOUT}"
CMD="${CMD} --action_head_hidden_dim ${ACTION_HEAD_HIDDEN_DIM}"
CMD="${CMD} --num_vlm_layers ${NUM_VLM_LAYERS}"
CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"

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
    # mmap 模式
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
fi

# Action Polyfit 配置
if [ "${USE_ACTION_POLYFIT}" = "true" ]; then
    CMD="${CMD} --use_action_polyfit"
    CMD="${CMD} --action_polyfit_degree ${ACTION_POLYFIT_DEGREE}"
fi

# Action from State Diff 配置
if [ "${USE_ACTION_FROM_STATE_DIFF}" = "true" ]; then
    CMD="${CMD} --use_action_from_state_diff"
    CMD="${CMD} --action_from_state_diff_degree ${ACTION_FROM_STATE_DIFF_DEGREE}"
    if [ -n "${STATE_DIFF_COLUMNS}" ]; then
        CMD="${CMD} --state_diff_columns ${STATE_DIFF_COLUMNS}"
    fi
    if [ -n "${ACTION_DIFF_TARGET_COLUMNS}" ]; then
        CMD="${CMD} --action_diff_target_columns ${ACTION_DIFF_TARGET_COLUMNS}"
    fi
    if [ -n "${ACTION_KEEP_ORIGINAL_COLUMNS}" ]; then
        CMD="${CMD} --action_keep_original_columns ${ACTION_KEEP_ORIGINAL_COLUMNS}"
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

# 禁用保存最佳模型
if [ "${NO_SAVE_BEST}" = "true" ]; then
    CMD="${CMD} --no_save_best"
fi

# ===== 打印配置信息 =====

echo "============================================================================"
echo "🌌 OFT1_0 Action Head - Cosmos Reason 2B VL (GR00T-N1.6-3B) 训练"
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
echo "  LR Scheduler Type: ${LR_SCHEDULER_TYPE}"
if [ "${LR_SCHEDULER_TYPE}" = "warmup_step_decay" ]; then
    echo "  LR Decay Step: ${LR_DECAY_STEP}"
    echo "  LR Decay Value: ${LR_DECAY_VALUE}"
fi
echo "  Adam Beta1: ${ADAM_BETA1}"
echo "  Adam Beta2: ${ADAM_BETA2}"
echo "  Gradient Accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo ""
echo "🌌 OFT 模型配置:"
echo "  Transformer Blocks: ${NUM_TRANSFORMER_BLOCKS}"
echo "  Attention Heads: ${NUM_ATTENTION_HEADS}"
echo "  Dropout: ${DROPOUT}"
echo "  Action Head Hidden Dim: ${ACTION_HEAD_HIDDEN_DIM}"
echo "  VLM Layers: ${NUM_VLM_LAYERS}"
echo "  VLM Output Dim: ${VLM_OUTPUT_DIM}"
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
    fi
    echo ""
fi
if [ "${USE_ACTION_POLYFIT}" = "true" ]; then
    echo "📈 Action Polyfit 配置:"
    echo "  Enable Action Polyfit: ${USE_ACTION_POLYFIT}"
    echo "  Polyfit Degree: ${ACTION_POLYFIT_DEGREE}"
    echo ""
fi
if [ "${USE_ACTION_FROM_STATE_DIFF}" = "true" ]; then
    echo "📈 Action from State Diff 配置:"
    echo "  Enable: ${USE_ACTION_FROM_STATE_DIFF}"
    echo "  Polyfit Degree: ${ACTION_FROM_STATE_DIFF_DEGREE}"
    echo "  State Diff Columns: ${STATE_DIFF_COLUMNS}"
    echo "  Action Target Columns: ${ACTION_DIFF_TARGET_COLUMNS}"
    echo "  Action Keep Original: ${ACTION_KEEP_ORIGINAL_COLUMNS}"
    echo ""
fi
if [ "${USE_WANDB}" = "true" ]; then
    echo "📊 Wandb 配置:"
    echo "  Project: ${WANDB_PROJECT}"
    echo "  Run Name: ${WANDB_RUN_NAME}"
    echo "  Log Freq: ${WANDB_LOG_FREQ} steps"
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

