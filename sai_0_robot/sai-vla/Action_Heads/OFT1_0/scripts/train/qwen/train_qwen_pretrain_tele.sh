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
# OFT1_0 Action Head - Qwen VL 多子集预训练脚本 (unitree_v2 recipe)
# ============================================================================
#
# 与 train_qwen_datasets_libero_plus_action7_chunk16_no_state_resume copy.sh
# 的区别:
#   ✓ USE_MULTI_DATASET="true": 自动扫 MULTI_DATASET_ROOT 下所有子数据集,
#     跨子集 shuffle vlm chunks, 不需要把数据物理合并成一个 lerobot
#   ✓ 多源 normalize + state right-pad 在 MultiLeRobotDataset.__getitem__ 完成
#   ✓ chunk_batch_cache 自动切到 MultiChunkBatchCache (跨子集 SHM 共享)
#
# 第一次运行强烈推荐先跑 dry-run:
#   USE_MULTI_DATASET="true"  +  MULTI_DATASET_DRY_RUN="true"
#   会扫描+打印每个子集的 action/state dim / vlm 完整性 / 全局帧数等, 并
#   写一份 manifest.json 然后正常退出。根据输出再回来确认 ACTION_DIM /
#   PROPRIO_DIM / NUM_ACTIONS_CHUNK 是否合理。
# ============================================================================

# ===== 基本配置 =====

# 多源 root: 下面是 N 个独立 LeRobot 子数据集 (每个有自己的 meta/, data/,
# vlm_hidden_states/), 不需要物理合并。
DATA_PATH="/data_disk2/hwl/datasets/dataset196/lerobot_dataset196_rightarm_state_joint_action_eefdelta_filter_by_velocity"

# ============================================================================
# 数据集维度对齐 (覆盖 Action_Heads/OFT1_0/constants.py 的默认值)
# ----------------------------------------------------------------------------
# 多源场景下:
#   - ACTION_DIM   = 所有子集共享的 action 维度 (MultiDatasetIndex 强制 uniform)
#   - PROPRIO_DIM  = 所有子集 state 的 max dim, 短的右补 0
# 第一次跑用 MULTI_DATASET_DRY_RUN="true" 扫一遍, 把扫描输出的:
#   "max action_dim: ?"   填到 ACTION_DIM
#   "max state_dim:  ?"   填到 PROPRIO_DIM
# ============================================================================
# 实测 unitree_train_v2_recipe_lerobot 17 个有效子集 (其余 5 个 vlm 不完整自动跳过):
#   max action_dim = 7   (所有子集统一)
#   max state_dim  = 8   (15/17 子集是 8, 另 2 子集是 7 → 自动右补 0 到 8)
#   全局 episodes  = 11690, frames = 2486667, vlm chunks = 22
export ACTION_DIM=7
export PROPRIO_DIM=8
export NUM_ACTIONS_CHUNK=16

# GPU 设备 ID (逗号分隔，如 "0,1,2,3")
GPU_IDS="0,1,2,3,4,5,6,7"

# GPU 数量 (必须与 GPU_IDS 数量一致)
NUM_GPUS=8

# Master 端口 (分布式训练通信端口)
MASTER_PORT=29510

# ===== 训练超参数 =====

# 批次大小 (每个 GPU 的 batch size)
# 有效 batch_size = batch_size * gradient_accumulation_steps * num_gpus
BATCH_SIZE=500

# 数据加载线程数 # 大 batch 需要更多数据，更多 worker 能跟上 # 使用webdataset的时候设置为1
NUM_WORKERS=8

# 每个 worker 预取的 batch 数量 (仅在 NUM_WORKERS>0 时生效)
# 预取队列大小 = NUM_WORKERS × PREFETCH_FACTOR
PREFETCH_FACTOR=4

# 验证集比例 (0.0 表示不使用验证集)
VAL_SPLIT=0.0

# 训练轮数
EPOCHS=9999999

# 最大训练步数 (达到此步数后自动停止)
STEPS=100000  # ! 上一次400000

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
LR_SCHEDULER_TYPE="cosine" # 第一次cosine 200000steps 第二次constant 从 200000 到 400000steps  # 第三次400000steps 到 600000steps重新做shuffle

# Step Decay 参数 (仅在 LR_SCHEDULER_TYPE="warmup_step_decay" 时生效)
# 从第几个 step 开始改变学习率
LR_DECAY_STEP=150000
# 改变后的学习率值
LR_DECAY_VALUE=1e-5

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

# VLM 隐藏层数量 (使用的 VLM 层数)
NUM_VLM_LAYERS=1

# VLM 输出维度
# Qwen2-VL-7B: 4096
# Qwen2-VL-2B: 2048
VLM_OUTPUT_DIM=2048

# ===== 保存配置 =====
GPU_NAME="p6000"
# WHICH_LAYER="14_16_18" # 只在文件命名起作用，不影响实际训练（多层用下划线隔开）
WHICH_LAYER="14" # 只在文件命名起作用，不影响实际训练
MODEL_NAME="qwen2b" # 只在文件命名起作用，不影响实际训练 # qwen3-2b, qwen3-4b, qwen3-8b, qwen3-32b, eagle2-5
DATASET_NAME="no_pretrain_tele"

# ============================================================================
# Run 日期 (用于 OUT_DIR / LOG_DIR / WANDB_RUN_NAME)
# ----------------------------------------------------------------------------
# - 新训练: 留空，自动取今天的日期 $(date +%Y%m%d)
# - resume: 固定为原 run 启动那天的日期 (如 "20260427")，否则会另起一个新的输出目录
#   和 wandb run，新的 step checkpoint 会跟原有的 step_xxxxx/ 散到两个目录里。
# ============================================================================
RUN_DATE=""
if [ -z "${RUN_DATE}" ]; then
    RUN_DATE=$(date +%Y%m%d)
fi

# ============================================================================
# 断点续训 (Resume)
# ----------------------------------------------------------------------------
# 留空 = 从头训练
# 填入 checkpoint 目录 = 从该 checkpoint resume
#   - 目录里需有 training_state.pt 才能完整恢复 optim/scheduler/scaler/step/RNG
#   - 只有 action_head.pt (旧版 checkpoint) 时退化为"仅加载权重，从 step 0 开始"
# resume 时务必把上面的 RUN_DATE 设为原 run 的启动日期，避免输出/日志/wandb 跑到新目录。
# ============================================================================
RESUME_FROM_CHECKPOINT=""

# ----------------------------------------------------------------------------
# 学习率重置开关 (仅在 RESUME_FROM_CHECKPOINT 非空时生效)
# "false" (默认): 完整恢复 → optimizer/scheduler/step/RNG 全部接上次 ckpt 继续,
#                 学习率按 ckpt 当时的 step 自动推算 (例如 step=300000 → cosine 已衰减大半).
# "true"        : 仅加载模型权重, 强制 step=0 / 全新 optimizer / 全新 scheduler,
#                 学习率从 0 重新走 warmup → decay (warmup_step_decay 也会从头计算).
#                 适用场景:
#                   - 想用某个 ckpt 当作"预训练初始化", 重新走完整的 warmup+cosine
#                   - 切换数据集 / loss / lr 调度 后继续 fine-tune
# 注意: "true" 时也意味着 wandb step 会重新从 0 开始 (新的 run / 新的 step 曲线).
# ----------------------------------------------------------------------------
RESUME_RESET_LR="false"

# 每 N 个 epoch 保存一次 checkpoint (设置很大的值=禁用)
SAVE_EVERY=9999999

# 每 N 个 step 保存一次 checkpoint (0=禁用)
SAVE_EVERY_STEPS=20000

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
USE_WEBDATASET="false"

# WebDataset 分片路径模式 (仅在 USE_WEBDATASET="true" 时生效)
# 使用 brace expansion 语法匹配多个分片文件:
#   {000000..000012} 会展开为 000000, 000001, ..., 000012
#
# 示例 (假设有 13 个分片，编号 000000 到 000012):
#   WEBDATASET_SHARD_PATTERN="/data/.../webdataset_shard5000_ac25/shard-{000000..000012}.tar"
#
# 分布式训练时，各 GPU 会自动分配不同的分片读取
WEBDATASET_SHARD_PATTERN="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_libero_spatial/libero_lerobot_spatial_sys1_qwen_2b_14/webdataset_shard5000_ac16/shard-{000000..000010}.tar"

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

# 是否使用共享内存缓存 (推荐用于多 GPU 训练)
# true: rank 0 预加载所有数据到共享内存，其他 GPU 零拷贝访问
# 自动检测 VLM hidden states 的形状（支持可变 seq_len）
# false: 使用 mmap 模式或不缓存
# 注意: 启用 USE_CHUNK_BATCH_CACHE 时此项会被忽略
USE_SHARED_CACHE="true"

# ============================================================================
# Chunk 分批共享内存缓存 (ChunkBatchCache, 推荐用于"内存装不下整个 vlm_hidden_states" 场景)
#
# 工作原理:
#   1. 启动时扫描 vlm_hidden_states/chunk-*.npz 与 meta/episodes.jsonl
#   2. 用 min(可用 RAM, /dev/shm 可用) × CHUNK_BATCH_SAFETY_RATIO 作为单批内存预算，
#      自动决定每批能装多少个 chunk-XXX.npz；
#   3. 把 chunks 分成 N 批，每个 epoch 重新 shuffle (seed+epoch)；
#   4. 训练时按批切换：rank 0 释放上批 SHM → 加载下批 chunks → 广播 → 其他 rank attach；
#   5. 批内由 DistributedSampler.set_epoch(epoch * N + batch_idx) 完成 sample 级 shuffle。
#
# 优势:
#   - 无需把数据转换成 WebDataset；直接吃 lerobot chunk-XXX.npz
#   - RAM 不够装下整个数据集时仍可使用 SHM 加速 + 跨批 shuffle
#   - 单批可装下时退化为 ~ 等价于 USE_SHARED_CACHE 的速度
#
# 注意: USE_CHUNK_BATCH_CACHE="true" 会自动禁用 USE_WEBDATASET_CACHED / USE_WEBDATASET / USE_SHARED_CACHE
# 同时也会被 USE_DIRECT_DISK_SHUFFLE="true" 覆盖关闭 (见下方 Disk-Shuffle 节)
# ============================================================================
USE_CHUNK_BATCH_CACHE="false"

# ============================================================================
# Disk-Shuffle 模式 (USE_DIRECT_DISK_SHUFFLE) —— 无 SHM, 全局 shuffle
# ----------------------------------------------------------------------------
# 设计目的:
#   ChunkBatchCache 模式下, 数据被切成多段依次进 SHM, 每段切换瞬间 loss 会
#   出现明显突刺 (sample 分布瞬变)。Disk-Shuffle 模式彻底跳过 SHM 缓存,
#   训练时 DataLoader worker 用 mmap 按需读单帧 hidden state, 让
#   DistributedSampler 在**整个数据集**上做全局 shuffle, 段切换问题消失.
#
# 原理:
#   1) 启动时扫描所有 chunk-XXX.npz 的中央目录, 建 vlm_index_local → (chunk, episode, frame) 的映射;
#   2) 每个 DataLoader worker lazy-mmap 自己第一次访问的 chunk 文件 (LRU 限句柄数);
#   3) 训练时每个 sample 都是一次小随机读 (单帧 hidden state, 一般 0.5-4 KB).
#
# !!!!!! HDD 警告 !!!!!!
#   本模式对存储**随机 IO 性能**极度敏感. NVMe SSD 实测能跑出数千 sample/s/worker,
#   HDD 上仅 30 sample/s/worker 量级, 大概率把训练拖到 1/100 速度.
#   建议: 先把数据搬到 NVMe (/data_disk2 或类似) 再开本开关. 启动时会自动探测,
#         若数据在 HDD 上会打印强烈警告. 想在 HDD 上"先小规模跑通验证" 可以接受.
#
# 互斥性: 开启此项时 USE_CHUNK_BATCH_CACHE / USE_WEBDATASET_CACHED / USE_WEBDATASET /
#         USE_SHARED_CACHE 会被自动关闭 (CLI 层兜底).
# ============================================================================
USE_DIRECT_DISK_SHUFFLE="false" # ! 到400000steps 重新做shuffle，这个之前都是false

# 每个 DataLoader worker 缓存的 npz mmap 句柄上限 (-1 表示不限, 推荐).
# 大数据集 chunk 数 >> 内存预算时再设正数 (例如 64).
DISK_SHUFFLE_NPZ_LRU_PER_WORKER=-1

# LeRobotDataset 内 parquet_cache 上限 (LRU 驱逐). 0 = 不限.
# disk-shuffle 模式下 worker 会触碰整个数据集的 parquet, 默认 4096 较稳.
DISK_SHUFFLE_PARQUET_CACHE_MAX=4096

# 内存预算安全系数（0~1）。预算 = min(RAM_avail, /dev/shm_avail) × ratio
# 推荐 0.45 ~ 0.65，越小越保守
# 实测 dataset196 + 1.6T SHM 下:
#   ratio=0.45 → 3 chunks/batch, 5 batches, 单批 ≤ 600 GiB（最稳，切批多）
#   ratio=0.55 → 4 chunks/batch, 4 batches, 单批 ≤ 800 GiB（推荐）
#   ratio=0.65 → 5 chunks/batch, 3 batches, 单批 ≤ 1000 GiB（更快但接近上限）
CHUNK_BATCH_SAFETY_RATIO=0.7

# 共享内存中存储的 VLM 数据类型 (float32 / float16 / bfloat16)
# float16 可减半 SHM 占用 → 一次能装下更多 chunk
CHUNK_BATCH_CACHE_DTYPE="float32"

# 每批最少 chunk 数（即使内存允许更少也至少这么多）
CHUNK_BATCH_MIN_CHUNKS=1

# 每批最多 chunk 数（-1 表示不限）
CHUNK_BATCH_MAX_CHUNKS=-1

# 直接指定每批 chunk 数（>0 时覆盖自动估算；-1 表示自动按内存预算估算）
# 适用于：你已经知道单 chunk 大约多大、想精确控制每批装多少
CHUNK_BATCH_MANUAL_CHUNKS=-1

# 估算每 chunk RAM 占用时的安全放大倍数
CHUNK_BATCH_INFLATION=1.05

# chunk 顺序的随机种子（每 epoch 实际用 seed+epoch shuffle）
CHUNK_BATCH_SEED=42

# ============================================================================
# Step-Segments 模式（推荐）—— 把整训按 step 切成 num_batches 段
# ----------------------------------------------------------------------------
# 旧逻辑：每个 epoch 内部都遍历完所有 ChunkBatch（每 epoch × num_batches 次磁盘 IO）。
#   STEPS=100000, num_batches=2 时 ≈ 387 次重新装载 chunk → SHM。
#
# 新逻辑（USE_STEP_BASED_SEGMENTS=true，默认开启）：
#   1) 整训用 chunk_batch_seed shuffle 一次后冻结 chunk 顺序；
#   2) num_segments = num_batches，每段长度 = ceil(STEPS / num_segments)；
#   3) 段开始时加载该段对应的 ChunkBatch；段内反复迭代 DataLoader 直到 step 跨过段尾；
#   4) 整训只触发 num_segments-1 次磁盘 IO（segment 0 在初始化时就装好了）。
#
# 例：STEPS=100000, num_batches=2 →
#       step 0~49999 用 ChunkBatch 0（已加载，0 次 IO）
#       step 50000~99999 用 ChunkBatch 1（仅 1 次 IO）
#
# 副作用：
#   - 跨段 chunk 不会被重新打乱（同段内 chunk 集合恒定），用户已用 train_sampler 段内
#     做 sample 级 shuffle 来弥补。
#   - resume 时由 step 推算回原段；首次 resume 会触发一次额外的段切换（约一次装载耗时）。
#
# 设为 "false" 可回退到旧逻辑。
# ============================================================================
USE_STEP_BASED_SEGMENTS="false"

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
CONVERT_QUAT_TO_AXISANGLE="false"

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
# 二值化阈值 (平均值 > threshold -> 1, 否则 -> -1)
HAND_BINARY_THRESHOLD=None

# 使用 minmax 归一化的列索引 (空格分隔)
# 从 stats.json 读取 observation.state 的 min/max
# 示例: gripper (0,1) + xyz (2,3,4) = "0 1 2 3 4"
STATE_NORM_COLUMNS_MINMAX=""

# ===== Action Polyfit 配置 =====
# 是否对 action chunk 进行多项式拟合
# true: 对每个 action chunk 进行多项式拟合后再训练
# false: 使用原始 action chunk 训练
USE_ACTION_POLYFIT="false"

# 多项式拟合阶数 (仅在 USE_ACTION_POLYFIT="true" 时生效)
# 常用值: 3 (三阶多项式), 4 (四阶多项式)
ACTION_POLYFIT_DEGREE=4

# ===== Action from State Diff 配置 =====
# 是否使用 state 差分替代 action
# true: 对 state chunk 进行多项式拟合后计算 state[t+1] - state[t] 替代原始 action
# false: 使用原始 action
USE_ACTION_FROM_STATE_DIFF="false"

# state 差分多项式拟合阶数 (仅在 USE_ACTION_FROM_STATE_DIFF="true" 时生效)
ACTION_FROM_STATE_DIFF_DEGREE=3

# 参与差分计算的 state 列索引 (与 ACTION_DIFF_TARGET_COLUMNS 一一对应)
# libero_plus: state[0:6] = eef pose (xyz + euler_rpy), state[6:8] = gripper qpos
STATE_DIFF_COLUMNS="0 1 2 3 4 5"

# 差分结果赋值的 action 列索引 (与 STATE_DIFF_COLUMNS 一一对应)
# libero_plus: action[0:6] = delta_xyz + delta_rpy, action[6] = gripper
ACTION_DIFF_TARGET_COLUMNS="0 1 2 3 4 5"

# 保持原始 action 值的列索引 (不被 state 差分替换)
# libero_plus: gripper 列 (index 6)
ACTION_KEEP_ORIGINAL_COLUMNS="6"

# ===== 混合 Loss 配置 =====
# 是否启用混合 loss (MAE + BCE)
# true: 指定列使用 BCEWithLogitsLoss，其余列使用 L1Loss
# false: 所有列统一使用 L1Loss
USE_MIXED_LOSS="false"

# 使用 BCE loss 的 action 列索引 (空格分隔)
# 适用于二值动作维度 (如夹爪: -1 或 1)
BCE_ACTION_COLUMNS="6"

# MAE loss 权重
MAE_LOSS_WEIGHT=1.0

# BCE loss 权重
BCE_LOSS_WEIGHT=1.0

# ===== Weights & Biases 配置 =====

# 是否使用 Wandb 记录训练
USE_WANDB="true"

# Wandb 项目名称(WANDB_PROJECT)
WANDB_PROJECT="unitree_v2_pretrain"

OUT_DIR="/data_disk2/hwl/checkpoints/no_pretrain_tele/action${ACTION_DIM}_chunk${NUM_ACTIONS_CHUNK}/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_layer${WHICH_LAYER}_${RUN_DATE}_wp${WANDB_PROJECT}_${DATASET_NAME}_CBC_${USE_CHUNK_BATCH_CACHE}_dtype${CHUNK_BATCH_CACHE_DTYPE}_USE_AMP_${USE_AMP}_MD_true/checkpoints"

LOG_DIR="/data_disk2/hwl/checkpoints/no_pretrain_tele/action${ACTION_DIM}_chunk${NUM_ACTIONS_CHUNK}/${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_layer${WHICH_LAYER}_${RUN_DATE}_wp${WANDB_PROJECT}_${DATASET_NAME}_CBC_${USE_CHUNK_BATCH_CACHE}_dtype${CHUNK_BATCH_CACHE_DTYPE}_USE_AMP_${USE_AMP}_MD_true/logs"

# Wandb 运行名称 (留空则自动生成)
WANDB_RUN_NAME="${MODEL_NAME}_layer${WHICH_LAYER}_${GPU_NAME}_bsz${BATCH_SIZE}*${GRADIENT_ACCUMULATION_STEPS}*${NUM_GPUS}_tb${NUM_TRANSFORMER_BLOCKS}_${STEPS}steps_oft1_${RUN_DATE}_${DATASET_NAME}_CBC_${USE_CHUNK_BATCH_CACHE}_dtype${CHUNK_BATCH_CACHE_DTYPE}_USE_AMP_${USE_AMP}_MD_true"

# ============================================================================
# Multi-Dataset (跨子集) 配置
# ----------------------------------------------------------------------------
# USE_MULTI_DATASET="true": 把 DATA_PATH 当 root 目录, 自动扫所有子集
# MULTI_DATASET_INCLUDE="" : 留空表示扫所有子目录; 非空时白名单生效, 例如:
#                            MULTI_DATASET_INCLUDE="set_a,set_b,set_c"
# MULTI_DATASET_EXCLUDE="" : 子集黑名单, 同样逗号分隔
# MULTI_DATASET_STATS_STRATEGY:
#     per_subset    每个子集自己的 minmax (推荐, 跨形态/相机鲁棒)
#     minmax_union  全局合并 minmax (相同机器人形态时再考虑)
# MULTI_DATASET_REQUIRE_COMPLETE_VLM:
#     "false" (默认): vlm 不完整子集只用"已生成的 chunk-XXX.npz" 部分加载训练。
#                     unitree_v2 实测下: kuka 用 158/581 chunks, language_table
#                     146/443, fractal 57/88, bc_z 8/44, furniture_bench 1/6 仍参训.
#     "true"        : 严格模式, 任一 chunk 缺失就整个子集 skip
# MULTI_DATASET_STRICT_STATE_DIM=false : 默认允许 state_dim 不一致, 自动右补 0
# MULTI_DATASET_DRY_RUN=true: 仅扫描+打印计划+写 manifest, 不真训练
# ============================================================================
USE_MULTI_DATASET="false"
MULTI_DATASET_INCLUDE=""
MULTI_DATASET_EXCLUDE=""
MULTI_DATASET_STATS_STRATEGY="per_subset"
MULTI_DATASET_REQUIRE_COMPLETE_VLM="false"
MULTI_DATASET_STRICT_STATE_DIM="false"
MULTI_DATASET_TARGET_ACTION_DIM=-1
MULTI_DATASET_SAVE_MANIFEST="${OUT_DIR}/../multi_dataset_manifest.json"
MULTI_DATASET_DRY_RUN="false"

# Wandb 基本日志记录频率 (每 N 步记录 loss, lr, batch_time 等)
WANDB_LOG_FREQ=1

# ===== 分布式训练环境变量 =====

# 解决 GPU 硬件问题导致的 NVML 错误
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # 按 PCI 总线顺序排列 GPU
export NCCL_NVLS_ENABLE=0                    # 禁用 NVLS (NVLink SHARP) 功能
export NCCL_P2P_DISABLE=1                    # 禁用 P2P，避免 NVML 拓扑检测访问坏的 GPU # 该设置会让 NCCL 使用较慢的通信方式，初始化时间可能会更长。
export NCCL_SHM_DISABLE=0                    # 启用共享内存通信 (替代 P2P)
export NCCL_IB_DISABLE=1                     # 禁用 InfiniBand # 该设置会让 NCCL 使用较慢的通信方式，初始化时间可能会更长。

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
# 注意: --rdzv_backend=c10d 时, --master_port 不会被 rendezvous server 使用,
# c10d 默认硬编码 29400. 必须显式 --rdzv_endpoint 才能换端口.
# 不显式指定会导致多个训练 job 抢占 29400, 出现 "EADDRINUSE / 连到旧 rdzv 卡住" 现象.
CMD="CUDA_VISIBLE_DEVICES=${GPU_IDS} torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} --rdzv_backend=c10d --rdzv_endpoint=localhost:${MASTER_PORT} --rdzv_conf timeout=7200 ${TRAIN_SCRIPT}"

# 数据相关参数
CMD="${CMD} --data_path ${DATA_PATH}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --num_workers ${NUM_WORKERS}"
CMD="${CMD} --prefetch_factor ${PREFETCH_FACTOR}"
CMD="${CMD} --val_split ${VAL_SPLIT}"

# Multi-Dataset (跨子集) 参数
if [ "${USE_MULTI_DATASET}" = "true" ]; then
    CMD="${CMD} --use_multi_dataset"
    CMD="${CMD} --multi_dataset_root ${DATA_PATH}"
    if [ -n "${MULTI_DATASET_INCLUDE}" ]; then
        CMD="${CMD} --multi_dataset_include ${MULTI_DATASET_INCLUDE}"
    fi
    if [ -n "${MULTI_DATASET_EXCLUDE}" ]; then
        CMD="${CMD} --multi_dataset_exclude ${MULTI_DATASET_EXCLUDE}"
    fi
    CMD="${CMD} --multi_dataset_stats_strategy ${MULTI_DATASET_STATS_STRATEGY}"
    if [ "${MULTI_DATASET_REQUIRE_COMPLETE_VLM}" = "true" ]; then
        CMD="${CMD} --multi_dataset_require_complete_vlm"
    else
        CMD="${CMD} --multi_dataset_no_require_complete_vlm"
    fi
    if [ "${MULTI_DATASET_STRICT_STATE_DIM}" = "true" ]; then
        CMD="${CMD} --multi_dataset_strict_state_dim"
    fi
    if [ -n "${MULTI_DATASET_TARGET_ACTION_DIM}" ] && [ "${MULTI_DATASET_TARGET_ACTION_DIM}" != "-1" ]; then
        CMD="${CMD} --multi_dataset_target_action_dim ${MULTI_DATASET_TARGET_ACTION_DIM}"
    fi
    if [ -n "${MULTI_DATASET_SAVE_MANIFEST}" ]; then
        CMD="${CMD} --multi_dataset_save_manifest \"${MULTI_DATASET_SAVE_MANIFEST}\""
    fi
    if [ "${MULTI_DATASET_DRY_RUN}" = "true" ]; then
        CMD="${CMD} --multi_dataset_dry_run"
    fi
fi

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

# 数据加载模式优先级 (高 → 低):
#   DiskShuffle (无 SHM, 全局 shuffle, NVMe 才好用) >
#   ChunkBatchCache > WebDataset 分批缓存 > WebDataset > 共享内存缓存 > mmap
if [ "${USE_DIRECT_DISK_SHUFFLE}" = "true" ]; then
    # Disk-Shuffle 模式: 跳过 SHM, 训练时 mmap 按需读单帧.
    # 注意: train_multigpu.py 会再次兜底关闭其它 cache, 即使用户同时把
    #       USE_CHUNK_BATCH_CACHE / USE_SHARED_CACHE 设成 true 也安全.
    CMD="${CMD} --use_disk_shuffle"
    CMD="${CMD} --disk_shuffle_npz_lru_per_worker ${DISK_SHUFFLE_NPZ_LRU_PER_WORKER}"
    CMD="${CMD} --disk_shuffle_parquet_cache_max ${DISK_SHUFFLE_PARQUET_CACHE_MAX}"
elif [ "${USE_CHUNK_BATCH_CACHE}" = "true" ]; then
    # ChunkBatchCache 模式（lerobot 原生分批共享内存）
    CMD="${CMD} --use_chunk_batch_cache"
    CMD="${CMD} --chunk_batch_safety_ratio ${CHUNK_BATCH_SAFETY_RATIO}"
    CMD="${CMD} --chunk_batch_cache_dtype ${CHUNK_BATCH_CACHE_DTYPE}"
    CMD="${CMD} --chunk_batch_min_chunks ${CHUNK_BATCH_MIN_CHUNKS}"
    CMD="${CMD} --chunk_batch_max_chunks ${CHUNK_BATCH_MAX_CHUNKS}"
    CMD="${CMD} --chunk_batch_manual_chunks ${CHUNK_BATCH_MANUAL_CHUNKS}"
    CMD="${CMD} --chunk_batch_inflation ${CHUNK_BATCH_INFLATION}"
    CMD="${CMD} --chunk_batch_seed ${CHUNK_BATCH_SEED}"
    if [ "${USE_STEP_BASED_SEGMENTS}" = "true" ]; then
        CMD="${CMD} --use_step_based_segments"
    else
        CMD="${CMD} --no_step_based_segments"
    fi
elif [ "${USE_WEBDATASET_CACHED}" = "true" ] && [ -n "${WEBDATASET_SHARD_PATTERN}" ]; then
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

# State 预处理参数 (独立于 State Mapper，用于数据加载时的预处理)
if [ -n "${STATE_PROCESS_ORDER}" ]; then
    CMD="${CMD} --state_process_order ${STATE_PROCESS_ORDER}"
fi
if [ -n "${HAND_BINARY_COLUMNS}" ]; then
    CMD="${CMD} --hand_binary_columns ${HAND_BINARY_COLUMNS}"
    CMD="${CMD} --hand_binary_threshold ${HAND_BINARY_THRESHOLD}"
fi
# State 归一化配置
if [ -n "${STATE_NORM_COLUMNS_MINMAX}" ]; then
    CMD="${CMD} --state_norm_columns_minmax ${STATE_NORM_COLUMNS_MINMAX}"
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

# 混合 Loss 配置
if [ "${USE_MIXED_LOSS}" = "true" ]; then
    CMD="${CMD} --use_mixed_loss"
    if [ -n "${BCE_ACTION_COLUMNS}" ]; then
        CMD="${CMD} --bce_action_columns ${BCE_ACTION_COLUMNS}"
    fi
    CMD="${CMD} --mae_loss_weight ${MAE_LOSS_WEIGHT}"
    CMD="${CMD} --bce_loss_weight ${BCE_LOSS_WEIGHT}"
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

# 断点续训
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    CMD="${CMD} --resume_from_checkpoint \"${RESUME_FROM_CHECKPOINT}\""
    if [ "${RESUME_RESET_LR}" = "true" ]; then
        CMD="${CMD} --resume_reset_lr"
    fi
fi

# ===== 打印配置信息 =====

echo "============================================================================"
echo "🤖 OFT1_0 Action Head - Qwen VL 训练"
echo "============================================================================"
echo ""
echo "📁 路径配置:"
echo "  数据集 root: ${DATA_PATH}"
echo "  输出目录: ${OUT_DIR}"
echo "  日志目录: ${LOG_DIR}"
echo "  Run 日期: ${RUN_DATE}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "  🔁 Resume from: ${RESUME_FROM_CHECKPOINT}"
    if [ "${RESUME_RESET_LR}" = "true" ]; then
        echo "  🔁 Reset LR:    true (仅加载权重, step/optim/scheduler/RNG 全部从 0 开始)"
    else
        echo "  🔁 Reset LR:    false (完整恢复 optim/scheduler/step)"
    fi
else
    echo "  🆕 从头训练 (未指定 RESUME_FROM_CHECKPOINT)"
fi
echo ""
if [ "${USE_MULTI_DATASET}" = "true" ]; then
    echo "🧩 Multi-Dataset 配置:"
    echo "  Root:                ${DATA_PATH}"
    echo "  Include:             ${MULTI_DATASET_INCLUDE:-(全部子目录)}"
    echo "  Exclude:             ${MULTI_DATASET_EXCLUDE:-(无)}"
    echo "  Stats Strategy:      ${MULTI_DATASET_STATS_STRATEGY}"
    echo "  Require Complete VLM:${MULTI_DATASET_REQUIRE_COMPLETE_VLM}"
    echo "  Strict State Dim:    ${MULTI_DATASET_STRICT_STATE_DIM}"
    echo "  Target Action Dim:   ${MULTI_DATASET_TARGET_ACTION_DIM} (-1=auto)"
    echo "  Save Manifest:       ${MULTI_DATASET_SAVE_MANIFEST:-(不保存)}"
    echo "  Dry Run:             ${MULTI_DATASET_DRY_RUN}"
    echo ""
fi
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
echo "🧠 OFT 模型配置:"
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
echo "🔄 State 配置:"
echo "  MinMax Norm Columns: ${STATE_NORM_COLUMNS_MINMAX}"
if [ -n "${STATE_PROCESS_ORDER}" ]; then
    echo "  State Preprocessing Order: ${STATE_PROCESS_ORDER}"
    if [ -n "${HAND_BINARY_COLUMNS}" ]; then
        echo "    hand_binary: columns=${HAND_BINARY_COLUMNS}, threshold=${HAND_BINARY_THRESHOLD}"
    fi
fi
echo ""
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
if [ "${USE_MIXED_LOSS}" = "true" ]; then
    echo "⚖️ 混合 Loss 配置:"
    echo "  Enable Mixed Loss: ${USE_MIXED_LOSS}"
    echo "  BCE Action Columns: ${BCE_ACTION_COLUMNS}"
    echo "  MAE Loss Weight: ${MAE_LOSS_WEIGHT}"
    echo "  BCE Loss Weight: ${BCE_LOSS_WEIGHT}"
    echo ""
fi
echo "📦 数据加载优化:"
if [ "${USE_CHUNK_BATCH_CACHE}" = "true" ]; then
    echo "  模式: ChunkBatchCache (lerobot chunk-XXX.npz 分批共享内存)"
    echo "  Safety Ratio: ${CHUNK_BATCH_SAFETY_RATIO}"
    echo "  Cache Dtype: ${CHUNK_BATCH_CACHE_DTYPE}"
    echo "  Min Chunks/Batch: ${CHUNK_BATCH_MIN_CHUNKS}"
    echo "  Max Chunks/Batch: ${CHUNK_BATCH_MAX_CHUNKS}"
    echo "  Manual Chunks/Batch: ${CHUNK_BATCH_MANUAL_CHUNKS} (-1=auto)"
    echo "  RAM Inflation: ${CHUNK_BATCH_INFLATION}"
    echo "  Seed: ${CHUNK_BATCH_SEED}"
    echo "  Step-Segments: ${USE_STEP_BASED_SEGMENTS}  ← true=按 step 分段加载（整训仅切批 num_batches-1 次）"
elif [ "${USE_WEBDATASET_CACHED}" = "true" ] && [ -n "${WEBDATASET_SHARD_PATTERN}" ]; then
    echo "  模式: WebDataset 分批缓存 (分片加载到内存)"
    echo "  分片路径: ${WEBDATASET_SHARD_PATTERN}"
    echo "  每批分片数: ${WEBDATASET_CACHE_SHARDS}"
    echo "  缓存 Dtype: ${WEBDATASET_CACHE_DTYPE}"
elif [ "${USE_WEBDATASET}" = "true" ] && [ -n "${WEBDATASET_SHARD_PATTERN}" ]; then
    echo "  模式: WebDataset (超大规模数据优化)"
    echo "  分片路径: ${WEBDATASET_SHARD_PATTERN}"
    echo "  Shuffle Buffer: ${WEBDATASET_SHUFFLE_BUFFER}"
elif [ "${USE_SHARED_CACHE}" = "true" ]; then
    echo "  模式: 共享内存缓存 (多 GPU 共享)"
    echo "  缓存 Dtype: ${CACHE_DTYPE}"
elif [ "${CACHE_VLM_STATES}" = "true" ]; then
    echo "  模式: mmap (OS 页缓存)"
else
    echo "  模式: 无缓存 (每次从磁盘读取)"
fi
echo "  Num Workers: ${NUM_WORKERS}"
echo "  Prefetch Factor: ${PREFETCH_FACTOR}"
echo ""
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
