#!/usr/bin/env bash
# ============================================================================
# Sai0_1 OFT1_0 - LIBERO-plus 多 GPU 数据并行评估
# ============================================================================
#
# 工作模式:
#   - 在 GPU_IDS 指定的 N 张卡上各起一个 python 进程, 每个进程独立加载完整模型
#   - target_task_ids 用 stride 方式切分给 N 张卡
#       GPU_IDS[0]: task_ids[0::N]
#       GPU_IDS[1]: task_ids[1::N]
#       ...
#       GPU_IDS[N-1]: task_ids[N-1::N]
#     stride 切分让每张卡的 category / difficulty 分布大致相同, 跑完时间接近.
#   - 每张卡输出独立 JSON 到 ${VIDEO_DIR}/eval_results_<suite>_shard{K}_of_{N}.json
#   - 全部跑完后, 自动调用 merge_shard_results 合并出总 JSON
#       ${VIDEO_DIR}/eval_results_<suite>_merged.json
#   - 视频和日志一起落到 ${VIDEO_DIR}/ 下
#
# 使用前置条件: 与 run_eval_libero_plus.sh 相同, 详见 README.md
#
# 用法:
#   # 用 GPU 0-7 跑 libero_spatial (默认)
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh
#
#   # 用 GPU 2,3,5,6 跑 libero_object
#   GPU_IDS="2,3,5,6" TASK_SUITE_NAME=libero_object \
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh
#
#   # 调试: 用 2 张卡各跑前 4 个 task (合计 8 个)
#   GPU_IDS="0,1" MAX_TASKS=8 \
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh
# ============================================================================

set -e

# ===== Conda 环境 =====
CONDA_ENV="${CONDA_ENV:-qwen_eagle_hwl}"

# ===== GPU 配置 =====
# 逗号分隔的物理 GPU id, 每个 id 启一个独立 python 进程
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_SHARDS=${#GPU_ID_ARRAY[@]}

# ===== 模型 / 数据集 路径 (按需修改) =====
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data_disk2/hwl/checkpoints/libero_plus_lerobot/action7_chunk16_pretrain/p6000_bsz500*1*8_tb4_200000steps_layer14_20260526_wplibero_plus_libero_plus_CBC_true_dtypefloat32_USE_AMP_true/checkpoints/step_200000/action_head.pt}"
DATASET_PATH="${DATASET_PATH:-/data_disk2/hwl/datasets/libero_plus_lerobot}"

# ===== VLM =====
VLM_TYPE="${VLM_TYPE:-qwen3_vl}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
VLM_LAYERS="${VLM_LAYERS:-14}"
VLM_OUTPUT_DIM="${VLM_OUTPUT_DIM:-2048}"

# ===== Prompt =====
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

# ===== LIBERO-plus =====
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_spatial}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
TASK_IDS="${TASK_IDS:-}"
MAX_TASKS="${MAX_TASKS:--1}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
MAX_STEPS="${MAX_STEPS:-600}"
ENV_SEED="${ENV_SEED:-}"
CATEGORIES="${CATEGORIES:-}"
DIFFICULTY_LEVELS="${DIFFICULTY_LEVELS:-}"

# ===== 推理 =====
EXECUTE_ALL_CHUNKS="${EXECUTE_ALL_CHUNKS:-true}"
ACTION_CHUNK_SIZE="${ACTION_CHUNK_SIZE:-16}"

# ===== 系统 / 视频 / 实时 =====
# FLIP_IMAGES: 是否把 LIBERO env 直出的图像 (OpenGL 颠倒方向) 翻 180° 后再给 VLM.
#   训练时模型看到的是"正向"图像 (LeRobot 数据集方向), LIBERO env 的 raw obs 是颠倒的,
#   所以推理时必须做这个翻转才能跟训练对齐.
FLIP_IMAGES="${FLIP_IMAGES:-true}"
# VIDEO_FLIP: 视频保存时是否翻转 (仅影响 mp4 展示, 不影响给模型的输入).
VIDEO_FLIP="${VIDEO_FLIP:-true}"
SAVE_VIDEOS="${SAVE_VIDEOS:-true}"
# SAVE_VIDEO_EVERY=1 表示每个 task 都保存视频 (按 trial 0 收帧).
# 想抽样减少磁盘占用可改大, 比如 50.
SAVE_VIDEO_EVERY="${SAVE_VIDEO_EVERY:-1}"
VERBOSE="${VERBOSE:-false}"
RESUME="${RESUME:-true}"
PRINT_PER_TASK="${PRINT_PER_TASK:-true}"
SUMMARY_EVERY="${SUMMARY_EVERY:-100}"

# ===== Env vars 必须在 import Action_Heads.OFT1_0 之前生效 =====
export LIBERO_PLUS_ROOT="${LIBERO_PLUS_ROOT:-/data_disk1/hwl/LIBERO-plus}"
export LIBERO_PLUS_CONFIG_DIR="${LIBERO_PLUS_CONFIG_DIR:-$HOME/.libero_plus_sai0}"
export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_DIR}"
export ROBOSUITE_LOG="${HOME}/.robosuite/robosuite.log"
mkdir -p "${HOME}/.robosuite"

export ACTION_DIM="${ACTION_DIM}"
export NUM_ACTIONS_CHUNK="${NUM_ACTION_CHUNKS}"
export PROPRIO_DIM="${PROPRIO_DIM:-8}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# 多进程竞争 OMP/MKL 时把每个 worker 的线程数压低, 防止 CPU 抖
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# ===== 输出路径 =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

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

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${VLM_TYPE}_${VLM_LAYERS}_${VLM_OUTPUT_DIM}_${ACT_TAG}_${TASK_SUITE_NAME}_${STEP_TAG}_x${NUM_SHARDS}gpu_$(date +%Y%m%d_%H%M%S)}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/experiments}"
VIDEO_DIR="${VIDEO_DIR:-${EXPERIMENTS_ROOT}/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${VIDEO_DIR}}"
mkdir -p "${VIDEO_DIR}"

# ===== 打印配置 =====
echo "============================================================================"
echo "🤖 Sai0_1 OFT1_0 - LIBERO-plus 多 GPU 评估"
echo "============================================================================"
echo "Conda env       : ${CONDA_ENV}"
echo "GPU IDs         : ${GPU_IDS} (共 ${NUM_SHARDS} 张卡)"
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

# ===== 第 1 步: 确保 LIBERO-plus 环境就绪 (单次, 不并发) =====
echo ""
echo "🛠️  Step 1/3: 确保 LIBERO-plus 环境配置好"
cd "${PROJECT_ROOT}"
"$(conda info --base)/envs/${CONDA_ENV}/bin/python" -m eval.Sai0_1.libero_plus.OFT1_0.setup_libero_plus \
    --libero_plus_root "${LIBERO_PLUS_ROOT}" \
    --config_dir "${LIBERO_PLUS_CONFIG_DIR}"

# ===== 第 2 步: 启动 N 个 worker, 每个 worker 一张 GPU =====
echo ""
echo "🚀 Step 2/3: 同时启动 ${NUM_SHARDS} 个 GPU worker"

PIDS=()
for SHARD_INDEX in "${!GPU_ID_ARRAY[@]}"; do
    GPU_ID="${GPU_ID_ARRAY[${SHARD_INDEX}]}"
    WORKER_LOG="${VIDEO_DIR}/worker_shard${SHARD_INDEX}_of_${NUM_SHARDS}_gpu${GPU_ID}.log"

    # 构建 worker 命令
    CMD="\"$(conda info --base)/envs/${CONDA_ENV}/bin/python\" -u -m eval.Sai0_1.libero_plus.OFT1_0.eval_libero_plus"
    CMD="${CMD} --checkpoint_path \"${CHECKPOINT_PATH}\""
    CMD="${CMD} --vlm_model_path \"${VLM_MODEL_PATH}\""
    CMD="${CMD} --vlm_type ${VLM_TYPE}"
    CMD="${CMD} --vlm_layers ${VLM_LAYERS}"
    CMD="${CMD} --vlm_output_dim ${VLM_OUTPUT_DIM}"
    CMD="${CMD} --content_order ${CONTENT_ORDER}"
    CMD="${CMD} --dataset_path \"${DATASET_PATH}\""
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
    CMD="${CMD} --device cuda:0"
    CMD="${CMD} --video_dir \"${VIDEO_DIR}\""
    CMD="${CMD} --log_dir \"${LOG_DIR}\""
    CMD="${CMD} --save_video_every ${SAVE_VIDEO_EVERY}"
    CMD="${CMD} --shard_index ${SHARD_INDEX}"
    CMD="${CMD} --num_shards ${NUM_SHARDS}"
    CMD="${CMD} --summary_every ${SUMMARY_EVERY}"

    [[ -n "${TASK_IDS}" ]] && CMD="${CMD} --task_ids \"${TASK_IDS}\""
    [[ -n "${CATEGORIES}" ]] && CMD="${CMD} --categories \"${CATEGORIES}\""
    [[ -n "${DIFFICULTY_LEVELS}" ]] && CMD="${CMD} --difficulty_levels \"${DIFFICULTY_LEVELS}\""
    [[ -n "${ENV_SEED}" ]] && CMD="${CMD} --env_seed ${ENV_SEED}"

    [[ "${FLIP_IMAGES}" == "true" ]] && CMD="${CMD} --flip_images" || CMD="${CMD} --no_flip_images"
    [[ "${VIDEO_FLIP}" == "true" ]] && CMD="${CMD} --video_flip" || CMD="${CMD} --no_video_flip"
    [[ "${LOWERCASE_INSTRUCTION}" == "true" ]] && CMD="${CMD} --lowercase_instruction" || CMD="${CMD} --no_lowercase_instruction"
    [[ "${ADD_GENERATION_PROMPT}" == "true" ]] && CMD="${CMD} --add_generation_prompt" || CMD="${CMD} --no_generation_prompt"
    [[ "${ADD_ACTION_PROMPT}" == "true" ]] && CMD="${CMD} --add_action_prompt" || CMD="${CMD} --no_action_prompt"
    [[ "${EXECUTE_ALL_CHUNKS}" == "true" ]] && CMD="${CMD} --execute_all_chunks" || CMD="${CMD} --no_execute_all_chunks"
    [[ "${SAVE_VIDEOS}" == "true" ]] && CMD="${CMD} --save_videos"
    [[ "${VERBOSE}" == "true" ]] && CMD="${CMD} --verbose"
    [[ "${PRINT_PER_TASK}" == "true" ]] && CMD="${CMD} --print_per_task" || CMD="${CMD} --no_print_per_task"
    [[ "${RESUME}" == "true" ]] && CMD="${CMD} --resume" || CMD="${CMD} --no_resume"

    echo "  ▶ shard ${SHARD_INDEX} 在 GPU ${GPU_ID}, 日志: ${WORKER_LOG}"

    # 把启动命令也写到日志开头, 方便重跑
    {
        echo "================ WORKER SHARD ${SHARD_INDEX}/${NUM_SHARDS} (GPU ${GPU_ID}) ================"
        echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "CUDA_VISIBLE_DEVICES=${GPU_ID}"
        echo "CMD: ${CMD}"
        echo "============================================================================"
    } > "${WORKER_LOG}"

    # 用 bash -c "..." 在子 shell 中执行命令, 显式把 stdout/stderr 重定向到 worker log,
    # 避免 launcher 自己被 nohup 重定向时 fd 串台导致 worker 输出被吞.
    # 内部先 exec 把 stdout/stderr 重新绑到 log file, 然后 export CUDA_VISIBLE_DEVICES,
    # 最后用 exec 启动真正的 python, 这样 PID 是真正 python 的 pid, 一切输出都进 log.
    # 注意: exec 是 shell builtin, 不能像 simple command 一样用 'VAR=val' 前缀赋值,
    # 因此必须用单独的 export 语句把 CUDA_VISIBLE_DEVICES 推到子进程 env.
    bash -c "exec >> '${WORKER_LOG}' 2>&1; export CUDA_VISIBLE_DEVICES=${GPU_ID}; exec ${CMD}" &
    WORKER_PID=$!
    PIDS+=("${WORKER_PID}")

    echo "    └─ PID = ${WORKER_PID}" | tee -a "${WORKER_LOG}"

    # 稍微错开启动, 避免 8 个 worker 同时打满磁盘 IO 加载 VLM 权重
    sleep 3
done

echo ""
echo "  → 已启动 ${NUM_SHARDS} 个 worker, PIDs: ${PIDS[*]}"
echo "  → 实时查看任意一个 worker:  tail -f ${VIDEO_DIR}/worker_shard0_of_${NUM_SHARDS}_gpu${GPU_ID_ARRAY[0]}.log"
echo "  → 实时聚合所有 worker:      tail -f ${VIDEO_DIR}/worker_shard*.log"
echo ""

# ===== 启动后台聚合监控: 每 ${MONITOR_EVERY_SEC} 秒读所有 shard JSON 汇总进度 =====
MONITOR_EVERY_SEC="${MONITOR_EVERY_SEC:-60}"
MONITOR_PID=""
if [[ "${MONITOR_EVERY_SEC}" -gt 0 ]]; then
    (
        while true; do
            sleep "${MONITOR_EVERY_SEC}"
            "$(conda info --base)/envs/${CONDA_ENV}/bin/python" - <<PYEOF || true
import json, glob, os, time
shards = sorted(glob.glob("${VIDEO_DIR}/eval_results_${TASK_SUITE_NAME}_shard*_of_${NUM_SHARDS}.json"))
if not shards:
    raise SystemExit
total_ep = total_succ = 0
shard_summary = []
for p in shards:
    try:
        with open(p) as f:
            d = json.load(f)
        m = d.get("metrics", {}).get("overall", {})
        ep = m.get("episodes", 0); succ = m.get("successes", 0)
        total_ep += ep; total_succ += succ
        sname = os.path.basename(p).replace("eval_results_${TASK_SUITE_NAME}_", "").replace(".json", "")
        sr = (succ / ep * 100) if ep else 0.0
        shard_summary.append(f"{sname}={succ}/{ep} ({sr:.1f}%)")
    except Exception:
        pass
overall_sr = (total_succ / total_ep * 100) if total_ep else 0.0
print(f"\n[$(echo $(date '+%H:%M:%S'))] 📡 全局进度: {total_succ}/{total_ep} = {overall_sr:.2f}% | " + " | ".join(shard_summary), flush=True)
PYEOF
        done
    ) &
    MONITOR_PID=$!
    echo "  → 全局进度聚合监控已启动 (每 ${MONITOR_EVERY_SEC}s 一次, PID ${MONITOR_PID})"
fi

echo ""
echo "⏳ 等待全部 worker 完成 ..."

# ===== 等待全部 worker 完成 =====
ANY_FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[${i}]}"
    if wait "${PID}"; then
        echo "  ✅ shard ${i} (pid ${PID}) 完成"
    else
        EXIT_CODE=$?
        echo "  ❌ shard ${i} (pid ${PID}) 退出码 ${EXIT_CODE}"
        ANY_FAILED=1
    fi
done

# 关掉聚合监控
if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
fi

# ===== 第 3 步: merge =====
echo ""
echo "📦 Step 3/3: 合并所有 shard 的结果"

MERGE_OUT="${VIDEO_DIR}/eval_results_${TASK_SUITE_NAME}_merged.json"
MERGE_LOG="${VIDEO_DIR}/merge.log"

if [[ ${ANY_FAILED} -eq 1 ]]; then
    echo "  ⚠️ 有 worker 异常退出, 用 --allow_partial 合并仍可用的 shard"
    "$(conda info --base)/envs/${CONDA_ENV}/bin/python" \
        -m eval.Sai0_1.libero_plus.OFT1_0.merge_shard_results \
        --video_dir "${VIDEO_DIR}" \
        --task_suite_name "${TASK_SUITE_NAME}" \
        --num_shards "${NUM_SHARDS}" \
        --output "${MERGE_OUT}" \
        --allow_partial 2>&1 | tee "${MERGE_LOG}"
else
    "$(conda info --base)/envs/${CONDA_ENV}/bin/python" \
        -m eval.Sai0_1.libero_plus.OFT1_0.merge_shard_results \
        --video_dir "${VIDEO_DIR}" \
        --task_suite_name "${TASK_SUITE_NAME}" \
        --num_shards "${NUM_SHARDS}" \
        --output "${MERGE_OUT}" 2>&1 | tee "${MERGE_LOG}"
fi

echo ""
echo "============================================================================"
echo "🎉 评估完成"
echo "  - 合并结果   : ${MERGE_OUT}"
echo "  - 各 shard   : ${VIDEO_DIR}/eval_results_${TASK_SUITE_NAME}_shard*_of_${NUM_SHARDS}.json"
echo "  - Worker 日志: ${VIDEO_DIR}/worker_shard*.log"
echo "  - 视频       : ${VIDEO_DIR}/task_*/<日期>/*.mp4"
echo "============================================================================"

if [[ ${ANY_FAILED} -eq 1 ]]; then
    exit 1
fi
