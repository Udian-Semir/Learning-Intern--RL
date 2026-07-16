#!/usr/bin/env bash
# ============================================================================
# Sai0_1 OFT1_0 - LIBERO-plus 4 个 task suite 全套串行评估
# ============================================================================
#
# 串行调用 run_eval_libero_plus_multi_gpu.sh 跑完 4 个 suite:
#   libero_spatial -> libero_object -> libero_goal -> libero_10
# 共 ~10,030 个 task. 8 GPU 估计 ~3.5h (训练抢资源会更慢).
#
# 每个 suite 跑完会有独立的 experiment 目录:
#   experiments/<vlm>_<step>_x8gpu_<时间戳>_<suite>/
# 内含 worker_shard*.log / eval_results_<suite>_shard*.json /
#       eval_results_<suite>_merged.json / task_*/<日期>/*.mp4
#
# 用法:
#   # 默认 4 个 suite, 8 GPU, 每个 task 都存视频
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_all_suites_multi_gpu.sh
#
#   # 换 checkpoint
#   CHECKPOINT_PATH='/data_disk2/.../step_180000/action_head.pt' \
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_all_suites_multi_gpu.sh
#
#   # 只跑指定的几个 suite
#   SUITES="libero_object libero_10" \
#   bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_all_suites_multi_gpu.sh
#
#   # 推荐配合 nohup, 避免 SSH 断开任务挂掉
#   nohup bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_all_suites_multi_gpu.sh \
#       > /tmp/libero_plus_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   disown
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_LAUNCHER="${SCRIPT_DIR}/run_eval_libero_plus_multi_gpu.sh"

# ===== 实验目录 (绝对路径, 透传给 launcher 用于 VIDEO_DIR) =====
export EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/experiments}"

# ===== 默认 4 个 suite, 想只跑部分 suite 用 SUITES env 覆盖 =====
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

# ===== 公共配置 (可以被外部 env 覆盖, 然后透传给 launcher) =====
export NUM_SHARDS="${NUM_SHARDS:-8}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export SAVE_VIDEO_EVERY="${SAVE_VIDEO_EVERY:-1}"
export RESUME="${RESUME:-true}"
export MONITOR_EVERY_SEC="${MONITOR_EVERY_SEC:-120}"

# 默认 step_80000, 用户可以覆盖到任何 step (路径里若含 *, 必须用单引号)
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data_disk2/hwl/checkpoints/libero_plus_lerobot/action7_chunk16_pretrain/p6000_bsz500*1*8_tb4_200000steps_layer14_20260526_wplibero_plus_libero_plus_CBC_true_dtypefloat32_USE_AMP_true/checkpoints/step_200000/action_head.pt}"

START_TS=$(date +%s)
START_HUMAN=$(date '+%Y-%m-%d %H:%M:%S')

echo "============================================================================"
echo "  LIBERO-plus 全套评估 (串行)"
echo "============================================================================"
echo "  开始时间        : ${START_HUMAN}"
echo "  Checkpoint      : ${CHECKPOINT_PATH}"
echo "  GPU IDs         : ${GPU_IDS}  (NUM_SHARDS=${NUM_SHARDS})"
echo "  SAVE_VIDEO_EVERY: ${SAVE_VIDEO_EVERY}"
echo "  Resume          : ${RESUME}"
echo "  Experiments root: ${EXPERIMENTS_ROOT}"
echo "  Suites          : ${SUITES}"
echo "============================================================================"
echo ""

declare -A SUITE_EXIT
declare -A SUITE_DURATION
declare -A SUITE_EXP_DIR

for SUITE in ${SUITES}; do
    SUITE_START=$(date +%s)
    echo ""
    echo "============================================================================"
    echo "  ▶ 开始 ${SUITE}  ($(date '+%H:%M:%S'))"
    echo "============================================================================"

    # 不传 EXPERIMENT_NAME, 让 launcher 自动用时间戳建独立目录
    # (每个 suite 各自一个目录, 互不干扰)
    unset EXPERIMENT_NAME

    set +e
    TASK_SUITE_NAME="${SUITE}" bash "${WORKER_LAUNCHER}"
    SUITE_EXIT[${SUITE}]=$?
    set -e

    SUITE_END=$(date +%s)
    SUITE_DURATION[${SUITE}]=$((SUITE_END - SUITE_START))

    # 找出本次 suite 真正落地的 experiment 目录 (按时间戳取最新带这个 suite 的)
    LATEST_EXP=$(ls -td "${EXPERIMENTS_ROOT}/"*"${SUITE}"* 2>/dev/null | head -1 || true)
    SUITE_EXP_DIR[${SUITE}]="${LATEST_EXP:-(未找到)}"

    if [[ "${SUITE_EXIT[${SUITE}]}" -eq 0 ]]; then
        echo ""
        echo "  ✅ ${SUITE} 完成, 耗时 $((SUITE_DURATION[${SUITE}] / 60)) min"
    else
        echo ""
        echo "  ⚠️  ${SUITE} 退出码 ${SUITE_EXIT[${SUITE}]}, 但继续跑下一个 suite"
    fi
done

END_TS=$(date +%s)
TOTAL_MIN=$(( (END_TS - START_TS) / 60 ))

echo ""
echo "============================================================================"
echo "🎉 全部 suite 跑完, 总耗时 ${TOTAL_MIN} min"
echo "============================================================================"
for SUITE in ${SUITES}; do
    EC="${SUITE_EXIT[${SUITE}]:-?}"
    DUR="${SUITE_DURATION[${SUITE}]:-0}"
    EXP="${SUITE_EXP_DIR[${SUITE}]:-(未找到)}"
    if [[ "${EC}" == "0" ]]; then
        ICON="✅"
    else
        ICON="❌"
    fi
    echo "  ${ICON} ${SUITE}  耗时 $((DUR / 60)) min  (exit=${EC})"
    echo "      实验目录: ${EXP}"
    if [[ -d "${EXP}" ]]; then
        MERGED="${EXP}/eval_results_${SUITE}_merged.json"
        if [[ -f "${MERGED}" ]]; then
            echo "      合并结果: ${MERGED}"
        fi
    fi
done
echo "============================================================================"
