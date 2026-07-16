#!/usr/bin/env bash
# 一键把 10 个已完成 VLM 抽取的数据集合并成一个 LeRobot 数据集
# （方案 A：物理合并，不依赖 train_multigpu.py 改代码）
#
# 运行完成后把 train_qwen_datasets_pretrain10_22.sh 里的 DATA_PATH
# 改成 $OUTPUT_DIR 就能直接跑 pretrain。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# 合并输出路径（建议放在大盘上）
OUTPUT_DIR="${OUTPUT_DIR:-/data_disk1/hwl/pretrain10_merged}"

# 源数据集清单
DATASET_LIST="${DATASET_LIST:-$SCRIPT_DIR/../datasets_pretrain10.txt}"

# chunks_size: 每个 chunk 装多少 episode（和单数据集一致，默认 1000）
CHUNKS_SIZE="${CHUNKS_SIZE:-1000}"

# 是否合并视频（默认不合并，SKIP_IMAGES=true 训练不需要）
INCLUDE_VIDEOS="${INCLUDE_VIDEOS:-false}"

# 已存在时是否清空
OVERWRITE="${OVERWRITE:-false}"

# 是否 dry-run
DRY_RUN="${DRY_RUN:-false}"

# 推荐用 qwen_eagle_hwl 环境（已装 pyarrow/pandas/numpy）
CONDA_ENV="${CONDA_ENV:-qwen_eagle_hwl}"

cd "$REPO_ROOT"

# 激活 conda 环境（可选）
if [[ -n "${CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
    if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
        echo "[run_merge] 使用 conda env: $CONDA_ENV"
    fi
fi

ARGS=(
    --datasets_from_file "$DATASET_LIST"
    --output "$OUTPUT_DIR"
    --chunks_size "$CHUNKS_SIZE"
)

if [[ "$INCLUDE_VIDEOS" == "true" ]]; then
    ARGS+=(--include_videos --link_mode hardlink)
fi

if [[ "$OVERWRITE" == "true" ]]; then
    ARGS+=(--overwrite)
fi

if [[ "$DRY_RUN" == "true" ]]; then
    ARGS+=(--dry_run)
fi

echo "[run_merge] 命令:"
echo "  python -m utils.mix_dataset.merge_lerobot_datasets ${ARGS[*]}"
echo ""

python -m utils.mix_dataset.merge_lerobot_datasets "${ARGS[@]}"
