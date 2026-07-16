#!/bin/bash
# LIBERO to LeRobot Conversion Script
# This script converts LIBERO datasets to LeRobot format

# Exit on error
set -e

echo "========================================"
echo "LIBERO to LeRobot Conversion Tool"
echo "========================================"
echo ""

# Configuration
BASE_INPUT_DIR="/home/dev/文档/huangwenlong/dataset/libero_github"
# BASE_OUTPUT_DIR="/data/HuangWenlong/datasets/libero_github_convert_for_Sai0-VLA_eagle25-only-libero_spatial_v3"
BASE_OUTPUT_DIR="/home/dev/文档/huangwenlong/dataset/libero_github_convert_for_Sai0-VLA_libero_all/libero_lerobot_all_sys0_eagle_-1"

# Check for --stats flag anywhere in arguments
GENERATE_STATS=false
ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--stats" ]; then
        GENERATE_STATS=true
    else
        ARGS+=("$arg")
    fi
done
# Reset positional parameters without --stats
set -- "${ARGS[@]}"

# Function to generate stats.json for a dataset
generate_stats() {
    local OUTPUT_PATH="$1"
    if [ "$GENERATE_STATS" = true ]; then
        echo ""
        echo "等待1秒后生成 stats.json..."
        sleep 1
        echo "正在生成 stats.json: $OUTPUT_PATH"
        python generate_stats_json.py --dataset-path "$OUTPUT_PATH" --force
    fi
}

# Check if input directory exists
if [ ! -d "$BASE_INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $BASE_INPUT_DIR"
    echo "Please update BASE_INPUT_DIR in this script."
    exit 1
fi

# Parse command line arguments
MODE=${1:-"help"}

case $MODE in
    "single")
        # Convert a single dataset
        DATASET=${2:-"libero_10"}
        echo "Converting single dataset: $DATASET"
        echo ""
        python batch_convert_libero.py \
            --dataset "$DATASET" \
            --base-input-dir "$BASE_INPUT_DIR" \
            --base-output-dir "$BASE_OUTPUT_DIR"
        
        # Generate stats if --stats flag is set
        if [ "$DATASET" = "libero_10" ]; then
            OUTPUT_NAME="libero_lerobot_10"
        elif [ "$DATASET" = "libero_90" ]; then
            OUTPUT_NAME="libero_lerobot_90"
        else
            OUTPUT_NAME="libero_lerobot_${DATASET#libero_}"
        fi
        generate_stats "$BASE_OUTPUT_DIR/$OUTPUT_NAME"
        ;;
    
    "all")
        # Convert all datasets (separate output directories)
        echo "Converting all datasets (separate directories)..."
        echo ""
        python batch_convert_libero.py \
            --all \
            --base-input-dir "$BASE_INPUT_DIR" \
            --base-output-dir "$BASE_OUTPUT_DIR"
        
        # Generate stats for each dataset if --stats flag is set
        generate_stats "$BASE_OUTPUT_DIR/libero_lerobot_10"
        generate_stats "$BASE_OUTPUT_DIR/libero_lerobot_90"
        generate_stats "$BASE_OUTPUT_DIR/libero_lerobot_goal"
        generate_stats "$BASE_OUTPUT_DIR/libero_lerobot_object"
        generate_stats "$BASE_OUTPUT_DIR/libero_lerobot_spatial"
        ;;
    
    "all_merge")
        # Convert all datasets merged into single directory
        MERGED_NAME=${2:-"libero_lerobot_merged"}
        echo "Converting all datasets (merged into single directory)..."
        echo "Merged output name: $MERGED_NAME"
        echo ""
        python batch_convert_libero.py \
            --all-merge \
            --base-input-dir "$BASE_INPUT_DIR" \
            --base-output-dir "$BASE_OUTPUT_DIR" \
            --merged-output-name "$MERGED_NAME"
        
        # Generate stats if --stats flag is set
        generate_stats "$BASE_OUTPUT_DIR/$MERGED_NAME"
        ;;
    
    "all_merge_but_remove")
        # Convert all datasets except specified ones, merged into single directory
        shift  # Remove the mode argument
        if [ $# -lt 1 ]; then
            echo "ERROR: all_merge_but_remove mode requires at least one dataset to exclude"
            echo "Usage: $0 all_merge_but_remove <dataset1> [dataset2] ... [--name output_name] [--stats]"
            echo "Available datasets: libero_10, libero_90, libero_goal, libero_object, libero_spatial"
            exit 1
        fi
        
        # Parse arguments: collect datasets to exclude and optional --name
        EXCLUDE_DATASETS=()
        MERGED_NAME="libero_lerobot_merged"
        while [ $# -gt 0 ]; do
            case "$1" in
                --name)
                    shift
                    MERGED_NAME="$1"
                    ;;
                *)
                    EXCLUDE_DATASETS+=("$1")
                    ;;
            esac
            shift
        done
        
        echo "Converting all datasets EXCEPT: ${EXCLUDE_DATASETS[*]}"
        echo "Merged output name: $MERGED_NAME"
        echo ""
        python batch_convert_libero.py \
            --all-merge-exclude ${EXCLUDE_DATASETS[@]} \
            --base-input-dir "$BASE_INPUT_DIR" \
            --base-output-dir "$BASE_OUTPUT_DIR" \
            --merged-output-name "$MERGED_NAME"
        
        # Generate stats if --stats flag is set
        generate_stats "$BASE_OUTPUT_DIR/$MERGED_NAME"
        ;;
    
    "custom")
        # Custom conversion with specified paths
        INPUT_DIR=${2:-""}
        OUTPUT_DIR=${3:-""}
        
        if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
            echo "ERROR: Custom mode requires input and output directories"
            echo "Usage: $0 custom <input_dir> <output_dir> [--stats]"
            exit 1
        fi
        
        echo "Custom conversion:"
        echo "  Input: $INPUT_DIR"
        echo "  Output: $OUTPUT_DIR"
        echo ""
        
        python libero_to_lerobot.py \
            --input-dir "$INPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --fps 10 \
            --chunk-size 100
        
        # Generate stats if --stats flag is set
        generate_stats "$OUTPUT_DIR"
        ;;
    
    "help"|*)
        echo "Usage: $0 <mode> [options] [--stats]"
        echo ""
        echo "Modes:"
        echo "  single <dataset>        Convert a single dataset"
        echo "                          Options: libero_10, libero_90, libero_goal,"
        echo "                                  libero_object, libero_spatial"
        echo ""
        echo "  all                     Convert all datasets (separate directories)"
        echo ""
        echo "  all_merge [name]        Convert all datasets (merged into single directory)"
        echo "                          Optional: specify merged output directory name"
        echo "                          Default: libero_lerobot_merged"
        echo ""
        echo "  all_merge_but_remove <dataset> [dataset2...] [--name output_name]"
        echo "                          Merge all datasets EXCEPT specified ones"
        echo "                          Can exclude multiple datasets"
        echo "                          Optional: --name to specify output directory name"
        echo ""
        echo "  custom <input> <output> Custom conversion with specified paths"
        echo ""
        echo "  help                    Show this help message"
        echo ""
        echo "Global Options:"
        echo "  --stats                 Generate stats.json after conversion (wait 1s then run)"
        echo "                          Can be placed anywhere in the command"
        echo ""
        echo "Examples:"
        echo "  $0 single libero_10"
        echo "  $0 single libero_10 --stats"
        echo "  $0 all"
        echo "  $0 all --stats"
        echo "  $0 all_merge"
        echo "  $0 all_merge my_merged_dataset --stats"
        echo "  $0 all_merge_but_remove libero_90"
        echo "  $0 all_merge_but_remove libero_90 libero_10"
        echo "  $0 all_merge_but_remove libero_90 --name my_merged_no90 --stats"
        echo "  $0 custom /path/to/input /path/to/output --stats"
        echo ""
        echo "Configuration (edit this script to change):"
        echo "  BASE_INPUT_DIR:  $BASE_INPUT_DIR"
        echo "  BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR"
        ;;
esac

echo ""
echo "========================================"
echo "Done!"
echo "========================================"
