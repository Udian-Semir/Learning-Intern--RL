# LIBERO 转 LeRobot 数据格式转换工具

本工具将 LIBERO 机器人演示数据集从 HDF5 格式转换为 LeRobot 格式，兼容现代视觉-语言-动作（VLA）模型。

## 功能特性

- ✅ 将 LIBERO HDF5 文件转换为 LeRobot parquet 格式
- ✅ 提取并保存 RGB 观测为 MP4 视频
- ✅ 添加 VLM hidden state 索引用于后续集成
- ✅ 生成所有必需的元数据（info.json, tasks.jsonl, episodes.jsonl）
- ✅ 支持批量转换多个数据集
- ✅ 支持合并所有数据集到单一目录（`all_merge` 模式）
- ✅ 支持合并时排除指定数据集（`all_merge_but_remove` 模式）
- ✅ 将数据组织为 chunks 以便高效加载

## 支持的数据集

- `libero_10`: 10 个任务
- `libero_90`: 90 个任务  
- `libero_goal`: 10 个目标条件任务
- `libero_object`: 10 个物体交互任务
- `libero_spatial`: 10 个空间推理任务

## 目录结构

```
libero_data_transform_for_Sai/
├── README.md                    # 本文件
├── libero_to_lerobot.py        # 主转换脚本
├── batch_convert_libero.py     # 批量处理脚本
└── run_conversion.sh           # 快速启动脚本
```

## 快速开始

### 1. 单个数据集转换

```bash
python libero_to_lerobot.py \
    --input-dir /data/HuangWenlong/datasets/libero_github/libero_10 \
    --output-dir /data/output/libero_10_lerobot \
    --fps 10 \
    --chunk-size 100
```

### 2. 批量转换

转换指定数据集：
```bash
python batch_convert_libero.py --dataset libero_10
```

转换所有数据集（每个数据集单独输出目录）：
```bash
python batch_convert_libero.py --all
```

转换所有数据集（合并到单一输出目录）：
```bash
python batch_convert_libero.py --all-merge
```

合并时排除指定数据集：
```bash
# 排除 libero_90
python batch_convert_libero.py --all-merge-exclude libero_90

# 排除多个数据集
python batch_convert_libero.py --all-merge-exclude libero_90 libero_10
```

### 3. 使用 Shell 脚本

```bash
# 转换所有数据集（分开保存）
./run_conversion.sh all

# 转换所有数据集（合并到单一目录）
./run_conversion.sh all_merge

# 转换所有数据集（合并，自定义输出目录名）
./run_conversion.sh all_merge my_merged_dataset

# 合并时排除指定数据集
./run_conversion.sh all_merge_but_remove libero_90

# 排除多个数据集
./run_conversion.sh all_merge_but_remove libero_90 libero_10

# 排除数据集并自定义输出目录名
./run_conversion.sh all_merge_but_remove libero_90 --name my_merged_no90

# 转换单个数据集
./run_conversion.sh single libero_spatial

# 转换后自动生成 stats.json（添加 --stats 参数）
./run_conversion.sh all_merge --stats
./run_conversion.sh single libero_10 --stats
./run_conversion.sh all_merge_but_remove libero_90 --name my_merged --stats
```

编辑脚本以自定义路径和数据集。

**`--stats` 参数**：可以在任何命令后添加 `--stats`，转换完成后会等待1秒然后自动运行 `generate_stats_json.py` 生成统计信息文件。

## 输出格式

转换后的数据遵循 LeRobot v2.0 格式：

```
output_dir/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       ├── observation.images.agentview/
│       │   ├── episode_000000.mp4
│       │   └── ...
│       └── observation.images.wrist/
│           ├── episode_000000.mp4
│           └── ...
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── tasks.jsonl        # Task descriptions
│   └── episodes.jsonl     # Episode information
└── vlm_hidden_states/     # Placeholder for VLM embeddings
```

## Data Schema

Each parquet file contains the following columns:

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| `observation.state` | float32 | (10,) | Robot state (EE pos, ori, gripper, joints) |
| `action` | float32 | (7,) | Robot action (6-DOF + gripper) |
| `timestamp` | float64 | (1,) | Frame timestamp |
| `episode_index` | int64 | (1,) | Episode ID |
| `index` | int64 | (1,) | Global frame index |
| `task_index` | int64 | (1,) | Task ID |
| `next.done` | bool | (1,) | Episode termination flag |
| `next.reward` | float64 | (1,) | Reward signal |
| `vlm_hidden_state_index` | int64 | (1,) | VLM embedding index |
| `annotation.human.action.task_description` | int64 | (1,) | Task description ID |
| `annotation.human.validity` | int64 | (1,) | Data validity flag |

**VLM Hidden State Index**: This index matches the global `index` and increments sequentially. It can be used to load corresponding VLM hidden state files stored in `vlm_hidden_states/hidden_state_{index:06d}.npy`.

## Parameters

### libero_to_lerobot.py

- `--input-dir`: Path to LIBERO HDF5 files directory
- `--output-dir`: Output directory for LeRobot format data
- `--fps`: Frames per second for video encoding (default: 10.0)
- `--chunk-size`: Number of episodes per chunk (default: 100)

### batch_convert_libero.py

- `--dataset`: Specific dataset to convert (libero_10, libero_90, etc.)
- `--all`: Convert all datasets (each to separate output directory)
- `--all-merge`: Convert all datasets merged into single output directory
- `--all-merge-exclude`: Merge all datasets EXCEPT specified ones (can specify multiple)
- `--base-input-dir`: Base directory for LIBERO datasets (default: `/data/HuangWenlong/datasets/libero_github`)
- `--base-output-dir`: Base directory for output (default: `/data/HuangWenlong/datasets`)
- `--merged-output-name`: Output directory name for merged mode (default: `libero_lerobot_merged`)

## Requirements

```
h5py
numpy
pandas
opencv-python (cv2)
pyarrow
tqdm
```

## Examples

### Example 1: Convert libero_10 with custom output

```bash
python libero_to_lerobot.py \
    --input-dir /data/HuangWenlong/datasets/libero_github/libero_10 \
    --output-dir /custom/output/path \
    --fps 15 \
    --chunk-size 50
```

### Example 2: Batch convert all datasets

```bash
python batch_convert_libero.py \
    --all \
    --base-input-dir /data/HuangWenlong/datasets/libero_github \
    --base-output-dir /data/HuangWenlong/datasets/lerobot_format
```

### Example 3: Convert specific datasets

```bash
# Convert only libero_goal and libero_spatial
python batch_convert_libero.py --dataset libero_goal
python batch_convert_libero.py --dataset libero_spatial
```

### Example 4: Merge all datasets into single directory

```bash
# Merge all 5 datasets (libero_10, libero_90, libero_goal, libero_object, libero_spatial)
# into a single output directory with unified task/episode indices
python batch_convert_libero.py \
    --all-merge \
    --base-input-dir /data/HuangWenlong/datasets/libero_github \
    --base-output-dir /data/HuangWenlong/datasets/lerobot_format \
    --merged-output-name libero_all_merged
```

This will create a single directory containing all 130 tasks from all datasets with:
- Continuous episode indices across all datasets
- Continuous task indices (0-129)
- Unified metadata files (tasks.jsonl with all 130 tasks)

### Example 5: Merge datasets excluding specific ones

```bash
# Merge all datasets except libero_90 (useful when libero_90 is too large)
python batch_convert_libero.py \
    --all-merge-exclude libero_90 \
    --base-input-dir /data/HuangWenlong/datasets/libero_github \
    --base-output-dir /data/HuangWenlong/datasets/lerobot_format \
    --merged-output-name libero_merged_no90

# Merge only goal, object, and spatial datasets (exclude libero_10 and libero_90)
python batch_convert_libero.py \
    --all-merge-exclude libero_10 libero_90 \
    --merged-output-name libero_small_merged
```

This will merge the remaining datasets with continuous indices.

## Notes

- The conversion preserves all demonstrations from the original HDF5 files
- Videos are encoded with mp4v codec at the specified FPS
- VLM hidden states directory is created but populated with placeholders (you need to generate embeddings separately)
- Episode indices are global and sequential across all tasks
- Chunk organization allows for efficient data loading in training

## Troubleshooting

**Problem**: `FileNotFoundError: No such file or directory`
- Solution: Check that input paths are correct and HDF5 files exist

**Problem**: `ModuleNotFoundError: No module named 'cv2'`
- Solution: Install opencv: `pip install opencv-python`

**Problem**: Out of disk space
- Solution: Ensure sufficient space (videos take ~100-200MB per task)

**Problem**: Conversion is slow
- Solution: This is normal - video encoding takes time. Use batch script for overnight runs.

## Integration with VLM

After conversion, you can generate VLM hidden states:

```python
# Pseudo-code for VLM integration
import numpy as np
from pathlib import Path

output_dir = Path("/output/libero_10_lerobot")
vlm_dir = output_dir / "vlm_hidden_states"

# For each frame, generate and save hidden state
for idx in range(total_frames):
    # Load corresponding video frame
    # Run through VLM encoder
    hidden_state = vlm_model.encode(frame)
    
    # Save with matching index
    np.save(vlm_dir / f"hidden_state_{idx:06d}.npy", hidden_state)
```

## License

This conversion tool is provided as-is for research purposes.

## Contact

For questions or issues, please contact the repository maintainer.
