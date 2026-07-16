# 视频帧数统计工具

该工具用于统计 LeRobot 格式数据集中指定任务索引的视频总帧数。

## 功能特点

- 支持统计指定任务索引的所有视频帧数
- 自动扫描数据集目录中的所有视频文件（.mp4, .avi）
- **自动检测并统计所有视角的视频**（如 agentview、wrist 等）
- 从对应的 parquet 文件中提取任务索引信息
- 提供详细的统计信息（视频数量、总帧数、平均帧数）
- 支持将结果导出为 JSON 格式

## 重要说明

**视频统计包含所有视角：**
- 该工具会自动检测数据集中的所有相机视角（如 `observation.images.agentview`、`observation.images.wrist` 等）
- 统计结果中的**视频数量和总帧数是所有视角的总和**
- 例如：如果一个任务有 50 个 episode，每个 episode 有 2 个视角（agentview 和 wrist），则会统计 100 个视频文件
- 如果需要区分不同视角的统计，可以根据输出 JSON 中的 `videos` 列表进行过滤

## 依赖项

```bash
pip install opencv-python pandas pyarrow
```

## 使用方法

### 基本用法

```bash
# 统计所有任务的视频帧数
python count_video_frames.py --dataset-path /path/to/dataset
```

### 统计指定任务

```bash
# 统计任务索引 0, 1, 2 的视频帧数
python count_video_frames.py --dataset-path /path/to/dataset --task-indices 0 1 2
```

### 保存结果到文件

```bash
# 将统计结果保存为 JSON 文件
python count_video_frames.py --dataset-path /path/to/dataset --output result.json
```

### 组合使用

```bash
# 统计指定任务并保存结果
python count_video_frames.py \
    --dataset-path /data/libero_10_lerobot \
    --task-indices 0 1 2 3 4 \
    --output task_stats.json
```

## 参数说明

- `--dataset-path`: **必需**，数据集根目录路径
- `--task-indices`: 可选，要统计的任务索引列表（空格分隔），不指定则统计所有任务
- `--output`: 可选，输出 JSON 文件路径

## 数据集结构

该工具适用于 LeRobot 格式的数据集，预期的目录结构如下：

```
dataset_path/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.top/
        │   ├── episode_000000.mp4
        │   ├── episode_000001.mp4
        │   └── ...
        └── observation.images.left_wrist/
            ├── episode_000000.mp4
            ├── episode_000001.mp4
            └── ...
```

## 输出示例

### 终端输出

```
正在扫描数据集: /data/libero_10_lerobot
找到 200 个视频文件

============================================================
视频帧数统计结果
============================================================

任务索引 0:
  视频数量: 100
  总帧数: 26596
  平均帧数: 265.96
  (注：包含 2 个视角 × 50 个 episode = 100 个视频文件)

任务索引 1:
  视频数量: 100
  总帧数: 25800
  平均帧数: 258.00

------------------------------------------------------------
总计:
  任务数量: 2
  视频数量: 200
  总帧数: 52396
  (注：统计了所有视角的视频文件)
============================================================
```

### JSON 输出格式

```json
{
  "dataset_path": "/data/libero_10_lerobot",
  "task_indices": [0, 1],
  "statistics": {
    "0": {
      "total_frames": 5000,
      "video_count": 20,
      "videos": [
        "videos/chunk-000/observation.images.top/episode_000000.mp4",
        "videos/chunk-000/observation.images.left_wrist/episode_000000.mp4",
        ...
      ]
    },
    "1": {
      "total_frames": 4800,
      "video_count": 20,
      "videos": [...]
    }
  },
  "summary": {
    "total_tasks": 2,
    "total_videos": 40,
    "total_frames": 9800
  }
}
```

## 注意事项

1. 确保数据集路径正确，且包含 `data/` 和 `videos/` 子目录
2. parquet 文件中必须包含 `task_index` 列
3. 视频文件名格式应为 `episode_XXXXXX.mp4` 或 `episode_XXXXXX.avi`
4. 如果视频文件损坏或无法读取，该文件将被跳过并显示警告
5. **统计结果包含所有相机视角的视频文件**：
   - 工具会自动扫描 `videos/chunk-XXX/` 下的所有 `observation.images.*` 目录
   - 每个 episode 通常有多个视角（如 agentview、wrist），每个视角都会被统计
   - 总帧数 = 所有视角的帧数之和

## 故障排除

### 找不到视频文件
- 检查数据集路径是否正确
- 确认 `videos/` 目录存在且包含视频文件

### 无法读取任务索引
- 确保 parquet 文件存在于 `data/` 目录
- 检查 parquet 文件中是否包含 `task_index` 列

### 视频帧数为 0
- 视频文件可能损坏，尝试用视频播放器打开验证
- 检查是否安装了正确的 OpenCV 版本

## 许可证

与 LIBERO 项目使用相同的许可证。
