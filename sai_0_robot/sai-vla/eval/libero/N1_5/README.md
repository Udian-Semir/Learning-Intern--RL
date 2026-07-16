# N1.5 LIBERO Evaluation

本目录包含 GR00T N1.5 在 LIBERO 基准测试上的评估脚本。

## 文件说明

- `eval_n1_5.py`: 主评估脚本，支持加载 fine-tuned N1.5 checkpoint 并在 LIBERO 环境中评估

## 使用方法

### 基本用法

```bash
python eval_n1_5.py --model-path /path/to/your/finetuned/checkpoint
```

### 完整参数示例

```bash
python eval_n1_5.py \
    --model-path /path/to/your/finetuned/checkpoint \
    --embodiment-tag new_embodiment \
    --benchmark libero_10 \
    --task-id 0 \
    --num-rollouts 50 \
    --max-steps 600 \
    --seed 42 \
    --device cuda:0 \
    --results-dir ./n1_5_eval_results \
    --save-video \
    --video-dir ./n1_5_eval_videos
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-path` | str | **必需** | Fine-tuned N1.5 checkpoint 目录路径 |
| `--embodiment-tag` | str | `new_embodiment` | 训练时使用的 embodiment tag |
| `--denoising-steps` | int | None | 覆盖扩散去噪步数 |
| `--action-norm` | str | `min_max` | Action 归一化模式 (`min_max` 或 `mean_std`) |
| `--benchmark` | str | `libero_10` | LIBERO 基准套件 |
| `--task-id` | int | None | 特定任务 ID（不指定则评估所有任务） |
| `--task-order-index` | int | `0` | 任务顺序索引 |
| `--num-rollouts` | int | `50` | 每个任务的 rollout 数量 |
| `--max-steps` | int | 自动 | 每个 rollout 的最大步数 |
| `--num-steps-wait` | int | `10` | 环境稳定等待步数 |
| `--resolution` | int | `256` | 相机分辨率 |
| `--results-dir` | str | `./n1_5_eval_results` | 结果保存目录 |
| `--save-video` | flag | False | 是否保存 rollout 视频 |
| `--video-dir` | str | `./n1_5_eval_videos` | 视频保存目录 |
| `--seed` | int | `42` | 随机种子 |
| `--device` | str | `cuda:0` | 推理设备 |

### 支持的 Benchmark

- `libero_spatial`: 空间推理任务 (max 220 steps)
- `libero_object`: 物体操作任务 (max 280 steps)
- `libero_goal`: 目标达成任务 (max 600 steps)
- `libero_10`: 10 个任务的综合测试 (max 1000 steps)
- `libero_90`: 90 个任务 (max 400 steps)
- `libero_100`: 100 个任务 (max 600 steps)

## 输出

### 结果文件

评估完成后会在 `--results-dir` 目录下生成 JSON 格式的结果文件，包含：

```json
{
  "config": {
    "model_path": "...",
    "benchmark": "libero_10",
    "num_rollouts": 50,
    ...
  },
  "overall": {
    "total_episodes": 500,
    "total_successes": 350,
    "success_rate": 0.7
  },
  "tasks": {
    "0": {
      "task_id": 0,
      "task_description": "pick up the block...",
      "success_rate": 0.8,
      "avg_steps": 150.5,
      ...
    },
    ...
  }
}
```

### 视频文件

如果启用 `--save-video`，会保存每个 rollout 的双视角（top + wrist）拼接视频：
- 文件名格式: `task{task_id}_rollout{rollout_id:03d}_{success|fail}.mp4`

## 注意事项

1. **Checkpoint 格式**: 确保 checkpoint 目录包含 `config.json` 和 `experiment_cfg/metadata.json` 文件
2. **Embodiment Tag**: 必须与训练时使用的 tag 一致，metadata.json 中需要有对应的归一化统计信息
3. **GPU 内存**: N1.5 模型需要较大 GPU 内存，建议使用至少 16GB 显存
4. **环境依赖**: 需要安装 LIBERO 和 Isaac-GR00T 相关依赖

## 示例

### 评估单个任务

```bash
python eval_n1_5.py \
    --model-path /home/user/checkpoints/libero_finetune \
    --benchmark libero_object \
    --task-id 3 \
    --num-rollouts 20 \
    --save-video
```

### 评估整个 benchmark

```bash
python eval_n1_5.py \
    --model-path /home/user/checkpoints/libero_finetune \
    --benchmark libero_10 \
    --num-rollouts 50 \
    --results-dir ./results/libero_10_eval
```

### 使用不同的去噪步数

```bash
python eval_n1_5.py \
    --model-path /home/user/checkpoints/libero_finetune \
    --denoising-steps 4 \
    --benchmark libero_spatial
```
