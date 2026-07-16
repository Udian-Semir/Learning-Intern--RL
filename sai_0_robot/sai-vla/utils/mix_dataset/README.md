# utils/mix_dataset

**方案 A：把多个 LeRobot v2.0 数据集物理合并成一个 LeRobot 目录**，
用来给 `Action_Heads/OFT1_0/train_multigpu.py` 做 **多数据集 pretrain**。

> 背景：`train_multigpu.py` 只支持单个 `--data_path`。要同时用 10+ 个已提取 VLM
> hidden states 的 LeRobot 数据集做 pretrain，最稳的做法就是先把它们合并成一个
> LeRobot 目录，再按正常流程训练。

## 目录结构

```
utils/mix_dataset/
├── __init__.py
├── merge_lerobot_datasets.py   # 主脚本（物理合并）
├── datasets_pretrain10.txt     # 10 个已完成数据集清单
├── scripts/
│   └── run_merge_pretrain10.sh # 一键合并
└── README.md
```

## 合并后的数据集结构

和普通 LeRobot v2.0 数据集完全一致：

```
<output_dir>/
├── meta/
│   ├── info.json           # total_episodes/frames/chunks 等（重算）
│   ├── episodes.jsonl      # 全局重编号后的 episode 列表
│   ├── tasks.jsonl         # 按文本去重后的全局 task 表
│   ├── stats.json          # observation.state / action 的 min/max/mean/std（重算）
│   └── modality.json       # 继承自第一个源数据集
├── data/
│   └── chunk-{NNN}/episode_{EEEEEE}.parquet
├── vlm_hidden_states/
│   └── chunk-{NNN}.npz     # 每个 chunk 内部 key=episode_{EEEEEE}
├── videos/                 # 仅 --include_videos 时才有（hardlink 或 symlink）
└── merge_manifest.json     # 记录每个源数据集的 ep_offset/idx_offset/vlm_offset
```

## 核心做的事情

1. **全局重编号**：每个源数据集都分配一段连续的 episode 区间和 frame index 区间。
   对 parquet 的这几列做偏移：
   - `episode_index`            += `ep_offset`
   - `index`                    += `idx_offset`
   - `vlm_hidden_state_index`   += `vlm_offset`  （当前源 `vlm_idx == index`）
2. **task 表合并**：按 task 文本去重，建立每个源的 `src_task_idx -> new_task_idx` 映射，
   对 parquet 里的 `task_index` / `annotation.human.*.task_description` 做 remap。
3. **episodes.jsonl**：复制源条目，更新 `episode_index`，并对 `tasks[]` 里的 task_index 做 remap。
4. **VLM `chunk-XXX.npz`**：按新 episode 号重新打包。**流式写入**（`zipfile.ZipFile` +
   `numpy.lib.format.write_array`），内存占用只等于单个 episode 的 hidden state，
   再大的数据集也能跑。
5. **stats.json**：在重写 parquet 的同时在线累加 `observation.state` / `action` 的
   min/max/sum/sumsq，结束后合成 mean/std。训练侧 `load_normalization_stats` 只用
   min/max，所以这套合并统计是正确的全局 min-max。
6. **info.json**：`features` 复制自第一个源，但把 `observation.state.shape` 和
   `action.shape` 校正到一致维度；`total_episodes / total_frames / total_chunks` 重算；
   `vlm_hidden_state_path` 写成 chunk-npz 模板。
7. **videos**：默认 **不合并**（训练 `SKIP_IMAGES=true` 不需要）。要合并就加
   `--include_videos`，默认用 **hardlink**（不占磁盘空间）。

## 使用

```bash
# 方式 1: 一键脚本（默认输出 /data_disk1/hwl/pretrain10_merged）
bash utils/mix_dataset/scripts/run_merge_pretrain10.sh

# dry-run 先看计划
DRY_RUN=true bash utils/mix_dataset/scripts/run_merge_pretrain10.sh

# 覆盖已有输出
OVERWRITE=true bash utils/mix_dataset/scripts/run_merge_pretrain10.sh

# 自定义输出路径
OUTPUT_DIR=/data_disk1/hwl/my_mix bash utils/mix_dataset/scripts/run_merge_pretrain10.sh

# 方式 2: 直接调 Python（在 qwen_eagle_hwl 环境里跑）
python -m utils.mix_dataset.merge_lerobot_datasets \
    --datasets_from_file utils/mix_dataset/datasets_pretrain10.txt \
    --output /data_disk1/hwl/pretrain10_merged \
    --overwrite
```

## 对接训练脚本

合并完成后，只需要改 `Action_Heads/OFT1_0/scripts/train/qwen/train_qwen_datasets_pretrain10_22.sh`
的 `DATA_PATH`：

```bash
# 原来（单数据集）：
# DATA_PATH="/data_disk1/hwl/unitree_train_v2_recipe_lerobot/austin_buds_dataset_converted_externally_to_rlds"

# 改成：
DATA_PATH="/data_disk1/hwl/pretrain10_merged"
```

保持 `SKIP_IMAGES="true"`（合并时默认没有复制视频）。其它参数保持不变。

## 预计输出规模（10 个数据集）

| 维度 | 数值（估算） |
|---|---|
| total_episodes | ~7,640 |
| total_frames   | ~1,080,000 |
| unique tasks   | 取决于去重 |
| parquet 大小   | ~几 GB（全拷贝；也可以改 `df.to_parquet` 为软链接原 parquet，但列会不匹配） |
| VLM 大小       | 和原来 10 个数据集总和相当（~几百 GB） |
| videos 大小    | 0（默认不合并）；若 `--include_videos`，使用 hardlink 不额外占用 |

## 运行环境

推荐 `qwen_eagle_hwl`（自带 `pyarrow / pandas / numpy / tqdm`）。

## 限制 / 说明

- 所有源数据集必须有相同的 `observation.state` 维度 和 `action` 维度（脚本会强校验）。
- `features` 使用第一个数据集的，`robot_type` 置为 `"mixed"`。
- 若源数据集的 `fps` 不同，merged `info.fps` 取第一个源的，会打 warning。
- `vlm_hidden_states` 必须是 **chunk-npz 格式**（每 `chunks_size` 个 episode 一个 npz）。
  如果源是旧的 `episode_XXXXXX.npy` 格式，请先跑 `utils/migrate_vlm_npy_to_chunk_npz.py` 迁移。
- 本脚本**不是增量合并**，每次都是全量重建输出目录。要继续加数据集就重新跑一次。
