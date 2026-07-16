# Sai0_1 OFT1_0 - LIBERO-plus 评估

本目录提供基于 [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus) 官方
benchmark 对 `Sai0_1` 项目下 `Action_Heads/OFT1_0` 训出的 action head 做
鲁棒性评估的代码,完全对齐论文 *"In-depth Robustness Analysis For
Vision-Language-Action Models"* 给出的 7 类扰动 + 难度分级。

```
eval/Sai0_1/libero_plus/OFT1_0/
├── README.md                              <- 本文档
├── __init__.py
├── setup_libero_plus.py                   <- 一键准备 LIBERO-plus 环境(无副作用)
├── eval_libero_plus.py                    <- 主评估脚本(单 GPU worker)
├── merge_shard_results.py                 <- 多 GPU 跑完后合并各卡 shard JSON
├── scripts/
│   ├── run_eval_libero_plus.sh            <- 单 GPU 启动脚本
│   └── run_eval_libero_plus_multi_gpu.sh  <- 多 GPU 启动脚本(推荐, 每卡独立 worker)
└── experiments/                           <- 评估结果输出(运行时自动生成)
```

---

## 1. 前置依赖

### 1.1 Conda 环境

只用 **`qwen_eagle_hwl`** 一个环境(同一个项目里 `eval/Sai0_1/libero/OFT1_0`
跑原版 LIBERO 用的也是这个环境)。

```bash
conda activate qwen_eagle_hwl
```

### 1.2 LIBERO-plus 仓库

仓库已经在 `/data_disk1/hwl/LIBERO-plus`,本仓库代码默认就指向这里。
如果路径不一样,可以通过环境变量 `LIBERO_PLUS_ROOT` 覆盖。

### 1.3 LIBERO-plus 额外依赖(qwen_eagle_hwl 默认没有)

LIBERO-plus 在原版 LIBERO 之外多用了 `Wand` 和 `scikit-image` 实现传感器
噪声扰动,且 `libero/libero/envs/venv.py` 里硬 import 了 `gym`(原版 LIBERO
能用 `gymnasium`,但 LIBERO-plus 沿用旧版 gym):

```bash
conda activate qwen_eagle_hwl
pip install Wand scikit-image "gym==0.25.2"

# Wand 还需要系统包 ImageMagick(若已装可跳过)
sudo apt install -y libmagickwand-dev libfontconfig1-dev libexpat1
```

> 已经在我配置过的机器上跑过 `pip install Wand scikit-image "gym==0.25.2"`,
> 装好就不要再动了。`scikit-image` 是 lazy import,第一次用到才会加载。

### 1.4 LIBERO-plus assets

| 子目录 | 来源 | 状态 |
|---|---|---|
| `articulated_objects/scenes/textures/...` | 与原版 LIBERO 完全一致 | `setup_libero_plus.py` 自动从 `qwen_eagle_hwl` 中已安装的 LIBERO 软链过来,**无需重新下载** |
| `new_objects/` | LIBERO-plus 独有,放在 [HuggingFace assets.zip](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main) | 默认**未下载**,带 `_add_X` / `_levelX` 后缀的任务(约 290 个,占总量 3%)会被 eval 脚本自动跳过 |

如果想跑全 ~10K 个任务,需要手动下载 `new_objects/` 等独有资产并解压到
`/data_disk1/hwl/LIBERO-plus/libero/libero/assets/`。
解压完再次运行 `setup_libero_plus.py` 不会重复创建链接,安全。

---

## 2. 快速使用

### 2.1 第一次先做一次环境准备

```bash
conda activate qwen_eagle_hwl
cd /home/dev/文档/huangwenlong/sai0-vla
python -m eval.Sai0_1.libero_plus.OFT1_0.setup_libero_plus
```

它会做:

1. 给 `/data_disk1/hwl/LIBERO-plus/libero/__init__.py` `touch` 一个空文件,
   让 LIBERO-plus 成为合法 Python 包(LIBERO-plus 仓库本身漏了这个文件,
   导致 `sys.path.insert` 之后 `import libero` 仍会找回 conda env 中的
   原版 LIBERO)
2. 把 `qwen_eagle_hwl/.../libero/libero/assets/*` 软链到
   `/data_disk1/hwl/LIBERO-plus/libero/libero/assets/`
3. 在 `~/.libero_plus_sai0/config.yaml` 写一份 LIBERO-plus 专用的配置,
   不污染原版 LIBERO 用的 `~/.libero/config.yaml`(所以原版 LIBERO 的评估
   还能正常跑)

> `eval_libero_plus.py` 内部也会做上述 1/3,因此真正"必须"手动做的只有 2,
> 也就是首次准备资产软链。

### 2.2 一键评估(推荐 — 多 GPU)

10K+ 个 task 单卡要跑 ~30 小时, 强烈建议用多卡数据并行:

```bash
# 默认用 GPU 0-7 全部 8 张卡, 每张卡独立加载完整模型, task 用 stride 切分
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh
```

底层做的事情:
1. **Stride 切分**: 共 N 张卡, 第 k 张拿到 `task_ids[k::N]`. 因为 LIBERO-plus task 列表是按 category 排序的,
   stride 切分能让每张卡的 category / difficulty 分布几乎相同, 避免某张卡都是难 task 而另一张卡早早闲下来.
2. **每张卡一个独立 nohup python 进程**, 都加载完整 VLM (~5 GB) + Action Head (~400 MB).
3. **每张卡独立写 JSON**: `eval_results_<suite>_shard{K}_of_{N}.json`.
4. **全部跑完后自动 merge**: 调用 `merge_shard_results.py` 生成 `eval_results_<suite>_merged.json`,
   重新计算 overall / by_category / by_difficulty.

8 卡并行预计 ~4 小时跑完一个套件 (vs 单卡 ~30 小时).

常见自定义:

```bash
# 只用 GPU 2,3,5,6 跑 libero_object
GPU_IDS="2,3,5,6" TASK_SUITE_NAME=libero_object \
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh

# 用 4 张卡 (GPU 0-3) 跑全 4 个套件
for SUITE in libero_spatial libero_object libero_goal libero_10; do
    GPU_IDS="0,1,2,3" TASK_SUITE_NAME=$SUITE \
    bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh
done
```

跑挂了/中断了之后想接着跑:`RESUME=true`(默认) 会让每张卡读自己的 `shard{K}_of_{N}.json`
跳过已完成的 task, 然后再 merge 一次:

```bash
GPU_IDS="0,1,2,3" RESUME=true VIDEO_DIR=/path/to/已有/experiment_dir \
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh
```

### 2.3 单 GPU(简单调试用)

```bash
# 单张卡 (默认 GPU 0)
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus.sh

# 单张卡用 GPU 5
GPU_ID=5 bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus.sh
```

单 GPU / 多 GPU 通用的常见自定义:

```bash
# 换 checkpoint
CHECKPOINT_PATH=/path/to/your/action_head.pt \
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh

# 只评摄像机视角扰动 + 机器人初态扰动
CATEGORIES="Camera Viewpoints,Robot Initial States" \
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh

# 只评难度等级 1-2 的简单 task
DIFFICULTY_LEVELS="1,2" \
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh

# 快速调试:只跑前 50 个 task, 关掉视频保存
MAX_TASKS=50 SAVE_VIDEOS=false \
bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh
```

支持的所有变量见 `scripts/run_eval_libero_plus.sh` / `run_eval_libero_plus_multi_gpu.sh` 顶部。

### 2.4 直接跑 Python(进阶)

```bash
CUDA_VISIBLE_DEVICES=0 python -m eval.Sai0_1.libero_plus.OFT1_0.eval_libero_plus \
    --checkpoint_path /path/to/action_head.pt \
    --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
    --vlm_type qwen3_vl \
    --vlm_layers 14 \
    --vlm_output_dim 2048 \
    --dataset_path /data_disk2/hwl/datasets/libero_plus_lerobot \
    --task_suite_name libero_spatial \
    --num_trials_per_task 1 \
    --max_steps 600 \
    --num_action_chunks 16 \
    --num_transformer_blocks 4 \
    --action_head_hidden_dim 4096 \
    --execute_all_chunks \
    --video_dir ./eval/Sai0_1/libero_plus/OFT1_0/experiments/my_run
```

---

## 3. 输出说明

每次评估会在 `--video_dir`(默认 `experiments/<EXPERIMENT_NAME>/`) 下生成:

**单 GPU 输出**:

| 文件 | 内容 |
|---|---|
| `eval_results_<suite>.json` | 主结果 JSON,含 `task_results`(逐 task 详情) + `metrics`(按 category / difficulty 聚合) |
| `eval_<suite>_<datetime>.log` | 运行时的 task 维度日志 |
| `eval_output.log` | bash 脚本启动时 dump 的完整入参 + Python 运行的 stdout/stderr |
| `task_<id>/<DATE>/*.mp4` | (可选) rollout 视频,关掉用 `SAVE_VIDEOS=false`,默认每 `SAVE_VIDEO_EVERY=50` 个 task 存一个 |

**多 GPU 输出** (假设 `GPU_IDS="0,1,2,3"`):

| 文件 | 内容 |
|---|---|
| `eval_results_<suite>_shard{0..3}_of_4.json` | 每张 GPU 各自的结果 (互不相交的 task 子集) |
| `eval_results_<suite>_merged.json` | **最终合并结果** (推荐看这个),含 `merged_from_shards` 列表 |
| `worker_shard{0..3}_of_4_gpu{X}.log` | 每张 GPU 的独立日志(每个 worker 的 stdout/stderr) |
| `merge.log` | merge 阶段的输出 |
| `eval_<suite>_<datetime>_shard{K}_of_N.log` | 每张 GPU 的 task 维度日志 |
| `task_<id>/<DATE>/*.mp4` | rollout 视频(默认每 50 个 task 存一个) |

`metrics` 字段的结构:

```json
{
  "overall": {
    "episodes": 2402,
    "successes": 1672,
    "success_rate": 0.696
  },
  "by_category": {
    "Camera Viewpoints":     {"episodes": 376, "successes": 212, "success_rate": 0.564},
    "Robot Initial States":  {"episodes": 350, "successes": 112, "success_rate": 0.319},
    "Language Instructions": {"episodes": 390, "successes": 310, "success_rate": 0.795},
    "Light Conditions":      {"episodes": 292, "successes": 259, "success_rate": 0.887},
    "Background Textures":   {"episodes": 258, "successes": 240, "success_rate": 0.933},
    "Sensor Noise":          {"episodes": 351, "successes": 266, "success_rate": 0.758},
    "Objects Layout":        {"episodes": 385, "successes": 285, "success_rate": 0.742}
  },
  "by_difficulty": {
    "L1": {"episodes": 480, "successes": 420, "success_rate": 0.875},
    "L2": {"episodes": 669, "successes": 530, "success_rate": 0.792},
    "L3": {"episodes": 630, "successes": 410, "success_rate": 0.651},
    "L4": {"episodes": 432, "successes": 232, "success_rate": 0.537},
    "L5": {"episodes": 191, "successes":  80, "success_rate": 0.419}
  }
}
```

可以直接拿 `by_category` 列填到 [LIBERO-plus leaderboard](https://github.com/sylvestf/LIBERO-plus#-libero-plus-benchmark-leaderboard)。

---

## 4. 与原版 `eval/Sai0_1/libero/OFT1_0` 的差异

| 项 | 原版 LIBERO | LIBERO-plus(本目录) |
|---|---|---|
| Task 数量 | 10 / suite | 2402(spatial) / 2518(object) / 2591(goal) / 2519(libero_10) |
| `num_trials_per_task` | 5~10 | **1**(每个 task 已经是一个独立扰动 variant,官方约定) |
| State 表示 | `[gripper(2), xyz(3), axis-angle(3)]` 8 维 | `[xyz(3), euler_rpy(3), gripper_qpos(2)]` 8 维(与 `libero_plus_lerobot` 数据集对齐) |
| Quat 转换 | `quat → axis-angle` | `quat → euler XYZ`(用 `scipy.spatial.transform.Rotation`) |
| `convert_quat_to_axisangle` | True | False |
| Resume | 不支持 | **支持**(中途中断后再跑会自动跳过 `eval_results_<suite>.json` 中已完成的 task) |
| Per-category 指标 | ✗ | ✓ 直接读 `task_classification.json` |
| Per-difficulty 指标 | ✗ | ✓ |

---

## 5. 实现要点(给以后维护的人)

`eval_libero_plus.py` 顶部的 `_bootstrap_libero_plus()` 是关键:

1. `sys.path.insert(0, '/data_disk1/hwl/LIBERO-plus')` —— 让 import 优先
   走 LIBERO-plus 副本
2. `touch /data_disk1/hwl/LIBERO-plus/libero/__init__.py` —— LIBERO-plus
   仓库漏了这个空文件,不 touch 的话 Python 会回退到 conda env 中已安装
   的旧 LIBERO,sys.path.insert 等于白做
3. `LIBERO_CONFIG_PATH=~/.libero_plus_sai0` —— 用独立 config,不和原版
   LIBERO 互相覆盖
4. `_fix_robosuite_log_permission()` —— monkey-patch `logging.FileHandler`,
   把 `/tmp/robosuite.log` 重定向到 `~/.robosuite/robosuite.log`,避免
   多用户机器上的 PermissionError
5. `_patch_torch_load_for_legacy_pickles()` —— monkey-patch `torch.load`,
   显式传 `weights_only=False`,以兼容 PyTorch 2.6+ 默认收紧后的反序列化策略
   (LIBERO-plus 的 `.pruned_init` 用 `numpy.core.multiarray._reconstruct`,
   PyTorch 2.6+ 默认不允许这个 global)

State 在 `Sai0Policy.get_action()` 里这样拼:

```python
xyz   = obs["robot0_eef_pos"]                                  # (3,)
euler = ScipyRotation.from_quat(obs["robot0_eef_quat"]).as_euler("xyz")  # (3,) RPY
grip  = obs["robot0_gripper_qpos"]                             # (2,)
state = np.concatenate([xyz, euler, grip], axis=0)             # (8,)
```

这个顺序与 `libero_plus_lerobot/meta/info.json` 中
`observation.state.names = [eef_pos_x/y/z, eef_euler_r/p/y,
gripper_qpos_left, gripper_qpos_right]` 完全一致,与训练时的
`load_normalization_stats` min/max 一一对应。

---

## 6. 常见问题

**Q1: 提示 `No module named 'libero'`**

A: `qwen_eagle_hwl` 没装 LIBERO,或者 `setup_libero_plus.py` 没跑过。先
`pip list | grep libero` 确认 conda env 中的原版 LIBERO 已经装了,然后
跑一次 `python -m eval.Sai0_1.libero_plus.OFT1_0.setup_libero_plus`。

**Q2: 能不能不动 conda env,完全用 LIBERO-plus 替换原版 LIBERO?**

A: 可以但不推荐(会覆盖 `qwen_eagle_hwl` 中已安装的原版 LIBERO,导致原版
LIBERO eval 任务列表也变成 2402 个):

```bash
conda activate qwen_eagle_hwl
cd /data_disk1/hwl/LIBERO-plus
pip install -e .
```

本目录的方案是 **不修改 conda env**,只是加 `sys.path` + 独立 config,
这样原版 LIBERO eval 与 LIBERO-plus eval 可以并存。

**Q3: 跑到一半 OOM / GPU 占用满 / 想换 GPU?**

A: 直接 `Ctrl+C`,然后:

```bash
GPU_ID=5 RESUME=true bash eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus.sh
```

`RESUME=true` 是默认值,会读取已有的 `eval_results_<suite>.json` 跳过
已完成的 task。

**Q4: 报错 `FileNotFoundError: assets/scenes/tabletop_table_<XXX>.xml`**

A: LIBERO-plus 独有的 `Background Textures` 资产没下载。可以选:
- 跳过(脚本会自动 skip 这些 task,记到 `eval_results_<suite>.json` 里
  `skipped: true`)
- 或者从 [HF](https://huggingface.co/datasets/Sylvest/LIBERO-plus/tree/main)
  把 `assets.zip` 下载下来解压到 `/data_disk1/hwl/LIBERO-plus/libero/libero/assets/`

**Q5: 报错 `_pickle.UnpicklingError: Weights only load failed`**

A: 该报错应该被 `_patch_torch_load_for_legacy_pickles()` 处理掉。如果
还出现,说明这个 patch 在某个并发场景被绕过了。可以在 shell 里手动
`export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0` 强制关掉(PyTorch 2.7+ 支持)。
