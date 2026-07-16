# LIBERO-plus 数据集原理速记

面向 sai0-vla 项目的内部说明，聚焦 OFT1_0 在 LIBERO-plus 上做评测时
最容易踩坑的几个点：

- 训练用 40 个 task，为什么评测要跑 ~10K 任务？
- `task_classification.json` 是什么？7 类扰动具体长什么样？
- 评测时模型收到的 instruction 为什么带 `table 15`、`initstate 363` 这种后缀？
- 评测资产为什么经常缺失，正确的目录结构应该是什么？
- 跑评测时如何分 shard、聚合结果。

> 路径示例都是当前机器上的实际位置：
>
> - LIBERO-plus 仓库 : `/data_disk1/hwl/LIBERO-plus`
> - LeRobot 训练集   : `/data_disk2/hwl/datasets/libero_plus_lerobot`

---

## 1. LIBERO-plus = 原版 LIBERO 的"扰动放大版"

| | 原版 LIBERO | LIBERO-plus |
|---|---|---|
| 目标 | 测 in-distribution 性能 | 测 OOD / robustness |
| Suite | 4 个: spatial / object / goal / 10 (LIBERO-90 略) | 同样 4 个 |
| 每 suite task 数 | **10** | **2402 / 2518 / 2591 / 2519** ≈ ~2500 |
| 总评测任务 | **40** | **10030** |
| 每 task trial 数 | 50 (原版做统计) | **1** (变体本身就够多了) |
| 目的 | 训练 + 测试 | 仅做测试 (论文 / leaderboard) |

> 训练数据集 `libero_plus_lerobot/meta/info.json` 里看到 `total_tasks: 40` 是因为
> 训练只用原版 LIBERO 的 40 个 base task × ~358 demos / task = 14347 episodes。
> 模型从来没见过那 10030 个变体，这是测试集的特意设计。

---

## 2. 7 大类扰动 (category) + 多档难度 (difficulty_level)

LIBERO-plus 在每个 base task 上沿 7 个维度生成大量变体，每个变体记录在
`task_classification.json` 里：

| category | 改了什么 | 名字后缀样例 |
|---|---|---|
| Background Textures | 替换桌面 / 背景纹理 | `_table_15`, `_tb_22` |
| Object Textures | 物体材质纹理 | `_objtex_*` |
| Object Layouts | 同 task 内物体摆放位置 | `_layout_*` |
| Robot Initial States | 机械臂起始 EEF 位姿 / 关节角 | `_initstate_283`, `_initstate_363` |
| Camera Viewpoints | agentview 相机外参 | `_view_0_0_100_0_0` |
| Lighting | 灯光强度 / 方向 / 色温 | `_light_*` |
| Language Instructions | 同义重述 prompt | task name 含 `_language_N_` |

每个 category 下还分 `difficulty_level` (1-5)，数字越大扰动越剧烈。

评测脚本里聚合方式是 `(suite × category)` 二维 + overall：

```python
# eval_libero_plus.py:_aggregate_metrics
{
    "overall":     {"episodes": ..., "successes": ..., "success_rate": ...},
    "by_category": {"Background Textures": {...}, "Lighting": {...}, ...},
    "by_difficulty": {1: {...}, 2: {...}, 3: {...}, ...},
}
```

---

## 3. task_classification.json 文件结构

```
/data_disk1/hwl/LIBERO-plus/libero/libero/benchmark/task_classification.json
```

格式：

```json
{
  "libero_spatial":  [ {id, name, category, difficulty_level}, ...  2402 个 ],
  "libero_object":   [ ...  2518 个 ],
  "libero_goal":     [ ...  2591 个 ],
  "libero_10":       [ ...  2519 个 ]
}
```

合计 10030 条记录，跟评测脚本里 `task_suite.n_tasks` 严格一致。

`name` 字段直接对应 bddl 文件名（无 `.bddl` 后缀），例如：

```
pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_table_15
└── 原版 base task ──────────────────────────────────────────────┘└ 后缀 ┘
```

---

## 4. ⚠️ instruction 带后缀这件事 (LIBERO-plus 的实现 quirk)

### 现象

评测时模型实际收到的 prompt 不是干净的指令，而是这种带后缀的字符串：

| 评测 task name | 模型收到的 instruction |
|---|---|
| `pick_up_..._on_the_plate_table_15` | `pick up ... on the plate table 15` |
| `pick_up_..._on_the_plate_tb_22`    | `pick up ... on the plate tb 22`    |
| `pick_up_..._view_0_0_100_0_0_initstate_283` | `pick up ... view 0 0 100 0 0 initstate 283` |

### 为什么

`/data_disk1/hwl/LIBERO-plus/libero/libero/benchmark/__init__.py:46-72`
里 `grab_language_from_filename()` 当 task name **不含 `_language_`** 时，
直接做了：

```python
language = " ".join(x.split("_"))   # 把整段 filename 转空格
return language[: language.find(".bddl")]
```

**没有任何"剥掉 perturbation 后缀"的逻辑**。所以扰动用的标识符直接成了 prompt 的一部分。

只有 task name 里含 `_language_` 标记的（`Language Instructions` 类别，
共 1537 个）才会进 `if` 分支，真正去 bddl 文件里读 `language_instruction`
字段——那个字段是论文作者手写的同义重述，用来测 prompt robustness。

### 影响

- 训练时模型看到干净的 40 条指令 (`tasks.jsonl` 里那种)
- 评测时模型多见到 `table 15` / `initstate 363` 这些训练里从未出现的 token
- 但**所有 LIBERO-plus 评测者都是这样**，leaderboard 上的数都受同样影响，公平
- 想做"清洗后的 upper bound"实验：可在 `Sai0Policy.get_action()` 加正则
  把 `_table_\d+`、`_tb_\d+`、`_view_[\d_]+`、`_initstate_\d+` 等剥掉后再喂模型。
  ⚠️ 一定要做成可选 flag 而不是默认行为，否则跟 leaderboard 不可比。

---

## 5. 训练 vs 评测的数据范围对比

```
┌─ 训练阶段 ─────────────────────────────────────────────────┐
│  数据集: libero_plus_lerobot (LeRobot 格式)                │
│    - total_tasks   = 40 (4 suite × 10 base task)           │
│    - total_episodes= 14347 (~358 demos / task)             │
│    - total_frames  = 2.24M                                 │
│    - 指令: tasks.jsonl 里 40 条, 干净自然语言              │
│    - 图像: agentview + wrist (256×256, h264 编码)          │
│  hidden states 提取: FLIP_IMAGES="false" (LeRobot 已为正向)│
└────────────────────────────────────────────────────────────┘

┌─ 评测阶段 ─────────────────────────────────────────────────┐
│  benchmark: LIBERO-plus task_classification.json           │
│    - libero_spatial: 2402 任务                             │
│    - libero_object : 2518                                  │
│    - libero_goal   : 2591                                  │
│    - libero_10     : 2519                                  │
│    - 合计 10030, 每个跑 1 次                               │
│  prompt: 原版指令 + perturbation 后缀 (如 "table 15")      │
│  图像: LIBERO env 直出 (OpenGL 颠倒方向)                   │
│  评测脚本里 FLIP_IMAGES=true 把图像翻 180° 才与训练对齐    │
└────────────────────────────────────────────────────────────┘
```

---

## 6. 资产目录结构 (踩过坑)

LIBERO-plus 的 `assets.zip` 解压后会带一段巨长的服务器路径，扩展资产被埋
在深路径里，需要合并到正确位置：

```
错误 (assets.zip 解压后)
  /data_disk1/hwl/LIBERO-plus/libero/libero/inspire/hdd/.../LIBERO-plus-0/assets/
                                       └─ scenes/(263 个), new_objects/(416 个), ...

正确 (合并后)
  /data_disk1/hwl/LIBERO-plus/libero/libero/assets/
    ├── scenes/         263 项 (含 tabletop250/ 子目录 517 个)
    ├── new_objects/    416 个 LIBERO-plus 独有物体
    ├── articulated_objects/, stable_hope_objects/, ...
    └── textures/       基础 + LIBERO-plus 扩展 549 项
```

修复办法：`setup_libero_plus.py --merge_libero_plus_assets`，会：

1. 物化指向 conda env 的 symlink → 在原位变成实目录，内部仍逐项 symlink 回 conda env
2. 把深路径里 LIBERO-plus 独有的项 symlink 进来 (零拷贝，不污染 conda env)

未合并时的症状：评测日志里大量

```
⚠️ Task XX 资产/bddl 缺失, 跳过: [Errno 2] No such file or directory:
'.../scenes/tabletop_table_Cobblestone01_GLOSS_6K.xml'
```

---

## 7. 评测时的工程要点

### 7.1 单卡 vs 多卡

- 每 task 1 trial，10K 个变体 → 单卡 spatial 大约要跑 5-8h
- 多 GPU 时按 stride 切分: shard k 的 task_ids = `range(n)[k::N]`
- 每张卡独立加载完整模型 + 独立写 `eval_results_<suite>_shardK_of_N.json`
- 全跑完后用 `merge_shard_results.py` 合并出 `eval_results_<suite>_merged.json`

### 7.2 视频保存

- `cfg.save_videos and (local_run_idx % save_every == 0)` 才保存
- `local_run_idx` 是**当前 shard 内已实际跑过的 task 计数** (不是绝对 task_id)，
  避免多 GPU stride 切分下部分 shard 永远不命中的 bug
- 路径: `${VIDEO_DIR}/task_${task_id:05d}/${DATE}/${DATE_TIME}--episode=N--success={True/False}--task=...mp4`

### 7.3 图像方向

- LIBERO env 直出的 `obs["agentview_image"]` 是 OpenGL 渲染原方向 (颠倒)
- 训练时 LeRobot 数据集已经是正向方向
- 评测必须 `flip_images=True`，backbone 内部翻 180° 才能跟训练对齐
- 视频保存方向由独立的 `video_flip` 控制，跟 `flip_images` 解耦

### 7.4 Resume

评测 JSON 累计写入，中断后再跑会跳过 `episodes >= num_trials_per_task` 的 task。

---

## 8. 关键代码索引

| 功能 | 文件 | 关键行 |
|---|---|---|
| benchmark 加载 / instruction 生成 | `LIBERO-plus/libero/libero/benchmark/__init__.py` | `grab_language_from_filename()` (line 46-72) |
| 4 个 suite 的 task 数 | 同上 | `task_num = [2402, 2518, 2591, 2519, 90]` (line 102) |
| OFT1_0 评测主循环 | `eval/Sai0_1/libero_plus/OFT1_0/eval_libero_plus.py` | `for task_id in pbar:` (~ line 940) |
| TaskClassificationInfo 加载 | 同上 | line 400-434 |
| metrics 聚合 | 同上 | `_aggregate_metrics()` |
| 任务 stride 切分 | 同上 | `_filter_task_ids()` |
| 多卡 launcher | `eval/Sai0_1/libero_plus/OFT1_0/scripts/run_eval_libero_plus_multi_gpu.sh` | `for SHARD_INDEX in ...` 循环 |
| 资产合并 | `eval/Sai0_1/libero_plus/OFT1_0/setup_libero_plus.py` | `merge_libero_plus_assets()` |
| 录原始方向视频 | `eval/Sai0_1/libero_plus/OFT1_0/tools/record_raw_env_video.py` | 全文件 |

---

## 9. 一句话总结

LIBERO-plus = "把原版 LIBERO 的 40 个 task 沿 7 个扰动维度复制 250 倍 → 10030 个评测任务"，
专门用来看模型在新桌面、新光照、新初始位姿、新相机角度、新指令重述下还能不能完成同样的 base task，
模型仍然只用原版 40 task 的训练数据训。比 leaderboard 公平的代价是 prompt 里会带一些
filename 残留的 perturbation 标识符，所有人都受同样影响。
