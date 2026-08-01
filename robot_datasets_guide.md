# 机器人数据集速查

这份笔记只放官方入口和最实用的查看方法。

## 一页版整理

| 数据集 / 基准 | 类型 | 规模 / 形态 | 数据格式 / 入口 | 适合先做什么 | 注意点 |
|---|---|---:|---|---|---|
| RoboTwin 2.0 | 仿真数据生成器 + 双臂 benchmark | 50 个双臂任务，5 种机器人形态；RoboTwin-OD 含 731 个实例、147 类 | GitHub / docs，兼容 LeRobot 生态 | 看任务配置、仿真采样、domain randomization | 更偏 sim-to-real 与可扩展生成，不是先看真实遥操作数据的首选 |
| RoboDojo | sim-and-real benchmark / 评测框架 | 任务集合 + 配置系统 | 官网 Wiki / GitHub | 跑 install、download、evaluation 三步 | 更像 benchmark 框架；如果目标是“研究数据字段”，优先级低于 DROID/FMB |
| DROID | 真实机器人遥操作数据集 | 76k demonstrations，350h，564 scenes，86 tasks | RLDS 1.7TB / raw 8.7TB / debug subset 2GB | 先下 `droid_100`，看 visualizer 和 `openpi/examples/droid/README.md` | 全量很大；先用 debug subset 跑通读取、归一化和动作字段 |
| FMB | 真实功能性操作数据集 | 22,550 expert trajectories；单物体 full dataset 约 545GB | `.npy` 轨迹文件 / Hugging Face | 下 Assembly 小包，用 `numpy.load(...).item()` 看轨迹 key | 字段偏底层传感和 TCP 状态；适合理解轨迹结构，不一定直接兼容 VLA 训练 |

## 怎么选

- 想最快理解真实机器人数据结构：先看 **DROID**，再看 **FMB**。
- 想复现 VLA / openpi 风格训练：优先看 **DROID**，因为仓库里已有对应 example。
- 想看仿真任务、LeRobot 格式和 sim-to-real：看 **RoboTwin 2.0**。
- 想跑统一 benchmark 或研究评测流程：看 **RoboDojo**。
- 存储上先按“最小可运行”处理：DROID 用 `droid_100`，FMB 下小包，RoboTwin/RoboDojo 先跑文档里的 quick demo。

## 1. RoboTwin 2.0

- 官网: https://robotwin-platform.github.io/
- GitHub: https://github.com/RoboTwin-Platform/RoboTwin

要点:
- 标题是 *A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation*
- RoboTwin-OD 物体库: 731 个实例, 147 个类别
- 50 个双臂任务
- 5 种机器人形态
- 官网摘要里强调 synthetic data + real demos 的 sim-to-real 提升

怎么看:
- 先看官网首页的 abstract
- 再看 GitHub README 和 `docs/`
- 目前我能稳定核实到的是官网和 repo，数据下载入口以 repo / docs 为准

## 2. RoboDojo

- 官网: https://robodojo-benchmark.com/
- Wiki: https://robodojo-benchmark.com/doc/
- GitHub: https://github.com/RoboDojo-Benchmark/RoboDojo

要点:
- 更像 unified sim-and-real benchmark
- 不是单一静态数据集，更偏评测框架和任务集合

怎么看:
- 先看官网文档首页
- 再看 `Usage` 和 `Configurations`
- 如果你要复现，优先找它的 install / download / evaluation 三段

## 3. DROID

- 官网: https://droid-dataset.github.io/
- 数据集页: https://droid-dataset.github.io/droid/the-droid-dataset
- Developer docs: https://droid-dataset.github.io/droid/
- GitHub: https://github.com/droid-dataset/droid
- Policy learning repo: https://github.com/droid-dataset/droid_policy_learning

要点:
- 76k demonstrations / 350h interaction data
- 564 scenes, 86 tasks
- 官方提供 RLDS 和 raw 两种格式

下载:
```bash
# Full dataset in RLDS, 1.7TB
gsutil -m cp -r gs://gresearch/robotics/droid <target_dir>

# Debug subset, 100 episodes, 2GB
gsutil -m cp -r gs://gresearch/robotics/droid_100 <target_dir>

# Raw stereo HD, 8.7TB
gsutil -m cp -r gs://gresearch/robotics/droid_raw <target_dir>
```

怎么看:
- 先看 interactive dataset visualizer
- 再看官方 Colab
- 如果你只想先跑通代码，先下 `droid_100`
- 本地训练入口可直接看仓库里的 `openpi/examples/droid/README.md`

## 4. FMB

- 官网: https://functional-manipulation-benchmark.github.io/
- Dataset page: https://functional-manipulation-benchmark.github.io/dataset/index.html
- Procedure: https://functional-manipulation-benchmark.github.io/procedure/index.html
- GitHub: https://github.com/rail-berkeley/fmb
- Hugging Face: https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark

要点:
- 22,550 expert demonstration trajectories
- 数据是 `.npy` 轨迹文件
- 单物体任务 full dataset 约 545 GB
- 多物体任务 assembly 约 86 / 77 / 70 GB

每条轨迹里常见字段:
- `obs/side_1`, `obs/side_2`, `obs/wrist_1`, `obs/wrist_2`
- `obs/tcp_pose`, `obs/tcp_vel`, `obs/tcp_force`, `obs/tcp_torque`
- `action`, `primitive`, `object_id` / `object_info`

怎么看:
- 先下 `Full Dataset` 之外的小包，比如 `Assembly 1`
- 直接用 `numpy.load(path, allow_pickle=True).item()` 打开 `.npy`
- 先检查 `traj.keys()`，再看图像、动作、`tcp_pose`
- 用官网 trajectory visualizer 看轨迹和点云

## 我建议你的查看顺序

1. DROID
2. FMB
3. RoboTwin 2.0
4. RoboDojo

原因:
- DROID 和 FMB 最容易先看懂真实机器人数据结构
- RoboTwin 2.0 适合看仿真/LeRobot 格式
- RoboDojo 更偏 benchmark 和评测流程

## 最快的本地查看法

### 看 Hugging Face 数据

```python
from datasets import load_dataset
ds = load_dataset("lerobot/robotwin_unified", split="train")
print(ds)
print(ds[0].keys())
```

### 看 `.npy` 轨迹

```python
import numpy as np
traj = np.load("xxx.npy", allow_pickle=True).item()
print(traj.keys())
print(traj["obs/side_1"].shape)
```

### 看 DROID

- 先用 `droid_100`
- 再看官方 visualizer
- 最后再考虑拉全量 1.7TB

### 看 RoboDojo

- 先跑 `Quick Evaluation`
- 再看 `env_cfg/` 下的 scene / robot / camera 配置
- 它更适合“跑 benchmark”，不是“翻一堆静态样本”
