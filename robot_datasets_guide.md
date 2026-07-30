# 机器人数据集速查

这份笔记只放官方入口和最实用的查看方法。

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
