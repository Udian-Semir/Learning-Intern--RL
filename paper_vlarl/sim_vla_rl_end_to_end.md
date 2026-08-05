# 仿真机器人数据采集 -> VLA -> RL 完整链路

本文只收录能够拼出下面这条闭环的项目：

```text
仿真环境/任务定义
  -> 专家、脚本或遥操作采集轨迹
  -> 数据格式转换与归一化
  -> VLA 监督微调（SFT/BC）
  -> 仿真在线 RL（PPO/GRPO/SAC 等）
  -> 固定 seed 闭环评测，必要时部署到真机
```

## 先给结论

目前最接近“拿下来就能通”的路线是：

```text
RoboTwin 2.0
  + RLinf
  + OpenPI pi0/pi0.5 或 OpenVLA-OFT
```

RoboTwin 负责任务、资产、域随机化和仿真数据采集；RLinf 负责 OpenPI/OpenVLA 的 SFT、PPO/GRPO 在线 RL、日志和评测。两边有专门的 `RLinf_support` 分支以及现成 YAML 配置，但不是一个仓库内的单条命令。

第二条值得做的是：

```text
RoboCasa365
  + RoboCasa HDF5/LeRobot 数据转换
  + OpenPI 或 GR00T
  + RLinf
```

它的场景和任务规模更大，数据接口更规范，但安装资源和动作 schema 更复杂。

## 项目筛选表

| 路线 | 仿真/采集 | VLA SFT | 在线 RL | 闭环评测/部署 | 完整度判断 |
| --- | --- | --- | --- | --- | --- |
| **RoboTwin 2.0 + RLinf + pi0/pi0.5** | 有 `collect_data.sh`，100,000+ 轨迹，支持自己采集 | RLinf 有 `robotwin_sft_openpi(_pi05)`；RoboTwin 还有 PI0/OpenVLA-OFT | RLinf 有 PPO/GRPO/DAgger，RoboTwin 有 RL reward 分支 | RLinf 有固定 train/eval seed、视频和 `env/success_once` | **首选，最完整** |
| **RoboCasa365 + RLinf + OpenPI/GR00T** | `collect_demos.py` 遥操作采集；自动轨迹和人类演示；HDF5 -> LeRobot | RoboCasa 有 Diffusion Policy、pi、GR00T；RLinf 有 OpenPI recipe | RLinf 有 RoboCasa/RoboCasa365 PPO 配置 | 有 task-soup、pretrain/target split 和评测脚本 | **完整，但资源重** |
| **Isaac Lab Mimic + GR00T + RLinf** | Isaac Lab 遥操作录制；Mimic 自动扩增示范 | Mimic 原生偏 BC/robomimic；GR00T 可吃 LeRobot；需自己接 VLA SFT | Isaac Lab 原生 PPO；RLinf 支持 IsaacLab/GR00T RL | Isaac Lab 有 replay/eval，GR00T 有 sim/real eval | **组件最全，适配工作最多** |
| **RL-VLA³ + RLinf 生态** | 支持 LIBERO、ManiSkill、MetaWorld、RoboCasa rollout；不是数据采集器 | 使用 π0/π0.5、GR00T、OpenVLA-OFT SFT checkpoint | 异步 Simulator-Generator-Trainer 在线 RL | 有 `run_libero.sh` 等脚本和多环境配置 | **RL 最强，需外接数据采集/SFT** |
| **ManiSkill + RLinf/OpenPI** | GPU 并行仿真、RGB-D/分割、可写 expert collector | RLinf 有 `pi0_maniskill`/`pi05_maniskill` SFT | RLinf 支持 ManiSkill PPO/RLT 等 | 可用 ManiSkill eval 和 RLinf 日志 | **最快做算法原型，非现成数据闭环** |

## 路线 A：RoboTwin 2.0 + RLinf（推荐）

### 代码与论文

- RoboTwin：<https://github.com/RoboTwin-Platform/RoboTwin>
- RoboTwin 的 RL 分支：<https://github.com/RoboTwin-Platform/RoboTwin/tree/RLinf_support>
- RLinf：<https://github.com/RLinf/RLinf>
- RLinf RoboTwin 文档：<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html>
- RoboTwin 2.0 论文：<https://arxiv.org/abs/2506.18088>
- RLinf-VLA 论文：<https://arxiv.org/abs/2510.06710>

### 1. 安装环境与资产

```bash
git clone https://github.com/RLinf/RLinf.git
cd RLinf

git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
cd RoboTwin
bash script/_download_assets.sh

export ROBOT_PLATFORM=ALOHA
export PYTHONPATH=$PWD:$PYTHONPATH
```

也可以直接使用 RLinf 的 RoboTwin Docker 镜像：

```bash
docker run -it --rm --gpus all --shm-size 32g --network host \
  -v "$PWD":/workspace/RLinf \
  rlinf/rlinf:agentic-rlinf0.3-robotwin
```

### 2. 在仿真中采集轨迹

```bash
cd /path/to/RoboTwin
bash collect_data.sh beat_block_hammer demo_randomized 0
```

脚本会先为目标采集数量寻找可行 seed，再回放 seed 生成轨迹。RoboTwin 2.0 官方同时提供 100,000+ 条预采集轨迹；由于任务配置、机器人 embodiment、相机和域随机化都可改，官方更建议自己采集。

RoboTwin 原生数据不是直接的 LeRobot 单目录：通常是 HDF5 加 JSON/PKL sidecar。做 VLA SFT 前必须使用对应 policy 的转换脚本，不能把 HDF5 随便改名后直接喂给 OpenPI。

### 3. VLA SFT

RLinf 已提供 OpenPI 的 RoboTwin dataconfig，配置名为 `pi0_aloha_robotwin` 和 `pi05_aloha_robotwin`。示例配置：

```text
examples/sft/config/robotwin_sft_openpi.yaml
examples/sft/config/robotwin_sft_openpi_pi05.yaml
```

启动入口：

```bash
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05
```

核心接口约定如下：

| 字段 | RoboTwin/RLinf 约定 |
| --- | --- |
| 视觉 | 头部 RGB，可选左右腕部 RGB，训练时通常 resize/crop 到 224 |
| state | ALOHA 双臂本体状态，RLinf env 文档中为 14 维 |
| action | 双臂 action chunk；示例 `num_action_chunks=50`、`action_dim=14` |
| language | 任务描述字符串 |
| normalization | `unnorm_key` 必须与 SFT 和 RL 配置完全一致 |

如果使用自己采集的数据，先把数据转换到 LeRobot/对应 dataconfig，再计算 state/action 的 norm stats；`unnorm_key` 不一致会导致反归一化错误。

### 4. 在线 RL 微调

以 `adjust_bottle` 和 π0.5 为例：

```bash
bash examples/embodiment/run_embodiment.sh \
  robotwin_adjust_bottle_ppo_openpi_pi05
```

OpenVLA-OFT + GRPO 的例子：

```bash
bash examples/embodiment/run_embodiment.sh \
  robotwin_place_empty_cup_grpo_openvlaoft
```

RLinf 的 embodied worker 会同时启动 actor、rollout 和 RoboTwin env：

```text
VLA rollout -> RoboTwin step -> task reward -> advantage -> PPO/GRPO update
```

目前 RLinf_support 分支已经给一批 RoboTwin 任务写了 RL reward；不是所有任务都已经覆盖。先选 `adjust_bottle`、`place_empty_cup`、`move_can_pot`、`lift_pot`、`handover_block` 等配置名能找到的任务。

### 5. 固定 seed 评测

```bash
bash evaluations/run_eval.sh \
  robotwin_adjust_bottle_openpi_pi05_eval
```

训练指标看 `env/success_once`；可在 YAML 中开启 `video_cfg.save_video` 保存评测视频。RLinf 文档报告的一个参考结果是，OpenVLA-OFT 在七个 RoboTwin 任务上的平均成功率由 SFT 的 28.79% 提高到 GRPO 的 86.16%，但这不是用户自采数据的保证值。

## 路线 B：RoboCasa365 + RLinf

### 代码与论文

- RoboCasa：<https://github.com/robocasa/robocasa>
- RoboCasa365 论文：<https://robocasa.ai/assets/robocasa365_iclr26.pdf>
- RLinf RoboCasa365 文档：<https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robocasa365.html>
- GR00T：<https://github.com/NVIDIA/Isaac-GR00T>

RoboCasa365 的特点是 365 tasks、2,500+ 厨房场景、3,200+ 物体，以及人类和自动生成的机器人示范数据。它更适合研究任务泛化和 task-soup，而不是先做最小可行原型。

### 采集与转换

```bash
python robocasa/scripts/collect_demos.py --env <env-name>
python robocasa/scripts/dataset_scripts/convert_hdf5_lerobot.py \
  --raw_dataset_path <raw.hdf5>
```

转换器当前的关键数据接口是：20 FPS、三路图像、16 维 state、12 维 action，并保存 `next.reward` 和 `next.done`。这与 RoboTwin 的 14 维 ALOHA action 不能直接混合，必须经过 embodiment/action adapter。

### RLinf 闭环入口

RLinf 的 RoboCasa365 配置默认使用 OpenPI、PandaOmron、12 维 action schema 和 `atomic_seen` task-soup：

```bash
bash requirements/install.sh embodied --model openpi --env robocasa365
source .venv/bin/activate
bash examples/embodiment/run_embodiment.sh \
  robocasa365_opendrawer_ppo_openpi
bash evaluations/run_eval.sh robocasa365_eval_openpi
```

如果要加入自己采集的数据，先用 RoboCasa 的 HDF5 -> LeRobot 转换器做 SFT，再把 SFT checkpoint 路径替换到 RLinf 的 actor/rollout 配置。官方 benchmark 数据的 task registry 与自采数据的 task id 不应混用。

## 路线 C：Isaac Lab Mimic + GR00T + RLinf

### 适合什么

这条路线适合你想研究“少量人工示范 -> 大量仿真合成轨迹 -> VLA/模仿学习 -> RL”的数据生成方法。Isaac Lab Mimic 本身主要解决示范扩增和 imitation learning，不是一个开箱即用的 VLA online-RL recipe，所以需要自己写数据 adapter。

### 官方数据生成链

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --dataset_file ./datasets/dataset.hdf5 \
  --num_demos 10 --teleop_device keyboard

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --enable_cameras --auto \
  --input_file ./datasets/dataset.hdf5 \
  --output_file ./datasets/annotated_dataset.hdf5

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --enable_cameras --num_envs 20 --generation_num_trials 10 \
  --input_file ./datasets/annotated_dataset.hdf5 \
  --output_file ./datasets/generated_dataset.hdf5
```

生成的数据可以转成 LeRobot/GR00T 所需格式，再走 GR00T 的 finetune 和 sim eval。需要在线 VLA RL 时，可把 Isaac Lab 作为 RLinf 的 env，或使用 RLinf 已有的 IsaacLab/GR00T 配置。这里的关键工作是把 action space、camera names、state layout 和 episode termination 对齐。

## 路线 D：RL-VLA³

- 仓库：<https://github.com/Haoran0301/RL-VLA3>
- 上游基础设施：<https://github.com/RLinf/RLinf>

RL-VLA³ 把 Simulator、Generator 和 Trainer 异步解耦，支持 LIBERO、ManiSkill、MetaWorld、RoboCasa，以及 π0/π0.5、GR00T N1.5、OpenVLA-OFT。入口包括：

```bash
bash scripts/run_libero.sh <config_name>
bash scripts/run_maniskill.sh <config_name>
bash scripts/run_robocasa.sh <config_name>
```

它是很好的高吞吐 RL 后端，但仓库没有像 RoboTwin 的 `collect_data.sh` 那样的仿真数据采集入口。因此正确的拼法是：

```text
RoboTwin/RoboCasa/ManiSkill 采集与转换
  -> OpenPI/OpenVLA/GR00T SFT
  -> RL-VLA³ 异步在线 RL
```

不要把 RL-VLA³ 单独宣传成“从数据采集到 VLA 预训练全部完成”。

## 统一数据接口建议

四个项目最终都应先落到下面的 episode 契约，再进入 VLA 或 RL：

```json
{
  "episode_id": "robotwin_adjust_bottle_000001",
  "task_id": "adjust_bottle",
  "language": "adjust the bottle",
  "robot": "aloha",
  "fps": 20,
  "observations": {
    "images": {"front": "...", "wrist_left": "...", "wrist_right": "..."},
    "state": "float32[T, state_dim]"
  },
  "actions": "float32[T, action_dim]",
  "action_type": "delta|absolute|chunk",
  "rewards": "float32[T]",
  "terminated": "bool[T]",
  "truncated": "bool[T]",
  "source": {"simulator": "robotwin", "seed": 0, "domain_randomization": {}}
}
```

融合时必须显式保存 `robot`、`action_type`、`state_dim`、`action_dim`、坐标系和 gripper 语义；不能只把不同数据集的 `action` 列拼起来。RoboTwin 的 ALOHA 双臂 14 维、RoboCasa/PandaOmron 的 12/16 维以及 ManiSkill 的任务特定 action space 应分别保留 adapter。

## 论文与代码目录

本目录已补充以下可直接阅读的 PDF：

| 文件 | 论文 | 用途 |
| --- | --- | --- |
| `05_RLinf_Flexible_Efficient_LargeScale_RL_2025.pdf` | RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation | 分布式 RL 基础设施 |
| `06_RLinf_VLA_Unified_Efficient_VLA_RL_2025.pdf` | RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models | VLA + RL 系统 |
| `07_piRL_Online_RL_Flow_VLA_2025.pdf` | πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models | π0/π0.5 风格 flow VLA 的在线 RL |
| `08_SAC_Flow_2025.pdf` | SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling | flow policy + SAC |
| `09_DSRL_Latent_Space_RL_2025.pdf` | Steering Your Diffusion Policy with Latent Space Reinforcement Learning | diffusion policy latent RL |
| `10_RoboTwin_2_Scalable_Data_Generator_2025.pdf` | RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation | 仿真采集与 benchmark |

已有的 ReinFlow、DPPO、QGF、STARE-VLA PDF 和中文精读也保留在本目录。RoboCasa365 PDF 的官方直链在当前网络下会返回截断文件，因此只保留官方论文链接，不把损坏 PDF 留在目录。

## 推荐执行顺序

如果目标是尽快验证完整闭环，按下面顺序：

1. 先用 LIBERO + RLinf/OpenPI 跑通单机环境和 PPO，确认训练入口、奖励、checkpoint、评测都正常。
2. 换到 RoboTwin `adjust_bottle`，用官方 SFT checkpoint 做一次 RL，再替换成自己采集的数据。
3. 再做 RoboCasa365 或 Isaac Lab Mimic，解决多相机、不同 embodiment 和 action adapter。
4. 最后把 RL-VLA³ 接到已经统一的数据和环境接口上，做多 GPU 异步吞吐实验。

