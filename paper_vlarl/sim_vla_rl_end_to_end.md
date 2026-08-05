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

### 6. 采数据的最小执行卡

如果你现在的目标只是“先把自己的仿真数据采起来”，建议先按这套最小闭环做，不要一上来追求全任务、多相机、多 embodiment：

```text
单任务
  -> 单机器人
  -> 单相机或双相机
  -> 脚本/规划器专家
  -> 成功轨迹导出
  -> 转成统一数据格式
  -> 先做 100 到 500 条成功 episode
```

第一版采集最好只保留两类任务：

```text
容易成功的结构化任务
  抓取、放置、搬运、开关、按钮、插拔

容易判定成功的任务
  物体进入目标区域、姿态对齐、夹爪闭合、目标被抬起
```

如果任务本身很难规划，先不要硬上纯 RL，先把 `privileged_state` 专家跑通，再把视觉观测和真机风格慢慢加回来。

### 7. 推荐的数据目录

建议每个数据集都显式保留下面这些文件，不要只存一堆裸 HDF5：

```text
dataset_root/
  meta/
    dataset.json
    tasks.json
    sensors.json
  episodes/
    episode_000001/
      observations.npz
      actions.npz
      rewards.npz
      dones.npz
      info.json
      video.mp4
  splits/
    train.txt
    val.txt
    test.txt
```

其中 `dataset.json` 至少要写清：

```json
{
  "name": "my_robot_sim_v1",
  "robot": "custom_arm",
  "action_type": "ee_delta",
  "action_dim": 7,
  "state_dim": 16,
  "fps": 20,
  "camera_names": ["front", "wrist"],
  "language_format": "instruction",
  "success_definition": "object_in_target_zone"
}
```

### 8. 采集循环的标准模板

你可以把第一版采集器写成下面这个逻辑：

```python
for seed in seeds:
    obs = env.reset(seed=seed)
    done = False
    while not done:
        action = expert(obs, privileged_state=env.get_privileged_state())
        next_obs, reward, terminated, truncated, info = env.step(action)

        writer.add({
            "obs": obs,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        })

        obs = next_obs
        done = terminated or truncated
```

这里最重要的是两件事：

1. 进入训练集的 `obs` 只能是模型未来真的会看到的东西，不能把仿真真值偷偷塞进去。
2. `action_type`、坐标系、控制频率、夹爪语义必须固定记录，否则后面混数据会炸。

### 9. 你现在最该做的顺序

```text
1. 选一个任务
2. 先只做仿真 reset / step / success 判定
3. 接一个脚本或规划器专家
4. 采 100 条成功轨迹
5. 导出统一格式
6. 再决定要不要接 VLA SFT
```

如果你愿意，我下一步可以直接把“你自己的机器人仿真采集规范”继续往下写成一份可落地的 `dataset schema + adapter` 草案。

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

## 对自有机器人数据的直接价值

### 1. 用少量真机示范换取大量仿真试错

自己的机器人只需要先采集少量高质量示范，用来让 π0/π0.5 学会基本动作顺序；大量失败尝试交给 RoboTwin 仿真完成。这样可以减少真机磨损、人工操作时间和安全风险。

### 2. 把机器人差异压缩到 adapter

VLA、PPO、rollout 和分布式训练代码可以复用。你主要需要实现：

```text
observation.images.*  -> 相机图像
observation.state     -> 关节/末端状态
action                -> 你的控制器输入
prompt                -> 任务语言
reward/termination    -> 成功判定和终止条件
```

例如自己的 7DoF 单臂可以统一为 `[dx, dy, dz, droll, dpitch, dyaw, gripper]`，再在仿真 wrapper 中转换为实际关节控制命令。

### 3. 强制明确 action 物理语义

自己的数据可能同时存在 absolute joint、joint delta、EE delta、velocity 和 torque。路线 A 要求在进入 SFT 前记录：

```text
action_type、action_dim、坐标系、控制频率、gripper 语义、chunk 长度
```

这可以避免把形状相同但含义不同的动作直接混合，也方便后续接入 DROID、FMB、RoboTwin、RoboDojo 等数据源。

### 4. 用仿真状态自动产生 reward

只要你的自有机器人仿真环境能判断物体位置、接触、姿态或是否进入目标区域，就可以自动生成 `reward/success/termination`，不需要逐帧人工标注“好动作”。真机数据负责行为先验，仿真状态负责 RL 信号。

### 5. 在仿真中注入 sim-to-real 扰动

应把真实设备的变化加入 RoboTwin 或自定义仿真环境：相机内外参误差、关节零位误差、控制延迟、动作噪声、物体摩擦和光照纹理。RL 训练得到的策略因此能在一组扰动内保持成功，而不是只记住一个理想场景。

### 6. 直接测试 VLA 的 action chunk 部署因素

π0.5 一次输出未来一段动作块。路线 A 能在仿真中调试 `chunk length`、重规划频率、控制器延迟、动作平滑和限幅，这些因素通常比单步 BC loss 更直接地决定真机长时序成功率。

### 7. 训练/评测 seed 分离，结果更可信

RLinf 为训练和评测提供独立 seed 文件。自有机器人做实验时也应保留这条规则，否则模型可能记住固定初始物体位置，造成成功率虚高。

## 自有机器人必须补的部分

路线 A 复用的是训练基础设施，不会自动替你完成机器人建模。至少需要补齐：

```text
URDF/仿真资产和控制器
观测字典与相机同步
action 到控制器的映射
success/reward/termination
HDF5/LeRobot 转换器
真机相机 topic、关节顺序、控制频率 adapter
动作限幅、碰撞检测和急停
```

建议顺序是：单臂单任务单相机 -> 100-500 条示范 SFT -> 32-64 个并行环境 PPO -> 加入延迟和视觉/动力学随机化 -> 独立 seed 评测 -> sim-to-real。

## 能否直接复刻自己的工作环境并自动部署到真机

可以，这条路线在研究上是成立的，通常叫 **digital twin + sim-to-real**。理想目标是：

```text
自己的机器人 CAD/URDF/USD
  -> 仿真工作站和任务
  -> 自动专家轨迹
  -> VLA SFT
  -> 仿真在线 RL
  -> 同一 checkpoint 的真机 rollout
```

但“直接部署”必须满足观测和动作接口一致。实际部署通常存在两个转换函数：

```text
o_real = T_obs(real_camera, real_joint_state, calibration)
a_real = T_action(a_policy, controller_rate, limits, calibration)
```

只有当 `T_obs` 和 `T_action` 已经把真机传感器、坐标系、关节顺序、控制频率、动作限幅对齐后，才可以复用同一个策略权重。RLinf/RoboTwin 负责训练闭环，不会自动完成这两个转换。

### 哪些情况下可行性高

- 刚性物体、桌面抓取、放置、搬运、按钮和开关等任务。
- 机器人运动学、关节限制、夹爪控制模式已知。
- 仿真和真机使用相同的 action 语义，例如都使用 EE delta 或 joint delta。
- 相机安装位置、分辨率、帧率和曝光能被标定或固定。
- 能通过真实日志做系统辨识，估计延迟、摩擦、控制误差和传感器噪声。

### 哪些情况下不能指望零适配

- 布料、液体、软体物体、强接触和遮挡严重的任务。
- 真机控制器内部有未知滤波、插值、阻抗环或安全限幅。
- 仿真使用绝对关节 target，而真机接口实际接收速度/力矩。
- 训练只有纯渲染图像，没有真实相机风格或少量真实数据校准。

### 对自有机器人的推荐实现

如果你的机器人不是 ALOHA 双臂，不建议强行把所有内容改成 RoboTwin 的 14 维接口。更稳妥的做法是：

```text
Isaac Lab / ManiSkill / SAPIEN
  -> 自定义机器人和工作站数字孪生
  -> 自定义 Gymnasium env + success reward
  -> RLinf env wrapper
  -> OpenPI π0.5 或 GR00T SFT/RL
  -> 真机 observation/action adapter
```

如果你的机器人就是 ALOHA 或 RoboTwin 已支持的 embodiment，则直接复用 RoboTwin 的任务和 RLinf 配置，工作量会小很多。

### 推荐的迁移顺序

```text
1. 先让仿真和真机使用同一套 action/state 字段。
2. 用真机采集少量示范，确认 VLA 在仿真中能完成基本任务。
3. 用仿真成功 reward 做 PPO/GRPO，加入视觉和动力学随机化。
4. 用真实传感器日志校准相机、关节零位、控制延迟和动作噪声。
5. 先在真机低速度、低力矩、空场景下做 1-step/短 horizon 测试。
6. 逐步放开动作范围，再进入完整任务 rollout。
```

因此结论是：**仿真自动生成数据并训练，再用相同 VLA 权重启动真机，是可行且值得做的路线；但必须保留一个明确的 sim-to-real adapter 和安全验证阶段，不能把仿真 action 文件直接当作真机控制命令。**

## 当前目标修正：只在仿真里自动采数据

如果当前目标是“复刻自己的机器人和工作环境，然后在仿真中自动生成训练数据”，那么 **第一阶段不需要 checkpoint**。

`checkpoint` 只是已经训练好的模型权重，例如：

```text
SFT checkpoint：VLA 学会专家示范后的权重
RL checkpoint：VLA 在仿真奖励上继续优化后的权重
```

它们用于“让模型自己 rollout”或“从已有策略继续训练”，不是仿真数据采集的必需品。

### 仿真采数据真正需要的东西

```text
数字孪生机器人 + 工作台/物体
        ↓
任务 reset 和成功判定
        ↓
专家生成器（规划器 / 状态机 / 特权状态 RL）
        ↓
仿真执行器
        ↓
图像、state、action、语言、seed、成功标签
```

专家生成器可以有三种形式：

1. **脚本 + IK/运动规划器**：适合抓取、搬运、放置、按按钮等结构化任务，最稳定，最适合作为第一版自动采集器。
2. **使用仿真特权状态训练的 PPO/SAC policy**：输入物体位姿、接触和机器人真实状态，先学一个 state policy，再用它执行时同步保存相机图像，得到视觉数据。
3. **少量遥操作示范 + Mimic/轨迹扩增**：当任务很难用规则规划时，先录少量示范，再在仿真中改变物体位置和场景生成更多轨迹。

### RoboTwin 的 `collect_data.sh` 属于哪一种

RoboTwin 的采集脚本不是依赖一个 VLA checkpoint，而是调用任务脚本和规划后端（例如 `mplib`/`curobo`），搜索可行 seed、执行专家动作并保存轨迹。因此路线 A 的正确顺序是：

```text
RoboTwin/自定义仿真专家
  -> 自动采集成功轨迹
  -> LeRobot 转换
  -> π0/π0.5 SFT 得到 checkpoint
  -> RLinf 在仿真中用 VLA checkpoint 做在线 RL
```

### 自己的机器人需要实现的采集接口

至少要有以下接口，RLinf 后续才能接上：

```python
obs = env.reset(seed=seed)
action = expert(obs, privileged_state=state)
next_obs, reward, terminated, truncated, info = env.step(action)
writer.add(obs=obs, action=action, language=task, info=info)
```

专家可以读取 `privileged_state`，但写入 VLA 数据的 observation 不应包含仿真专属的物体真值，否则训练出的 VLA 会发生信息泄漏。采集时应同时保存：

```text
相机 RGB/深度（按最终 VLA 输入选择）
机器人 state
真正发送给控制器的 action
任务语言
episode seed 和随机化参数
success、terminated、truncated
```

### 什么时候才需要 checkpoint

```text
阶段 0：规划器/状态机采集数据        不需要 checkpoint
阶段 1：用数据训练 π0/π0.5            产生 SFT checkpoint
阶段 2：用 SFT VLA 在仿真 rollout     需要 checkpoint
阶段 3：RLinf PPO/GRPO 在线微调        从 SFT checkpoint 开始
阶段 4：评测或真机部署                 使用最终 RL checkpoint
```

所以你现在要做的是“先把自有机器人仿真和专家采集器打通”，而不是先找一个 checkpoint。checkpoint 会在第一批数据采完并完成 SFT 之后才出现。
