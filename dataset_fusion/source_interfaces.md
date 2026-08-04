# 四个来源的接口与格式

本文把已知事实和需要在实际样本上核验的部分分开写。不同下载版本可能增删字段，最终以实际 `meta/info.json`、RLDS spec 或轨迹字典为准。

## 官方入口与规模速查

| source | 官方入口 | 规模/版本备注 |
| --- | --- | --- |
| DROID | [dataset](https://droid-dataset.github.io/droid/the-droid-dataset) · [repo](https://github.com/droid-dataset/droid) | 约 76k demonstrations、350 h、564 scenes、86 tasks；RLDS 约 1.7 TB，raw 约 8.7 TB；完整训练版本常写作 `droid/1.0.1` / builder `droid_101`，`droid_100` 是 debug subset |
| FMB | [dataset](https://functional-manipulation-benchmark.github.io/dataset/index.html) · [repo](https://github.com/rail-berkeley/fmb) | 约 22,550 expert trajectories；raw `.npy` 和官方 RLDS builder 并存 |
| RoboTwin 2.0 | [repo](https://github.com/RoboTwin-Platform/RoboTwin) · [HF data](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset) | 当前公开 bundle 约 460 个 zip，50 tasks、230 task/embodiment 组合，按 clean/randomized 名义约 126,500 episodes |
| RoboDojo | [repo](https://github.com/RoboDojo-Benchmark/RoboDojo) · [download docs](https://robodojo-benchmark.com/doc/usage/install-and-download/) | sim HDF5 约 523 GB；RGB-D HDF5 约 4.5 TB；LeRobot v3 joint/EE 各约 120 GB；real HDF5 约 273 GB；公开 sim training export 是 35 tasks，不等于完整 benchmark task 数 |

## 1. DROID

### 原始载体

- 官方大规模版本：RLDS / TensorFlow Datasets trajectory。
- 原始高清版本：按记录会话组织的图像、机器人状态、动作和 annotation。
- 本仓库的 full-DROID loader 使用 `DroidRldsDataset`，见 [`droid_rlds_dataset.py`](../pi_0_robot/openpi/src/openpi/training/droid_rlds_dataset.py)。

### 本仓库 loader 已核验的字段

```text
observation.exterior_image_1_left
observation.exterior_image_2_left
observation.wrist_image_left
observation.joint_position
observation.gripper_position
action_dict.joint_position       # 可选训练动作空间
action_dict.joint_velocity       # 可选训练动作空间
action_dict.gripper_position
language_instruction
language_instruction_2
language_instruction_3
```

官方 `droid_101` RLDS 的每个 step 还有 `is_first`、`is_last`、`is_terminal`、`reward`、`discount`。三路 RGB 都是左目 JPEG 解码后的 `uint8[180,320,3]`；状态包括 `joint_position[7]`、`gripper_position[1]`、`cartesian_position[6]`。可靠动作应明确选择 `action_dict` 下的字段：`joint_position[7]`、`joint_velocity[7]`、`cartesian_position[6]`、`cartesian_velocity[6]`、`gripper_position[1]` 或 `gripper_velocity[1]`。不要把顶层 `action[7]` 的历史注释当作唯一语义来源。

原始 DROID 记录层级还包含 `metadata_*.json`、`trajectory.h5`、MP4 / SVO 录像。原始 HDF5 中可有 robot/control/camera 的独立时间戳，以及相机内外参；RLDS 为训练便利而不保证保留这些标定与时间信息。DROID 的机器人控制频率为 15 Hz；不能把视频帧率和 control dt 混为一谈。

本地 loader 会随机选择一个 exterior camera，并从三个语言标注中随机选择一个。融合时应保留所有可用相机，用 `image_mask` 表示有效性；随机 view augmentation 放到训练阶段。

### 动作注意事项

- DROID 可使用关节**绝对位置**或关节**速度**，两者不能混为一种标签。
- `openpi` 的 RLDS joint-position 配置在模型输入前转为相对当前 state 的 delta，输出时再转回 absolute joint target。
- 转到 `cartesian_delta_abs_gripper_v1` 时，需使用 Franka 的准确 URDF、TCP 定义、关节顺序、时间戳和单位，对相邻目标位姿做 FK 再求 delta。

## 2. FMB（Functional Manipulation Benchmark）

### 原始载体

- 每条演示通常是可 pickle 的 `.npy` trajectory dictionary，而不是 RLDS / JSON。
- 视觉、状态和动作字段在 `.npy` 内；常见字段如下。

```text
obs/side_1, obs/side_2
obs/wrist_1, obs/wrist_2
obs/tcp_pose, obs/tcp_vel
obs/tcp_force, obs/tcp_torque
action
primitive
object_id / object_info
```

官方 builder 的常见统一字段还包括 RGB / depth 四视角、`joint_pos[7]`、`joint_vel[7]`、`eef_pose[7]`、`eef_vel[6]`、`eef_force[3]`、`eef_torque[3]`、gripper 标量和 `action[7]`。`eef_pose` 使用 `[x,y,z,qx,qy,qz,qw]`，必须显式记录 `quat_xyzw`。公开轨迹未必带逐 step timestamp，通常只能按 nominal 10 Hz 构造时间轴；相机采集频率不能取代 action dt。

### 接入原则

- `tcp_pose` 是转换 Cartesian action 的优先依据，但必须核验 pose 的 frame、旋转编码、单位和 action 是 target、delta 还是 velocity。
- 力、力矩、primitive、object id 是可选 supervision，应放在 optional stream，并有独立 mask，不能假定四个 source 都有。
- 侧视/腕视相机要按实际安装位置映射到 `base_0`、`left_wrist_0`、`right_wrist_0`；不能只按文件名猜左右手。

FMB 原生控制动作已知为 `[dx,dy,dz,droll,dpitch,dyaw,gripper]`：前三项为米、后三项为 Euler 增量 radians；控制为约 10 Hz。最后一维语义是 `1=closed, 0=open`，而本规范的 canonical gripper 定义为 `0=closed, 1=open`，转换时必须显式取反并记录。力/力矩坐标系在不同公开版本和 checkpoint 约定中可能不同，必须标注 frame，不能跨版本直接混用。

FMB 的相机内参可能作为独立 calibration asset 提供，轨迹 builder 不一定将内外参写入每条 trajectory；没有可信 `T_base_camera` 时，`calibration_mask=false`，而不是填单位矩阵。

## 3. RoboTwin 2.0

### 原始载体

- 官方原生数据是 task / embodiment / variation 打包的 zip，不是原生 LeRobot。
- 典型结构：`seed.txt`、`scene_info.json`、`_traj_data/episodeN.pkl`、`data/episodeN.hdf5`、`instructions/episodeN.json`、`video/episodeN.mp4`。
- 官方提供的 LeRobot 转换脚本是 baseline 示例，不是无损原始格式；它会选择 prompt、重编码图像并丢失部分 calibration / scene metadata。

### 已核验的原始 HDF5 接口

```text
/joint_action/left_arm, /joint_action/right_arm
/joint_action/left_gripper, /joint_action/right_gripper
/joint_action/vector
/endpose/left_endpose, /endpose/right_endpose       # 依 recorder config 可选
/endpose/left_gripper, /endpose/right_gripper
/observation/{camera}/rgb                            # NUL padded JPEG bytes
/observation/{camera}/intrinsic_cv                   # [T,3,3]
/observation/{camera}/extrinsic_cv                   # [T,3,4]
/observation/{camera}/cam2world_gl                   # [T,4,4]
```

可选字段有 `depth`、第三视角 RGB、mesh / actor segmentation、pointcloud。adapter 必须递归枚举真实 HDF5 路径，不能把某个 sample 的字段写死为全体 schema。JPEG byte string 需截去 NUL padding 后用 `cv2.imdecode` 解码。

### 必须导出的最低字段

```text
task language / task id
robot joint targets
camera frames + camera name
robot model / URDF version
end-effector convention definition
camera intrinsics and extrinsics
sim seed, task variation, collection success provenance
```

### 动作注意事项

- `/joint_action/vector` 的顺序是 `[left arm joints, left gripper, right arm joints, right gripper]`。6-DoF 双臂 embodiment 常为 14 维；Franka 双臂可为 16 维，不能直接补零混训。
- 这些 joint action 存的是 sampled drive target / qpos-style target，官方训练转换使用 `state[t] -> action[t]=state[t+1]`，即下一时刻绝对 joint target；不是 Cartesian delta，也不是 velocity。
- `endpose/*_endpose` 的 7 维顺序为 `[x,y,z,qw,qx,qy,qz]`，且包含 RoboTwin 特有的 embodiment / gripper offset 变换。不得把它直接标成通用 TCP pose；须保留其原始 convention 与来源 config。
- 原始 HDF5 无真实 timestamp。sim timestep 是 1/250 s、默认 `save_freq=15`，但分段首尾会额外记录帧，故只可标为 `timestamp_source=synthetic_nominal_15hz`，不能宣称严格同步。
- 每 episode 的 `instructions/episodeN.json` 是 `seen` / `unseen` prompt 列表，不是唯一 prompt。adapter 在转换时必须选择并写入 chosen prompt，不能在每次训练无记录随机抽取。
- 仿真相机外参应逐帧保留；`extrinsic_cv` 与 `cam2world_gl` 坐标约定不同，不能直接混用。

## 4. RoboDojo

### 数据性质

- RoboDojo 同时提供 sim-and-real benchmark / environment 和官方训练、评测轨迹；不能再把它当作“只有环境、没有数据”的来源。
- 官方可下载 source 至少包括：sim HDF5（约 3500 episodes、35 tasks、ARX X5）、RGB-D HDF5、LeRobot v2.1、LeRobot v3.0 joint-action、LeRobot v3.0 EE-action，以及 real HDF5（约 1740 episodes、18 个 real tasks、多个 embodiment）。完整 benchmark 的 runnable tasks 多于公开训练轨迹 tasks，因此不要默认每个 benchmark task 都有可训练 demonstration。

### 原生 HDF5 接口

```text
data_format_version
instruction / instructions, optional subtasks
additional_info/frequency
vision/{camera}/colors
vision/{camera}/depths                              # optional
vision/{camera}/intrinsic_matrix
vision/{camera}/extrinsic_matrix                    # 有 singular/plural 文档不一致
action/{left,right}_arm_joint_states
action/{left,right}_ee_joint_states
state/{left,right}_arm_joint_states
state/{left,right}_ee_joint_states
state/{left,right}_ee_poses, tcp_poses, delta_ee_poses   # optional
```

默认 ARX X5 的 joint action 是 14D：`[left_6_joints, left_gripper, right_6_joints, right_gripper]`。RGB HDF5 中常见 JPEG byte stream，须先验证 decoder 色彩通道后统一为 RGB；官方不同读取器存在 OpenCV BGR 与 PIL RGB 的差异。

### 发布的 LeRobot 版本

RoboDojo LeRobot v3.0 joint export 为 25 Hz，典型字段如下：

```text
observation.state: float32[14]
action: float32[14]
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
timestamp, frame_index, episode_index, index, task_index
```

EE export 的 state/action 是 16D，每只手为 `[x,y,z,qw,qx,qy,qz,gripper]`。选择 joint export 还是 EE export 必须写入 `action_spec`；不允许把 14D joint action 与 16D EE pose 直接拼接。

### 接入合同

每条 RoboDojo trajectory 需要保留：

```text
environment_id, source format and raw episode id
task_id, task language, task family and variation
robot_id, embodiment, controller type and action semantics
per-step observation timestamp and action timestamp
all camera calibration metadata and transform direction
state joint order / units / limits
sim / real flag, layout seed and data source (teleop / datagen)
```

RoboDojo 环境接受 absolute joint target 或 EE pose + gripper。EE pose 使用 `[x,y,z,qw,qx,qy,qz]`，且 state 以 environment-local / world-origin-relative convention 表示；相机 getter 给出的常是 `T_world_camera`（camera-to-world），不是常见 world-to-camera。adapter 必须显式保存 `transform_direction`。

RoboDojo 的 sim dt 为 0.004 秒，采集间隔通常为 10，即 25 Hz。训练时不得让同 task / 同 layout seed 的轨迹进入对应 RoboDojo benchmark 的 test split；real 与 sim 也应带独立 domain label。

## 接口差异总表

| 项目 | DROID | FMB | RoboTwin 2.0 | RoboDojo |
| --- | --- | --- | --- | --- |
| 数据性质 | 真机遥操作轨迹 | 真机专家轨迹 | 仿真生成器/轨迹 | 环境/benchmark，需先导出轨迹 |
| 原始格式 | RLDS/TFDS | `.npy` dict | HDF5 + JSON/PKL sidecars | HDF5，或官方 LeRobot v2/v3 export |
| 相机 | 双 exterior + wrist 常见 | side / wrist 多视角 | 多相机、双臂 | 每个 task/机器人不同 |
| 状态 | joint + gripper 基础字段 | TCP、速度、力/力矩可用 | joint targets、config-dependent endpose | joint / EE states，依导出版本 |
| 动作 | joint pos 或 joint vel + gripper | Cartesian delta + gripper，需核验版本 | next absolute joint target | absolute joint target 或 EE pose + gripper |
| 语言 | 多个自然语言 annotation | task / primitive / object metadata | task templates | task config / language wrapper |
| 直接混入条件 | 可 | 可，先核验语义 | native adapter 后可 | 选定 HDF5/v3 action spec、并做 benchmark split 后可 |

## 原始格式不是 JSON 的原因

四个来源中，JSON/YAML 通常只承载元数据。DROID 训练数据是 RLDS，FMB 轨迹是 `.npy`，视频是编码视频文件；把所有帧和数组转成 JSON 会导致文件巨大、读写慢且丢失 dtype。统一 JSON 应描述**数组在哪里、是什么语义、是否有效**，而不是携带数组本体。
