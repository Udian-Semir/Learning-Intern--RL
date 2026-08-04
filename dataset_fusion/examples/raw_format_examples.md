# 四源原始元数据结构示例

以下是接口形状示例，不是完整数据内容。数值数组、图像和深度不应复制进 JSON。

## DROID：RLDS step 的概念投影

```json
{
  "steps": {
    "observation": {
      "wrist_image_left": "uint8[180,320,3]",
      "exterior_image_1_left": "uint8[180,320,3]",
      "exterior_image_2_left": "uint8[180,320,3]",
      "joint_position": "float64[7]",
      "gripper_position": "float64[1]",
      "cartesian_position": "float64[6]"
    },
    "action_dict": {
      "joint_position": "float64[7]",
      "joint_velocity": "float64[7]",
      "cartesian_position": "float64[6]",
      "cartesian_velocity": "float64[6]",
      "gripper_position": "float64[1]",
      "gripper_velocity": "float64[1]"
    },
    "language_instruction": "...",
    "language_instruction_2": "...",
    "language_instruction_3": "..."
  }
}
```

## FMB：`.npy` trajectory dictionary 的概念投影

```json
{
  "container": "numpy.load(path, allow_pickle=true).item()",
  "obs/side_1": "uint8[T,H,W,3]",
  "obs/wrist_1": "uint8[T,H,W,3]",
  "obs/tcp_pose": "float32[T,7], xyz+qx+qy+qz",
  "obs/q": "float32[T,7]",
  "obs/dq": "float32[T,7]",
  "obs/tcp_force": "float32[T,3]",
  "obs/tcp_torque": "float32[T,3]",
  "action": "float32[T,7], delta_xyz+delta_euler+gripper",
  "primitive": "string[T]",
  "object_info": "optional metadata"
}
```

## RoboTwin：原生 sidecar 与 HDF5

`instructions/episodeN.json` 的实际结构：

```json
{
  "seen": ["instruction paraphrase 1", "instruction paraphrase 2"],
  "unseen": ["held-out paraphrase 1", "held-out paraphrase 2"]
}
```

HDF5 的概念结构：

```json
{
  "joint_action/vector": "float64[T,14_or_16]",
  "endpose/left_endpose": "optional float64[T,7], xyz+qw+qx+qy+qz",
  "observation/head_camera/rgb": "NUL-padded JPEG byte strings[T]",
  "observation/head_camera/intrinsic_cv": "float32[T,3,3]",
  "observation/head_camera/extrinsic_cv": "float32[T,3,4]",
  "observation/head_camera/cam2world_gl": "float32[T,4,4]"
}
```

## RoboDojo：LeRobot v3 episode metadata 的概念投影

```json
{
  "source_format": "lerobot_v3.0",
  "domain": "sim",
  "embodiment": "arx_x5",
  "fps": 25,
  "action_representation": "joint_position_absolute",
  "action": "float32[14]",
  "observation.state": "float32[14]",
  "video_keys": [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist"
  ],
  "task_text_source": "task_index -> meta/tasks.parquet"
}
```

若使用 RoboDojo `v3_ee`，state/action 改为每臂 `[x,y,z,qw,qx,qy,qz,gripper]`，总维度 16。若需要相机 K 或外参，不应使用 joint-only LeRobot v3 反推，必须读完整 HDF5。
