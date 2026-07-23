# Left Arm URDF Architecture

## Scope

This document describes `urdf/left_arm_with_wrist_camera.urdf` in the
`robot_model` ROS2 package. It is the left-arm-only model used by:

- `launch/left_arm_rviz.launch.py`
- `scripts/start_left_arm_foxglove.sh`

The model contains one fixed main camera, one AR5 left arm, one wrist camera,
and the RH6 left dexterous hand. The former parallel gripper is not part of
this URDF.

## Coordinate Rules

- Every `<origin>` describes the child frame relative to the parent frame.
- `xyz` is in meters.
- `rpy` is in radians, ordered as roll, pitch, yaw.
- Task and mechanical frames use `+X` as the forward direction.
- The `visual` and `collision` origins only place CAD meshes. They do not
  change the TF frame of a link.
- The world frame is the root frame.

The model deliberately keeps `camera_color_frame` aligned with
`camera_link`, so it also uses `+X` forward. This is different from the ROS
optical-frame convention (`+Z` forward, `+X` right, `+Y` down). A standard
`sensor_msgs/CameraInfo` FOV visualizer may require a separate optical frame
if it assumes that ROS convention.

## TF Tree

```text
world
|- camera_base_link
|  `- camera_link
|     `- camera_color_frame
`
  basement_base_link
  `- AR5-5_07L_base
     `- AR5-5_07L_link1
        `- AR5-5_07L_link2
           `- AR5-5_07L_link3
              `- AR5-5_07L_link4
                 `- AR5-5_07L_link5
                    `- AR5-5_07L_link6
                       `- AR5-5_07L_link7
                          `- AR5-5_07L_tcp
                             |- left_wrist_camera_link
                             |  `- left_wrist_camera_color_frame
                             `- rh6_left_base_link
                                |- rh6_left_fz11_Link -> fz12 -> fz13 -> fz14
                                |- rh6_left_fz21_Link -> fz22 -> fz23
                                |- rh6_left_fz31_Link -> fz32 -> fz33
                                |- rh6_left_fz41_Link -> fz42 -> fz43
                                |- rh6_left_fz51_Link -> fz52 -> fz53
                                `- rh6_left_grasp_center
```

`AR5-5_07L_tcp` is the tool center point, not a visual mesh. It is the common
parent of the wrist camera and RH6 hand, so both follow every arm joint.

## Fixed Frames

| Joint | Parent -> child | `xyz` | `rpy` | Purpose |
| --- | --- | --- | --- | --- |
| `fixed` | `world` -> `basement_base_link` | `0 0 0` | `0 0 0` | Robot pedestal |
| `left_fixed` | `basement_base_link` -> `AR5-5_07L_base` | `-0.10645 0 1.40` | `pi pi/2 0` | Left arm mounting pose |
| `AR5-5_07L_tcp_joint` | `AR5-5_07L_link7` -> `AR5-5_07L_tcp` | `0.001668 0.000142 0.097` | `0 0 0` | Tool center point |
| `left_rh6_mount_joint` | `AR5-5_07L_tcp` -> `rh6_left_base_link` | `0 0 0.01025` | `0 0 0` | RH6 hand mounting pose |
| `rh6_left_grasp_center_joint` | `rh6_left_base_link` -> `rh6_left_grasp_center` | `0.02 0 0.1` | `0 0 0` | Nominal hand grasp point |

The RH6 mount has zero rotation, therefore the hand root inherits the TCP
orientation and the model's `+X`-forward convention. The `0.01025 m` offset
is retained from `robot_model_v4_700`.

## Main Camera

```text
world -> camera_base_link -> camera_link -> camera_color_frame
```

| Joint | Transform | Meaning |
| --- | --- | --- |
| `camera_base_fixed` | `xyz="0.033 0.14 1.475"`, `rpy="0 0.7853981633974483 1.570796"` | Mounts the main camera 1.475 m above world origin. `camera_link +X` points toward world `+Y` and 45 degrees downward. |
| `camera_fixed` | identity | Camera body shares the base frame. |
| `camera_mount_to_color_joint` | identity | `camera_color_frame` shares the body frame and uses `+X` forward. |

The camera CAD meshes are `Camera_base_40d.stl` and `Camera_435.stl`. Their
visual origins contain a `+90 deg` yaw solely to align the STL files; that is
not the camera TF orientation.

## Left Arm

The AR5 chain uses seven revolute joints. All are named
`AR5-5_07L_joint_N`, where `N` is 1 through 7.

| Joint | Parent -> child | Axis in parent joint frame |
| --- | --- | --- |
| `AR5-5_07L_joint_1` | base -> link1 | `0 0 1` |
| `AR5-5_07L_joint_2` | link1 -> link2 | `0 1 0` |
| `AR5-5_07L_joint_3` | link2 -> link3 | `0 0 1` |
| `AR5-5_07L_joint_4` | link3 -> link4 | `0 1 0` |
| `AR5-5_07L_joint_5` | link4 -> link5 | `0 0 1` |
| `AR5-5_07L_joint_6` | link5 -> link6 | `0 1 0` |
| `AR5-5_07L_joint_7` | link6 -> link7 | `1 0 0` |

The position of `AR5-5_07L_tcp` changes with these seven joint states. Its
local orientation is the same as link7 because its fixed joint has zero RPY.

## Wrist Camera

```text
AR5-5_07L_tcp
  -> left_wrist_camera_link
    -> left_wrist_camera_color_frame
```

| Joint | Transform | Notes |
| --- | --- | --- |
| `left_wrist_camera_fixed` | `xyz="-0.1 0 0"`, `rpy="0 -1.570796327 0"` | Current preliminary TCP-to-camera extrinsic. Replace with hand-eye calibration. |
| `left_wrist_camera_color_fixed` | identity | Color frame uses the same `+X`-forward convention. |

The wrist camera's `Camera_435.stl` is visual-only. Its mesh origin must not
be used as a substitute for the calibrated TCP-to-camera transform.

## RH6 Left Dexterous Hand

The RH6 hand was transplanted from
`robot_model_v4_700/urdf/half_finish.urdf`. Its left-hand geometry is stored
under `meshes/rh6_ctrl/` in this package.

There are 16 actuated RH6 finger joints:

```text
Thumb:  rh6_left_fz11, rh6_left_fz12, rh6_left_fz13, rh6_left_fz14
Finger 2: rh6_left_fz21, rh6_left_fz22, rh6_left_fz23
Finger 3: rh6_left_fz31, rh6_left_fz32, rh6_left_fz33
Finger 4: rh6_left_fz41, rh6_left_fz42, rh6_left_fz43
Finger 5: rh6_left_fz51, rh6_left_fz52, rh6_left_fz53
```

All RH6 finger joints are revolute. Several terminal joints have equal lower
and upper limits, so they remain fixed in the exported model. The green
`rh6_left_grasp_center` sphere is a convenience frame for grasp planning and
visual inspection, not a physical fingertip.

## ROS2 Topics And Visualization

`left_arm_rviz.launch.py` isolates this model from globally published robot
descriptions by using the following remappings:

| Purpose | Topic |
| --- | --- |
| URDF description | `/left_arm/robot_description` |
| Arm and hand joint states | `/left_arm/joint_states` |
| Dynamic transforms | `/tf` |
| Fixed transforms | `/tf_static` |

The one-command Foxglove launcher uses a dedicated ROS domain and local DDS
only:

```bash
./scripts/start_left_arm_foxglove.sh 102
```

It sets `ROS_DOMAIN_ID=102` and `ROS_LOCALHOST_ONLY=1`, starts the left-arm
publishers, then exposes the data through Foxglove Bridge at
`ws://localhost:8765`.

For RViz:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=102
export ROS_LOCALHOST_ONLY=1
ros2 launch robot_model left_arm_rviz.launch.py use_rviz:=true
```

## Calibration Checklist

1. Update `camera_base_fixed` when the physical main camera mounting pose
   changes.
2. Update `left_wrist_camera_fixed` with the measured `T_tcp_camera`.
3. Update `left_rh6_mount_joint` only when the hand adapter position or
   orientation differs from the V4 mounting geometry.
4. Keep CAD `visual` offsets separate from TF calibration transforms.
5. If using standard ROS `CameraInfo`, add a dedicated optical frame with
   `+Z` forward and set `CameraInfo.header.frame_id` to that optical frame.
