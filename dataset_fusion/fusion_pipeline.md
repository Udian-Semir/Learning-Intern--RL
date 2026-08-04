# 融合流程、mask 与验收

## 0. 先决定目标，而不是先 merge 文件

目标 `cartesian_delta_abs_gripper_v1`：

```text
arm_0: [dx, dy, dz, dRx, dRy, dRz, gripper_target]
arm_1: [dx, dy, dz, dRx, dRy, dRz, gripper_target]
```

所有平移使用米、旋转使用 radians、参考系使用机器人基座、夹爪使用 `[0,1]` 的绝对开合目标。最终 action 是 `[14]`，但每个 episode 都有 `[14]` 的 `action_mask`。

## 1. Source adapter

每个 source adapter 必须输出：

1. `episode metadata`：来源、robot、task、坐标系、原始文件和转换版本。
2. `frame stream`：时间、图像引用、状态、动作、所有有效性 mask。
3. `calibration`：相机内参/畸变/相对 base 或 end-effector 的外参。
4. `conversion report`：单位、FK/IK 版本、被拒绝帧数、原因。

不要在 adapter 内做随机 camera dropout、随机裁剪或数据增强；它们属于训练 dataloader。

## 2. 时间对齐与重采样

设目标控制频率为 $f_{target}$，时间步为 $\Delta t=1/f_{target}$。

- 图像：取与 target timestamp 最近且时间差不超过阈值的帧；否则该 view 的 mask 为 false。
- 连续状态：线性插值，四元数使用 SLERP；插值跨越大 gap 时拒绝该帧。
- 绝对目标动作：先在原始时间轴插值 / 重建，再转换为相对于当前状态的 delta。
- 速度动作：按时间单位转换并根据 target step 积分或保留为控制量，二者不可混用。
- 夹爪：最近邻或分段常数；保持 absolute target 语义。

每帧保存 `observation_timestamp`、`action_timestamp` 和 `sync_error_s`。若相机和 action 差超过阈值，该样本不可用于 action loss。

## 3. 动作转换

### 绝对关节目标 -> Cartesian delta

对当前 joint state $q_t$ 和下一个目标 $q_{t+1}^{target}$：

$$
{}^B T_{tcp,t}=FK(q_t),\qquad
{}^B T_{tcp,t+1}=FK(q_{t+1}^{target}).
$$

平移标签：

$$
\Delta p={}^B p_{tcp,t+1}-{}^B p_{tcp,t}.
$$

旋转标签使用相对旋转：

$$
\Delta R=({}^B R_{tcp,t})^T{}^B R_{tcp,t+1},
$$

再将 $\Delta R$ 编码成 axis-angle 三维向量。FK 必须使用该数据源的真实 robot URDF、joint order、joint offset、TCP offset 和单位。

### 原生 Cartesian delta

必须核验它的 reference frame：base、TCP/local frame、camera frame 之间不能混。若为 TCP-local delta，先用当前 TCP orientation 转到 base frame，或明确把目标契约改为 local-frame delta。

### 双臂与单臂

单臂轨迹映射到 `arm_0`；`arm_1` 填零仅为 tensor shape 对齐，但对应 mask 必须为 false。双臂轨迹的两个 arm 都为 true。无效值不参与 loss 和 normalization statistics。

### 四个 source 的具体转换

| source | native action | canonical 转换 |
| --- | --- | --- |
| DROID | `action_dict.joint_position[7]` 或 `joint_velocity[7]` + gripper | joint-position 先 FK；joint-velocity 必须用真实 dt 积分，不能直接当 pose；若缺少 URDF/calibration，保留 native-only |
| FMB | `[dxyz, dEuler, gripper]`，约 10 Hz | 将 Euler 增量转旋转矩阵/axis-angle；将 gripper 的 `1=closed,0=open` 映射为 canonical 的 `0=closed,1=open` |
| RoboTwin | `joint_action/vector[t+1]`，绝对 next drive target；6+gripper 或 7+gripper per arm | 用对应 embodiment URDF/TCP 做 FK；`endpose` 不是无条件的通用 TCP，须记录 RoboTwin transformed-pose convention |
| RoboDojo | joint export 的绝对 target 14D，或 EE export 的绝对 pose+gripper 16D | joint target 用 FK；EE target 用当前 pose 求相对变换；pose 顺序记录为 `xyz_qwqxqyqz`，并保留 environment-local convention |

每一行都同时保存 `action_native_uri`、`native_action_representation`、`native_rotation_repr`、`native_reference_frame` 和 `action_conversion_valid`。canonical 转换失败时仍可保留观察数据，但 `action_mask=false` 且不能进入 action loss。

## 4. mask 规则

| mask | shape | false 的含义 | 训练行为 |
| --- | --- | --- | --- |
| `image_mask` | `[V]` | view 不存在、不同步或质量不合格 | 不编码该 view / attention mask |
| `proprio_mask` | `[D_p]` | state 字段没有测量值 | 不计 state stats；输入缺失 token |
| `action_mask` | `[H, 14]` | action 维度没有真实监督 | 不计 action / flow / diffusion loss |
| `time_mask` | `[H]` | action chunk 已超过 episode 结尾 | 不计该时间步损失 |
| `language_mask` | `[1]` | 无可靠语言标注 | no-language token，不训文本条件对齐 |
| `calibration_mask` | `[V]` | view 缺少可信 calibration | 不做几何监督，不等于图像不可用 |

动作损失必须是 masked loss：

$$
\mathcal L_{action}=
\frac{\sum_{h,d}m^{time}_h m^{action}_{h,d}\,\ell(\hat a_{h,d},a_{h,d})}
{\sum_{h,d}m^{time}_h m^{action}_{h,d}+\epsilon}.
$$

对 flow matching，$\ell$ 是 velocity / noise 的误差而非直接 action MSE，但 mask 的位置相同。**不要把缺失 action 维度的零当作 no-op 监督。**

## 5. 归一化

- 先转换到共同单位和语义，再计算统计量。
- 仅在 `action_mask=true` 的元素上计算均值、标准差、分位数。
- 按 `(embodiment_id, action_spec_id)` 保存统计量；跨 embodiment 训练可额外保存 global stats，但不能以 global stats 覆盖未对齐的原始关节空间。
- 夹爪 target 已定义到 `[0,1]`，不应按各 source 的原始毫米 / rad 值统一 normalize。

## 6. 混合采样

不按帧数直接 concat。推荐两阶段采样：

$$
P(source=i)\propto w_i\,n_i^{\alpha},\quad 0\leq\alpha\leq1,
$$

其中 $n_i$ 是有效 episode 数，$w_i$ 是人为设定的 source weight。起始可用：

```text
DROID    0.35
FMB      0.30
RoboTwin 0.25
RoboDojo 0.10  # 仅在导出并验收合格 demonstrations 后启用
```

每个 source 内再按 task 均衡采样，并保留 `dataset_id` / `embodiment_id` 作为训练条件 token 或 adapter selector。

## 7. 验收检查

转换器必须逐 episode 输出并检查：

```text
1. schema-valid JSON metadata
2. 时间单调、无重复、图像/状态/action 对齐
3. action 单位、范围和 max speed/rotation 合理
4. mask 与实际字段存在性一致
5. 用 FK 或环境 rollout 检查动作转换
6. 视觉手工抽查每个 source 至少 20 条轨迹
7. train / val / test 按 episode 切分，绝不按 frame 随机切分
8. source、robot、task、raw episode id 可追溯
```

## 8. 物理合并的最后一步

只有四个 adapter 都已导出相同 canonical fields、相同 action dimension 和相同 target FPS 后，才可转换为 LeRobot 目录并使用现有 `merge_lerobot_datasets.py` 处理 episode index、task index、Parquet、视频和 `stats.json`。现有脚本不能替代本流程。

## 9. RoboDojo 的评测隔离与版本选择

RoboDojo 的公开 sim joint、sim EE、real 和 depth 版本不是同一数据视图。若研究几何/手眼标定，优先使用完整 HDF5；joint-only LeRobot 导出没有相机 K/外参和 EE pose，不能从它们反推这些字段。若评测 RoboDojo，必须按 `task_id + layout_seed + domain` 做 episode-level holdout，不能把同一任务或布局变体的一部分帧放入训练。

## 10. 许可与溯源

每个 canonical episode 都要保留 source URL、source commit/version、原始路径和 checksum。RoboDojo 的公开数据卡、GitHub LICENSE 与 README 对再分发许可存在不一致，建立混合数据集后对外发布前必须单独核实许可；内部训练也不要丢失原始 license metadata。
