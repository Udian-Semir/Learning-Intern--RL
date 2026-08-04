# 四数据集融合规范：DROID、FMB、RoboTwin 2.0、RoboDojo

本目录定义把四个来源整理为可供 VLA / Flow Policy 训练的共同数据契约。它解决的是**语义统一**，不是把文件直接拼在一起。

## 结论先行

- `DROID`、`FMB`、`RoboTwin 2.0`、`RoboDojo` 都有可转换的轨迹来源，但其原始容器与动作语义不同。
- RoboDojo 同时是 benchmark / 环境和公开轨迹数据集；它已发布 HDF5 与 LeRobot 导出。仍须按 task、embodiment、sim/real、joint/EE action 版本选择 source，不能把所有下载包当成同一份数据。
- 绝不可只把不同长度的 `action` 向量补零后训练。必须先对齐动作类型、单位、参考系、控制频率和有效维度，并在损失中使用 mask。
- 推荐最终动作语义为 `cartesian_delta_abs_gripper_v1`：每只手使用基座坐标系中的末端平移增量、旋转增量和绝对夹爪目标。

## 目录内容

| 文件 | 用途 |
| --- | --- |
| [`source_interfaces.md`](source_interfaces.md) | 四个来源的原始格式、字段、接口差异和风险 |
| [`fusion_pipeline.md`](fusion_pipeline.md) | 转换、校验、重采样、归一化、训练 mask 与混合采样步骤 |
| [`schemas/unified_episode.schema.json`](schemas/unified_episode.schema.json) | 单条统一 episode 元数据 JSON Schema |
| [`schemas/dataset_manifest.schema.json`](schemas/dataset_manifest.schema.json) | 融合数据集总 manifest JSON Schema |
| [`configs/source_adapters.json`](configs/source_adapters.json) | 每个来源到统一契约的字段映射与转换要求 |
| [`configs/canonical_frame_table.json`](configs/canonical_frame_table.json) | Parquet 帧表列、dtype、mask 与不变量 |
| [`configs/mix_plan.example.json`](configs/mix_plan.example.json) | 可修改的四源混合计划示例 |
| [`examples/raw_format_examples.md`](examples/raw_format_examples.md) | 四种原始接口的结构化示例 |
| [`examples/droid_episode.example.json`](examples/droid_episode.example.json) | 一条符合统一 JSON Schema 的 episode metadata 示例 |

## 统一数据层次

```text
原始数据
  DROID: RLDS / TFDS
  FMB:   .npy trajectory dictionaries
  RoboTwin: native HDF5 + JSON / PKL sidecars
  RoboDojo: native HDF5 or published LeRobot v2/v3 export
        |
        v
Source adapter
  字段重命名、时间对齐、动作语义转换、坐标转换、语言清洗
        |
        v
Canonical episode v1
  JSON metadata + Parquet numeric streams + 视频文件
        |
        v
LeRobot / 训练视图
  meta/info.json, tasks.jsonl, episodes.jsonl, stats.json
  data/*.parquet, videos/*
```

JSON 只承载 metadata、字段定义、文件索引和校验结果；图像、点云、大数组动作放在 Parquet、视频或二进制数组文件中。

## 推荐的目标动作空间

每只机械臂占 7 维：

```text
[dx, dy, dz, dRx, dRy, dRz, gripper_target]
```

- `dx, dy, dz`：米，机器人基座坐标系。
- `dRx, dRy, dRz`：轴角旋转增量，弧度。
- `gripper_target`：绝对开合目标，归一化到 `[0, 1]`；`0` 为闭合、`1` 为张开。
- 双臂为 `[arm_0(7), arm_1(7)]`，总共 14 维。
- 单臂只填 `arm_0`，`arm_1` 的 `action_mask` 全为 `false`。

无法获得可靠正运动学、TCP 定义或手眼外参的轨迹，不能把关节动作臆测成末端动作。应暂留在 native-only split，或先补齐 URDF、TCP offset、joint limits 和坐标变换。

## 最小可行路径

1. 用少量样本跑通 DROID 与 FMB adapter，生成 schema-valid episode metadata。
2. 验证时间、相机、状态和动作在同一时间轴；人工检查每个 source 的至少 20 条轨迹。
3. 固定 RoboTwin 的 robot / task / controller 子集后接入；直接读取 native HDF5，不复用有信息丢失的 baseline converter。
4. 在 RoboDojo 中固定 sim/real、task、embodiment 和 joint/EE action export；为 benchmark 留出按 task 和 layout seed 隔离的 test split。
5. 使用 dataset-balanced sampler，不按原始帧数混采。

## 与现有代码的关系

仓库已有的 [`merge_lerobot_datasets.py`](../sai_0_robot/sai-vla/utils/mix_dataset/merge_lerobot_datasets.py) 只适用于 **state/action 维度已经完全一致** 的 LeRobot v2 数据集。它不执行本目录规定的语义转换、坐标变换、重采样或 per-dimension mask，因此必须在 adapter 阶段之后使用。
