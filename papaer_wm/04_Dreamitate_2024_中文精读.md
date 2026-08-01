# Dreamitate 中文精读

> 原论文：[Dreamitate: Real-World Visuomotor Policy Learning via Video Generation](04_Dreamitate_2024.pdf)  
> 作者：Junbang Liang, Ruoshi Liu, Ege Ozguroglu, Sruthi Sudhakar, Achal Dave, Pavel Tokmakov, Shuran Song, Carl Vondrick  
> 年份：2024  
> 关键词：video generation、visuomotor policy、human demonstrations、tool tracking、embodiment gap

## 0. 这篇论文在讲什么

Dreamitate 的核心思路是：

```text
输入当前场景图像
-> 视频生成模型想象“人用工具完成任务”的视频
-> 在生成视频中追踪工具 6D 位姿
-> 把工具轨迹交给机器人执行
```

它不是让机器人直接模仿人手，而是让人和机器人都使用同一种可追踪工具。这样可以绕开 human hand 和 robot manipulator 之间的 embodiment gap。

一句话：

> 用互联网视频预训练的 video diffusion model 负责泛化，用可追踪工具把生成视频转换成机器人动作。

## 1. 摘要翻译

操作任务中的关键挑战，是学习能在多样视觉环境中鲁棒泛化的策略。一个有前景的机制是利用在大规模互联网视频上预训练的视频生成模型。本文提出 Dreamitate，一个视觉运动策略学习框架：它在某个任务的人类演示视频上微调视频扩散模型。测试时，模型根据新场景图像生成一段任务执行视频，然后直接用这段合成视频控制机器人。关键洞见是，使用常见工具可以轻松弥合人手和机器人之间的 embodiment gap。作者在四个复杂度递增的任务上评估方法，展示互联网规模生成模型的先验能让策略比现有行为克隆方法获得显著更强的泛化能力。

## 2. 为什么需要 Dreamitate

传统 behavior cloning 把视觉运动策略学习写成：

```text
observation -> action
```

它需要真实机器人动作标签。问题是：

- 机器人遥操作数据采集贵；
- 场景、物体、光照、桌面材质变化会让策略泛化困难；
- 人类视频很多，但人手动作不能直接变成机器人动作；
- 机器人视频可执行，但数据规模远小于人类视频。

Dreamitate 提出一个中间路线：

- 用人类演示收集数据，但演示者使用带 CAD 模型的工具；
- 视频模型生成未来工具使用视频；
- 用 3D tracking 把工具轨迹变成机器人末端轨迹；
- 机器人也装同样或对应的工具执行。

## 3. 方法总览

给定当前视频帧 `v_0`，目标是通过视频生成规划并执行机器人动作：

```text
a_t = T(vhat_t),  where {vhat_t} = f_theta(v_0)
```

其中：

- `f_theta`：视频生成模型；
- `vhat_t`：生成的视频帧；
- `T`：工具轨迹追踪器；
- `a_t ∈ SE(3)`：机器人执行的工具 6D 位姿动作。

整个 pipeline：

1. 人在桌面场景中用工具完成任务，录制双目视频。
2. 用这些视频微调预训练视频生成模型。
3. 测试时输入新场景双目图像，生成双目执行视频。
4. 在生成视频里用 CAD 模型追踪工具 6D pose。
5. 把工具轨迹映射到机器人末端执行。

## 4. 视频生成部分

作者从大规模互联网视频预训练的 Stable Video Diffusion 初始化，然后在任务特定的人类工具演示视频上微调。

训练数据是双目视频：

```text
(v^1, v^2) ∈ V
```

每个任务单独训练一个视频模型，比如 sweeping、scooping、rotation。

训练目标可以写成：

```text
min_theta E_{v∈V} sum_t ||vhat_t^1 - v_t^1||_2^2 + ||vhat_t^2 - v_t^2||_2^2
for {vhat_t} = f_theta(v_0)
```

实现细节：

- encoder 和 decoder 冻结；
- 只微调 spatial / temporal attention layers；
- 根据输出视角修改每帧图像 embedding；
- 前半段帧生成一个视角，后半段帧生成另一个视角；
- 测试时给双目图像对，生成双目任务执行视频。

## 5. Track then Act

生成视频只是中间表示，最终要变成机器人动作。

Dreamitate 使用已知 CAD 模型的工具，在生成视频中追踪工具 pose：

- 输入：生成的 RGB 双目视频；
- 追踪器：MegaPose；
- 分辨率：`768 x 448`；
- 相机参数：根据 Intel RealSense 默认内参推导；
- 输出：每一帧工具相对相机的 6D pose；
- 执行：机器人末端安装工具，按 `a_0, ..., a_T` 执行轨迹。

由于工具是刚体且 CAD 已知，工具位姿比人手位姿更容易精确追踪。这是论文最关键的工程选择。

## 6. 实验设置

作者在四个真实世界任务上评估：

| 任务 | 训练对象 / 演示数 | 测试对象 / trials | 任务难点 |
|---|---:|---:|---|
| Rotation | 31 个对象，371 demos | 10 个 unseen objects，40 trials | 双端协调、选择稳定接触点 |
| Scooping | 17 个碗、8 种颗粒，368 demos | 8 个碗、4 种颗粒，40 trials | 精确 3D 操作、识别满碗和空碗、避开干扰物 |
| Sweeping | 6 种颗粒，356 demos | 6 种新颗粒，40 trials | 多模态轨迹、避障、扫到目标位置 |
| Push-Shape | 26 个字母形状，727 demos | 8 个 unseen shapes，32 trials | 长时序推形状，同时对齐位置和角度 |

为了测试泛化，训练和测试在不同地点、不同桌面和光照条件下进行，训练对象和测试对象不重叠。

baseline 是 Diffusion Policy，使用同样训练数据和双目图像输入。作者用 MegaPose 预处理演示视频，为 Diffusion Policy 生成动作轨迹监督。

## 7. 任务一：Object Rotation

任务：随机放置一个对象，用夹具抓住合适位置，逆时针旋转最多 45 度。

成功标准：

- 机器人在旋转过程中保持和物体接触；
- 至少逆时针旋转 25 度。

结果：

- Diffusion Policy：22 / 40，约 55%；
- Dreamitate：37 / 40，约 92.5%。

错误分析：

- Diffusion Policy 容易没接触到物体，或者选到不稳定抓取点导致打滑；
- Dreamitate 更稳定地接触物体，但对透明袋子等复杂形状仍可能选点失败。

## 8. 任务二：Granular Material Scooping

任务：把颗粒物从满容器舀到空容器，同时避开干扰物。

成功标准：

- 只要有颗粒被转移到空碗就算成功。

测试时比训练更难：

- 训练时每次只有 1 个干扰物；
- 测试时每次有 5 个干扰物；
- bowl 和 particles 都是 unseen。

结果：

- Diffusion Policy：22 / 40，约 55%；
- Dreamitate：34 / 40，约 85%。

这说明双目视频生成可以给出足够准确的 3D 工具轨迹。Diffusion Policy 常见失败是被场景干扰物吸引，或者误判碗的高度。

## 9. 任务三：Table Top Sweeping

任务：用刷子把随机放置的颗粒扫到星形标记目标附近，同时避开障碍物。

成功标准：

- 任意颗粒被扫到目标 50mm 范围内。

这个任务难在训练数据存在多模态行为：同一个场景中可能有多个颗粒，选择哪个颗粒、从哪条路径扫过去都可以。

结果：

- Diffusion Policy：5 / 40，约 12.5%；
- Dreamitate：37 / 40，约 92.5%。

论文认为，互联网视频预训练给视频模型带来的物理和行为先验，帮助它更好处理多模态演示和场景泛化。

## 10. 任务四：Push-Shape

任务：把泡沫形状推到目标 mask 位置和朝向。它是 Push-T 的更难长时序版本。

训练：

- 26 个字母形状。

测试：

- 8 个未见形状，包括数字和多边形。

指标：

- mIoU：最终形状和目标 mask 的重合度；
- rotation error：朝向误差。

结果：

| 方法 | mIoU | Rotation Error |
|---|---:|---:|
| Diffusion Policy | 0.550 | 48.2° |
| Dreamitate | 0.731 | 8.0° |

Diffusion Policy 往往能把物体推到目标附近，但不能有效调整角度。Dreamitate 生成的视频计划更能表达连续接触和姿态调整。

## 11. 总量化结果

| 模型 | Rotation | Scooping | Sweeping | Push-Shape mIoU | Push-Shape Rot. Error |
|---|---:|---:|---:|---:|---:|
| Diffusion Policy | 22 / 40 | 22 / 40 | 5 / 40 | 0.550 | 48.2° |
| Dreamitate | 37 / 40 | 34 / 40 | 37 / 40 | 0.731 | 8.0° |

Dreamitate 在四个任务上都更强，尤其 sweeping 和 Push-Shape 的差距很明显。

## 12. 数据规模消融

作者在 rotation 任务上减少训练集：

- full dataset；
- 2/3 dataset；
- 1/3 dataset。

结果趋势：

- Diffusion Policy 数据减少后性能明显下降；
- Dreamitate 即使用 1/3 数据仍保持较高成功率。

解释：视频模型继承了互联网视频预训练的操作和物理先验，所以对任务数据量更不敏感。

## 13. 局限

论文列出几个限制：

- 依赖可视觉追踪的工具；
- 如果工具严重遮挡，tracking 会失败；
- 刚体工具适合 sweeping、scooping、pushing，但不适合细粒度灵巧操作；
- 视频模型推理成本较高，暂时不适合实时闭环控制；
- 当前方法更像 open-loop trajectory execution，遇到执行偏差时自我纠正能力有限。

## 14. 和 RoboDreamer / DreamZero 的关系

| 方法 | 视频的作用 | 动作怎么来 |
|---|---|---|
| RoboDreamer | 生成机器人未来视频计划 | 单独 inverse dynamics model |
| Dreamitate | 生成人用工具的未来视频 | 追踪工具 6D pose，直接执行 |
| DreamZero | 联合生成未来视频和连续动作 | 同一个模型直接输出 action chunk |

Dreamitate 的独特之处是“工具桥接”：

- RoboDreamer 主要解决语言组合泛化；
- Dreamitate 主要解决人类视频如何变机器人动作；
- DreamZero 试图把视频和动作统一进一个 WAM。

## 15. 最短总结

Dreamitate 可以记成一句话：

> 用视频扩散模型先想象人类工具使用过程，再追踪生成视频中的工具轨迹，让机器人执行同样的工具 6D 运动，从而用少量人类演示学到更泛化的真实机器人策略。
