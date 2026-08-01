# RoboDreamer 中文精读

> 原论文：[RoboDreamer: Learning Compositional World Models for Robot Imagination](03_RoboDreamer_2024.pdf)  
> 作者：Siyuan Zhou, Yilun Du, Jiaben Chen, Yandong Li, Dit-Yan Yeung, Chuang Gan  
> 年份：2024  
> 关键词：robot imagination、text-to-video、diffusion、compositional generation、inverse dynamics

## 0. 这篇论文在讲什么

RoboDreamer 把机器人规划改写成视频生成问题：

```text
当前图像 + 语言任务 -> 生成未来视频计划 -> 逆动力学模型把视频变成动作
```

它的核心不是“再训练一个更大视频模型”，而是解决文本到视频在机器人任务里的泛化问题。机器人指令往往是细粒度空间关系，比如：

```text
move pepsi can near plastic bottle
place water bottle into bottom drawer
pick apple from top drawer
```

普通 text-to-video 模型容易只抓住大概动作，却搞错物体关系。RoboDreamer 的办法是把语言指令拆成组件，然后组合多个条件生成模型，让模型能泛化到训练时没见过的组合。

## 1. 摘要翻译

Text-to-video 模型在机器人决策中表现出很大潜力，因为它们可以想象真实的未来动作计划，也可以较准确地模拟环境。然而，这类模型的一个主要问题是泛化：模型通常只能合成与训练时语言指令相似的视频。这严重限制了机器人决策，因为我们希望 world model 能为未见过的物体和动作组合合成计划，从而在新环境中解决新任务。

为解决这个问题，作者提出 RoboDreamer，一种通过分解视频生成过程来学习组合式 world model 的方法。它利用语言天然的组合性，把指令解析为一组更低层的 primitive，然后用这些 primitive 条件化一组模型来生成视频。这种分解自然带来组合泛化：新的自然语言指令可以由训练中见过的组件重新组合而成。作者还展示了这种分解能加入多模态目标，比如同时给语言指令和目标图像来生成视频。实验表明，RoboDreamer 能在 RT-X 的未见目标上成功合成视频计划，在仿真机器人执行中有效，并显著超过整体式视频生成 baseline。

## 2. 背景：视频生成如何做机器人规划

论文使用 UPDP，也就是 Unified Predictive Decision Process 的抽象。可以理解为：

```text
G = <X, C, H, rho>
```

其中：

- `X`：图像观测空间；
- `C`：文本任务描述空间；
- `H`：规划 horizon；
- `rho(. | x0, c)`：条件视频生成器，根据初始图像 `x0` 和任务描述 `c` 生成未来视频。

生成出来的视频计划：

```text
tau = [x_1, ..., x_H]
```

再交给轨迹任务条件策略或 inverse dynamics model：

```text
pi(. | {x_h}_{h=0}^H, c) -> actions
```

也就是说，规划本身被转成 text-conditioned video generation。

## 3. 为什么普通视频模型不够

内容创作里的视频生成通常只需要整体动作合理，比如“一个人走路”“狗在跑”。机器人任务更难，因为它需要精确的物体关系：

- 哪个物体被移动；
- 放到哪个物体旁边；
- 放进哪个抽屉；
- 是 top drawer 还是 middle drawer；
- 是 near、into、from 还是 on。

如果训练集中没有某个组合，普通模型可能把语义搞混。例如要求“把 pepsi can 移到 plastic bottle 旁边”，模型可能把 pepsi can 放到 green can 旁边。RoboDreamer 认为根本问题是：整体式条件 `p(video | full sentence)` 对 unseen composition 不友好。

## 4. 方法一：Text Parser

RoboDreamer 使用预训练 parser 加规则，把语言指令拆成：

- verb phrase：动作和被操作物体；
- prepositional phrase：空间关系和目标物体。

例子：

```text
place water bottle into bottom drawer
```

可拆成：

```text
place water bottle
into bottom drawer
```

再比如：

```text
pick orange from bottom drawer and place on counter
```

可拆出：

```text
pick orange
from bottom drawer
place on counter
```

这样的拆分让每个模型组件只负责一部分约束。只要单个组件在训练中见过，推理时就可以组合成新任务。

## 5. 方法二：组合式视频生成

给定自然语言指令 `L`，解析成组件：

```text
{l_i}_{i=1:N}
```

RoboDreamer 把整体视频生成分布写成各组件条件分布的乘积：

```text
p_theta(tau | L) ∝ Π_i p_theta(tau | l_i)^(1/N)
```

直觉：

- 每个 `p(video | l_i)` 负责满足一个局部条件；
- 最终视频要同时满足所有局部条件；
- 因此模型能把“见过的动作”和“见过的关系”重新组合。

在 diffusion 训练里，这会变成 score function 的组合。每个组件条件对应一个 score：

```text
epsilon(tau_t, t | l_i)
```

组合 score 约等于这些 score 的平均：

```text
1/N * sum_i epsilon(tau_t, t | l_i)
```

训练目标是标准 denoising diffusion MSE：

```text
L_MSE = || 1/N * sum_i epsilon(tau_t, t | l_i) - epsilon ||^2
```

论文还指出，如果只训练组合目标，每个单独组件不一定学得好。所以它使用混合目标：随机采样组件子集，让模型既会单独理解组件，也会组合组件。

## 6. 方法三：多模态组合

语言有时不够精确。目标图像、草图能提供更清楚的空间约束。RoboDreamer 把多模态条件也纳入组合：

```text
language components: {l_i}
multimodal components: {m_j}
```

整体条件分布：

```text
p_theta(tau | L, M)
∝ Π_i p_theta(tau | l_i)^(1/(N+K)) * Π_j p_theta(tau | m_j)^(1/(N+K))
```

这让推理时可以灵活组合：

- 只有语言；
- 语言 + goal image；
- 语言 + goal sketch；
- 多个语言片段 + 多个视觉目标。

## 7. 实现细节

RoboDreamer 基于 AVDC 和 Imagen 的视频扩散框架：

- U-Net 中使用时空卷积；
- 使用 temporal attention；
- 使用三阶段 cascaded diffusion 做超分辨率；
- 文本编码器用 frozen T5-XXL；
- goal image / sketch 用 Stable Diffusion VQ-VAE 的图像编码器；
- 各模态 embedding 输入 PerceiverSampler，再通过 cross-attention 注入 U-Net。

附录里的训练设置：

- 基础视频：`8 x 64 x 64`；
- 逐级超分到 `8 x 128 x 128` 和 `8 x 256 x 256`；
- batch size 256；
- learning rate `5e-5`；
- 约 100 张 V100 训练。

## 8. 实验一：RT-1 视频生成

### 数据

使用 RT-1 真实机器人数据：

- 约 70k demonstrations；
- 平均长度 44；
- 约 500 个任务；
- 每 5 帧采样一次。

任务示例：

- pick；
- pick ... from ...；
- place；
- open；
- close；
- knock；
- pull。

### Baseline

比较对象：

- AVDC：机器人视频生成模型；
- HiP：latent video diffusion 机器人模型；
- RoboDreamer w/o：不使用 text parsing 的版本；
- RoboDreamer：完整方法。

### 指标

论文认为现有自动指标不够可靠，所以使用 human evaluation。每个样本至少 3 人评分：

- 0：生成视频计划不合理或不能完成任务；
- 1：计划可执行且能完成任务。

### 结果

| 方法 | Seen | Unseen |
|---|---:|---:|
| AVDC | 63.1 | 46.9 |
| HiP | 70.3 | 50.1 |
| RoboDreamer w/o | 85.5 | 68.8 |
| RoboDreamer | 90.1 | 81.3 |

重点是 unseen 上提升很大。说明拆语言组件后，模型确实更能处理训练中没见过的动作-物体-关系组合。

## 9. 实验二：多模态生成

RoboDreamer 测试了：

- `RoboDreamer (t)`：只有 text；
- `RoboDreamer (t+s)`：text + sketch；
- `RoboDreamer (t+i)`：text + goal image。

结果：

| 模型 | Human ↑ | FVD ↓ |
|---|---:|---:|
| AVDC | 46.9 | 517.1 |
| RoboDreamer (t) | 81.3 | 487.8 |
| RoboDreamer (t+s) | 94.7 | 454.7 |
| RoboDreamer (t+i) | 95.8 | 444.3 |

解释：

- goal image 和 sketch 提供了明确空间布局；
- 语言负责“要做什么”；
- 图像/草图负责“做到什么样”；
- 组合式 score 让这些条件能一起约束生成过程。

## 10. 实验三：RLBench 机器人规划

### 设置

使用 RLBench：

- Franka Panda + gripper；
- 7 DoF；
- 8 维动作空间，加 gripper state；
- 只用前视 RGB 图像，这比多视角更难；
- 使用 macro-step，让任务更偏高层规划。

比较方法：

- Image-BC；
- Hiveformer；
- UniPi；
- RoboDreamer。

### 结果

| 模型 | lamp off | lamp on | stack blocks | lift block | take shoes | close box | Average |
|---|---:|---:|---:|---:|---:|---:|---:|
| Image-BC | 60.1 | 47.0 | 0 | 0 | 0 | 82.4 | 31.6 |
| Hiveformer | 81.2 | 53.2 | 10.6 | 28.2 | 1.0 | 90.8 | 44.2 |
| UniPi | 70.6 | 47.1 | 7.1 | 23.3 | 3.8 | 94.1 | 41.0 |
| RoboDreamer | 96.3 | 51.9 | 18.5 | 22.2 | 10.5 | 96.3 | 49.3 |

RoboDreamer 平均成功率最高，说明合成的视频计划能帮助机器人执行。

## 11. 局限

论文列出几个限制：

- 当前只用单相机，很多机器人任务需要多视角或 3D 信息；
- 对真实世界图像泛化仍不够强，机器人数据多样性有限；
- moving-camera 场景下视频生成仍不稳定；
- 方法依赖 inverse dynamics model，视频计划质量和动作执行之间仍有误差。

## 12. 和其他 world model 的关系

RoboDreamer 和 DreamerV3 不同：

- DreamerV3 学 latent dynamics，并在 latent 中训练 RL agent；
- RoboDreamer 直接生成未来视频，把视频作为计划；
- 它更像“视频世界模型 + inverse dynamics policy”。

和后来的 WAM / DreamZero 的区别：

- RoboDreamer 是先生成视频，再用单独 inverse dynamics 变动作；
- DreamZero 把视频和动作放进一个模型里联合生成，减少视频-action 对齐问题。

## 13. 最短总结

RoboDreamer 可以记成一句话：

> 把机器人语言指令拆成可组合组件，用多个条件 score 组合生成未来视频计划，从而提升未见任务和多模态目标下的机器人想象能力。
