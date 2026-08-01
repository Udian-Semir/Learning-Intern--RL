# World Action Models are Zero-shot Policies 中文精读

> 原论文：[World Action Models are Zero-shot Policies](05_World_Action_Models_are_Zero-Shot_Policies_2026.pdf)  
> 项目名：DreamZero  
> 机构：NVIDIA  
> arXiv：2602.15922v1，2026-02-17  
> 关键词：World Action Model、WAM、video diffusion、flow matching、zero-shot policy、cross-embodiment transfer、real-time control

## 0. 这篇论文在讲什么

这篇论文提出 DreamZero，一种 World Action Model，也就是 WAM。它和传统 VLA 的区别是：

```text
VLA:  observation + language -> action
WAM:  observation + language -> future video + action
```

DreamZero 用 14B image-to-video diffusion backbone 初始化，同时预测未来视觉状态和连续动作。作者认为，视频预训练带来的物理时空先验，可以让机器人策略更好泛化到未见动作、未见环境和新 embodiment。

最核心的主张：

> world model 不只是用来规划的模拟器；如果它同时生成未来世界和动作，它本身就可以成为 zero-shot policy。

## 1. 摘要翻译

当前最先进的 Vision-Language-Action 模型在语义泛化上表现很好，但面对新环境中的未见物理动作时泛化能力不足。作者提出 DreamZero，一个建立在预训练视频扩散骨干上的 World Action Model。和 VLA 不同，WAM 通过预测未来世界状态和动作来学习物理动力学，把视频作为世界如何演化的密集表示。通过联合建模视频和动作，DreamZero 能从异构机器人数据中有效学习多样技能，而不依赖重复演示。在真实机器人实验中，与最先进 VLA 相比，它在新任务和新环境泛化上获得超过 2 倍提升。关键的是，作者通过模型和系统优化，让 14B 自回归视频扩散模型能以 7Hz 做实时闭环机器人控制。最后，作者展示两种跨 embodiment 迁移：来自其他机器人或人类的视频-only 演示，只需 10 到 20 分钟数据，就能让 unseen task 表现相对提升超过 42%；更令人惊讶的是，DreamZero 只用 30 分钟 play data 就能适配新 embodiment，并保留 zero-shot 泛化能力。

## 2. 为什么 VLA 不够

VLA 从 VLM 继承语言和视觉语义先验，所以擅长：

- 识别新物体；
- 理解语言中的语义；
- 把已学技能应用到新目标。

但 VLA 对“未见物理动作”容易失败。例如：

- `move coke can to Taylor Swift`：VLA 可以用互联网语义知识理解 Taylor Swift 相关目标；
- `untie the shoelace`：如果训练数据中没有解鞋带动作，VLA 往往不知道动作该怎么执行。

论文认为问题在于：

> VLM 先验告诉模型 what to do，但没有足够表达 how to do，尤其缺少几何、动力学、接触和运动控制层面的时空表示。

WAM 的思路是让策略显式预测未来视觉状态。预测视频迫使模型学习物体和世界如何随动作变化，再把动作和未来视频对齐。

## 3. WAM 的定义

DreamZero 要预测：

```text
future video: o_{l:l+H}
actions:      a_{l:l+H}
```

条件包括：

- 语言指令 `c`；
- 当前本体状态 `q_l`；
- 当前和历史视觉观测 `o_{0:l}`。

论文把联合预测分解成：

```text
π0(o_{l:l+H}, a_{l:l+H} | o_{0:l}, c, q_l)
= π0(o_{l:l+H} | o_{0:l}, c, q_l)
  π0(a_{l:l+H} | o_{0:l+H}, q_l)
```

直觉：

- 第一项是 video prediction：想象接下来世界该变成什么样；
- 第二项像 implicit inverse dynamics model：根据想象出来的视觉未来推断动作。

但 DreamZero 不训练两个独立模型，而是端到端训练一个模型同时预测视频和动作。作者认为这样能更好保持 video-action alignment。

## 4. 模型架构

DreamZero 基于 Wan2.1-I2V-14B-480P，一个 14B image-to-video diffusion model。

新增参数尽量少：

- state encoder；
- action encoder；
- action decoder。

多个摄像头视角的处理也很朴素：把多视角拼成一张大 frame，而不是大改 backbone。

### 为什么用自回归架构

DreamZero 使用 autoregressive video generation，原因：

1. 可以用 KV cache，加速推理；
2. 可以利用视觉历史作为下一段生成的上下文；
3. 可以保留原生 FPS，避免 subsampling 破坏 video-action 对齐；
4. 在闭环执行中，每执行完一个 action chunk，就能用真实观测替换 cache 里的预测帧，降低误差积累。

重要细节：

- 自回归主要用于 video modality；
- action 不做闭环式自回归预测，避免动作误差传播；
- 视频按 chunk 生成，每个 chunk 的 latent frames 数为 `K`，匹配 action horizon。

## 5. 训练目标：joint flow matching

DreamZero 使用 flow matching 训练视频 latent 和动作。

给定 chunk index `k` 和 denoising timestep `t_k ∈ [0, 1]`：

```text
z^k_{t_k} = t_k z^k_1 + (1 - t_k) z^k_0
a^k_{t_k} = t_k a^k_1 + (1 - t_k) a^k_0
```

其中：

- `z^k_0 ~ N(0, I)`：视频 latent 噪声；
- `a^k_0 ~ N(0, I)`：动作噪声；
- `z^k_1`：干净视频 latent；
- `a^k_1`：归一化后的干净动作。

clean context：

```text
C_k = {(z^j_1, a^j_1)}_{j=1}^{k-1}
```

模型 `u_theta` 预测 joint velocity：

```text
v_k = [z^k_1, a^k_1] - [z^k_0, a^k_0]
```

训练目标：

```text
L(theta) = E[ 1/K * sum_k w(t_k)
              || u_theta([z^k_{t_k}, a^k_{t_k}]; C_k, c, q_k, t_k) - v_k ||^2 ]
```

直觉：

- 视频和动作在同一个 denoising timestep 上联合去噪；
- 模型必须让生成的视频未来和动作 chunk 互相一致；
- teacher forcing 让当前 noisy chunk 可以看干净的 previous chunks。

## 6. 推理：闭环执行

推理时，DreamZero 联合去噪视频和动作 chunk。执行完动作 chunk 后，系统把真实新观测放回 KV cache，替换先前生成的视频帧。

这非常关键：

- 普通自回归视频生成会越滚越偏；
- 机器人闭环执行天然会不断得到真实观测；
- 所以 DreamZero 能用真实观测刷新历史，减少 compounding error。

论文也指出，DreamZero 作为 stateful policy 能利用视觉历史，但本论文没有专门评估必须依赖长期记忆才能完成的任务。

## 7. 实时控制难点

朴素实现太慢：

- 16 diffusion steps；
- 14B DiT backbone；
- 推理和机器人执行顺序阻塞；
- 单 GPU 每个 action chunk 约 5.7 秒。

这对闭环控制不可用。DreamZero 的目标是让推理在 action chunk 过期前完成。实验中：

- 控制频率：30Hz；
- action horizon：48 steps；
- 每个 chunk 约 1.6 秒；
- 目标推理延迟：低于约 200ms；
- 最终实时闭环约 7Hz。

## 8. 加速方案

### 8.1 异步闭环执行

机器人持续执行最新 action chunk，同时模型并行根据最新观测推理下一段 action。

约束从：

```text
推理必须完成 -> 机器人才能动
```

变成：

```text
推理必须在当前 action chunk 执行完之前完成
```

### 8.2 系统级优化

| 优化 | 作用 |
|---|---|
| CFG Parallelism | classifier-free guidance 需要 conditional / unconditional 两次 forward，把它们分到两张 GPU |
| DiT Caching | flow matching 中相邻 velocity 方向相似时复用 cached velocity，把有效 DiT steps 从 16 降到 4 |

### 8.3 实现级优化

| 优化 | 作用 |
|---|---|
| `torch.compile` + CUDA Graphs | 消除 CPU overhead，融合算子 |
| Quantization | Blackwell 上使用 NVFP4，QKV/Softmax 等敏感部分保留更高精度 |
| Kernel & Scheduler | 用 cuDNN attention，把 scheduler 搬到 GPU，减少同步阻塞 |

### 8.4 DreamZero-Flash

即使做了系统优化，diffusion steps 仍然是瓶颈。直接把 steps 降到 1 会让动作质量下降，因为视频还很 noisy，动作会被错误视觉条件影响。

DreamZero-Flash 的核心是：

> 训练时解耦 video 和 action 的 noise schedule，让模型学会在 noisy video context 下预测 clean action。

普通 DreamZero：

```text
t_video = t_action = t_k ~ Uniform(0, 1)
```

DreamZero-Flash：

```text
t_video = 1 - eta,  eta ~ Beta(7, 1)
t_action ~ Uniform(0, 1)
```

结果是 video 偏向高噪声状态，action 仍然均匀采样。模型因此适应 few-step / single-step inference。

Flash 后：

- diffusion steps 从 4 降到 1；
- 推理从约 350ms 降到约 150ms；
- 性能损失很小。

### 8.5 累计加速

| 优化 | H100 | GB200 |
|---|---:|---:|
| Baseline | 1x | 1.1x |
| + CFG Parallelism | 1.9x | 1.8x |
| + DiT Caching | 5.5x | 5.4x |
| + Torch Compile + CUDA Graphs | 8.9x | 10.9x |
| + Kernel & Scheduler Opts. | 9.6x | 14.8x |
| + Quantization (NVFP4) | N/A | 16.6x |
| + DreamZero-Flash | N/A | 38x |

最终 GB200 上从 5.7s 降到约 150ms。

## 9. 数据和训练

### AgiBot G1 数据

作者用 AgiBot G1 收集约 500 小时遥操作数据：

- 22 个真实环境；
- 包括家庭、餐厅、超市、咖啡店、办公室；
- 7,193 个 episodes；
- 平均每条 episode 约 4.4 分钟；
- 平均约 42.4 个 subtasks；
- 强调 task diversity，而不是每个任务重复很多遍。

技能覆盖包括：

- navigation；
- torso adjustment；
- pick and place；
- folding；
- cleaning；
- restocking；
- table bussing 等。

### DROID / Franka 数据

作者也在 Franka 单臂机器人上用 DROID 训练，目的是验证 WAM 能否利用公开异构机器人数据，并方便复现。

### 训练设置

- backbone：Wan2.1-I2V-14B-480P；
- AgiBot：100K steps，global batch size 128；
- DROID：100K steps，global batch size 128；
- 更新：所有 DiT blocks、state encoder、action encoder、action decoder；
- 冻结：text encoder、image encoder、VAE；
- 动作表示：默认相对关节位置；
- 过滤 idle actions；
- LoRA 实验效果不佳，所以主要用全参数更新。

## 10. 实验设置

baseline：

- GR00T N1.6；
- `π0.5`。

每个 baseline 有两种初始化：

- scratch：从预训练 VLM 权重开始，没有先验机器人数据；
- pretrained：使用官方在大量跨 embodiment 机器人数据上预训练的 checkpoint。

所有方法在相同机器人数据上继续训练，并尽量匹配 batch size 和 gradient steps。

默认评估条件很严格：

- unseen environments；
- unseen objects；
- 训练和评估地点不同；
- 所以测试是 OOD generalization，不是训练分布插值。

## 11. 主结果 Q1：WAM 是否更能从多样非重复数据学习

在 AgiBot G1 seen tasks 上，DreamZero 平均 task progress 达到 `62.2%`，超过最好的 pretrained VLA baseline `27.4%`，超过 2 倍。

在 DROID-Franka 上，DreamZero 也优于 pretrained baseline。论文认为差距来自联合 video-action formulation：

- VLA 要直接学习 observation-to-action 映射；
- WAM 有视频生成先验，能把动作学习变成更接近 inverse dynamics 的问题；
- 多样数据对 WAM 更有价值，因为它丰富了状态-动作-未来变化的对应关系。

## 12. 主结果 Q2：能否泛化到 unseen tasks

AgiBot G1 的 unseen tasks 包括：

- 解鞋带；
- 熨衣服；
- 用刷子画；
- 握手；
- 从 mannequin 取帽子；
- 堆方块等。

结果：

- scratch VLA：接近 0；
- DreamZero：平均 `39.5%` task progress；
- pretrained VLA baseline：`16.3%` 左右；
- DreamZero 明显更强。

DROID-Franka：

- DreamZero：`49%` task progress，`22.5%` success rate；
- GR00T N1.6 pretrained：`31%` task progress，`12.5%` success rate；
- `π0.5` pretrained：`33%` task progress，`7.5%` success rate。

作者观察到，pretrained VLA 遇到新任务时经常默认去 reach/grasp，像是过拟合到 pick-and-place 主行为。DreamZero 更像先生成视觉计划，再执行对应动作。

## 13. 主结果 Q3：post-training 后是否仍保持泛化

作者在 AgiBot 上做三个下游任务：

| 任务 | 数据量 | 描述 |
|---|---:|---|
| Shirt folding | 33 hrs | 5 个 sequential stages |
| Fruit packing | 12 hrs | 把 10 个水果装进袋子 |
| Table bussing | 40 hrs | 垃圾进垃圾桶，餐具进餐具箱 |

每个任务 post-train 50K steps。

结论：

- DreamZero 在所有任务上匹配或超过 VLA baseline；
- fruit packing 上优势明显；
- post-training 后仍能在 unseen environment 上保持泛化。

## 14. 主结果 Q4：跨 embodiment video-only transfer

作者测试两种迁移：

1. Robot-to-robot：YAM -> AgiBot；
2. Human-to-robot：人类第一视角 -> AgiBot。

数据很少：

- 9 个 unseen tasks；
- 每个任务 8 条 demonstration；
- YAM 约 20 分钟；
- human 约 12 分钟；
- 只用视频，不用动作标签。

训练方式：

- 从 DreamZero-AgiBot checkpoint 出发；
- 和原 pretraining data 做 1:1 mix；
- co-train 10K steps；
- cross-embodiment 数据只用 video prediction objective；
- AgiBot 数据仍用 joint video-action objective。

结果：

| 方法 | Unseen task progress |
|---|---:|
| DreamZero | 38.3% ± 7.6% |
| DreamZero + Human2Robot Transfer | 54.3% ± 10.4% |
| DreamZero + Robot2Robot Transfer | 55.4% ± 9.5% |

这个结果很重要：没有动作标签的视频，也能帮助 WAM 学更好的 task dynamics 和预期行为。

## 15. 主结果 Q5：30 分钟适配新 embodiment

作者把 AgiBot 预训练好的 DreamZero 迁移到新双臂机器人 YAM：

- 只有 55 条 trajectory；
- 11 个 unique tasks；
- 约 30 分钟 play data。

结果：模型仍保留较强语言跟随能力，并泛化到训练中没见过的物体，比如 pumpkin、teddy bear、cup noodles、paper bag。

作者解释可能原因：

- AgiBot G1 和 YAM 都是双臂 parallel gripper，视觉形态相近；
- WAM 学的是“从视觉未来到动作”的 implicit inverse dynamics，这可能比直接 policy learning 更省数据；
- video backbone 已经理解很多物理动力学，新 embodiment 主要补动作映射。

## 16. 主结果 Q6：DreamZero-Flash 是否保持性能

Table bussing 上比较：

| 方法 | Denoising steps | Task progress | Inference speed |
|---|---:|---:|---:|
| DreamZero | 4 | 83% ± 6.1% | 350ms |
| DreamZero | 1 | 52% ± 10.2% | 150ms |
| DreamZero-Flash | 1 | 74% ± 10.1% | 150ms |

说明 Flash 的 decoupled noise schedule 能恢复大部分单步推理损失。

## 17. 消融实验

### 数据多样性

同样 500 小时：

| 数据 | Task progress |
|---|---:|
| repetitive data | 33% ± 4.2% |
| diverse data | 50% ± 6.3% |

WAM 更喜欢多样状态-动作-未来对应，而不是同一任务反复演示。

### 模型规模

| 模型 | 规模 | Task progress |
|---|---:|---:|
| DreamZero | 5B | 21% ± 4.2% |
| DreamZero | 14B | 50% ± 6.3% |
| VLA | 5B | 0% |
| VLA | 14B | 0% |

WAM 对视频 backbone scale 更敏感，14B 明显优于 5B。单纯把 VLA 做大并不能解决从多样数据中学动作的问题。

### 自回归 vs 双向

| 架构 | Task progress |
|---|---:|
| Bidirectional WAM | 50% ± 14.4% |
| Autoregressive WAM | 50% ± 6.3% |

任务进度类似，但 AR：

- 动作更平滑；
- KV cache 让推理快 3 到 4 倍；
- 更适合实时闭环。

## 18. 讨论与未来方向

### 18.1 WAM scaling laws

作者认为 WAM 可能有自己的 scaling law：模型规模、数据规模、训练计算量如何影响机器人动作能力，还没有系统研究。WAM 的动作 scaling 规律可能不同于 VLA。

### 18.2 从野外人类视频学习

本文只用了 12 分钟 in-lab human egocentric data，但互联网上和第一视角数据集中有大量人类操作视频。因为 WAM 本来就从视频模型初始化，作者认为大规模人类视频可能比当前 VLA 更适合迁移到机器人技能。

### 18.3 更快推理

DreamZero 已经能用 2 张 GB200 跑到 7Hz，但相比很多 VLA 在消费级 GPU 上 20Hz 以上，仍然昂贵。未来需要更小但泛化仍强的视频 backbone。

### 18.4 长时序推理

当前 DreamZero 更像 System 1 policy，视觉记忆约 6 秒。长时序任务可能需要：

- System 2 planner；
- 或更长 context window 的 WAM。

### 18.5 高精度任务

DreamZero 的广泛泛化不等于能做好亚厘米精度任务，比如插钥匙、精密装配。这类任务可能仍需要密集演示。不过作者也提到，近期 WAM 在毫米级精度任务上有积极信号。

### 18.6 embodiment 设计

作者提出两个影响 WAM embodiment 的因素：

- DoF 越高，implicit IDM 需要更多 play data；
- 越像人类的 embodiment，越可能从海量人类视频中受益。

这两个方向有张力：机械简单更易学习动作映射，但人形机器人更容易吸收人类视频先验。

## 19. 和其他 world model 的区别

论文附录比较了几类 world model：

| 类型 | 代表 | 特点 | 和 WAM 的区别 |
|---|---|---|---|
| Latent-space world model | Dreamer、V-JEPA | 在抽象 latent 中预测未来 | 通常需要 goal-conditioned planning 或 search |
| 3D point cloud world model | PointWorld | 在 3D spatial domain 中预测 scene dynamics | 部署时需要 MPC/MPPI 等优化 |
| Video world model + IDM | RoboDreamer、UniPi 类方法 | 先生成视频，再用 inverse dynamics 变动作 | 视频和动作分离，可能对齐不足 |
| WAM | DreamZero | 联合预测未来视频和动作 | 直接输出 action trajectory，无需测试时搜索 |

关键区别：

```text
传统 world model: p(s_{t+1} | s_t, a_t) + planner
WAM:              p(o_{t:t+H}, a_{t:t+H} | o_{0:t}, c)
```

所以 WAM 更像一种“会想象未来的策略”，而不是单独模拟器。

## 20. 对 VLA 研究的启发

这篇论文对 VLA 路线有几个直接启发：

1. 只把 VLM 接动作头可能不够，动作泛化需要时空动力学预训练。
2. 视频生成质量会直接影响 action quality，机器人策略可能跟视频 backbone scaling 强绑定。
3. 训练数据不一定要每个任务反复演示；多样、真实、长时序数据对 WAM 可能更有价值。
4. 人类视频和其他机器人视频即使没有动作标签，也能通过 video prediction objective 帮助机器人学任务动力学。
5. 实时部署是 WAM 的硬门槛，模型结构和系统优化同样重要。

## 21. 最短总结

DreamZero 可以记成一句话：

> WAM 用视频扩散模型的世界先验，同时生成未来视频和连续动作，把“世界模型”从模拟器推进成可实时闭环执行的 zero-shot robot policy。
