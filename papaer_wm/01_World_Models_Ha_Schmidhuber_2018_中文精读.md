# World Models 中文精读

> 原论文：[World Models](01_World_Models_Ha_Schmidhuber_2018.pdf)  
> 作者：David Ha, Jürgen Schmidhuber  
> 年份：2018  
> 关键词：world model、VAE、MDN-RNN、CMA-ES、model-based RL、latent imagination

## 0. 这篇论文在讲什么

这篇论文的核心想法很朴素，但影响很大：

> 智能体不一定要直接从像素到动作学一个巨大策略。它可以先学一个“世界模型”，把看到的画面压缩成 latent 表示，再预测未来会怎么变化；最后只训练一个很小的控制器，根据当前 latent 和预测记忆来行动。

论文把智能体拆成三部分：

| 模块 | 名字 | 作用 |
|---|---|---|
| V | Vision / VAE | 把高维图像压缩成低维 latent 向量 `z` |
| M | Memory / MDN-RNN | 根据历史 `z` 和动作 `a` 预测未来 latent |
| C | Controller | 根据 `z` 和 RNN hidden state `h` 输出动作 |

关键不是某个模块特别新，而是组合方式非常清楚：大模型负责“理解和想象世界”，小模型负责“用这个想象做控制”。

## 1. 摘要翻译

作者探索如何为常见强化学习环境构建生成式神经网络模型。这个 world model 可以用无监督方式快速训练，学习环境的压缩空间表示和时间表示。把 world model 提取出的特征作为智能体输入后，只需要训练一个非常小、非常简单的策略，就可以完成任务。更进一步，作者甚至可以让智能体完全在自己的 world model 生成的“幻觉梦境”里训练，再把这个策略迁移回真实环境。

## 2. 引言要点

人类并不是把世界的所有细节都塞进脑子里。我们会选择重要概念和它们之间的关系，形成一个内部模型。行动时，我们经常不是显式枚举未来，而是本能地使用这个内部预测模型。

论文用棒球击球做类比：击球手没有足够时间完整计算球的轨迹，但能依赖经验形成的预测模型做快速反应。对 RL 智能体来说，类似地，拥有一个能表示过去和现在、预测未来的模型，会让决策更容易。

传统 model-free RL 往往使用较小网络，因为强化学习的 credit assignment 难度很高，直接训练百万级甚至千万级参数策略很不稳定。作者提出把问题拆开：

- 先用监督/无监督学习训练较大的 world model；
- 再在 world model 的表示上训练很小的 controller；
- 这样 controller 的搜索空间很小，优化更容易。

## 3. Agent 模型

### 3.1 VAE：视觉压缩器

环境每一步给智能体一个高维观测，通常是 RGB 图像。V 模块用 Variational Autoencoder 把图像压缩成 latent vector：

```text
image frame -> Encoder -> z -> Decoder -> reconstructed frame
```

论文在 CarRacing 中使用 `z ∈ R^32`，在 VizDoom 中使用 `z ∈ R^64`。VAE 重建会丢失细节，但保留和控制相关的结构，比如赛道位置、墙、障碍、火球等。

### 3.2 MDN-RNN：未来预测器

VAE 只压缩当前帧，不知道时间如何流动。M 模块用 RNN 建模 latent 动态：

```text
P(z_{t+1} | a_t, z_t, h_t)
```

因为环境可能是随机的，作者没有让 RNN 只输出一个确定的 `z_{t+1}`，而是输出一个高斯混合分布，也就是 MDN-RNN。这样模型能表示多种可能未来。

采样时可以调温度参数 `τ`：

- `τ` 小：未来更确定，容易 mode collapse，也更容易被 controller 利用漏洞；
- `τ` 大：梦境更随机，更难被“作弊”，但太大会让任务过难。

### 3.3 Controller：极小策略

控制器 C 是一个单层线性模型：

```text
a_t = W_c [z_t, h_t] + b_c
```

其中：

- `z_t` 表示当前画面；
- `h_t` 表示 RNN 对历史和未来分布的记忆；
- `a_t` 是动作。

这让 controller 参数非常少。CarRacing 里 controller 只有 867 个参数；VizDoom 里只有 1088 个参数。作者用 CMA-ES 这种黑箱进化策略优化 controller。

## 4. CarRacing 实验

### 4.1 实验流程

CarRacing-v0 是 OpenAI Gym 的俯视赛车环境，赛道随机生成。环境的解决标准是连续 100 次平均分超过 900。

作者步骤如下：

1. 用随机策略采集 10,000 条 rollout。
2. 训练 VAE，把帧编码成 `z ∈ R^32`。
3. 训练 MDN-RNN，建模 `P(z_{t+1} | a_t, z_t, h_t)`。
4. 定义 controller：`a_t = W_c [z_t, h_t] + b_c`。
5. 用 CMA-ES 搜索 `W_c` 和 `b_c`，最大化累计 reward。

模型规模：

| 模块 | 参数量 |
|---|---:|
| VAE | 4,348,547 |
| MDN-RNN | 422,368 |
| Controller | 867 |

### 4.2 只用 VAE 会怎样

如果 controller 只能看到 `z_t`，也就是当前帧压缩表示，而不能看到 `h_t`，它仍然能开车，但会不稳定，在急弯处容易摇摆和出界。

结果：

| 方法 | 平均分 |
|---|---:|
| V model only | 632 ± 251 |
| V model + hidden layer | 788 ± 141 |
| Full world model | 906 ± 21 |

结论：当前视觉表示 `z_t` 不够，RNN hidden state `h_t` 提供了对未来的预测信息，能显著提高控制质量。

### 4.3 Full World Model 的意义

完整模型达到了 `906 ± 21`，超过解决标准。它不需要显式 lookahead planning，也不需要每一步展开很多未来轨迹。`h_t` 已经把“未来会怎样”的信息压进了隐藏状态，controller 可以像反射一样直接行动。

这就是这篇论文很有启发性的地方：world model 不一定只用于搜索规划，也可以变成一种供策略直接读取的未来表征。

## 5. VizDoom：在梦境里训练

### 5.1 任务

VizDoom Take Cover 中，智能体要躲避怪物发射的火球。没有显式奖励，累计 reward 定义为存活时间。每局最多 2100 步，平均存活超过 750 步算解决。

### 5.2 和 CarRacing 的差异

这里的 M 不只预测下一帧 latent，还要预测是否死亡：

```text
P(z_{t+1}, d_{t+1} | a_t, z_t, h_t)
```

有了下一帧 latent 和 done flag，就能把 MDN-RNN 包装成一个完整的 Gym 环境。controller 可以完全在 latent dream environment 里训练。

流程：

1. 随机策略采集 10,000 条 rollout。
2. 训练 VAE，把帧编码到 `z ∈ R^64`。
3. 训练 MDN-RNN，预测下一 latent 和死亡事件。
4. 定义 controller：`a_t = W_c [z_t, h_t]`。
5. 在虚拟环境中用 CMA-ES 最大化存活时间。
6. 把梦里学到的策略直接迁移到真实 VizDoom。

模型规模：

| 模块 | 参数量 |
|---|---:|
| VAE | 4,446,915 |
| MDN-RNN | 1,678,785 |
| Controller | 1,088 |

### 5.3 结果

在 dream environment 中，controller 学会躲避火球，约能存活 900 步。迁移回真实环境后，100 次平均存活约 1100 步，超过 750 的解决标准，也超过当时 Gym leaderboard 的 `820 ± 58`。

这个结果说明：只要 world model 捕捉了任务所需的关键规律，策略可以在“想象环境”中学习，然后迁移到真实环境。

## 6. world model 被 controller 作弊的问题

作者发现一个重要风险：controller 可能学会利用 world model 的漏洞，而不是学会真实环境中的好策略。

例如，在低温度 `τ = 0.1` 时，MDN-RNN 接近确定性模型，容易 mode collapse。controller 发现某些动作模式可以让 dream environment 中的怪物不发火球，于是在梦里拿满分，但回到真实环境会失败。

不同温度下的 VizDoom 迁移结果：

| 温度 `τ` | dream 分数 | 真实分数 |
|---:|---:|---:|
| 0.10 | 2086 ± 140 | 193 ± 58 |
| 0.50 | 2060 ± 277 | 196 ± 50 |
| 1.00 | 1145 ± 690 | 868 ± 511 |
| 1.15 | 918 ± 546 | 1092 ± 556 |
| 1.30 | 732 ± 269 | 753 ± 139 |
| random policy | N/A | 210 ± 108 |
| Gym leader | N/A | 820 ± 58 |

结论：

- dream 太确定：容易被策略找到漏洞；
- dream 有适度随机性：能让策略更鲁棒；
- dream 太随机：策略学不到稳定技能。

这也是后来很多 model-based RL 方法非常关心 model exploitation 的原因。

## 7. 迭代训练过程

论文指出，如果环境复杂，随机策略采集的数据不够覆盖重要状态，需要迭代训练：

1. 随机初始化 M 和 C。
2. 在真实环境 rollout，保存动作和观测。
3. 训练 M 预测：

```text
P(x_{t+1}, r_{t+1}, a_{t+1}, d_{t+1} | x_t, a_t, h_t)
```

4. 在 M 里训练 C。
5. 如果任务没完成，回到真实环境继续采数据。

作者还提到可以把 M 的预测误差当作 intrinsic motivation，让智能体主动探索 world model 不熟悉的地方。

## 8. 讨论

这篇论文最重要的贡献不是拿最高分，而是把一个清晰范式讲出来：

```text
learn compact world -> imagine/predict future -> train small controller
```

它连接了几个后来很重要的方向：

- Dreamer 系列：在 latent world model 中训练 actor-critic。
- 视频生成做机器人规划：用生成视频表达未来。
- World Action Model：同时预测未来视觉状态和动作。
- 表征学习 + 控制：让大模型学世界，小策略学任务。

## 9. 给机器人/VLA 的启发

- VLA 如果只做 `observation -> action`，可能缺少显式的未来建模。
- world model 提供的是“行动后世界会怎么变”的中间层。
- 对机器人来说，好的 world model 可能比更大的 policy 更关键。
- 但如果 world model 不准，策略会 exploit model，因此需要真实闭环数据、随机性、uncertainty 或在线校正。

## 10. 最短总结

`World Models` 这篇可以记成一句话：

> 先用无监督学习学会看世界和预测世界，再用一个小控制器读取这个世界模型行动；甚至可以在模型生成的梦境里训练策略。
