# DreamerV3 中文精读

> 原论文：[Mastering Diverse Domains through World Models](02_DreamerV3_2023.pdf)  
> 常用名：DreamerV3  
> 作者：Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap  
> 年份：2023 / 2024 revision  
> 关键词：world model、RSSM、imagined trajectories、actor-critic、symlog、twohot、Minecraft

## 0. 这篇论文在讲什么

DreamerV3 的目标是做一个“不要为每个环境重新调参”的通用强化学习算法。它通过学习 world model，在模型内部想象未来轨迹，然后用 actor-critic 学行为。

最重要的主张：

> 同一套超参数，DreamerV3 在 150+ 个任务、8 类领域上超过很多专门调过的算法，并且不用人类数据、不用课程学习，从零在 Minecraft 里挖到钻石。

这篇论文可以看作 `World Models` 思路的工程化和算法化升级：

- `World Models`：VAE + MDN-RNN + 小 controller；
- `DreamerV3`：RSSM + imagined actor-critic + 一整套跨领域稳定训练技巧。

## 1. 摘要翻译

构建一个能在广泛应用中学习解决任务的通用算法，是人工智能的基础挑战。当前强化学习算法虽然可以用于相似任务，但迁移到新应用领域往往需要大量专家调参与实验。作者提出 DreamerV3，一个使用单一配置就能在 150 多个多样任务上超过专门方法的通用算法。Dreamer 学习环境模型，并通过想象未来场景来改进行为。基于归一化、平衡和变换的一系列鲁棒训练技术，使它能跨领域稳定学习。直接使用默认配置时，Dreamer 成为第一个不用人类数据或课程学习、从零在 Minecraft 中采集钻石的算法。该工作让强化学习更容易用于困难控制问题。

## 2. 问题背景

PPO 是强化学习里常用的通用算法，但性能往往不如为特定领域设计和调优的方法。不同任务的难点很不一样：

- 连续控制；
- 离散动作；
- 稀疏奖励；
- 图像输入；
- 3D 空间推理；
- 程序生成环境；
- 长期探索。

过去通常需要大量人类经验去调超参数。DreamerV3 想解决的是：能不能固定一套配置，跨很多环境都好用？

## 3. DreamerV3 的整体结构

DreamerV3 有三个神经网络：

| 模块 | 作用 |
|---|---|
| World Model | 预测未来表征、reward、episode 是否继续，并重建观测 |
| Critic | 判断想象轨迹中每个状态的价值 |
| Actor | 选择动作，让未来进入高价值状态 |

训练流程：

1. 智能体和真实环境交互，把经验存进 replay buffer。
2. 从 replay buffer 采样序列，训练 world model。
3. 从 replay 中的状态出发，在 world model 里 rollout 想象轨迹。
4. 用想象轨迹训练 critic。
5. 用 critic 的价值信号训练 actor。
6. 用 actor 继续和环境交互，循环进行。

## 4. World Model：RSSM

DreamerV3 的 world model 是 Recurrent State-Space Model，也就是 RSSM。它把状态拆成：

- `h_t`：确定性的 recurrent hidden state，负责记忆历史；
- `z_t`：随机 latent state，负责表示当前观测的不确定性和压缩信息。

RSSM 的核心形式：

```text
Sequence model:     h_t = f_phi(h_{t-1}, z_{t-1}, a_{t-1})
Encoder:            z_t ~ q_phi(z_t | h_t, x_t)
Dynamics predictor: zhat_t ~ p_phi(zhat_t | h_t)
Reward predictor:   rhat_t ~ p_phi(rhat_t | h_t, z_t)
Continue predictor: chat_t ~ p_phi(chat_t | h_t, z_t)
Decoder:            xhat_t ~ p_phi(xhat_t | h_t, z_t)
```

直觉：

- encoder 看到真实观测 `x_t`，得到 posterior latent `z_t`；
- dynamics predictor 不看当前真实观测，只根据历史预测 prior latent；
- decoder/reward/continue predictor 让 latent 必须保留对控制有用的信息。

训练目标由三类 loss 组成：

```text
L(phi) = E[sum_t beta_pred L_pred + beta_dyn L_dyn + beta_rep L_rep]
```

其中：

- `L_pred`：重建观测、预测 reward、预测 continuation；
- `L_dyn`：让 dynamics prior 接近 encoder posterior；
- `L_rep`：让 posterior 更可预测，避免 latent 太任性。

论文使用权重：

```text
beta_pred = 1
beta_dyn  = 1
beta_rep  = 0.1
```

## 5. 为什么 DreamerV3 比之前更稳

DreamerV3 的贡献很大一部分不是新结构，而是一组“跨领域不炸”的训练技巧。

### 5.1 Free Bits

KL loss 太强会导致表示塌缩，太弱又会导致 dynamics 学不会。DreamerV3 使用 free bits，把 dynamics loss 和 representation loss 在低于 1 nat 时裁掉：

```text
L_dyn = max(1, KL(sg(q_phi(z_t | h_t, x_t)) || p_phi(z_t | h_t)))
L_rep = max(1, KL(q_phi(z_t | h_t, x_t) || sg(p_phi(z_t | h_t))))
```

意思是：如果 KL 已经够小，就不要继续压它，把优化重点留给预测任务。

### 5.2 Categorical latent 加 uniform mixture

论文把 encoder 和 dynamics predictor 的 categorical distribution 参数化为：

```text
99% neural network output + 1% uniform distribution
```

这样能防止分布变成完全确定，从而避免 KL loss 突刺。

### 5.3 Symlog

不同环境的观测、reward、return 尺度可能差很多。平方损失直接预测大数值容易发散，归一化又会带来非平稳性。DreamerV3 使用 symlog：

```text
symlog(x) = sign(x) * ln(|x| + 1)
symexp(x) = sign(x) * (exp(|x|) - 1)
```

它像 log 一样压缩大数值，但能处理负数，并且在 0 附近接近恒等映射。

### 5.4 Symexp twohot

对于 reward 和 value 这类可能随机且尺度跨度大的目标，DreamerV3 不直接回归一个标量，而是预测指数间隔 bins 上的 softmax 分布。输出值是 bins 的加权平均。

训练时用 twohot 编码连续目标：

- 目标落在两个相邻 bin 之间；
- 两个 bin 分配权重，权重和为 1；
- 用 categorical cross entropy 训练。

好处是梯度大小和目标数值大小解耦，更稳。

### 5.5 Return normalization

Actor 用 advantage 训练，但稀疏奖励下普通 advantage normalization 会放大奖励噪声，影响探索。DreamerV3 对 returns 做带下限的 normalization，既能在稀疏奖励下探索，也能在密集奖励下收敛到高性能。

## 6. Actor 和 Critic 怎么在想象中学习

Actor 和 critic 不直接在真实像素上训练，而是在 world model 的 latent state 上训练：

```text
s_t = {h_t, z_t}
a_t ~ pi_theta(a_t | s_t)
v_psi(R_t | s_t)
```

从 replay 的真实状态开始，world model 和 actor 生成未来 imagined trajectory：

```text
s_1:T, a_1:T, r_1:T, c_1:T
```

Critic 学 bootstrapped lambda-return。Actor 的目标是选择动作，让 imagined return 最大，同时保留一定 entropy 促进探索。

重要区别：

> DreamerV3 和 MuZero 这类搜索式方法不同。环境交互时 DreamerV3 不做 lookahead planning，而是直接从 actor 采样动作。规划压力被转移到训练阶段的 imagination。

## 7. 实验结果

DreamerV3 在 8 类领域、150+ 任务上用固定超参数评测：

| 领域 | 特点 | 结论 |
|---|---|---|
| Atari | 57 个游戏，200M frames | 超过 MuZero、Rainbow、IQN 等强方法 |
| ProcGen | 程序生成关卡，测试泛化 | 匹配 PPG，超过 Rainbow |
| DMLab | 3D 空间和时间推理 | 100M steps 超过 IMPALA/R2D2+ 在 1B steps 的表现 |
| Minecraft | 稀疏奖励、长视野探索 | 从零学会拿钻石 |
| Atari100k | 低数据预算 | 超过 IRIS、TWM、SPR、SimPLe 等多数方法 |
| Proprio Control | 低维 proprio 输入、连续控制 | 达到新 SOTA |
| Visual Control | 图像输入连续控制 | 超过 DrQ-v2、CURL |
| BSuite | credit assignment、scale robustness 等诊断 | 达到新 SOTA，尤其奖励尺度鲁棒性强 |

### Minecraft 钻石任务

Minecraft 钻石任务很难，因为：

- 每局是随机生成的无限 3D 世界；
- episode 最多 36,000 步，约 30 分钟；
- 需要按顺序发现 12 类物品；
- reward 稀疏；
- 有经验的人类也大约需要 20 分钟获得钻石。

DreamerV3 在默认配置下，不用人类数据、不用课程学习，在 100M environment steps 内所有训练出的 agent 都发现钻石。对比方法能进展到铁镐等高级物品，但没有拿到钻石。

## 8. 消融实验

论文的消融显示：

- KL objective、return normalization、symexp twohot 都有贡献；
- world model 的无监督 reconstruction loss 非常关键；
- 如果停止 reconstruction gradients，性能明显下降；
- DreamerV3 不只是靠 reward/value 监督，而是很依赖任务无关的世界建模。

这点很重要，因为它暗示：

> 未来可以用大量无监督视频或交互数据预训练 world model，再迁移到具体控制任务。

## 9. Scaling 结果

DreamerV3 测试了 12M 到 400M 参数的模型规模，也测试了不同 replay ratio。

结论：

- 更大的模型带来更高任务性能；
- 更大的模型也需要更少环境交互；
- 更高 replay ratio 也能提高性能和数据效率；
- 固定超参数下 scaling 表现稳定。

这让 DreamerV3 不只是一个小实验算法，而是一个能随计算资源提升而变强的框架。

## 10. 和其他方法对比

| 方法 | 核心 | DreamerV3 的区别 |
|---|---|---|
| PPO | model-free 通用基线 | 更通用但数据效率低，性能通常更低 |
| SAC | 连续控制常用 | 高维图像和熵系数调参困难 |
| MuZero | value model + search | 复杂且未开源完整实现；Dreamer 不在交互时搜索 |
| Gato | 多任务专家数据 imitation | 需要专家数据；Dreamer 从环境交互学习 |
| VPT | Minecraft 行为克隆 + RL | 需要人类数据和大量 GPU；Dreamer 从零学习 |

## 11. 结论翻译

作者提出第三代 Dreamer，一个能用固定超参数掌握广泛领域任务的通用强化学习算法。Dreamer 不仅在 150 多个任务上表现强，而且在不同数据和计算预算下稳定扩展，使强化学习更接近实际应用。默认配置下，Dreamer 是第一个从零在 Minecraft 中获得钻石的算法。作为一个基于 learned world model 的高性能算法，Dreamer 为未来研究打开了方向，包括教智能体更强的世界理解和在更复杂环境中学习。

## 12. 论文可以怎么记

最短版：

> DreamerV3 学一个 RSSM world model，在 latent space 中想象未来，用想象轨迹训练 actor-critic，并通过 symlog、twohot、free bits 等技巧实现跨领域稳定训练。

和机器人/VLA 的关系：

- 如果 VLA 只是模仿 `image + language -> action`，泛化主要来自语言和视觉先验；
- DreamerV3 强调 `action -> future state` 的动力学建模；
- 后续 World Action Model 可以看作把 Dreamer 的“想象未来”和 VLA 的“直接输出动作”合到一个更大的生成模型里。
