# ReinFlow 中文精读

原文: *ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning*

链接: https://arxiv.org/abs/2505.22094

项目页: https://reinflow.github.io/

## 一句话

ReinFlow 是一个用在线强化学习微调 flow matching 机器人策略的方法。它的核心技巧是给原本确定性的 flow 轨迹注入可学习高斯噪声，把动作生成过程变成离散时间 Markov process，从而可以精确计算 likelihood，再用 PPO 这类 policy gradient 方法微调。

## 摘要翻译

作者提出 ReinFlow，一个简单但有效的在线 RL 框架，用来微调一类 flow matching policies，使其适用于连续机器人控制。ReinFlow 从 RL 理论出发，在 flow policy 的确定性路径中注入可学习噪声，把 flow 转换成一个离散时间 Markov process。这样就可以精确、直接地计算 likelihood。

这个转换同时带来探索能力和训练稳定性，使 ReinFlow 能稳定微调不同类型的 flow model，包括 Rectified Flow 和 Shortcut Models，尤其是在去噪步数很少，甚至只有一步时。作者在 locomotion 和 manipulation 任务上测试，包括视觉输入、稀疏奖励和长时程规划。实验显示，Rectified Flow 策略在困难腿式 locomotion 任务上 fine-tuning 后平均 episode reward 净增长 135.36%，相比 diffusion RL fine-tuning 方法 DPPO 节省 denoising steps 和 82.63% wall time。Shortcut Model 策略在状态和视觉 manipulation 任务上，在 4 步甚至 1 步 denoising 时，成功率平均净提升 40.34%，性能接近 fine-tuned DDIM policy，同时平均节省 23.20% 计算时间。

## 背景问题

Flow matching policy 很适合机器人动作生成，因为它训练简单、推理快、表达能力强。但它有两个问题：

1. 它通常来自 imitation learning，数据质量有限时，BC 策略会卡在专家数据水平，不能主动探索。
2. Flow 的采样路径通常是确定性 ODE。RL 里的 policy gradient 需要 `log π(a|s)`，但 flow 的最终动作 likelihood 难算，尤其 denoising step 很少时，连续 flow 的 divergence / trace 估计误差会很大。

ReinFlow 解决的是：**怎样让 flow policy 既能保持快推理，又能像随机策略一样用在线 RL 微调。**

## 方法核心

普通 flow inference 是：

```text
a_{k+1} = a_k + v_θ(t_k, a_k, o) Δt_k
```

这是确定性的。给定 `a_k` 后，`a_{k+1}` 是唯一值，transition 是 Dirac delta，policy likelihood 很难直接用于 PPO。

ReinFlow 改成：

```text
a_0 ~ N(0, I)
a_{k+1} ~ N(
  a_k + v_θ(t_k, a_k, o) Δt_k,
  σ^2_{θ'}(t_k, a_k, o)
)
```

其中 `σ_{θ'}` 是一个 noise injection network。这样每一步都是高斯转移，log probability 有闭式表达：

```text
log π(a_0,...,a_K | o)
= log N(a_0; 0, I)
  + Σ_k log N(
      a_{k+1};
      a_k + v_θ(t_k,a_k,o)Δt_k,
      σ^2_{θ'}(t_k,a_k,o)
    )
```

关键点：机器人真正执行的是最终动作 `a_K`，但训练时把整个 denoising trajectory `a_0,...,a_K` 看成内部 Markov process，用每一步 Gaussian likelihood 做 policy gradient。

## Policy Gradient 公式含义

作者证明了一个针对 Markov process policy 的 policy gradient theorem。直觉是：

```text
∇J(π_θ)
≈ E[ A(o,a) ∇_θ Σ_k log π_θ(a_{k+1}|a_k,o) ]
```

也就是说，优势函数 `A(o,a)` 评价最终执行动作好不好，而梯度则分配到内部每个 denoising step 的 log probability 上。

这和 DPPO 的思想相似：DPPO 把 diffusion denoising 看成内部 MDP；ReinFlow 把 flow denoising 加噪后也变成内部 MDP。

## 算法流程

1. 初始化预训练 flow velocity network `v_θ`。
2. 初始化 noise network `σ_{θ'}`。
3. rollout 时，从 `a_0 ~ N(0,I)` 开始，逐步执行加噪 flow。
4. 执行最终动作 `a_K`，收集 reward。
5. 用 PPO clipped surrogate loss 更新 `θ` 和 `θ'`。
6. 训练后可以丢弃 noise net，只保留 fine-tuned deterministic flow policy。

## Regularization

论文讨论了两类正则：

1. `W2 regularization`
   约束 fine-tuned policy 不要离 pre-trained policy 太远。

2. `Entropy regularization`
   鼓励更高 entropy，让策略探索更多动作。

实验里 locomotion 任务中 entropy regularization 往往更有效。

## 实验结论

测试环境：

- OpenAI Gym: Hopper, Walker2d, Ant, Humanoid
- Franka Kitchen
- Robomimic: Can, Square, Transport

主要发现：

- ReinFlow 在 Gym 和 Franka Kitchen 上整体效率和性能最好。
- 在 Robomimic 视觉 manipulation 中，ReinFlow-S 和 ReinFlow-R 平均成功率提升 45.77%。
- ReinFlow 用更少 denoising steps，Can/Square 可做到 1 step，Transport 用 4 steps；DPPO 对应使用 5-step DDIM。
- 预训练数据更多或 inference steps 更多并不一定稳定提升表现，但 RL fine-tuning 提供了另一条 scaling path。
- noise level 很关键：太小探索不够，中等噪声提升最快，太大也可能影响精细操作。

## 局限

作者自己列了几个：

- 当前主要是 on-policy PPO，wall-time 可以靠并行省，但 sample efficiency 还可以提升。
- 真实机器人 online RL 还没充分验证。
- 对 noise magnitude 比较敏感，需要自动调参。
- 目前实验网络相对小，扩展到大规模 flow-based VLA 还是未来工作。

## 和你当前学习的关系

这篇就是把你前面学的东西直接用到 flow policy 上：

- `policy gradient`
- `advantage`
- `PPO clipped loss`
- `Markov process`
- `likelihood`
- `exploration`

最值得你重点看的是 Section 4.1 和 4.2：它们解释了为什么要给 flow 加噪，以及为什么加噪后可以用 PPO。

