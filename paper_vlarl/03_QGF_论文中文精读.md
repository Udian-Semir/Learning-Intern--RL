# QGF 中文精读

原文: *Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning*

链接: https://arxiv.org/abs/2606.11087

代码: https://github.com/zhouzypaul/qgf

## 一句话

QGF 不在训练时用 RL 更新 flow policy，而是在测试时用 critic 的 action gradient 引导 flow denoising，让行为克隆训练出来的 flow policy 生成更高 Q-value 的动作。

## 摘要翻译

Diffusion 和 flow models 这类表达能力强的连续控制策略，是近期机器人 imitation learning 的重要基础。它们在监督学习下稳定可扩展，但放到 RL pipeline 里做 policy improvement 很困难，往往需要专门目标或者穿过 denoising process 反向传播，带来稳定性和扩展性问题。

作者研究一个问题：能不能只在测试时做简单 policy improvement，而保留稳定的 supervised policy training？为此，作者提出 QGF。它先用行为克隆训练一个 reference flow policy，再训练一个 value function critic。测试时，QGF 用 value gradient 引导 reference policy 生成更高价值动作，不需要额外 policy learning。

实验显示，QGF 在高维动作空间的单任务和 goal-conditioned offline RL benchmark 上超过已有 test-time RL 方法，并且接近 state-of-the-art training-time algorithms，但运行成本更低。由于不需要 actor-critic 联合训练，它还随模型规模扩展得更稳定。

## 背景问题

传统 RL actor-critic 有一个难点：

```text
critic 一边变，actor 一边追 critic
```

这容易不稳定。对 flow/diffusion policy 更麻烦，因为动作是多步 denoising 生成的。如果要优化 actor，就要：

- 对 denoising chain 做 backprop through time，贵且不稳；
- 或者在 noisy action 上查 critic gradient，但 critic 只在 clean action 上训练过，容易 OOD。

QGF 的思路是：

```text
policy 训练时只做 BC
critic 训练时只做 value learning
测试时再用 critic gradient 改动作生成
```

## KL-regularized RL 直觉

论文用一个经典目标：

```text
maximize reward
同时不要离 behavior/reference policy 太远
```

其解有形式：

```text
π(a|s) ∝ π_hat(a|s) * exp(Q(s,a) / β)
```

也就是：好的策略等于 reference policy 乘上一个 `Q` 倾斜项。取 log gradient：

```text
∇_a log π(a|s)
= ∇_a log π_hat(a|s) + (1/β) ∇_a Q(s,a)
```

直觉：采样时沿着 reference flow 的方向走，同时加上一点让 `Q` 变大的方向。

## 为什么 naive Q guidance 不行

Flow denoising 中间会有 noisy action `a_t`。一个简单做法是：

```text
∇_{a_t} Q(s, a_t)
```

问题：critic 训练时见的是最终 clean action，不是中间 noisy action。因此 `Q(s,a_t)` 可能是 OOD 查询，gradient 会误导。

另一个更严谨做法：

```text
∇_{a_t} Q(s, ODE(a_t))
```

也就是把 noisy action 完整 denoise 成 clean action 后再算 Q。但这需要穿过完整 ODE 做反传，也就是 BPTT，贵且高方差。

## QGF 的核心近似

QGF 用一步 Euler 近似，把当前 noisy action 近似 denoise 到 clean action：

```text
a_hat_1 = a_t + v_θ(s, a_t, t) * (1 - t)
```

然后在这个近似 clean action 上算 critic gradient：

```text
g = ∇_{a_hat_1} Q(s, a_hat_1)
```

最后把这个 gradient 加到 flow velocity 上：

```text
a_{t+δ} = a_t + δ * ( v_θ(s,a_t,t) + (1/β) g )
```

这就是 Q-Guided Flow。

## 为什么不乘 Jacobian

更完整的链式法则应该有：

```text
J^T ∇_{a_hat_1} Q(s,a_hat_1)
```

其中：

```text
J = ∂a_hat_1 / ∂a_t
```

但作者发现 Jacobian 需要对 velocity field 求导，数值上更敏感、更高方差。直接把 `J` 近似成 identity 反而效果更好：

```text
J ≈ I
```

所以 QGF 的梯度估计器看起来更粗糙，但实验中更稳。

## 算法流程

测试时：

1. 从 `a_0 ~ N(0,I)` 开始。
2. 对每个 denoising time `t`：
   - 用一步 Euler 得到近似 clean action `a_hat_1`。
   - 计算 `g = ∇Q(s,a_hat_1)`。
   - 用 `v_θ + (1/β)g` 更新 noisy action。
3. 输出最终 `a_1`。

训练时：

- reference flow policy 用 BC / flow matching loss 训练；
- critic 用 IQL 或其他 Q learning 方法训练；
- policy 参数不通过 RL 更新。

## 实验结论

实验环境：

- OGBench 的 offline RL 任务
- 单任务和 goal-conditioned 设置
- 高维 action chunking

对比方法：

- Test-time methods: BFN, GradStep, QFQL, BPTT, CFGRL, RobustQ
- Training-time methods: FQL, EDP, QAM, DAC, QSM+BC

主要发现：

- QGF 明显超过之前的 test-time guidance 方法。
- QGF 不训练 reward-seeking actor，但能接近甚至略超强 training-time baseline。
- 相比 best-of-N，QGF 计算量低很多。QGF + BFN 可以在更小 sample 数下达到高 compute BFN 的效果。
- 在更难的 goal-conditioned long-horizon 任务上，QGF 相比 QFQL/BPTT 更稳。
- 模型变大时，QGF 比 QAM 更受益，因为它避免了 actor-critic 训练不稳定。
- 如果 critic 更强，比如 QAM-based Q，QGF 表现还会更好。

## 局限

- QGF 依赖 critic 质量。critic 错，guidance 也会错。
- 它是 test-time 方法，每次推理会额外算 Q gradient。
- 目前主要在 offline RL benchmark 上验证，还不是直接真实机器人在线训练框架。

## 和 DPPO / ReinFlow 的区别

DPPO / ReinFlow 是 training-time RL：

```text
更新 policy 参数
```

QGF 是 test-time RL：

```text
不更新 policy 参数，只在推理时改采样轨迹
```

所以 QGF 更像一种“测试时动作优化器”。它和 best-of-N、MCTS、test-time scaling 是同一类思路。

