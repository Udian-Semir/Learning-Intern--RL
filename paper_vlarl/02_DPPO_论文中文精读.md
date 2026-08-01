# DPPO 中文精读

原文: *Diffusion Policy Policy Optimization*

链接: https://arxiv.org/abs/2409.00588

项目页: https://diffusion-ppo.github.io/

## 一句话

DPPO 把 Diffusion Policy 的 denoising 过程看成一个内部 MDP，再把环境 MDP 和 diffusion MDP 组合成 two-layer MDP，从而可以用 PPO 对 diffusion policy 做 policy gradient fine-tuning。

## 摘要翻译

作者提出 Diffusion Policy Policy Optimization，简称 DPPO。这是一个用于 fine-tune diffusion-based policies 的算法框架和实践经验集合，面向连续控制和机器人学习任务。虽然 policy gradient 方法广泛用于其他 policy parameterization，但此前有人认为它不适合 diffusion policy。作者反而发现，DPPO 在常见 benchmark 上，相比其他 diffusion policy RL fine-tuning 方法以及其他 policy parameterization 的 PG 微调，表现和效率都很强。

实验显示，DPPO 利用了 RL fine-tuning 和 diffusion parameterization 之间的独特协同作用，带来结构化探索、稳定训练和更强鲁棒性。作者还在视觉输入机器人任务、长时程多阶段 manipulation，以及 sim-to-real 零样本部署中验证了 DPPO。

## 背景问题

Diffusion Policy 通过行为克隆学 expert demonstrations，能表达复杂多模态动作分布。但如果 expert data 覆盖不足或质量不高，BC 策略表现会受限。机器人能和环境交互，因此 RL 是自然的进一步优化工具。

难点是：

- diffusion policy 没有普通 Gaussian policy 那样直接的 `π(a|s)` likelihood；
- 直接把最终动作看成 policy output 很难对它做 policy gradient；
- denoising 过程增加了有效 horizon，可能导致方差更大。

DPPO 的回答是：不要只看最终动作，把 diffusion denoising 的每一步都看成一个 MDP transition。

## Two-layer MDP

外层是环境 MDP：

```text
s_t -> a_t -> r_t -> s_{t+1}
```

内层是 diffusion denoising MDP：

```text
a^K_t -> a^{K-1}_t -> ... -> a^0_t
```

其中 `a^K_t` 是初始高斯噪声，`a^0_t` 是最终给环境执行的动作。

DPPO 把它们合起来：

```text
环境每一步动作生成 = 一个完整 diffusion MDP
```

内部 denoising step 的 transition 是 Gaussian：

```text
x_{k-1} ~ N( μ_k(x_k, ε_θ(x_k,k)), σ_k^2 I )
```

因此每一步都有 tractable Gaussian likelihood，可以写进 policy gradient。

## PPO 实例化

普通 PPO 使用：

```text
min(
  ratio * A,
  clip(ratio, 1-ε, 1+ε) * A
)
```

DPPO 在 diffusion MDP 里使用同样思想，只是 `(s,a)` 换成内部 denoising state/action：

```text
s_bar = (environment state, noisy action)
a_bar = next denoised action
```

优势估计考虑 two-layer 结构。环境 reward 只在最终动作执行后出现，内部 denoising steps 没有 reward。作者用 denoising discount 下调更 noisy 的早期 denoising step 对 policy gradient 的贡献。

## 关键实践经验

1. 只 fine-tune 最后几个 denoising steps

早期 denoising steps 可以冻结，最后几个 steps 用 PPO 更新。这样显存和时间更省，性能不明显下降。

2. 用 DDIM 减少采样步数

像视觉任务和 Furniture-Bench 这种更贵的任务，可以用少量 DDIM steps fine-tune。

3. 噪声 schedule 很重要

DPPO 需要 diffusion noise 提供探索。如果最终噪声太小，探索不足；如果 likelihood 里方差太小，log prob 过大，训练不稳定。作者会 clip `σ_k` 到一个最小值。

4. Value estimator 只依赖环境 state

作者发现 value function 不依赖 denoised action，训练更稳定。这和 diffusion policy 的高随机性有关。

## 实验结论

测试环境包括：

- OpenAI Gym: Hopper, Walker2D, HalfCheetah
- Franka Kitchen
- Robomimic: Lift, Can, Square, Transport
- Furniture-Bench
- 真实家具装配 sim-to-real

主要结果：

- DPPO 相比 DIPO、IDQL、DQL、QSM 等 diffusion RL 方法训练更稳定，尤其在 Robomimic Transport 这种困难任务上更强。
- 相比 RLPD、Cal-QL、IBRL 等 demo-augmented RL，DPPO 在 Robomimic 上最终表现更好，wall-clock 也更有优势。
- 相比 Gaussian / GMM policy + PPO，DPPO 在 Square 和 Transport 上明显更强。
- 在 Furniture-Bench 中，DPPO 在 6 个设置里都提升了策略性能。
- sim-to-real One-leg 任务中，DPPO 零样本真实机器人成功率达到 80%；Gaussian policy 尽管仿真成功率高，真实部署失败严重。

## 作者对 DPPO 成功原因的解释

1. Diffusion policy 带来靠近 demonstrations manifold 的结构化探索。
2. 多步 denoising 让策略分布可以逐步更新，不容易 collapse。
3. Fine-tuned policy 对 dynamics perturbation 和 initial state distribution 更鲁棒。

这点很重要：DPPO 不只是“PPO 套在 diffusion 上”，它利用了 diffusion policy 的动作分布形状。

## 局限

- 仍然是 on-policy fine-tuning，真实机器人上 sample cost 会高。
- 超参数较多，特别是 denoising steps、noise clipping、advantage estimator。
- two-layer MDP 增加 horizon，理论上仍可能带来方差问题，只是实验中被实践技巧缓解。

## 和 ReinFlow 的关系

DPPO 是 diffusion 版本：

```text
diffusion denoising -> Gaussian transitions -> PPO
```

ReinFlow 是 flow 版本：

```text
flow denoising + learnable noise -> Gaussian transitions -> PPO
```

所以读这两篇时可以对照看。DPPO 是前辈，ReinFlow 借鉴了它“把生成过程看成 MDP”的路线，但针对 flow 的确定性路径设计了 noise injection。

