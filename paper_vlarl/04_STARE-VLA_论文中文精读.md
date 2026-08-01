# StARe-VLA 中文精读

原文: *StARe-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models*

链接: https://arxiv.org/abs/2512.05107

项目页: https://sites.google.com/view/starevla

## 一句话

StARe-VLA 的核心是：VLA 输出的是长时程动作轨迹，不应该像语言序列一样只给整条轨迹一个偏好或奖励，而应该按操作阶段拆开，比如 Reach、Grasp、Transport、Place，再给每个阶段更细的奖励和偏好信号。

## 摘要翻译

近期 VLA 模型在大语言模型和 RL fine-tuning 推动下，在机器人 manipulation 中有明显进展。现有方法通常把长时程动作当成语言序列，使用 TPO 或 PPO 这类 trajectory-level optimization，导致 credit assignment 粗糙、训练不稳定。

但动作轨迹和语言不同。语言句子可以有比较灵活的语序，整体语义仍成立；动作轨迹则由因果链条上的不同阶段组成，而且每个阶段难度不同。因此作者提出 progressive stage optimization。

作者提出 Stage-Aware Reinforcement，简称 StARe。它把长时程动作轨迹分解成有语义意义的阶段，并提供 dense、可解释、与阶段对齐的强化信号。把 StARe 接入 TPO 和 PPO 后，分别得到 StA-TPO 和 StA-PPO。再基于 SFT 初始化，作者提出 Imitation -> Preference -> Interaction，简称 IPI 的串行 fine-tuning pipeline。实验在 SimplerEnv 和 ManiSkill3 上取得很高成功率：SimplerEnv 98.0%，ManiSkill3 96.4%。

## 背景问题

很多 VLA fine-tuning 直接借鉴 LLM：

- SFT
- RLHF
- DPO
- PPO
- GRPO

但 VLA 动作轨迹不是普通文本。比如 pick-and-place：

```text
Reach -> Grasp -> Transport -> Place
```

这些阶段有严格顺序，而且难度不同。整条轨迹成功/失败的信号太粗：

- 失败了，可能是没抓住；
- 也可能是抓住了但放偏；
- 也可能是接近成功但释放不稳。

只给整条轨迹一个 reward 或 preference，模型不知道该改哪一段。

## StARe 模块

StARe 有两个组件：

1. Stage Separator
   根据任务相关事件切分轨迹阶段。

2. Stage Calculator
   对每个阶段计算 stage cost 和 dense reward。

### Stage Separator

切分不是按固定时间，而是按语义事件。例如：

```text
Reach -> Grasp:
end-effector 接触物体

Grasp -> Transport:
物体被抓起并高过阈值

Transport -> Place:
物体接近目标位置

Place -> Success:
释放后物体稳定留在目标区域
```

这样得到：

```text
τ -> { τ^(1), τ^(2), ..., τ^(K) }
```

每段对应一个语义阶段。

### Stage Calculator

以 Reach 阶段为例，stage cost 可以是 end-effector 和目标物体的平均距离：

```text
ℓ_k(τ^(k))
= (1/T_k) Σ_t || x_ee(t) - x_obj(t) ||_2
```

cost 越小，说明该阶段做得越好。

Dense reward 使用 potential-based shaping：

```text
r'_t = r_t + γ Φ_{t+1}(s_{t+1}) - Φ_t(s_t)
```

例如 Reach 的 potential 衡量 end-effector 接近物体的程度。这样 sparse terminal reward 被变成每一步都有反馈的 progressive signal。

## StA-TPO

普通 TPO 是 DPO 在轨迹上的扩展：

```text
L_TPO = - E log σ( β(q(τ+) - q(τ-)) )
```

其中：

```text
q(τ) = average_t [
  log π_new(a_t|s_t) - log π_ref(a_t|s_t)
]
```

问题是 TPO 对整条轨迹做 preference，credit assignment 粗。

StA-TPO 改成对每个阶段做 preference，并把 stage cost 加进去：

```text
q_hat(τ^(k)) = q(τ^(k)) - λ ℓ_k(τ^(k))
```

然后对阶段级 pair 做 preference loss。这样不仅能区分成功/失败，还能区分“成功但不够好”和“成功且动作质量高”。

## StA-PPO

普通 PPO 用环境 reward 算 advantage：

```text
GAE(r_t)
```

StA-PPO 用 StARe shaped reward 替换：

```text
GAE(r'_t)
```

其中：

```text
r'_t = r_t + γΦ_{t+1}(s_{t+1}) - Φ_t(s_t)
```

因此 online RL 中，每个阶段都有更密的反馈，特别适合长时程稀疏奖励任务。

## IPI Pipeline

作者把三个阶段串起来：

```text
Imitation -> Preference -> Interaction
```

对应：

1. SFT
   用专家 demonstrations 安全 warm start。

2. StA-TPO
   用离线 stage-wise preferences 对齐策略。

3. StA-PPO
   在线交互，用 stage-aware dense reward 进一步提升鲁棒性。

## 实验结论

Benchmark：

- SimplerEnv / WidowX
- ManiSkill3 / Franka

Backbone：

- OpenVLA-7B
- pi0.5_base

主要结果：

- OpenVLA-7B + IPI 在 SimplerEnv 平均成功率达到 98.0%。
- OpenVLA-7B + IPI 在 ManiSkill3 平均成功率达到 96.4%。
- pi0.5 based 方法也从 StARe 中受益，显著超过普通 SFT / GRAPE / πRL baseline。
- StA-PPO 相比普通 PPO，在高精度和接触丰富任务上优势最明显，比如 StackCube、LiftPegUpright。
- 对简单 pick-and-place 或 push/pull，PPO 和 StA-PPO 最终成功率接近，但 StA-PPO 收敛更快、方差更低。

## 为什么有效

StARe 解决的是 credit assignment：

```text
整条轨迹失败 -> 不知道哪一步错
阶段级 reward/preference -> 知道 Reach/Grasp/Place 哪段有问题
```

这和 RL 课里的 reward shaping、dense reward、advantage estimation 都有关。它不是发明一个全新的 RL 算法，而是给 VLA 任务设计更合理的奖励/偏好结构。

## 局限

- Stage rules 是 rule-based，需要针对任务设计。
- 对真实世界复杂任务，stage separator 是否稳定仍是问题。
- 如果任务没有清晰阶段，StARe 的优势可能下降。
- 实验主要在模拟 benchmark 上，真实机器人泛化还需要更多验证。

## 和你当前方向的关系

这篇更靠近 VLA RL fine-tuning：

- 如果你研究 `π0.5 / π0.6 / OpenVLA`，这篇很相关。
- 它关注的不是 flow/diffusion likelihood，而是“长时程动作轨迹怎么给 RL 信号”。
- 适合和 RLHF 里的 process reward / step-wise reward 对照看。

