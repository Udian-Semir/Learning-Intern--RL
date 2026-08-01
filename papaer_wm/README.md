# World Model 论文中文整理

这个目录收了 5 篇 world model / video world model / world action model 相关论文。下面按阅读顺序整理。

| 顺序 | 论文 | 中文稿 | 重点 |
|---:|---|---|---|
| 1 | [World Models](01_World_Models_Ha_Schmidhuber_2018.pdf) | [01 中文精读](01_World_Models_Ha_Schmidhuber_2018_中文精读.md) | 经典 VAE + MDN-RNN + 小控制器，把环境压成可预测 latent world |
| 2 | [DreamerV3](02_DreamerV3_2023.pdf) | [02 中文精读](02_DreamerV3_2023_中文精读.md) | RSSM 世界模型 + imagined actor-critic，固定超参跨 150+ 任务 |
| 3 | [RoboDreamer](03_RoboDreamer_2024.pdf) | [03 中文精读](03_RoboDreamer_2024_中文精读.md) | 把机器人指令拆成可组合短语，用视频扩散模型做机器人想象 |
| 4 | [Dreamitate](04_Dreamitate_2024.pdf) | [04 中文精读](04_Dreamitate_2024_中文精读.md) | 先生成“人用工具完成任务”的视频，再追踪工具轨迹让机器人模仿 |
| 5 | [World Action Models are Zero-shot Policies](05_World_Action_Models_are_Zero-Shot_Policies_2026.pdf) | [05 中文精读](05_World_Action_Models_are_Zero-Shot_Policies_2026_中文精读.md) | DreamZero / WAM，联合生成未来视频和动作，做到实时闭环控制 |

## 一句话脉络

1. **World Models**：先学一个压缩世界和预测未来的模型，控制器只需要读 latent state。
2. **DreamerV3**：把这个思路变成通用 model-based RL 算法，在世界模型里 imagination，然后训练 actor-critic。
3. **RoboDreamer**：把视频生成器当机器人世界模型，用语言组合性提升新任务生成能力。
4. **Dreamitate**：把生成视频变成可执行策略，用工具轨迹绕开人手和机器人之间的 embodiment gap。
5. **DreamZero / WAM**：不再只生成视频计划，而是同时生成视频和连续动作，把 world model 直接变成 zero-shot policy。

## 建议阅读法

- 如果你在补 RL 基础：先读 01 和 02。
- 如果你在看机器人视频生成：读 03 和 04。
- 如果你关心 VLA 下一步怎么和 world model 合流：重点读 05。
- 公式最重要的是三类：`VAE/RSSM latent state`、`diffusion/flow matching`、`video-action joint prediction`。
