# VLA 与流匹配策略强化学习论文

以下 PDF 均从对应 arXiv 论文页下载，并已验证可读取。

| 文件 | 正式论文名 | 来源 |
| --- | --- | --- |
| `01_ReinFlow_Online_RL_for_Flow_Matching_Policy_2025.pdf` | *ReinFlow: Fine-tuning Flow Matching Policy with Online Reinforcement Learning* | https://arxiv.org/abs/2505.22094 |
| `02_DPPO_Diffusion_Policy_Policy_Optimization_2024.pdf` | *Diffusion Policy Policy Optimization* | https://arxiv.org/abs/2409.00588 |
| `03_QGF_Test-Time_Gradient_Guidance_of_Flow_Policies_2026.pdf` | *Test-Time Gradient Guidance of Flow Policies in Reinforcement Learning* | https://arxiv.org/abs/2606.11087 |
| `04_STARE-VLA_Progressive_Stage-Aware_Reinforcement_2025.pdf` | *STARE-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models* | https://arxiv.org/abs/2512.05107 |

## 仿真采集 -> VLA -> RL 完整链路

最接近完整闭环的推荐组合是 **RoboTwin 2.0 + RLinf + OpenPI π0/π0.5**：RoboTwin 负责仿真任务、域随机化和轨迹采集，RLinf 负责 LeRobot/对应 dataconfig 的 SFT、PPO/GRPO 在线 RL、固定 seed 评测。第二条完整组合是 **RoboCasa365 + RoboCasa 数据转换 + RLinf + OpenPI/GR00T**。Isaac Lab Mimic 和 RL-VLA³ 也已整理，但前者需要自己接 VLA/RL adapter，后者是 RL 后端而不是采集器。

详细的项目筛选、命令、数据接口和限制见：

[sim_vla_rl_end_to_end.md](./sim_vla_rl_end_to_end.md)

本轮新增并核验的论文 PDF：

| 文件 | 正式论文 | 来源 |
| --- | --- | --- |
| `05_RLinf_Flexible_Efficient_LargeScale_RL_2025.pdf` | *RLinf: Flexible and Efficient Large-scale Reinforcement Learning via Macro-to-Micro Flow Transformation* | https://arxiv.org/abs/2509.15965 |
| `06_RLinf_VLA_Unified_Efficient_VLA_RL_2025.pdf` | *RLinf-VLA: A Unified and Efficient Framework for Reinforcement Learning of Vision-Language-Action Models* | https://arxiv.org/abs/2510.06710 |
| `07_piRL_Online_RL_Flow_VLA_2025.pdf` | *πRL: Online RL Fine-tuning for Flow-based Vision-Language-Action Models* | https://arxiv.org/abs/2510.25889 |
| `08_SAC_Flow_2025.pdf` | *SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling* | https://arxiv.org/abs/2509.25756 |
| `09_DSRL_Latent_Space_RL_2025.pdf` | *Steering Your Diffusion Policy with Latent Space Reinforcement Learning* | https://arxiv.org/abs/2506.15799 |
| `10_RoboTwin_2_Scalable_Data_Generator_2025.pdf` | *RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation* | https://arxiv.org/abs/2506.18088 |

## 中文精读

| 论文 | 中文精读 |
| --- | --- |
| ReinFlow | [01_ReinFlow_论文中文精读.md](./01_ReinFlow_论文中文精读.md) |
| DPPO | [02_DPPO_论文中文精读.md](./02_DPPO_论文中文精读.md) |
| QGF | [03_QGF_论文中文精读.md](./03_QGF_论文中文精读.md) |
| StARe-VLA | [04_STARE-VLA_论文中文精读.md](./04_STARE-VLA_论文中文精读.md) |

## 仍需注意的名称

`RL-VLA³` 的官方代码仓库是 https://github.com/Haoran0301/RL-VLA3，但仓库当前只给出 `arXiv--2602` 的 BibTeX 占位，没有可唯一核验的 arXiv 编号，因此没有伪造或下载一个不确定的 PDF。`SA-VLA`、`FlowPRO` 仍可能对应多个同名项目，需要作者或论文链接后再补入。
