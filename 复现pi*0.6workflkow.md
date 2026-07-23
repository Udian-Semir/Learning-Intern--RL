## Workflow
1.先验证pi 0.5 + RECAP在仿真中优于 BC/DAgger/reward-weighted BC
2.当前代码是pi 0.5 Flow matching + RECAP基础 
3.对比论文，缺少Gemma 34B, 860M Action Expert, FAST/KI 联合目标loss,
视觉语言 value function与自动rollout编排

接下来的工作流程：
1. LIBERO 

<!-- 1：π0.5 BC -->
2：π0.5 + RECAP
3：加入 BC + rollout、DAgger、reward-weighted BC 对照
4：加入 AWR / PPO，对齐 π*0.6 论文比较
<!-- 5：π0.5 + Residual SAC，作为独立扩展 -->