# Robot RL Study Pack

This folder collects three robot RL references:

1. Humanoid / upright robot
2. Quadruped / robot dog
3. Wheeled-legged robot

## 1. Humanoid

- Paper: [Humanoid-Gym: Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer](./humanoid/humanoid_rl_paper.pdf)
- Paper link: https://arxiv.org/abs/2404.05695
- Code: [humanoid-gym](./humanoid/humanoid-gym)
- Project page: https://sites.google.com/view/humanoid-gym/

Start here:
- `humanoid/humanoid-gym/README.md`
- `humanoid/humanoid-gym/humanoid/envs/base/legged_robot.py`
- `humanoid/humanoid-gym/humanoid/envs/custom/humanoid_env.py`
- `humanoid/humanoid-gym/humanoid/algo/ppo/ppo.py`

What to learn:
- PPO loop
- reward design for balance / locomotion
- observation structure for humanoid control

## 2. Quadruped

- Paper: [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](./quadruped/quadruped_rl_paper.pdf)
- Paper link: https://arxiv.org/abs/2109.11978
- Code: [legged_gym](./quadruped/legged_gym)
- Project page: https://leggedrobotics.github.io/legged_gym/

Start here:
- `quadruped/legged_gym/README.md`
- `quadruped/legged_gym/legged_gym/envs/base/legged_robot.py`
- `quadruped/legged_gym/legged_gym/envs/anymal_c/anymal.py`
- `quadruped/legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_config.py`

What to learn:
- how reward scales are built
- terrain randomization
- PPO training for locomotion

## 3. Wheeled-legged robot

- Paper: [Arm-Constrained Curriculum Learning for Loco-Manipulation of a Wheel-Legged Robot](./wheeled_legged/wheeled_legged_rl_paper.pdf)
- Paper link: https://arxiv.org/abs/2403.16535
- Code: [legged-robots-manipulation](./wheeled_legged/legged-robots-manipulation)
- Project page: https://acodedog.github.io/wheel-legged-loco-manipulation/

Start here:
- `wheeled_legged/legged-robots-manipulation/README.md`
- `wheeled_legged/legged-robots-manipulation/loco_manipulation_gym/envs/`
- `wheeled_legged/legged-robots-manipulation/loco_manipulation_gym/envs/go2w/go2w_config.py`
- `wheeled_legged/legged-robots-manipulation/loco_manipulation_gym/envs/go2_arx/go2_arx_config.py`
- `wheeled_legged/legged-robots-manipulation/loco_manipulation_gym/envs/airbot/airbot_config.py`

What to learn:
- how locomotion and manipulation are mixed
- curriculum learning for harder tasks
- how robot-specific configs change action/reward design

## Reading order

If your goal is to understand RL basics first:

1. Quadruped
2. Humanoid
3. Wheeled-legged

Reason:
- quadruped code is the cleanest entry point for locomotion RL
- humanoid adds a harder body and richer observation/reward setup
- wheeled-legged adds loco-manipulation and curriculum design

## Notes

- All three repos use Isaac Gym style training.
- They are good for learning:
  - PPO
  - reward shaping
  - observation/action design
  - sim2real intuition

