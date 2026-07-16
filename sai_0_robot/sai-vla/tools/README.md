# tools/ — 辅助分析脚本

本目录收纳 Sai0-VLA 项目中**不参与主训练/推理流程**、但在调试、可视化、评估时频繁使用的独立脚本。

## 命名 / 组织规范

1. **所有文件、目录名一律 `snake_case`**：仅允许 `[a-z0-9_]`，禁止 `-`、空格、中文、特殊符号。
2. **按功能分子目录**，子目录与其下脚本前缀一致（`calc_discrete/` ↔ `calc_discrete_*.py`）。
3. **所有脚本默认产物目录** = `tools/outputs/<脚本名>/`，由脚本内部的 `SCRIPT_DIR.parent / "outputs" / SCRIPT_NAME` 计算得到；用户也可通过 `--output_dir` 显式覆盖。
4. **同一脚本不同实验产物**用扁平子目录区分，参数化命名（如 `degree05_chunk20`），数字补 0 到 2 位以便排序。

## 目录布局

```
tools/
├── README.md                        ← 本文件
│
├── calc_discrete/                   离散化基线 / 误差 / 跟踪误差分析
│   ├── calc_discrete_ce_baseline.py
│   ├── calc_discrete_error.py
│   ├── calc_discrete_error_batch.py
│   ├── calc_discrete_tracking.py
│   └── calc_discrete_tracking_batch.py
│
├── check_data_format/               数据集格式校验（LeRobot parquet / hdf5 / npy / safetensors）
│   ├── check_libero_hdf5_column_action_info.py
│   ├── check_npy_shape.py
│   ├── check_parquet_column_action.py
│   ├── check_parquet_column_action_specified_delta.py
│   └── check_safetensors.py
│
├── plot_action/                     动作 / 状态曲线可视化
│   ├── plot_3d_trajectory.py
│   ├── plot_action_columns.py
│   ├── plot_action_polyfit.py
│   ├── plot_state_action_compare.py
│   └── plot_state_columns.py
│
├── evaluate_checkpoint/             单个 checkpoint 的离线评估
│   ├── evaluate_oft_action_trajectory.py
│   └── plot_cumulative_error_per_dim.py
│
├── openloop_eval/                   整段开环预测评估
│   ├── openloop_eval.py
│   └── run.sh                       一键运行入口（含默认超参）
│
├── model_structures/                Eagle / Qwen 等模型权重结构 dump
│   ├── model_structure_869830fc_20251229.txt
│   └── model_structure_d0814e7e_20251229.txt
│
├── test_oft_cumulative_error.py     OFT 端到端累积误差测试（独立 E2E）
│
└── outputs/                         所有脚本的默认产物目录
    ├── plot_action_columns/
    ├── plot_action_polyfit/
    │   └── degree<DD>_chunk<CC>[_<variant>]/
    ├── plot_state_columns/
    ├── plot_state_action_compare/
    └── openloop_eval/
        ├── dataset196_filter_step300000_seed42/
        ├── no_pretrain_step<N>/
        └── then_tele_reset_lr_step<N>/
```

## 常用入口

| 任务 | 命令 |
|---|---|
| 校验 LeRobot 数据集 parquet | `python tools/check_data_format/check_parquet_column_action.py --parquet_path <file>` |
| 看 action 各列曲线 | `python tools/plot_action/plot_action_columns.py --parquet_path <file>` |
| 看 polyfit 拟合效果 | `python tools/plot_action/plot_action_polyfit.py --parquet_path <file> --degree 5 --chunk_size 20` |
| 看 state vs action 对比 | `python tools/plot_action/plot_state_action_compare.py --parquet_path <file>` |
| 单 checkpoint 离线 L1 误差 | `python tools/evaluate_checkpoint/evaluate_oft_action_trajectory.py --checkpoint <ckpt> --dataset <ds>` |
| 整段开环评估（5 episode） | `bash tools/openloop_eval/run.sh` |
| 离散化 baseline CE | `python tools/calc_discrete/calc_discrete_ce_baseline.py --data_dir <dir>` |

> 详细参数请直接 `python <script> --help`。
