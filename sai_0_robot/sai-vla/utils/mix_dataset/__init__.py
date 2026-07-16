"""
utils.mix_dataset
=================

把多个 LeRobot v2.0 数据集 **物理合并** 成一个 LeRobot 目录，
用于 OFT action head 的多数据集 pretrain
(train_multigpu.py 仅支持单个 --data_path，所以先做物理合并再喂给训练)。

核心脚本：merge_lerobot_datasets.py
"""
