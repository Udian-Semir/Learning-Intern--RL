import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any

# ============================================================================
# [MODIFIED] 使用本地定义替代 gr00t 包导入
# 原始代码:
#     from gr00t.configs.base_config import Config
#     from gr00t.data.collator import BasicDataCollator
#     from gr00t.data.dataset.factory import DatasetFactory
#     from gr00t.data.interfaces import BaseProcessor
# 
# 修改原因: 避免依赖外部 gr00t 包，使用本地定义
# 修改时间: 2025-01
# ============================================================================

# 本地定义 Config 占位符
@dataclass
class Config:
    """配置基类占位符"""
    model: Any = None
    data: Any = None
    training: Any = None

# 本地定义 BasicDataCollator 占位符
class BasicDataCollator:
    """基础数据整理器占位符"""
    def __call__(self, features):
        return features

# 本地定义 DatasetFactory 占位符
class DatasetFactory:
    """数据集工厂占位符"""
    def __init__(self, config):
        self.config = config
        
    def build(self, processor):
        return None, None

# 本地定义 BaseProcessor 占位符
class BaseProcessor:
    """基础处理器占位符"""
    pass

# ============================================================================

import numpy as np
import torch
from transformers import PreTrainedModel


class ModelPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None
        self.data_collator = None

    def setup(self):
        pass

    def return_model(self):
        return self.model

    def return_dataset(self):
        return self.train_dataset, self.eval_dataset

    def return_collator(self):
        return self.data_collator

    def return_processor(self):
        return self.processor


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


class BasicPipeline(ModelPipeline):
    """A simple pipeline that works for diffusion and flowmatching-based models."""

    model_class: type[PreTrainedModel]
    processor_class: type[BaseProcessor]
    data_collator_class: type[BasicDataCollator] = BasicDataCollator

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        # Load model
        model = self.model_class(self.config.model)
        print("Model Config: ", model.config)

        # unfreeze the model first
        for name, param in model.named_parameters():
            param.requires_grad = True

        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        return model

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        self.processor = self.processor_class(
            modality_configs=self.config.data.modality_configs,
            statistics=None,  # This will be computed and set later.
            **self.config.model.processor_kwargs,
        )
        dataset_factory = DatasetFactory(self.config)
        train_dataset, eval_dataset = dataset_factory.build(self.processor)

        # Save dataset statistics for inference
        stats = train_dataset.get_dataset_statistics()
        stats_dict = convert_tensors_to_lists(stats)
        # Save statistics
        with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
            json.dump(stats_dict, f, indent=2)
        logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.data_collator_class()
        return data_collator
