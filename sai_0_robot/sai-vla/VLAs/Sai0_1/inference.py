"""
Sai0_1 推理模块

提供便捷的推理接口，支持：
- 单帧推理
- 数据集批量推理 (评估)
- 实时推理 (部署)

使用示例:
    from VLAs.Sai0_1 import Sai0Inference, Sai0Config
    
    # 创建推理器
    inference = Sai0Inference.from_checkpoint(
        checkpoint_path="./checkpoints/best/action_head.pt",
        vlm_type="qwen3_vl",
        vlm_model_path="Qwen/Qwen3-VL-2B-Instruct",
    )
    
    # 单帧推理
    actions = inference.predict(images, instruction, state)
    
    # 数据集评估
    results = inference.evaluate_dataset(dataset_path)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import json
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# 添加路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# Sai0 推理器
# ============================================================================

class Sai0Inference:
    """
    Sai0 推理器
    
    提供便捷的推理接口，封装 VLM 和 Action Head 的调用。
    """
    
    def __init__(
        self,
        model: "Sai0Model" = None,
        config: "Sai0Config" = None,
        normalizers: Dict = None,
        device: str = "cuda:0",
    ):
        """
        初始化推理器
        
        Args:
            model: Sai0Model 实例
            config: Sai0Config 配置
            normalizers: 归一化器字典
            device: 设备
        """
        self.model = model
        self.config = config
        self.normalizers = normalizers
        self.device = device
        
        if model is not None:
            self.model.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        vlm_type: str = "qwen3_vl",
        vlm_model_path: str = None,
        dataset_path: str = None,
        device: str = "cuda:0",
        **kwargs
    ) -> "Sai0Inference":
        """
        从 checkpoint 创建推理器
        
        Args:
            checkpoint_path: Action Head checkpoint 路径
            vlm_type: VLM 类型
            vlm_model_path: VLM 模型路径
            dataset_path: 数据集路径 (用于加载归一化统计)
            device: 设备
            **kwargs: 其他配置参数
        
        Returns:
            Sai0Inference 实例
        """
        from .config import Sai0Config, VLMConfig, ActionHeadConfig
        from .sai0_model import Sai0Model
        from .data_utils import load_normalization_stats
        
        # 创建配置
        vlm_config = VLMConfig(
            model_type=vlm_type,
            model_path=vlm_model_path,
            device=device,
            **{k: v for k, v in kwargs.items() if hasattr(VLMConfig, k)}
        )
        
        ah_config = ActionHeadConfig(
            pretrained_weights=checkpoint_path,
            **{k: v for k, v in kwargs.items() if hasattr(ActionHeadConfig, k)}
        )
        
        config = Sai0Config(vlm=vlm_config, action_head=ah_config)
        
        # 创建模型
        model = Sai0Model(config=config, device=device)
        model.eval()
        
        # 加载归一化统计
        normalizers = None
        if dataset_path:
            normalizers = load_normalization_stats(
                dataset_path,
                convert_quat_to_axisangle=ah_config.convert_quat_to_axisangle
            )
        
        return cls(model=model, config=config, normalizers=normalizers, device=device)
    
    @classmethod
    def from_config(cls, config: "Sai0Config", device: str = "cuda:0") -> "Sai0Inference":
        """从配置创建推理器"""
        from .sai0_model import Sai0Model
        from .data_utils import load_normalization_stats
        
        model = Sai0Model(config=config, device=device)
        model.eval()
        
        normalizers = None
        if config.data.dataset_path:
            normalizers = load_normalization_stats(
                config.data.dataset_path,
                convert_quat_to_axisangle=config.action_head.convert_quat_to_axisangle
            )
        
        return cls(model=model, config=config, normalizers=normalizers, device=device)
    
    def predict(
        self,
        images: List[Union[Image.Image, np.ndarray, str]],
        instruction: str,
        state: Union[np.ndarray, torch.Tensor],
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        单帧推理
        
        Args:
            images: 图像列表 (PIL.Image, numpy array, 或文件路径)
            instruction: 任务指令
            state: 当前状态 (state_dim,) 或 (1, state_dim)
            denormalize: 是否反归一化输出
        
        Returns:
            预测的动作 (num_chunks, action_dim)
        """
        # 处理图像
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            processed_images.append(img)
        
        # 处理 state
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        # 归一化 state
        if self.normalizers and 'state' in self.normalizers:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = self.normalizers['state'].normalize(state)
            state = state.squeeze(0)
        
        # 预测
        with torch.no_grad():
            actions = self.model.predict(
                images=processed_images,
                instruction=instruction,
                state=state,
            )
        
        # 反归一化
        actions = actions.cpu()
        if denormalize and self.normalizers and 'action' in self.normalizers:
            actions = self.normalizers['action'].denormalize(actions)
        
        return actions.numpy()
    
    def predict_batch(
        self,
        images_batch: List[List],
        instructions: List[str],
        states: Union[np.ndarray, torch.Tensor],
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        批量推理
        
        Args:
            images_batch: 图像批次列表，每个元素是一个图像列表
            instructions: 指令列表
            states: 状态批次 (batch, state_dim)
            denormalize: 是否反归一化
        
        Returns:
            预测的动作 (batch, num_chunks, action_dim)
        """
        results = []
        for images, instruction, state in zip(images_batch, instructions, states):
            actions = self.predict(images, instruction, state, denormalize=denormalize)
            results.append(actions)
        
        return np.stack(results, axis=0)
    
    def evaluate_dataset(
        self,
        dataset_path: str,
        num_samples: int = None,
        save_results: bool = True,
        output_path: str = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        在数据集上评估模型
        
        Args:
            dataset_path: 数据集路径
            num_samples: 评估样本数 (None=全部)
            save_results: 是否保存结果
            output_path: 结果保存路径
            verbose: 详细输出
        
        Returns:
            评估结果字典
        """
        from .data_utils import create_sai0_dataloader, get_dataset_info
        
        # 获取数据集信息
        info = get_dataset_info(dataset_path)
        if verbose:
            print(f"Evaluating on dataset: {dataset_path}")
            print(f"Total frames: {info['total_frames']}")
        
        # 创建 dataloader
        dataloader, normalizers = create_sai0_dataloader(
            dataset_path=dataset_path,
            batch_size=1,  # 评估时使用单样本
            shuffle=False,
            num_workers=0,
            verbose=False,
        )
        
        if normalizers:
            self.normalizers = normalizers
        
        # 评估
        all_losses = []
        all_errors = []
        
        num_batches = len(dataloader) if num_samples is None else min(num_samples, len(dataloader))
        
        with torch.no_grad():
            for i, (backbone_output, action_head_inputs) in enumerate(tqdm(dataloader, total=num_batches, disable=not verbose)):
                if num_samples is not None and i >= num_samples:
                    break
                
                # 移动到设备
                for k in backbone_output:
                    if isinstance(backbone_output[k], torch.Tensor):
                        backbone_output[k] = backbone_output[k].to(self.device)
                for k in action_head_inputs:
                    if isinstance(action_head_inputs[k], torch.Tensor):
                        action_head_inputs[k] = action_head_inputs[k].to(self.device)
                
                # 计算损失
                output = self.model(backbone_output, action_head_inputs)
                loss = output["loss"].item()
                all_losses.append(loss)
                
                # 计算动作误差
                gt_actions = action_head_inputs["action"]
                pred_actions = output.get("predicted_actions", None)
                if pred_actions is not None:
                    error = torch.mean(torch.abs(pred_actions - gt_actions)).item()
                    all_errors.append(error)
        
        # 汇总结果
        results = {
            "dataset_path": dataset_path,
            "num_samples": len(all_losses),
            "mean_loss": np.mean(all_losses),
            "std_loss": np.std(all_losses),
            "min_loss": np.min(all_losses),
            "max_loss": np.max(all_losses),
        }
        
        if all_errors:
            results.update({
                "mean_action_error": np.mean(all_errors),
                "std_action_error": np.std(all_errors),
            })
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Mean Loss: {results['mean_loss']:.6f} ± {results['std_loss']:.6f}")
            if 'mean_action_error' in results:
                print(f"  Mean Action Error: {results['mean_action_error']:.6f}")
        
        # 保存结果
        if save_results:
            if output_path is None:
                output_path = Path(dataset_path) / "eval_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            if verbose:
                print(f"Results saved to: {output_path}")
        
        return results


# ============================================================================
# 实时推理器 (用于部署)
# ============================================================================

class RealtimeInference:
    """
    实时推理器
    
    针对部署场景优化，支持：
    - 状态缓存
    - 预热
    - 性能监控
    """
    
    def __init__(
        self,
        inference: Sai0Inference,
        warmup_steps: int = 3,
    ):
        """
        初始化实时推理器
        
        Args:
            inference: Sai0Inference 实例
            warmup_steps: 预热步数
        """
        self.inference = inference
        self.warmup_steps = warmup_steps
        
        self._warmed_up = False
        self._latency_history = []
    
    def warmup(self, dummy_images: List = None, dummy_instruction: str = "pick up object"):
        """
        预热模型
        
        Args:
            dummy_images: 虚拟图像 (用于预热)
            dummy_instruction: 虚拟指令
        """
        if dummy_images is None:
            # 创建虚拟图像
            dummy_images = [
                Image.new("RGB", (256, 256), color=(128, 128, 128))
                for _ in range(2)
            ]
        
        dummy_state = np.zeros(8, dtype=np.float32)
        
        print(f"Warming up model ({self.warmup_steps} steps)...")
        for _ in range(self.warmup_steps):
            self.inference.predict(
                images=dummy_images,
                instruction=dummy_instruction,
                state=dummy_state,
            )
        
        self._warmed_up = True
        print("✓ Model warmed up")
    
    def predict(
        self,
        images: List,
        instruction: str,
        state: np.ndarray,
        track_latency: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """
        实时预测
        
        Args:
            images: 图像列表
            instruction: 指令
            state: 状态
            track_latency: 是否跟踪延迟
        
        Returns:
            (actions, latency_ms) 元组
        """
        if not self._warmed_up:
            self.warmup()
        
        start_time = time.time()
        
        actions = self.inference.predict(
            images=images,
            instruction=instruction,
            state=state,
        )
        
        latency = (time.time() - start_time) * 1000  # ms
        
        if track_latency:
            self._latency_history.append(latency)
        
        return actions, latency
    
    def get_latency_stats(self) -> Dict[str, float]:
        """获取延迟统计"""
        if not self._latency_history:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}
        
        return {
            "mean": np.mean(self._latency_history),
            "std": np.std(self._latency_history),
            "min": np.min(self._latency_history),
            "max": np.max(self._latency_history),
            "count": len(self._latency_history),
        }
    
    def reset_latency_history(self):
        """重置延迟历史"""
        self._latency_history = []


# ============================================================================
# 便捷函数
# ============================================================================

def quick_inference(
    images: List,
    instruction: str,
    state: np.ndarray,
    checkpoint_path: str,
    vlm_type: str = "qwen3_vl",
    vlm_model_path: str = None,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    快速推理 (一次性使用)
    
    Args:
        images: 图像列表
        instruction: 指令
        state: 状态
        checkpoint_path: Action Head checkpoint 路径
        vlm_type: VLM 类型
        vlm_model_path: VLM 模型路径
        device: 设备
    
    Returns:
        预测的动作
    """
    inference = Sai0Inference.from_checkpoint(
        checkpoint_path=checkpoint_path,
        vlm_type=vlm_type,
        vlm_model_path=vlm_model_path,
        device=device,
    )
    
    return inference.predict(images, instruction, state)


def evaluate_checkpoint(
    checkpoint_path: str,
    dataset_path: str,
    vlm_type: str = "qwen3_vl",
    vlm_model_path: str = None,
    num_samples: int = None,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """
    评估 checkpoint
    
    Args:
        checkpoint_path: Action Head checkpoint 路径
        dataset_path: 数据集路径
        vlm_type: VLM 类型
        vlm_model_path: VLM 模型路径
        num_samples: 评估样本数
        device: 设备
    
    Returns:
        评估结果
    """
    inference = Sai0Inference.from_checkpoint(
        checkpoint_path=checkpoint_path,
        vlm_type=vlm_type,
        vlm_model_path=vlm_model_path,
        dataset_path=dataset_path,
        device=device,
    )
    
    return inference.evaluate_dataset(
        dataset_path=dataset_path,
        num_samples=num_samples,
    )

