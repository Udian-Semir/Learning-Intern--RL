#!/usr/bin/env python3
"""
Evaluate GR00T N1.5 model on LIBERO benchmarks.

This script provides a standalone evaluation pipeline for fine-tuned N1.5 models
on LIBERO simulation tasks. It supports:
- Loading fine-tuned checkpoints from local path
- Multiple LIBERO benchmark suites (libero_spatial, libero_object, libero_goal, etc.)
- Batch evaluation with configurable rollouts
- Video recording of rollouts
- JSON result logging
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import imageio
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
LIBERO_ROOT = EVAL_DIR.parents[4]  # /home/.../LIBERO
GROOT_ROOT = LIBERO_ROOT / "Isaac-GR00T"

if str(LIBERO_ROOT) not in sys.path:
    sys.path.insert(0, str(LIBERO_ROOT))
if str(GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(GROOT_ROOT))

# Fix for PyTorch 2.6+ security change
try:
    torch.serialization.add_safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        np.dtypes.Float64DType,
    ])
except (AttributeError, TypeError):
    pass

_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

# LIBERO imports
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

# GR00T imports
from gr00t.model.policy import Gr00tPolicy
from gr00t.data.transform.base import ComposedModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import VideoToTensor, VideoCrop, VideoResize, VideoToNumpy
from gr00t.model.transforms import GR00TTransform
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.data.dataset import ModalityConfig


# ===========================================================================
# Libero Data Config (matching train preprocessing)
# ===========================================================================

class LiberoDataConfig(BaseDataConfig):
    """Data configuration for LIBERO environment compatible with N1.5."""
    
    video_keys = [
        "video.image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self, action_norm: str = "min_max") -> ComposedModalityTransform:
        if action_norm == "min_max":
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={key: "min_max" for key in self.action_keys},
            )
        else:
            action_transform = StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "mean_std",
                    "action.y": "mean_std",
                    "action.z": "mean_std",
                    "action.roll": "mean_std",
                    "action.pitch": "mean_std",
                    "action.yaw": "mean_std",
                    "action.gripper": "min_max",
                },
            )
        transforms = [
            # video transforms (eval mode: no jitter)
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            action_transform,
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


# ===========================================================================
# Utility Functions
# ===========================================================================

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle format.
    Copied from robosuite for consistency with official N1.5 eval.
    
    Args:
        quat: (x, y, z, w) quaternion
    
    Returns:
        (ax, ay, az) axis-angle exponential coordinates
    """
    quat = quat.copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Changes gripper action (last dimension) from [0,1] to [+1,-1].
    
    Normalization formula: y = 1 - 2 * (x - orig_low) / (orig_high - orig_low)
    """
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 1 - 2 * (action[..., -1] - orig_low) / (orig_high - orig_low)

    if binarize:
        action[..., -1] = np.sign(action[..., -1])

    return action


def get_libero_image(obs: Dict[str, np.ndarray], rotate: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract images from observation and preprocess.
    
    Args:
        obs: LIBERO observation dictionary
        rotate: If True, rotate 180 degrees to match train preprocessing.
                If False, return original images (for video display).
    
    Returns:
        img: agentview image
        wrist_img: wrist camera image
    """
    img = obs["agentview_image"]
    wrist_img = obs["robot0_eye_in_hand_image"]
    
    if rotate:
        # IMPORTANT: Rotate 180 degrees to match train preprocessing
        img = img[::-1, ::-1]
        wrist_img = wrist_img[::-1, ::-1]
    
    return img, wrist_img


def get_libero_env(task, resolution: int = 256):
    """Initialize LIBERO environment."""
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_dummy_action() -> List[float]:
    """Get dummy/no-op action for settling."""
    return [0, 0, 0, 0, 0, 0, -1]


def save_rollout_video(
    top_view: List[np.ndarray],
    wrist_view: List[np.ndarray],
    video_path: Path,
    fps: int = 30,
) -> None:
    """Save rollout as side-by-side video."""
    if not top_view:
        print(f"[WARN] No frames captured for {video_path.name}")
        return
    
    video_path.parent.mkdir(parents=True, exist_ok=True)
    
    with imageio.get_writer(str(video_path), fps=fps) as writer:
        for img1, img2 in zip(top_view, wrist_view):
            combined = np.hstack((img1, img2))
            writer.append_data(combined)
    
    print(f"Saved video: {video_path}")


# ===========================================================================
# GR00T Policy Wrapper
# ===========================================================================

class GR00TLiberoPolicy:
    """GR00T N1.5 Policy wrapper for LIBERO environments."""
    
    LIBERO_CONFIG = {
        "proprio_size": 8,
        "state_key_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "roll": 3,
            "pitch": 4,
            "yaw": 5,
            "gripper": (6, 8),
        },
    }

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "new_embodiment",
        device: str = "cuda",
        denoising_steps: Optional[int] = None,
        action_norm: str = "min_max",
        base_model_path: str = "nvidia/GR00T-N1.5-3B",
    ):
        """
        Initialize GR00T policy for LIBERO evaluation.
        
        Args:
            model_path: Path to fine-tuned checkpoint directory (action_head.pt)
            embodiment_tag: Embodiment tag used during training
            device: Device to run inference on
            denoising_steps: Override number of diffusion denoising steps
            action_norm: Action normalization mode ("min_max" or "mean_std")
            base_model_path: Path to base N1.5 model (HuggingFace or local)
        """
        self.device = device
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.action_keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
        
        # Create data config and transforms
        data_config = LiberoDataConfig()
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform(action_norm=action_norm)
        
        # Check if model_path contains a full model or just action_head.pt
        model_path_obj = Path(model_path)
        action_head_path = model_path_obj / "action_head.pt"
        has_action_head_only = action_head_path.exists() and not (model_path_obj / "model.safetensors.index.json").exists()
        
        if has_action_head_only:
            # Load base model first, then load fine-tuned action head
            print(f"Detected action_head.pt checkpoint format")
            print(f"Loading base GR00T N1.5 model from: {base_model_path}")
            
            # Check if checkpoint has its own metadata
            checkpoint_metadata_path = model_path_obj / "experiment_cfg" / "metadata.json"
            if checkpoint_metadata_path.exists():
                print(f"Using metadata from checkpoint: {checkpoint_metadata_path}")
                # We need to load base model structure, but use checkpoint's metadata
                # First, create a temporary policy loader
                from gr00t.model.gr00t_n1 import GR00T_N1_5
                from gr00t.model.policy import COMPUTE_DTYPE
                from gr00t.data.schema import DatasetMetadata
                import json
                
                # Load base model structure
                model = GR00T_N1_5.from_pretrained(base_model_path, torch_dtype=COMPUTE_DTYPE)
                model.eval()
                model.to(device=device)
                
                # Load checkpoint metadata
                with open(checkpoint_metadata_path, "r") as f:
                    metadatas = json.load(f)
                metadata_dict = metadatas.get(embodiment_tag)
                if metadata_dict is None:
                    raise ValueError(f"No metadata found for embodiment tag: {embodiment_tag}")
                metadata = DatasetMetadata.model_validate(metadata_dict)
                modality_transform.set_metadata(metadata)
                modality_transform.eval()
                
                # Create a simple policy wrapper
                class SimpleGr00tPolicy:
                    def __init__(self, model, modality_config, modality_transform):
                        self.model = model
                        self._modality_config = modality_config
                        self._modality_transform = modality_transform
                    
                    def get_action(self, observations):
                        import numpy as np
                        obs_copy = observations.copy()
                        # Check if batched
                        is_batch = True
                        for k, v in obs_copy.items():
                            if "state" in k and len(v.shape) < 3:
                                is_batch = False
                                break
                        if not is_batch:
                            for k, v in obs_copy.items():
                                if isinstance(v, np.ndarray):
                                    obs_copy[k] = np.expand_dims(v, axis=0)
                                elif isinstance(v, list):
                                    obs_copy[k] = np.expand_dims(np.array(v), axis=0)
                        
                        for k, v in obs_copy.items():
                            if not isinstance(v, np.ndarray):
                                obs_copy[k] = np.array(v)
                        
                        normalized_input = self._modality_transform(obs_copy)
                        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
                            model_pred = self.model.get_action(normalized_input)
                        normalized_action = model_pred["action_pred"].float()
                        unnormalized_action = self._modality_transform.unapply({"action": normalized_action.cpu()})
                        
                        if not is_batch:
                            for k, v in unnormalized_action.items():
                                if isinstance(v, np.ndarray):
                                    unnormalized_action[k] = np.squeeze(v, axis=0)
                                elif isinstance(v, torch.Tensor):
                                    unnormalized_action[k] = v.squeeze(0)
                        return unnormalized_action
                
                self.policy = SimpleGr00tPolicy(model, modality_config, modality_transform)
            else:
                print(f"No checkpoint metadata found, using base model metadata")
                self.policy = Gr00tPolicy(
                    model_path=base_model_path,
                    embodiment_tag=embodiment_tag,
                    modality_config=modality_config,
                    modality_transform=modality_transform,
                    denoising_steps=denoising_steps,
                    device=device,
                )
            
            # Load fine-tuned action head weights
            print(f"Loading fine-tuned action head from: {action_head_path}")
            action_head_state_dict = torch.load(action_head_path, map_location=device)
            self.policy.model.action_head.load_state_dict(action_head_state_dict, strict=False)
            print(f"Fine-tuned action head loaded successfully!")
        else:
            # Load full model directly
            print(f"Loading full GR00T N1.5 model from: {model_path}")
            self.policy = Gr00tPolicy(
                model_path=model_path,
                embodiment_tag=embodiment_tag,
                modality_config=modality_config,
                modality_transform=modality_transform,
                denoising_steps=denoising_steps,
                device=device,
            )
        
        print(f"Model loaded successfully. Action horizon: {self.policy.model.action_horizon}")

    def get_action(self, obs: Dict[str, np.ndarray], lang: str) -> np.ndarray:
        """
        Get action from GR00T policy given LIBERO observation.
        
        Args:
            obs: LIBERO environment observation
            lang: Language instruction
            
        Returns:
            7-dim action array: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        obs_dict = self._process_observation(obs, lang)
        action_chunk = self.policy.get_action(obs_dict)
        return self._convert_to_libero_action(action_chunk, idx=0)

    def _process_observation(self, obs: Dict[str, np.ndarray], lang: str) -> Dict[str, np.ndarray]:
        """Convert LIBERO observation to GR00T format."""
        xyz = obs["robot0_eef_pos"]
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        gripper = obs["robot0_gripper_qpos"]
        img, wrist_img = get_libero_image(obs)
        
        new_obs = {
            "video.image": np.expand_dims(img, axis=0),
            "video.wrist_image": np.expand_dims(wrist_img, axis=0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(gripper, axis=0),
            "annotation.human.action.task_description": [lang],
        }
        return new_obs

    def _convert_to_libero_action(self, action_chunk: Dict[str, np.ndarray], idx: int = 0) -> np.ndarray:
        """
        Convert GR00T action chunk to LIBERO format.
        
        Args:
            action_chunk: Action dictionary from GR00T policy
            idx: Index of action to extract from horizon (default: 0 for first action)
            
        Returns:
            7-dim action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        action_components = [
            np.atleast_1d(action_chunk[f"action.{key}"][idx])[0] 
            for key in self.action_keys
        ]
        action_array = np.array(action_components, dtype=np.float32)
        action_array = normalize_gripper_action(action_array, binarize=True)
        
        assert len(action_array) == 7, f"Expected 7-dim action, got {len(action_array)}"
        return action_array


# ===========================================================================
# Evaluation Functions
# ===========================================================================

@dataclass
class TimingStats:
    """Track timing statistics."""
    inference_times: List[float]
    
    def __init__(self):
        self.inference_times = []
    
    def add(self, inference_time: float):
        self.inference_times.append(inference_time)
    
    def summary(self) -> Dict[str, float]:
        if not self.inference_times:
            return {}
        return {
            "mean_inference_time": float(np.mean(self.inference_times)),
            "std_inference_time": float(np.std(self.inference_times)),
            "min_inference_time": float(np.min(self.inference_times)),
            "max_inference_time": float(np.max(self.inference_times)),
            "total_inferences": len(self.inference_times),
        }


def get_max_steps(task_suite_name: str) -> int:
    """Get max steps based on task suite."""
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 600,
        "libero_10": 1000,
        "libero_90": 400,
        "libero_100": 600,
    }
    return max_steps_map.get(task_suite_name, 600)


def run_single_rollout(
    policy: GR00TLiberoPolicy,
    env,
    task_description: str,
    initial_state: np.ndarray,
    max_steps: int,
    num_steps_wait: int = 10,
    save_video: bool = False,
) -> Tuple[bool, int, Optional[List[np.ndarray]], Optional[List[np.ndarray]], TimingStats]:
    """
    Run a single rollout episode.
    
    Returns:
        success: Whether task was completed
        steps: Number of steps taken
        top_view_frames: List of top camera frames (if save_video)
        wrist_view_frames: List of wrist camera frames (if save_video)
        timing_stats: Inference timing statistics
    """
    env.reset()
    obs = env.set_init_state(initial_state)
    
    timing_stats = TimingStats()
    top_view_frames = [] if save_video else None
    wrist_view_frames = [] if save_video else None
    
    done = False
    steps = 0
    
    for t in range(max_steps + num_steps_wait):
        # Wait for objects to settle
        if t < num_steps_wait:
            obs, _, _, _ = env.step(get_dummy_action())
            continue
        
        # Capture frames for video (use original images without rotation for natural view)
        if save_video:
            img, wrist_img = get_libero_image(obs, rotate=False)
            top_view_frames.append(img.copy())
            wrist_view_frames.append(wrist_img.copy())
        
        # Get action from policy
        start_time = time.time()
        action = policy.get_action(obs, task_description)
        inference_time = time.time() - start_time
        timing_stats.add(inference_time)
        
        # Execute action
        obs, reward, done, info = env.step(action.tolist())
        steps += 1
        
        if done:
            break
    
    return done, steps, top_view_frames, wrist_view_frames, timing_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GR00T N1.5 model on LIBERO benchmarks"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned N1.5 checkpoint directory (can be action_head.pt only or full model)",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="nvidia/GR00T-N1.5-3B",
        help="Path to base N1.5 model (HuggingFace hub or local path). Used when loading action_head.pt checkpoints.",
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        default="new_embodiment",
        help="Embodiment tag used during training",
    )
    parser.add_argument(
        "--denoising-steps",
        type=int,
        default=None,
        help="Override number of diffusion denoising steps",
    )
    parser.add_argument(
        "--action-norm",
        type=str,
        default="min_max",
        choices=["min_max", "mean_std"],
        help="Action normalization mode",
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero_10",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", "libero_100"],
        help="LIBERO benchmark suite to evaluate",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Specific task ID to evaluate (default: all tasks)",
    )
    parser.add_argument(
        "--task-order-index",
        type=int,
        default=0,
        help="Task order index for benchmark",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=50,
        help="Number of rollouts per task",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Max steps per rollout (default: auto based on benchmark)",
    )
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=10,
        help="Number of settling steps before evaluation",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Camera resolution",
    )
    
    # Output arguments
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./n1_5_eval_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save rollout videos",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./n1_5_eval_videos",
        help="Directory to save videos",
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directories
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.video_dir)
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    policy = GR00TLiberoPolicy(
        model_path=args.model_path,
        embodiment_tag=args.embodiment_tag,
        device=args.device,
        denoising_steps=args.denoising_steps,
        action_norm=args.action_norm,
        base_model_path=args.base_model_path,
    )
    
    # Get benchmark
    benchmark = get_benchmark(args.benchmark)(args.task_order_index)
    num_tasks = benchmark.n_tasks
    
    # Determine which tasks to evaluate
    if args.task_id is not None:
        task_ids = [args.task_id]
    else:
        task_ids = list(range(num_tasks))
    
    # Get max steps
    max_steps = args.max_steps or get_max_steps(args.benchmark)
    
    # Print configuration
    print("\n" + "=" * 60)
    print("GR00T N1.5 LIBERO Evaluation")
    print("=" * 60)
    print(f"Model Path: {args.model_path}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Tasks: {task_ids}")
    print(f"Rollouts per Task: {args.num_rollouts}")
    print(f"Max Steps: {max_steps}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Print all tasks
    print(f"\nAll Tasks in {args.benchmark}:")
    print("-" * 60)
    for i in range(num_tasks):
        task = benchmark.get_task(i)
        marker = " <<< SELECTED" if i in task_ids else ""
        print(f"  [{i}] {task.language}{marker}")
    print("-" * 60 + "\n")
    
    # Run evaluation
    all_results = {}
    total_successes = 0
    total_episodes = 0
    
    for task_id in task_ids:
        task = benchmark.get_task(task_id)
        initial_states = benchmark.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=args.resolution)
        
        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_description}")
        print(f"{'='*60}")
        
        task_successes = []
        task_steps = []
        task_timing = TimingStats()
        
        for rollout_idx in range(args.num_rollouts):
            state_idx = rollout_idx % len(initial_states)
            
            success, steps, top_frames, wrist_frames, timing = run_single_rollout(
                policy=policy,
                env=env,
                task_description=task_description,
                initial_state=initial_states[state_idx],
                max_steps=max_steps,
                num_steps_wait=args.num_steps_wait,
                save_video=args.save_video,
            )
            
            task_successes.append(int(success))
            task_steps.append(steps)
            task_timing.inference_times.extend(timing.inference_times)
            
            total_successes += int(success)
            total_episodes += 1
            
            status = "SUCCESS" if success else "FAIL"
            print(f"  Rollout {rollout_idx + 1}/{args.num_rollouts}: {status} ({steps} steps)")
            
            # Save video
            if args.save_video and top_frames and wrist_frames:
                status_str = "success" if success else "fail"
                video_name = f"task{task_id}_rollout{rollout_idx + 1:03d}_{status_str}.mp4"
                save_rollout_video(
                    top_frames, 
                    wrist_frames, 
                    video_dir / video_name,
                )
        
        env.close()
        
        # Task results
        task_success_rate = np.mean(task_successes)
        task_avg_steps = np.mean(task_steps)
        
        print(f"\nTask {task_id} Results:")
        print(f"  Success Rate: {task_success_rate:.3f} ({sum(task_successes)}/{args.num_rollouts})")
        print(f"  Avg Steps: {task_avg_steps:.1f}")
        
        timing_summary = task_timing.summary()
        if timing_summary:
            print(f"  Avg Inference Time: {timing_summary['mean_inference_time']:.3f}s")
        
        all_results[task_id] = {
            "task_id": task_id,
            "task_name": task.name,
            "task_description": task_description,
            "num_rollouts": args.num_rollouts,
            "successes": task_successes,
            "steps": task_steps,
            "success_rate": float(task_success_rate),
            "avg_steps": float(task_avg_steps),
            "timing": timing_summary,
        }
    
    # Overall results
    overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"Total Episodes: {total_episodes}")
    print(f"Total Successes: {total_successes}")
    print(f"Overall Success Rate: {overall_success_rate:.3f}")
    print("=" * 60)
    
    # Save results
    final_results = {
        "config": {
            "model_path": args.model_path,
            "benchmark": args.benchmark,
            "task_ids": task_ids,
            "num_rollouts": args.num_rollouts,
            "max_steps": max_steps,
            "seed": args.seed,
            "embodiment_tag": args.embodiment_tag,
            "action_norm": args.action_norm,
        },
        "overall": {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "success_rate": overall_success_rate,
        },
        "tasks": all_results,
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"{args.benchmark}_seed{args.seed}_{timestamp}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {result_file}")
    if args.save_video:
        print(f"Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()
