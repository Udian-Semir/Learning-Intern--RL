#!/usr/bin/env python3
"""Evaluate a Flow Matching 1 action head inside the LIBERO simulator."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
import imageio

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer

# ---------------------------------------------------------------------------
# Resolve project roots so we can import Sai0-VLA modules without installing.
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
SAI0_ROOT = EVAL_DIR.parents[2]  # .../custom_hwl/sai0-vla
if str(SAI0_ROOT) not in sys.path:
    sys.path.insert(0, str(SAI0_ROOT))

from VLAs.Sai0 import End2EndGr00tPipeline  # noqa: E402 (path prepared above)


@dataclass
class TimingAccumulator:
    vlm: float = 0.0
    head: float = 0.0
    total: float = 0.0
    comms: float = 0.0
    calls: int = 0

    def add(self, timing: Dict[str, float]) -> None:
        if not timing:
            return
        self.vlm += timing.get("vlm_time", 0.0)
        self.head += timing.get("action_head_time", 0.0)
        self.total += timing.get("total_time", 0.0)
        self.comms += timing.get("communication_time", 0.0)
        self.calls += 1

    def summary(self) -> Dict[str, float]:
        if self.calls == 0:
            return {}
        return {
            "calls": self.calls,
            "vlm_time_avg": self.vlm / self.calls,
            "action_head_time_avg": self.head / self.calls,
            "total_time_avg": self.total / self.calls,
            "communication_time_avg": self.comms / self.calls,
        }


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_csv_int_list(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]



def ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        scaled = (image * 255.0).clip(0, 255)
    else:
        scaled = image.clip(0, 255)
    return scaled.astype(np.uint8)


def extract_images(obs_entry: Dict[str, np.ndarray], camera_keys: Sequence[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for key in camera_keys:
        if key not in obs_entry:
            raise KeyError(f"Observation is missing camera key '{key}'")
        rgb = ensure_uint8(np.asarray(obs_entry[key]))
        img = Image.fromarray(np.ascontiguousarray(rgb))
        # Mirror vertically to stay consistent with training data orientation.
        images.append(ImageOps.flip(img))
    return images


def extract_state(obs_entry: Dict[str, np.ndarray], state_keys: Sequence[str]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for key in state_keys:
        if key not in obs_entry:
            raise KeyError(f"Observation is missing state key '{key}'")
        value = np.asarray(obs_entry[key]).reshape(-1)
        chunks.append(value)
    state = np.concatenate(chunks, axis=0).astype(np.float32)
    return state


def compose_dual_view_frame(
    obs_entry: Dict[str, np.ndarray],
    camera_pair: Sequence[str],
) -> np.ndarray:
    if len(camera_pair) != 2:
        raise ValueError("Dual-view video requires exactly two camera keys")

    pil_views: List[Image.Image] = []
    for camera_name in camera_pair:
        if camera_name not in obs_entry:
            raise KeyError(f"Observation is missing camera key '{camera_name}' for video export")
        raw = ensure_uint8(np.asarray(obs_entry[camera_name]))
        view = Image.fromarray(np.ascontiguousarray(raw))
        pil_views.append(ImageOps.flip(view))

    target_height = max(img.height for img in pil_views)
    resized_arrays: List[np.ndarray] = []
    for img in pil_views:
        if img.height != target_height:
            new_width = int(img.width * (target_height / img.height))
            img = img.resize((new_width, target_height))
        resized_arrays.append(np.asarray(img))

    return np.concatenate(resized_arrays, axis=1)


def save_video_frames(frames: Sequence[np.ndarray], file_path: Path, fps: int = 30) -> None:
    if not frames:
        print(f"[WARN] No frames captured for video {file_path.name}, skipping save")
        return
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(file_path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def plan_actions(
    pipeline: End2EndGr00tPipeline,
    obs_entry: Dict[str, np.ndarray],
    camera_keys: Sequence[str],
    state_keys: Sequence[str],
    prompt: str,
) -> Tuple[np.ndarray, Dict[str, float], Optional[List[int]]]:
    images = extract_images(obs_entry, camera_keys)
    state_vec = extract_state(obs_entry, state_keys)
    actions, timing = pipeline.predict(
        images=images,
        state=np.expand_dims(state_vec, axis=0),
        prompt=prompt,
        return_numpy=True,
    )
    if actions.ndim == 3:
        horizon = actions[0]
    elif actions.ndim == 2:
        horizon = actions
    else:
        raise ValueError(f"Unexpected action tensor shape: {actions.shape}")
    return horizon, timing, pipeline.get_last_input_ids()


def build_env(args, task_bddl: str) -> SubprocVectorEnv:
    env_args = {
        "bddl_file_name": task_bddl,
        "camera_heights": args.camera_height,
        "camera_widths": args.camera_width,
    }
    return SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(args.num_envs)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LIBERO evaluation with Sai0 Flow Matching 1 action head")
    parser.add_argument("--action-head-ckpt", required=True, help="Path to the Flow Matching checkpoint directory")
    parser.add_argument("--action-head-version", type=str, default=None, help="Force the action head version (e.g., flow_matching_1, oft)")
    parser.add_argument("--benchmark", default="libero_10", choices=[
        "libero_10", "libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_100"
    ])
    parser.add_argument("--task-id", type=int, default=0, help="Task index inside the benchmark")
    parser.add_argument("--task-order-index", type=int, default=0, help="Benchmark task order (matches LIBERO defaults)")
    parser.add_argument("--num-rollouts", type=int, default=50, help="Number of sequential rollouts to perform (default 50 as requested)")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel simulator instances (set to 1 for sequential rollouts)")
    parser.add_argument("--max-steps", type=int, default=600, help="Maximum control steps per rollout")
    parser.add_argument("--execute-all-horizon", action="store_true", help="Execute the full predicted horizon instead of replanning every step")
    parser.add_argument("--execute-all-chunks", action="store_true", help="Alias for --execute-all-horizon (for compatibility with OFT eval script)")
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-keys", type=str, default="agentview_image", help="Comma separated camera obs keys")
    parser.add_argument(
        "--state-keys",
        type=str,
        default="robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos",
        help="Comma separated proprio keys to build the state vector",
    )
    parser.add_argument("--settle-steps", type=int, default=5, help="Physics settle steps before evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vlm-backend", type=str, default="qwen3-vl")
    parser.add_argument("--vlm-model-name", type=str, default=None)
    parser.add_argument(
        "--vlm-layer-indices",
        type=str,
        default=None,
        nargs="+",
        help="Comma or space separated list of VLM layer ids",
    )
    parser.add_argument("--concat-mode", type=str, default="sequence", choices=["sequence", "feature", "last"])
    parser.add_argument("--prompt", type=str, default=None, help="Override the language prompt (defaults to task language)")
    parser.add_argument("--embodiment-id", type=int, default=31, help="Embodiment id passed to the action head")
    parser.add_argument("--results-dir", type=str, default="./flow_matching_eval_results")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="./flow_matching_eval_videos")
    parser.add_argument(
        "--video-camera",
        type=str,
        default="agentview_image,robot0_eye_in_hand_image",
        help="Comma separated pair of camera keys to combine when saving video",
    )
    parser.add_argument(
        "--append-token-ids",
        type=str,
        default=None,
        help="Comma separated list of token ids to append to the VLM input sequence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execute_full_horizon = args.execute_all_horizon or args.execute_all_chunks
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    camera_keys = parse_csv_list(args.camera_keys)
    state_keys = parse_csv_list(args.state_keys)
    if not camera_keys:
        raise ValueError("At least one camera key is required")
    if not state_keys:
        raise ValueError("At least one state key is required")

    vlm_layers_arg = args.vlm_layer_indices
    if isinstance(vlm_layers_arg, list):
        vlm_layers_arg = ",".join(vlm_layers_arg)
    vlm_layers = parse_csv_list(vlm_layers_arg) if vlm_layers_arg else None
    if vlm_layers:
        layer_indices = [int(idx) for idx in vlm_layers]
    else:
        layer_indices = None

    append_token_ids = parse_csv_int_list(args.append_token_ids) if args.append_token_ids else None

    pipeline = End2EndGr00tPipeline(
        vlm_backend=args.vlm_backend,
        vlm_model_name=args.vlm_model_name,
        action_head_ckpt=args.action_head_ckpt,
        action_head_version=args.action_head_version,
        vlm_layer_indices=layer_indices,
        concat_mode=args.concat_mode,
        device=args.device,
        embodiment_id=args.embodiment_id,
        append_token_ids=append_token_ids,
    )

    action_dim = pipeline.action_head_config.get("action_dim")
    action_horizon = pipeline.action_head_config.get("action_horizon")
    if action_dim is None or action_horizon is None:
        raise ValueError("Action head config must include 'action_dim' and 'action_horizon'")

    benchmark = get_benchmark(args.benchmark)(args.task_order_index)
    task = benchmark.get_task(args.task_id)
    prompt = args.prompt or task.language
    task_bddl = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    print("\n" + "="*60)
    print("LIBERO Evaluation Configuration")
    print("="*60)
    print(f"Benchmark: {args.benchmark}")
    print(f"Task ID: {args.task_id}")
    print(f"Task Name: {task.name}")
    print(f"Task Language: {task.language}")
    print(f"Action Dim: {action_dim} | Action Horizon: {action_horizon}")
    print(f"\nAll Available Tasks in {args.benchmark}:")
    print("-" * 60)
    num_tasks = len(benchmark.get_task_names())
    for i in range(num_tasks):
        t = benchmark.get_task(i)
        marker = " <<< SELECTED" if i == args.task_id else ""
        print(f"  [{i}] {t.language}{marker}")
    print("-" * 60)
    print(f"\nNumber of Rollouts: {args.num_rollouts}")
    print(f"Max Steps per Rollout: {args.max_steps}")
    print(f"Execution Mode: {'Execute ALL horizon actions' if execute_full_horizon else 'Execute FIRST action then replan'}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("="*60 + "\n")

    os.makedirs(args.results_dir, exist_ok=True)
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    init_states = benchmark.get_task_init_states(args.task_id)
    video_camera_keys = parse_csv_list(args.video_camera)
    if args.save_video and len(video_camera_keys) != 2:
        raise ValueError("--video-camera must specify exactly two comma-separated keys when saving video")

    all_success = []
    all_steps = []
    global_timing_acc = TimingAccumulator()
    vlm_tokens_printed = False

    action_buffer = None
    buffer_pos = 0

    zero_action = np.zeros((1, int(action_dim)), dtype=np.float32)

    for rollout_id in range(args.num_rollouts):
        print(f"\n===== Starting Rollout {rollout_id + 1}/{args.num_rollouts} =====")

        env = build_env(args, task_bddl)
        env.seed(args.seed + rollout_id)
        env.reset()

        state_idx = rollout_id % init_states.shape[0]
        obs = env.set_init_state(init_states[state_idx:state_idx+1])

        if args.settle_steps > 0:
            for _ in range(args.settle_steps):
                obs, _, _, _ = env.step(zero_action)

        done = False
        steps_taken = 0
        timing_acc = TimingAccumulator()
        action_buffer = None
        buffer_pos = 0

        rollout_frames: List[np.ndarray] = []

        with Timer() as rollout_timer:
            for step in range(args.max_steps):
                if done:
                    break

                if action_buffer is None or buffer_pos >= len(action_buffer):
                    trajectory, timing, input_tokens = plan_actions(
                        pipeline=pipeline,
                        obs_entry=obs[0],
                        camera_keys=camera_keys,
                        state_keys=state_keys,
                        prompt=prompt,
                    )

                    if (not vlm_tokens_printed) and input_tokens:
                        tokens_str = ", ".join(str(token) for token in input_tokens)
                        print(
                            "[INFO] VLM input token ids ("
                            f"{len(input_tokens)} tokens):\n[{tokens_str}]"
                        )
                        vlm_tokens_printed = True

                    timing_acc.add(timing)
                    global_timing_acc.add(timing)

                    action_buffer = trajectory
                    buffer_pos = 0

                if execute_full_horizon:
                    action = action_buffer[buffer_pos]
                    buffer_pos += 1
                else:
                    action = action_buffer[0]
                    action_buffer = None
                    buffer_pos = 0

                action = action.reshape(1, int(action_dim)).astype(np.float32)
                obs, reward, done_arr, info = env.step(action)
                done = bool(done_arr[0])

                if args.save_video:
                    dual_view = compose_dual_view_frame(obs_entry=obs[0], camera_pair=video_camera_keys)
                    rollout_frames.append(dual_view)

                steps_taken += 1

        env.close()

        success = int(done)
        all_success.append(success)
        all_steps.append(steps_taken)

        rollout_timing = timing_acc.summary()
        print(f"Rollout {rollout_id + 1}: Success={success}, Steps={steps_taken}, Time={rollout_timer.get_elapsed_time():.2f}s")
        if rollout_timing:
            print(f"  VLM: {rollout_timing['vlm_time_avg']:.3f}s | Action Head: {rollout_timing['action_head_time_avg']:.3f}s")
        if args.save_video:
            status = "success" if success else "fail"
            video_name = f"task{args.task_id}_rollout{rollout_id + 1:03d}_{status}.mp4"
            save_video_frames(rollout_frames, video_dir / video_name)
            print(f"  Video saved to: {video_dir / video_name}")

    success_rate = float(np.mean(all_success)) if all_success else 0.0
    avg_steps = float(np.mean(all_steps)) if all_steps else 0.0
    global_timing_summary = global_timing_acc.summary()
    timing_summary_serializable = (
        {key: float(value) for key, value in global_timing_summary.items()}
        if global_timing_summary
        else {}
    )

    stats = {
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "task_name": task.name,
        "task_language": task.language,
        "prompt": prompt,
        "num_rollouts": args.num_rollouts,
        "max_steps": args.max_steps,
        "all_success": all_success,
        "all_steps": all_steps,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "timing_summary": timing_summary_serializable,
        "action_dim": int(action_dim),
        "action_horizon": int(action_horizon),
    }

    result_name = f"{args.benchmark}_task{args.task_id}_seed{args.seed}_rollouts{args.num_rollouts}.json"
    save_path = Path(args.results_dir) / result_name
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print("\n===== Evaluation Complete =====")
    print(f"Task: {task.language}")
    print(f"Total Rollouts: {args.num_rollouts}")
    print(f"Success Rate: {success_rate:.3f} ({sum(all_success)}/{args.num_rollouts})")
    print(f"Average Steps: {avg_steps:.1f}")
    if global_timing_summary:
        print(
            f"Avg VLM: {global_timing_summary['vlm_time_avg']:.3f}s | "
            f"Avg Action Head: {global_timing_summary['action_head_time_avg']:.3f}s | "
            f"Avg Total: {global_timing_summary['total_time_avg']:.3f}s"
        )
    print(f"Stats saved to {save_path}")
    print(f"Videos saved to {video_dir}")


if __name__ == "__main__":
    main()

'''
Example CLI (mirrors parse_args defaults/options):

python eval_benchmark.py \
    --action-head-ckpt /path/to/checkpoint \
    --action-head-version flow_matching_1 \
    --benchmark libero_object \
    --task-id 0 \
    --task-order-index 0 \
    --num-rollouts 50 \
    --num-envs 1 \
    --max-steps 600 \
    --execute-all-horizon \
    --camera-height 128 \
    --camera-width 128 \
    --camera-keys agentview_image,robot0_eye_in_hand_image \
    --state-keys robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos \
    --settle-steps 5 \
    --seed 42 \
    --device cuda:0 \
    --vlm-backend qwen3-vl \
    --vlm-model-name Qwen/Qwen3-VL-8B-Instruct \
    --vlm-layer-indices 0,17,35 \
    --concat-mode sequence \
    --prompt "pick up the block" \
    --embodiment-id 31 \
    --results-dir ./flow_matching_eval_results \
    --save-video \
    --video-dir ./flow_matching_eval_videos \
    --video-camera agentview_image,robot0_eye_in_hand_image \
    --append-token-ids "151644,77091,198"
'''
