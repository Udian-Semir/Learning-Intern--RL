#!/usr/bin/env python3
"""
Convert LIBERO HDF5 datasets to LeRobot format

Usage:
    python libero_to_lerobot.py --input-dir /data/HuangWenlong/datasets/libero_github/libero_10 \
                                  --output-dir /data/output/libero_10_lerobot \
                                  --fps 10
"""

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import contextlib


def read_hdf5_structure(hdf5_file: Path) -> Dict:
    """Read HDF5 file structure to understand data format"""
    with h5py.File(hdf5_file, 'r') as f:
        demos = list(f['data'].keys())
        
        # Get info from first demo
        first_demo = f['data'][demos[0]]
        info = {
            'num_demos': len(demos),
            'keys': list(first_demo.keys()),
            'action_shape': first_demo['actions'].shape,
            'obs_keys': list(first_demo['obs'].keys()) if 'obs' in first_demo else []
        }
        
        if 'obs' in first_demo:
            obs = first_demo['obs']
            info['obs_shapes'] = {k: v.shape for k, v in obs.items()}
    
    return info


def extract_demo_data(hdf5_file: Path, demo_key: str) -> Dict[str, np.ndarray]:
    """Extract data from a single demo"""
    with h5py.File(hdf5_file, 'r') as f:
        demo = f['data'][demo_key]

        actions = demo['actions'][:]  # (T, 7)
        robot_states = demo['robot_states'][:]  # (T, 9)
        agentview_rgb = demo['obs']['agentview_rgb'][:]  # (T, 128, 128, 3)
        eye_in_hand_rgb = demo['obs']['eye_in_hand_rgb'][:]  # (T, 128, 128, 3)
        dones = demo['dones'][:]

        rewards_dataset = demo.get('rewards')
        if rewards_dataset is not None:
            rewards = rewards_dataset[:]
        else:
            rewards = np.zeros(len(actions), dtype=actions.dtype)

        data = {
            'actions': actions,
            'robot_states': robot_states,
            'agentview_rgb': agentview_rgb,
            'eye_in_hand_rgb': eye_in_hand_rgb,
            'dones': dones,
            'rewards': rewards,
        }

    return data


def save_episode_parquet(
    data: Dict[str, np.ndarray],
    episode_idx: int,
    task_idx: int,
    task_name: str,
    output_chunk_dir: Path,
    global_index_start: int,
    fps: float = 10.0
) -> Tuple[pd.DataFrame, int]:
    """Save episode data as parquet file"""

    num_frames = len(data['actions'])
    if num_frames == 0:
        raise ValueError("Episode contains no frames and cannot be written to parquet.")

    actions = np.asarray(data['actions'])
    robot_states = np.asarray(data['robot_states'])
    rewards = np.asarray(data['rewards'])
    dones = np.asarray(data['dones'])

    if not (len(robot_states) == len(actions) == len(rewards) == len(dones) == num_frames):
        raise ValueError("Inconsistent frame counts across data arrays.")

    timestamps = np.arange(num_frames, dtype=np.float64) / float(fps)
    task_indices = np.full(num_frames, task_idx, dtype=np.int64)
    task_description_indices = np.full(num_frames, task_idx, dtype=np.int64)
    annotation_validity = np.ones(num_frames, dtype=np.int64)
    episode_indices = np.full(num_frames, episode_idx, dtype=np.int64)
    global_indices = np.arange(global_index_start, global_index_start + num_frames, dtype=np.int64)
    vlm_hidden_state_indices = global_indices.copy()

    reward_values = rewards.copy()
    done_values = dones.astype(dones.dtype, copy=True)
    if done_values.size > 0:
        done_values[-1] = np.array(True, dtype=done_values.dtype).item()

    observation_element_type = pa.from_numpy_dtype(robot_states.dtype)
    action_element_type = pa.from_numpy_dtype(actions.dtype)
    reward_type = pa.from_numpy_dtype(reward_values.dtype)
    done_type = pa.from_numpy_dtype(done_values.dtype)

    schema = pa.schema([
        pa.field("observation.state", pa.list_(observation_element_type)),
        pa.field("action", pa.list_(action_element_type)),
        pa.field("timestamp", pa.float64()),
        pa.field("task_index", pa.int64()),
        pa.field("annotation.human.action.task_description", pa.int64()),
        pa.field("annotation.human.validity", pa.int64()),
        pa.field("episode_index", pa.int64()),
        pa.field("index", pa.int64()),
        pa.field("next.reward", reward_type),
        pa.field("next.done", done_type),
        pa.field("vlm_hidden_state_index", pa.int64()),
    ])

    table = pa.Table.from_arrays(
        [
            pa.array(robot_states.tolist(), type=pa.list_(observation_element_type)),
            pa.array(actions.tolist(), type=pa.list_(action_element_type)),
            pa.array(timestamps, type=pa.float64()),
            pa.array(task_indices, type=pa.int64()),
            pa.array(task_description_indices, type=pa.int64()),
            pa.array(annotation_validity, type=pa.int64()),
            pa.array(episode_indices, type=pa.int64()),
            pa.array(global_indices, type=pa.int64()),
            pa.array(reward_values, type=reward_type),
            pa.array(done_values, type=done_type),
            pa.array(vlm_hidden_state_indices, type=pa.int64()),
        ],
        schema=schema,
    )

    output_file = output_chunk_dir / f"episode_{episode_idx:06d}.parquet"
    pq.write_table(table, output_file)

    df = pd.DataFrame({
        'observation.state': list(robot_states),
        'action': list(actions),
        'timestamp': timestamps,
        'task_index': task_indices,
        'annotation.human.action.task_description': task_description_indices,
        'annotation.human.validity': annotation_validity,
        'episode_index': episode_indices,
        'index': global_indices,
        'next.reward': reward_values,
        'next.done': done_values,
        'vlm_hidden_state_index': vlm_hidden_state_indices,
    })

    return df, global_index_start + num_frames


def save_videos(
    data: Dict[str, np.ndarray],
    episode_idx: int,
    output_chunk_dir: Path,
    fps: float = 10.0,
    codec_candidates: Optional[List[str]] = None
) -> Tuple[int, int, str]:
    """Save RGB observations as video files and return video dimensions plus codec used"""

    codec_candidates = codec_candidates or ["FFV1", "mp4v"]

    # Save agentview video
    agentview_dir = output_chunk_dir / "observation.images.agentview"
    agentview_dir.mkdir(parents=True, exist_ok=True)
    agentview_video = agentview_dir / f"episode_{episode_idx:06d}.mp4"

    # Save eye_in_hand video
    wrist_dir = output_chunk_dir / "observation.images.wrist"
    wrist_dir.mkdir(parents=True, exist_ok=True)
    wrist_video = wrist_dir / f"episode_{episode_idx:06d}.mp4"

    # Get video properties from actual data shape
    height, width = data['agentview_rgb'].shape[1:3]
    frame_size = (width, height)

    def _codec_supported(codec: str) -> bool:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        for path in (agentview_video, wrist_video):
            writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
            if not writer.isOpened():
                writer.release()
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()
                return False
            writer.release()
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        return True

    selected_codec: Optional[str] = None
    for codec in codec_candidates:
        if _codec_supported(codec):
            selected_codec = codec
            break

    if selected_codec is None:
        raise RuntimeError(
            f"Unable to find a supported codec from candidates {codec_candidates} for episode {episode_idx}."
        )

    if selected_codec != "FFV1":
        logging.warning(
            "Falling back to codec '%s' for episode %d because FFV1 is unavailable for mp4 container.",
            selected_codec,
            episode_idx,
        )

    fourcc = cv2.VideoWriter_fourcc(*selected_codec)

    out_agent = cv2.VideoWriter(str(agentview_video), fourcc, fps, frame_size)
    if not out_agent.isOpened():
        raise RuntimeError(f"Failed to open video writer for {agentview_video} using codec {selected_codec}.")
    for frame in data['agentview_rgb']:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_agent.write(frame_bgr)
    out_agent.release()

    out_wrist = cv2.VideoWriter(str(wrist_video), fourcc, fps, frame_size)
    if not out_wrist.isOpened():
        raise RuntimeError(f"Failed to open video writer for {wrist_video} using codec {selected_codec}.")
    for frame in data['eye_in_hand_rgb']:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out_wrist.write(frame_bgr)
    out_wrist.release()

    return height, width, selected_codec.lower()


def save_metadata(
    task_names: List[str],
    total_episodes: int,
    total_frames: int,
    total_chunks: int,
    chunks_size: int,
    fps: float,
    output_dir: Path,
    video_shape: Tuple[int, int] = (128, 128),
    dtype_info: Optional[Dict[str, np.dtype]] = None,
    video_codec: str = "ffv1"
):
    """Save metadata files for LeRobot format"""
    
    dtype_info = dtype_info or {}
    
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    height, width = video_shape

    def _dtype_name(key: str, default: np.dtype) -> str:
        return np.dtype(dtype_info.get(key, default)).name
    
    # info.json
    info = {
        "codebase_version": "v2.0",
        "robot_type": "franka",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_names),
        "total_videos": total_episodes * 2,  # agentview + wrist
        "total_chunks": total_chunks,
        "chunks_size": chunks_size,
        "fps": fps,
        "splits": {"train": "0:100"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.agentview": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": video_codec,
                    "video.pix_fmt": "rgb24",
                    "video.is_depth_map": False,
                    "has_audio": False,
                    "lossless": video_codec.lower() == "ffv1"
                }
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "video.fps": fps,
                    "video.codec": video_codec,
                    "video.pix_fmt": "rgb24",
                    "video.is_depth_map": False,
                    "has_audio": False,
                    "lossless": video_codec.lower() == "ffv1"
                }
            },
            "observation.state": {
                "dtype": _dtype_name("observation.state", np.float32),
                "shape": [9],
                "names": ["gripper_qpos_left", "gripper_qpos_right",
                         "ee_pos_x", "ee_pos_y", "ee_pos_z",
                         "ee_quat_w", "ee_quat_x", "ee_quat_y", "ee_quat_z"]
            },
            "action": {
                "dtype": _dtype_name("action", np.float32),
                "shape": [7],
                "names": ["delta_x", "delta_y", "delta_z", 
                         "delta_roll", "delta_pitch", "delta_yaw", "gripper"]
            },
            "timestamp": {"dtype": "float64"},
            "task_index": {"dtype": "int64"},
            "annotation.human.action.task_description": {"dtype": "int64"},
            "annotation.human.validity": {"dtype": "int64"},
            "episode_index": {"dtype": "int64"},
            "index": {"dtype": "int64"},
            "next.reward": {"dtype": _dtype_name("next.reward", np.float64)},
            "next.done": {"dtype": _dtype_name("next.done", np.bool_)},
            "vlm_hidden_state_index": {
                "dtype": "int64",
                "description": "Index to load corresponding VLM hidden state file"
            }
        },
        "vlm_hidden_state_path": "vlm_hidden_states/hidden_state_{vlm_hidden_state_index:06d}.npy"
    }
    
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # tasks.jsonl - remove underscores from task names
    with open(meta_dir / "tasks.jsonl", 'w') as f:
        for idx, task_name in enumerate(task_names):
            # Remove underscores and replace with spaces
            cleaned_task_name = task_name.replace('_', ' ')
            # Remove leading uppercase letters (e.g., "KITCHEN_SCENE_" prefix)
            cleaned_task_name = cleaned_task_name.lstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
            # Remove leading numbers and spaces (e.g., "3 " from "3 turn on the stove")
            cleaned_task_name = cleaned_task_name.lstrip('0123456789 ')
            task_data = {"task_index": idx, "task": cleaned_task_name}
            f.write(json.dumps(task_data) + '\n')
    
    # modality.json
    modality = {
        "observation.state": {
            "gripper_qpos": {"start": 0, "end": 2},
            "ee_pos": {"start": 2, "end": 5},
            "ee_quat": {"start": 5, "end": 9}
        },
        "action": {
            "ee_delta": {"start": 0, "end": 6},
            "gripper": {"start": 6, "end": 7}
        },
        "observation.images": {
            "agentview": {"original_key": "observation.images.agentview"},
            "wrist": {"original_key": "observation.images.wrist"}
        },
        "annotation.human": {
            "action.task_description": {},
            "validity": {}
        }
    }
    
    with open(meta_dir / "modality.json", 'w') as f:
        json.dump(modality, f, indent=2)
    
    print(f"✓ Saved metadata to {meta_dir}")


def convert_libero_to_lerobot(
    input_dir: Path,
    output_dir: Path,
    fps: float = 10.0,
    chunk_size: int = 100
):
    """Main conversion function"""
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} task files")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track statistics
    task_names = []
    total_episodes = 0
    total_frames = 0
    episode_metadata = []
    video_shape = None  # Will be set from first video
    
    global_index = 0
    current_chunk = 0
    episodes_in_chunk = 0
    dtype_registry: Dict[str, np.dtype] = {}
    video_codec_used: Optional[str] = None

    def update_dtype(key: str, new_dtype: np.dtype) -> None:
        new_dtype = np.dtype(new_dtype)
        existing = dtype_registry.get(key)
        if existing is None:
            dtype_registry[key] = new_dtype
        elif existing != new_dtype:
            raise ValueError(
                f"Inconsistent dtype for '{key}': existing {existing} vs new {new_dtype}"
            )
    
    # Process each task file
    for task_idx, hdf5_file in enumerate(tqdm(hdf5_files, desc="Converting tasks")):
        task_name = hdf5_file.stem.replace('_demo', '')
        task_names.append(task_name)
        
        # Read file structure
        info = read_hdf5_structure(hdf5_file)
        num_demos = info['num_demos']
        
        print(f"\nTask {task_idx}: {task_name} ({num_demos} demos)")
        
        # Process each demo
        for demo_idx in range(num_demos):
            demo_key = f"demo_{demo_idx}"
            
            # Check if need new chunk
            if episodes_in_chunk >= chunk_size:
                current_chunk += 1
                episodes_in_chunk = 0
            
            # Create chunk directories
            data_chunk_dir = output_dir / "data" / f"chunk-{current_chunk:03d}"
            video_chunk_dir = output_dir / "videos" / f"chunk-{current_chunk:03d}"
            data_chunk_dir.mkdir(parents=True, exist_ok=True)
            video_chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract demo data
            data = extract_demo_data(hdf5_file, demo_key)
            num_frames = len(data['actions'])

            update_dtype("observation.state", data['robot_states'].dtype)
            update_dtype("action", data['actions'].dtype)
            update_dtype("next.reward", data['rewards'].dtype)
            update_dtype("next.done", data['dones'].dtype)
            
            # Save parquet
            _, global_index = save_episode_parquet(
                data, total_episodes, task_idx, task_name,
                data_chunk_dir, global_index, fps
            )
            
            # Save videos and get dimensions
            height, width, codec_used = save_videos(data, total_episodes, video_chunk_dir, fps)
            if video_shape is None:
                video_shape = (height, width)
            if video_codec_used is None:
                video_codec_used = codec_used
            elif video_codec_used != codec_used:
                logging.warning(
                    "Episode %d switched video codec from %s to %s; keeping metadata codec as %s.",
                    total_episodes,
                    video_codec_used,
                    codec_used,
                    video_codec_used,
                )
            
            # Track metadata
            episode_metadata.append({
                "episode_index": total_episodes,
                "task_index": task_idx,
                "length": num_frames,
                "chunk": current_chunk
            })
            
            total_episodes += 1
            total_frames += num_frames
            episodes_in_chunk += 1
    
    # Save episodes metadata
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "episodes.jsonl", 'w') as f:
        for ep in episode_metadata:
            f.write(json.dumps(ep) + '\n')
    
    # Save metadata
    save_metadata(
        task_names, total_episodes, total_frames,
        current_chunk + 1, chunk_size, fps, output_dir,
        video_shape if video_shape else (128, 128),
        dtype_info=dtype_registry,
        video_codec=(video_codec_used or "ffv1")
    )
    
    # Create placeholder VLM hidden states directory
    vlm_dir = output_dir / "vlm_hidden_states"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Conversion completed!")
    print(f"  Total tasks: {len(task_names)}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 datasets to LeRobot format"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing LIBERO HDF5 files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LeRobot format data"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for videos (default: 10.0)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Number of episodes per chunk (default: 100)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input_dir.exists():
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return 1
    
    # Run conversion
    convert_libero_to_lerobot(
        args.input_dir,
        args.output_dir,
        args.fps,
        args.chunk_size
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
