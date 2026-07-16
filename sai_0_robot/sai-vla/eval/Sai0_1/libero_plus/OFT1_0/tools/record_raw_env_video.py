#!/usr/bin/env python3
"""
LIBERO-plus env 原生方向录像工具
================================

启动一个 LIBERO-plus 任务, 用零动作让机械臂"自由下垂", 把 env.step() 直出的
``obs["agentview_image"]`` 和 ``obs["robot0_eye_in_hand_image"]`` 不做任何翻转
直接录成 mp4. 用来人工核对"模型评估时看到的图像方向"是否符合预期.

输出:
    <output_dir>/raw_top.mp4         (top  view, 完全不翻转)
    <output_dir>/raw_wrist.mp4       (wrist view, 完全不翻转)
    <output_dir>/raw_combined.mp4    (top + wrist 横拼,
                                      上面加了 "RAW NO-FLIP" 标签便于辨识)
    <output_dir>/flipped_top.mp4     (top  做 [::-1, ::-1, :] 之后)
    <output_dir>/flipped_wrist.mp4   (wrist 做 [::-1, ::-1, :] 之后)

用法:
    conda activate qwen_eagle_hwl
    python -m eval.Sai0_1.libero_plus.OFT1_0.tools.record_raw_env_video \\
        --task_suite_name libero_spatial \\
        --task_id 0 \\
        --num_steps 80 \\
        --output_dir /tmp/libero_raw_env

可选:
    --task_suite_name {libero_spatial, libero_object, libero_goal, libero_10}
    --task_id INT       (默认 0)
    --num_steps INT     (默认 80, 控制视频长度)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 必须在 import libero / robosuite 之前 fix 路径 + log 权限
THIS_FILE = Path(__file__).resolve()
EVAL_OFT_DIR = THIS_FILE.parent.parent
sys.path.insert(0, str(EVAL_OFT_DIR.parents[3]))

from eval.Sai0_1.libero_plus.OFT1_0.eval_libero_plus import (  # noqa: E402
    _bootstrap_libero_plus,
    _fix_robosuite_log_permission,
)

_fix_robosuite_log_permission()
_bootstrap_libero_plus()

import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


def _add_text_banner(img: np.ndarray, text: str, color=(255, 50, 50)) -> np.ndarray:
    """在图像顶部画一行文字, 方便区分 raw / flipped"""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw.rectangle([(0, 0), (img.shape[1], 22)], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=color, font=font)
    return np.asarray(pil)


def _stack(top: np.ndarray, wrist: np.ndarray, label: str) -> np.ndarray:
    """top 左, wrist 右, 顶部加一行说明"""
    h = max(top.shape[0], wrist.shape[0])
    if top.shape[0] != h:
        top = np.pad(top, ((0, h - top.shape[0]), (0, 0), (0, 0)))
    if wrist.shape[0] != h:
        wrist = np.pad(wrist, ((0, h - wrist.shape[0]), (0, 0), (0, 0)))
    combined = np.hstack([top, wrist])
    return _add_text_banner(combined, label)


def _save_video(frames, path: str, fps: int = 20) -> None:
    if not frames:
        print(f"⚠️  没有帧, 跳过 {path}")
        return
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p"
    )
    for f in frames:
        if f.dtype != np.uint8:
            f = f.astype(np.uint8)
        writer.append_data(f)
    writer.close()
    print(f"  ✓ 已保存: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--init_state_idx", type=int, default=0,
                        help="选第几个 init_state (默认 0)")
    parser.add_argument("--num_steps", type=int, default=80,
                        help="录多少 step (默认 80, fps=20 即 4 秒)")
    parser.add_argument("--num_steps_wait", type=int, default=10,
                        help="reset 后先用零动作走多少 step 让机械臂稳一下 (默认 10)")
    parser.add_argument("--output_dir", type=str, default="/tmp/libero_raw_env")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 LIBERO-plus env 原生方向录像")
    print(f"   suite       : {args.task_suite_name}")
    print(f"   task_id     : {args.task_id}")
    print(f"   init_state  : {args.init_state_idx}")
    print(f"   num_steps   : {args.num_steps}")
    print(f"   output_dir  : {out_dir}")
    print()

    bm = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not (0 <= args.task_id < bm.n_tasks):
        print(f"❌ task_id={args.task_id} 越界, 该 suite 共 {bm.n_tasks} 个 task")
        return 1
    task = bm.get_task(args.task_id)
    bddl = bm.get_task_bddl_file_path(args.task_id)
    print(f"   语言指令    : {task.language}")
    print()

    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    init_states = bm.get_task_init_states(args.task_id)
    if not (0 <= args.init_state_idx < len(init_states)):
        print(f"❌ init_state_idx={args.init_state_idx} 越界, 共 {len(init_states)} 个 init_state")
        return 1
    obs = env.set_init_state(init_states[args.init_state_idx])

    raw_top_frames, raw_wrist_frames = [], []
    flipped_top_frames, flipped_wrist_frames = [], []
    combined_raw_frames, combined_flipped_frames = [], []

    zero_action = np.zeros(7, dtype=np.float32)
    zero_action[-1] = -1.0  # 夹爪打开

    print("📸 录帧...")
    total_steps = args.num_steps_wait + args.num_steps
    for step in range(total_steps):
        raw_top = obs["agentview_image"]
        raw_wrist = obs["robot0_eye_in_hand_image"]

        if step >= args.num_steps_wait:
            flipped_top = raw_top[::-1, ::-1, :].copy()
            flipped_wrist = raw_wrist[::-1, ::-1, :].copy()

            raw_top_frames.append(raw_top.copy())
            raw_wrist_frames.append(raw_wrist.copy())
            flipped_top_frames.append(flipped_top)
            flipped_wrist_frames.append(flipped_wrist)
            combined_raw_frames.append(_stack(raw_top, raw_wrist, "RAW (no flip) - top | wrist"))
            combined_flipped_frames.append(_stack(flipped_top, flipped_wrist, "FLIPPED 180 - top | wrist"))

        obs, _, done, _ = env.step(zero_action.tolist())
        if done:
            break

    env.close()

    print(f"\n💾 保存视频 (fps={args.fps}, 共 {len(raw_top_frames)} 帧):")
    _save_video(raw_top_frames, str(out_dir / "raw_top.mp4"), fps=args.fps)
    _save_video(raw_wrist_frames, str(out_dir / "raw_wrist.mp4"), fps=args.fps)
    _save_video(flipped_top_frames, str(out_dir / "flipped_top.mp4"), fps=args.fps)
    _save_video(flipped_wrist_frames, str(out_dir / "flipped_wrist.mp4"), fps=args.fps)
    _save_video(combined_raw_frames, str(out_dir / "raw_combined.mp4"), fps=args.fps)
    _save_video(combined_flipped_frames, str(out_dir / "flipped_combined.mp4"), fps=args.fps)

    print()
    print(f"✅ 录像完成. 重点对比:")
    print(f"   {out_dir / 'raw_combined.mp4'}      ← 这就是模型评估时实际看到的方向")
    print(f"   {out_dir / 'flipped_combined.mp4'}  ← 之前 backbone flip 后给模型的(错误)方向")
    return 0


if __name__ == "__main__":
    sys.exit(main())
