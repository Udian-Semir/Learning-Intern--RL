"""
OFT 遥操作推理服务器 (兼容 robot_control 客户端)

基于 tele_parad 的 VLM + OFT Pipeline 架构，
API 接口兼容 deploy_dataset197_jointangle.py，
可直接被 robot_control_no_hand_status_joint_roi_new_bgr2rgb.py 调用。

使用方法:
    python server.py --config config.yaml
    python server.py --config config.yaml --offline

API 端点:
    POST /predict  - 接收 image_arrays + state (+ instruction / world_point)，返回 actions
    POST /act      - /predict 的别名
    GET  /health   - 健康检查
    GET  /info     - 模型信息
    GET  /debug    - 实时监控面板
    GET  /debug/data - 最新推理快照 (JSON)
"""

import os
import sys
import time
import json
import logging
import traceback
import base64
import configparser
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
import argparse

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from VLMs.S0_1.backbone import create_vlm_backbone
from Action_Heads.OFT1_0.vlm2oft_pipeline import create_vlm2oft_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==================== Debug 辅助 ====================

def _pil_to_base64_jpeg(img: Image.Image, quality: int = 85) -> Optional[str]:
    """PIL Image -> base64 JPEG string"""
    if img is None:
        return None
    try:
        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _np_to_base64_jpeg(arr: np.ndarray, quality: int = 85) -> Optional[str]:
    """RGB uint8 numpy -> base64 JPEG string"""
    if arr is None:
        return None
    try:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        return _pil_to_base64_jpeg(img, quality)
    except Exception:
        return None


# ==================== ROI 提取 ====================

class CameraProjector:
    """将世界坐标投影到像素坐标 (相机内参 + 外参)。"""

    def __init__(self, intrinsics_path: str, extrinsics_path: str,
                 intrinsic_section: str = "ColorIntrinsic"):
        config = configparser.ConfigParser()
        config.read(str(intrinsics_path), encoding="utf-8")
        self.fx = float(config[intrinsic_section]["fx"])
        self.fy = float(config[intrinsic_section]["fy"])
        self.cx = float(config[intrinsic_section]["cx"])
        self.cy = float(config[intrinsic_section]["cy"])

        with open(extrinsics_path, "r", encoding="utf-8") as f:
            extr = json.load(f)
        self.R = np.array(extr["R"], dtype=np.float64)
        self.t = np.array(extr["t"], dtype=np.float64)

    def world_to_pixel(self, world_point) -> Tuple[int, int]:
        world = np.array(world_point[:3], dtype=np.float64)
        cam = self.R @ world + self.t
        if abs(cam[2]) < 1e-6:
            return -1, -1
        u = self.fx * cam[0] / cam[2] + self.cx
        v = self.fy * cam[1] / cam[2] + self.cy
        return int(round(u)), int(round(v))


def extract_roi(image: np.ndarray, pixel_point: Tuple[int, int],
                margins: Tuple[int, int, int, int],
                target_size: int = 256) -> np.ndarray:
    """
    从图像中根据 pixel_point 裁剪 ROI 区域并 resize。

    Args:
        image: (H, W, 3) uint8
        pixel_point: (u, v)
        margins: (left, right, top, bottom)
        target_size: 输出正方形边长
    Returns:
        (target_size, target_size, 3) uint8
    """
    h, w = image.shape[:2]
    u, v = pixel_point
    m_left, m_right, m_top, m_bottom = margins

    roi_x_min, roi_x_max = u - m_left, u + m_right
    roi_y_min, roi_y_max = v - m_top, v + m_bottom
    roi_w, roi_h = roi_x_max - roi_x_min, roi_y_max - roi_y_min

    roi = np.zeros((roi_h, roi_w, 3), dtype=np.uint8)

    ix0, ix1 = max(0, roi_x_min), min(w, roi_x_max)
    iy0, iy1 = max(0, roi_y_min), min(h, roi_y_max)
    rx0 = ix0 - roi_x_min
    ry0 = iy0 - roi_y_min

    if ix0 < ix1 and iy0 < iy1:
        roi[ry0 : ry0 + (iy1 - iy0), rx0 : rx0 + (ix1 - ix0)] = image[iy0:iy1, ix0:ix1]

    return cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_AREA)


def _draw_roi_box(img: np.ndarray, pixel_point: Tuple[int, int],
                  margins: Tuple[int, int, int, int]) -> np.ndarray:
    """在图像副本上绘制 ROI 框和中心点 (用于 debug 可视化)。"""
    vis = img.copy()
    u, v = pixel_point
    m_l, m_r, m_t, m_b = margins
    cv2.rectangle(vis, (u - m_l, v - m_t), (u + m_r, v + m_b), (0, 255, 0), 2)
    cv2.circle(vis, (u, v), 6, (0, 0, 255), -1)
    cv2.putText(vis, f"({u},{v})", (u + 10, v - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return vis


# ==================== State 预处理 ====================

def apply_state_preprocessing(
    observation_state: np.ndarray,
    state_process_order: List[str],
    hand_binary_columns: Optional[List[int]] = None,
    hand_binary_threshold: float = 442.0,
    minmax_columns: Optional[List[int]] = None,
    minmax_min: Optional[np.ndarray] = None,
    minmax_max: Optional[np.ndarray] = None,
    gripper_binarize_columns: Optional[List[int]] = None,
    gripper_binarize_threshold: float = 0.5,
) -> np.ndarray:
    """按配置顺序对 state 向量做预处理 (归一化 / 二值化 / hand_binary 等)"""
    if not state_process_order:
        return observation_state

    index_offset = 0

    for processor_name in state_process_order:
        if processor_name == "hand_binary" and hand_binary_columns is not None:
            hand_offset = 0
            for group_idx in range(0, len(hand_binary_columns), 2):
                if group_idx + 1 < len(hand_binary_columns):
                    start = hand_binary_columns[group_idx]
                    end = hand_binary_columns[group_idx + 1]
                    adj_start = start + index_offset + hand_offset
                    adj_end = end + index_offset + hand_offset
                    if adj_start < len(observation_state) and adj_end <= len(observation_state):
                        hand_data = observation_state[adj_start:adj_end]
                        hand_binary = np.array(
                            [1.0 if np.mean(hand_data) > hand_binary_threshold else -1.0],
                            dtype=np.float32,
                        )
                        observation_state = np.concatenate([
                            observation_state[:adj_start],
                            hand_binary,
                            observation_state[adj_end:],
                        ])
                        hand_offset -= (end - start - 1)
            index_offset += hand_offset

        elif processor_name == "minmax_normalize" and minmax_columns is not None:
            if minmax_min is not None and minmax_max is not None:
                for col in minmax_columns:
                    if col < len(observation_state):
                        col_range = minmax_max[col] - minmax_min[col]
                        if col_range > 1e-8:
                            observation_state[col] = (
                                2.0 * (observation_state[col] - minmax_min[col]) / col_range - 1.0
                            )
                        else:
                            observation_state[col] = 0.0

        elif processor_name == "gripper_binarize" and gripper_binarize_columns is not None:
            for col in gripper_binarize_columns:
                if col < len(observation_state):
                    observation_state[col] = 1 if observation_state[col] > gripper_binarize_threshold else -1

    return observation_state


# ==================== 图像预处理 ====================

def preprocess_image(
    image: Image.Image,
    resize: Optional[List[int]] = None,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    rotate_180: bool = False,
) -> Image.Image:
    if resize is not None and len(resize) == 2:
        image = image.resize((resize[0], resize[1]), Image.Resampling.LANCZOS)
    if rotate_180:
        image = image.rotate(180)
    if flip_horizontal:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_vertical:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return image


# ==================== 推理引擎 ====================

class OFTInferenceEngine:
    """VLM + OFT Pipeline 推理引擎，根据 config.yaml 做全部预/后处理。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cuda:0")

        self.vlm_backbone = None
        self.oft_pipeline = None

        self.chunk_size = config.get("chunk_size", 50)
        self.action_dim = config.get("action_dim", 14)
        self.proprio_dim = config.get("proprio_dim", 14)
        self.use_proprio = config.get("use_proprio", False)
        self.execute_horizon = config.get("execute_horizon", None)

        # Debug 快照 (供前端面板读取)
        self._debug: Dict[str, Any] = {}
        self._request_count = 0
        self._start_time = time.time()

        # ---------- State 索引提取 ----------
        self.state_select_indices = config.get("state_select_indices", None)
        self.state_process_order = config.get("state_process_order", [])
        self.hand_binary_columns = config.get("hand_binary_columns", [])
        self.hand_binary_threshold = config.get("hand_binary_threshold", 442.0)
        self.state_dim = config.get("state_dim", 14)

        if self.state_select_indices is not None:
            if len(self.state_select_indices) != self.state_dim:
                raise ValueError(
                    f"state_select_indices 长度 ({len(self.state_select_indices)}) "
                    f"与 state_dim ({self.state_dim}) 不一致"
                )
            logger.info(f"State 索引提取: {self.state_select_indices} -> dim={self.state_dim}")

        # ---------- State MinMax 归一化 ----------
        state_normalize_config = config.get("state_normalize", {})
        self.minmax_columns = state_normalize_config.get("columns", [])
        self.minmax_min = None
        self.minmax_max = None

        stats_file = state_normalize_config.get("stats_file", "")
        stats_key = state_normalize_config.get("stats_key", "observation.state")
        if stats_file and os.path.exists(stats_file) and self.minmax_columns:
            try:
                with open(stats_file, "r") as f:
                    stats_data = json.load(f)
                if stats_key in stats_data:
                    state_stats = stats_data[stats_key]
                    if "min" in state_stats and "max" in state_stats:
                        self.minmax_min = np.array(state_stats["min"], dtype=np.float32)
                        self.minmax_max = np.array(state_stats["max"], dtype=np.float32)
                        logger.info(f"State 归一化 stats 加载成功: {stats_file}")
            except Exception as e:
                logger.error(f"加载 state stats 失败: {e}")

        # ---------- State Gripper 二值化 ----------
        gripper_binarize_config = config.get("state_gripper_binarize", {})
        self.gripper_binarize_columns = gripper_binarize_config.get("columns", [])
        self.gripper_binarize_threshold = gripper_binarize_config.get("threshold", 0.5)

        # ---------- 图像预处理 ----------
        image_preprocess = config.get("image_preprocess", {})
        self.image_resize = image_preprocess.get("resize", None)
        self.flip_horizontal = image_preprocess.get("flip_horizontal", False)
        self.flip_vertical = image_preprocess.get("flip_vertical", False)
        self.rotate_180 = image_preprocess.get("rotate_180", False)
        self.bgr_to_rgb = image_preprocess.get("bgr_to_rgb", False)

        # ---------- Action 反归一化 ----------
        action_postprocess = config.get("action_postprocess", {})
        action_denorm_config = action_postprocess.get("action_denormalize", {})
        self.action_denormalize_enabled = action_denorm_config.get("enabled", False)
        self.action_min = None
        self.action_max = None
        if self.action_denormalize_enabled:
            denorm_stats_file = action_denorm_config.get("stats_file", "")
            denorm_stats_key = action_denorm_config.get("stats_key", "action")
            if denorm_stats_file and os.path.exists(denorm_stats_file):
                try:
                    with open(denorm_stats_file, "r") as f:
                        sd = json.load(f)
                    if denorm_stats_key in sd:
                        action_stats = sd[denorm_stats_key]
                        if "min" in action_stats and "max" in action_stats:
                            self.action_min = np.array(action_stats["min"], dtype=np.float32)
                            self.action_max = np.array(action_stats["max"], dtype=np.float32)
                            logger.info(f"Action 反归一化 stats 加载成功: {denorm_stats_file}")
                        else:
                            self.action_denormalize_enabled = False
                    else:
                        self.action_denormalize_enabled = False
                except Exception as e:
                    logger.error(f"加载 action stats 失败: {e}")
                    self.action_denormalize_enabled = False
            else:
                self.action_denormalize_enabled = False

        # ---------- Action Gripper 二值化 ----------
        gripper_out = action_postprocess.get("gripper_binarize", {})
        self.action_gripper_binarize_enabled = gripper_out.get("enabled", False)
        self.action_gripper_binarize_columns = gripper_out.get("columns", [])
        self.action_gripper_binarize_threshold = gripper_out.get("threshold", 0.0)
        self.action_gripper_bce_mode = gripper_out.get("bce_mode", False)

        # ---------- Action 零填充 ----------
        padding_cfg = action_postprocess.get("padding_zeros", {})
        self.padding_zeros_enabled = padding_cfg.get("enabled", False)
        self.padding_zeros_prepend_count = padding_cfg.get("prepend_count", 0)
        self.padding_zeros_insert_before_last = padding_cfg.get("insert_before_last", 0)

        # ---------- Aggregate Sum ----------
        agg_cfg = action_postprocess.get("aggregate_sum", {})
        self.agg_enabled = agg_cfg.get("enabled", False)
        self.agg_horizon = agg_cfg.get("horizon", None)

        if self.agg_enabled:
            h = self.agg_horizon or self.chunk_size
            logger.info(f"Aggregate Sum 已启用: 对前 {h} 步求和为 1 步动作")

        # ---------- ROI 提取 ----------
        roi_cfg = config.get("roi", {})
        self.roi_enabled = roi_cfg.get("enabled", False)
        self.roi_projector: Optional[CameraProjector] = None
        self.roi_margins: Optional[Tuple[int, int, int, int]] = None
        self.roi_target_size = roi_cfg.get("target_size", 256)

        if self.roi_enabled:
            server_dir = Path(__file__).parent
            intrinsics_path = roi_cfg.get("intrinsics_path",
                                          str(server_dir / "preprocessing" / "config" / "intrinsics.ini"))
            extrinsics_path = roi_cfg.get("extrinsics_path",
                                          str(server_dir / "preprocessing" / "config" / "solvepnp_params_from_csv.json"))
            intrinsic_section = roi_cfg.get("intrinsic_section", "ColorIntrinsic")

            if not os.path.exists(intrinsics_path):
                raise FileNotFoundError(f"ROI intrinsics 文件不存在: {intrinsics_path}")
            if not os.path.exists(extrinsics_path):
                raise FileNotFoundError(f"ROI extrinsics 文件不存在: {extrinsics_path}")

            self.roi_projector = CameraProjector(intrinsics_path, extrinsics_path, intrinsic_section)
            logger.info(f"ROI CameraProjector 初始化: intrinsics={intrinsics_path}, extrinsics={extrinsics_path}")

            roi_config_path = roi_cfg.get("config_path",
                                          str(server_dir / "preprocessing" / "config" / "config.json"))
            if not os.path.exists(roi_config_path):
                raise FileNotFoundError(f"ROI config 文件不存在: {roi_config_path}")
            with open(roi_config_path, "r", encoding="utf-8") as f:
                rc = json.load(f)
            p = rc.get("parameters", {})
            ml = p["roi_margin_left"]
            mr = p["roi_margin_right"]
            mt = p["roi_margin_top"]
            mb = p["roi_margin_bottom"]

            hand_side = roi_cfg.get("hand_side", "right")
            if hand_side == "right":
                self.roi_margins = (mr, ml, mt, mb)
            else:
                self.roi_margins = (ml, mr, mt, mb)

            logger.info(f"ROI margins (L,R,T,B): {self.roi_margins}, target_size={self.roi_target_size}, hand_side={hand_side}")

        # ---------- 加载模型 ----------
        self._load_models()

    # ------------------------------------------------------------------ #
    #  模型加载
    # ------------------------------------------------------------------ #

    def _load_models(self):
        cfg = self.config
        vlm_cfg = cfg.get("vlm", {})

        logger.info(f"加载 VLM Backbone: {vlm_cfg.get('model_path')}")
        self.vlm_backbone = create_vlm_backbone(
            model_type=vlm_cfg.get("type", "eagle2_5_vl"),
            model_path=vlm_cfg.get("model_path"),
            device=self.device,
            layers=vlm_cfg.get("layers", [-1]),
            flip_images=vlm_cfg.get("flip_images", False),
            content_order=vlm_cfg.get("content_order", "images_first"),
            prompt_template=vlm_cfg.get("prompt_template", "simple"),
            lowercase_instruction=vlm_cfg.get("lowercase_instruction", True),
            add_generation_prompt=vlm_cfg.get("add_generation_prompt", True),
        )
        logger.info("VLM Backbone 加载完成")

        hidden_dim = cfg.get("hidden_dim", 2048)

        oft_cfg = cfg.get("oft", {})
        oft_checkpoint = oft_cfg.get("checkpoint")
        if not oft_checkpoint or not os.path.exists(oft_checkpoint):
            raise ValueError(f"OFT checkpoint 不存在: {oft_checkpoint}")

        logger.info(f"加载 OFT Pipeline: {oft_checkpoint}")

        num_transformer_blocks = oft_cfg.get("num_transformer_blocks", 2)
        num_attention_heads = oft_cfg.get("num_attention_heads", 8)
        num_vlm_layers = oft_cfg.get("num_vlm_layers", 1)
        action_head_hidden_dim = oft_cfg.get("action_head_hidden_dim", 4096)

        ckpt_config_path = Path(oft_checkpoint).parent / "config.json"
        if ckpt_config_path.exists():
            with open(ckpt_config_path, "r") as f:
                ckpt_cfg = json.load(f)
            self.chunk_size = ckpt_cfg.get("chunk_size", self.chunk_size)
            self.action_dim = ckpt_cfg.get("action_dim", self.action_dim)
            hidden_dim = ckpt_cfg.get("vlm_output_dim", hidden_dim)
            num_transformer_blocks = ckpt_cfg.get("num_transformer_blocks", num_transformer_blocks)
            num_attention_heads = ckpt_cfg.get("num_attention_heads", num_attention_heads)
            num_vlm_layers = ckpt_cfg.get("num_vlm_layers", num_vlm_layers)
            logger.info(
                f"  从 config.json 加载: chunk={self.chunk_size}, action_dim={self.action_dim}, "
                f"hidden={hidden_dim}, blocks={num_transformer_blocks}, heads={num_attention_heads}"
            )

        self.oft_pipeline = create_vlm2oft_pipeline(
            num_transformer_blocks=num_transformer_blocks,
            num_attention_heads=num_attention_heads,
            num_vlm_layers=num_vlm_layers,
            vlm_output_dim=hidden_dim,
            action_head_hidden_dim=action_head_hidden_dim,
        ).to(self.device)

        state_dict = torch.load(oft_checkpoint, map_location=self.device)
        self.oft_pipeline.load_state_dict(state_dict)
        self.oft_pipeline.eval()
        logger.info(f"OFT Pipeline 加载完成 (chunk={self.chunk_size}, action_dim={self.action_dim})")

    # ------------------------------------------------------------------ #
    #  预测
    # ------------------------------------------------------------------ #

    def predict(self, payload: dict) -> dict:
        """
        兼容 robot_control 客户端的预测接口。

        payload 格式 (与 deploy_dataset197_jointangle.py 一致):
        {
            "image_arrays": [[[r,g,b], ...], ...],   # list of HxWx3 uint8
            "state": [float, ...],                    # 原始 state 向量
            "instruction": "pick up the bottle",      # 可选
            "world_point": [x, y, z],                 # 可选 (当前未使用，保留兼容性)
        }

        返回:
        {
            "actions": [[...], ...],  # [chunk_size, final_action_dim]
            "num_actions": int,
            "action_dim": int,
        }
        """
        verbose = self.config.get("verbose", True)
        t_start = time.time()
        self._request_count += 1

        # ---------- 解析输入 ----------
        instruction = payload.get("instruction", self.config.get("default_instruction", ""))
        raw_state = payload.get("state", None)
        image_arrays = payload.get("image_arrays", [])
        world_point = payload.get("world_point", None)

        if not image_arrays:
            return {"error": "image_arrays is empty"}

        # ---------- 图像 ----------
        images: List[Image.Image] = []
        raw_np_images: List[np.ndarray] = []
        for idx, arr in enumerate(image_arrays):
            np_arr = np.array(arr, dtype=np.float32)
            if np_arr.max() <= 1.0:
                np_arr = (np_arr * 255.0)
            np_arr = np.clip(np_arr, 0, 255).astype(np.uint8)
            if self.bgr_to_rgb:
                np_arr = np_arr[:, :, ::-1].copy()
            raw_np_images.append(np_arr.copy())

            # resize 统一用 cv2.INTER_AREA (一次)
            if self.image_resize is not None and len(self.image_resize) == 2:
                target_w, target_h = self.image_resize
                if (np_arr.shape[1], np_arr.shape[0]) != (target_w, target_h):
                    np_arr = cv2.resize(np_arr, (target_w, target_h), interpolation=cv2.INTER_AREA)

            pil_img = Image.fromarray(np_arr, mode="RGB")
            # 只做几何变换 (flip/rotate)，resize 已在上面 cv2 完成
            pil_img = preprocess_image(
                pil_img,
                resize=None,
                flip_horizontal=self.flip_horizontal,
                flip_vertical=self.flip_vertical,
                rotate_180=self.rotate_180,
            )
            images.append(pil_img)
            if verbose:
                logger.info(f"  image[{idx}]: raw={raw_np_images[-1].shape} -> cv2.INTER_AREA -> {pil_img.size}")

        # ---------- ROI 提取 ----------
        roi_np: Optional[np.ndarray] = None
        pixel_point: Optional[Tuple[int, int]] = None
        annotated_np: Optional[np.ndarray] = None

        if self.roi_enabled and self.roi_projector is not None and self.roi_margins is not None:
            if len(image_arrays) == 1 and world_point is not None:
                pixel_point = self.roi_projector.world_to_pixel(world_point)

                # extract_roi 内部用 cv2.INTER_AREA 一次 resize 到模型输入尺寸
                roi_final_size = self.image_resize[0] if self.image_resize else self.roi_target_size
                roi_np = extract_roi(raw_np_images[0], pixel_point, self.roi_margins, roi_final_size)
                annotated_np = _draw_roi_box(raw_np_images[0], pixel_point, self.roi_margins)

                roi_pil = Image.fromarray(roi_np, mode="RGB")
                roi_pil = preprocess_image(
                    roi_pil,
                    resize=None,
                    flip_horizontal=self.flip_horizontal,
                    flip_vertical=self.flip_vertical,
                    rotate_180=self.rotate_180,
                )
                images.append(roi_pil)

                if verbose:
                    logger.info(f"  [ROI] world={world_point} -> pixel={pixel_point}, "
                                f"crop -> cv2.INTER_AREA -> {roi_np.shape}")
                    h_raw, w_raw = raw_np_images[0].shape[:2]
                    if pixel_point[0] < 0 or pixel_point[0] >= w_raw or pixel_point[1] < 0 or pixel_point[1] >= h_raw:
                        logger.warning(f"  [ROI] pixel_point {pixel_point} 超出图像范围 ({w_raw}x{h_raw})!")
                    if roi_np.max() == 0:
                        logger.warning("  [ROI] ROI 图像全黑，world_point 可能投影到了图像外部!")
            elif len(image_arrays) == 1 and world_point is None:
                logger.warning("  [ROI] ROI 已启用但 world_point 为空，跳过 ROI 提取")

        if verbose:
            logger.info(f"  instruction: \"{instruction}\"")
            logger.info(f"  总共输入图像数: {len(images)}")
            if raw_state is not None:
                logger.info(f"  raw state dim={len(raw_state)}")

        # ---------- State 预处理 ----------
        processed_state = None
        if raw_state is not None:
            observation_state = np.array(raw_state, dtype=np.float32)

            if self.state_select_indices is not None:
                observation_state = observation_state[self.state_select_indices]
                if verbose:
                    logger.info(f"  state 索引提取后 dim={len(observation_state)}: {observation_state.tolist()}")

            processed_state = apply_state_preprocessing(
                observation_state.copy(),
                state_process_order=self.state_process_order,
                hand_binary_columns=self.hand_binary_columns,
                hand_binary_threshold=self.hand_binary_threshold,
                minmax_columns=self.minmax_columns,
                minmax_min=self.minmax_min,
                minmax_max=self.minmax_max,
                gripper_binarize_columns=self.gripper_binarize_columns,
                gripper_binarize_threshold=self.gripper_binarize_threshold,
            )
            if verbose:
                logger.info(f"  state 处理后 dim={len(processed_state)}: {processed_state.tolist()}")

        # ---------- VLM 前向 ----------
        t_vlm = time.time()
        vlm_output = self.vlm_backbone.get_hidden_states(
            images=images,
            instruction=instruction,
        )
        vlm_hidden_states = vlm_output.hidden_states
        vlm_time = time.time() - t_vlm

        if verbose:
            total_tokens = sum(h.shape[1] for h in vlm_hidden_states)
            logger.info(f"  VLM: {len(vlm_hidden_states)} layers, {total_tokens} tokens, {vlm_time:.3f}s")

        # ---------- OFT Pipeline 前向 ----------
        with torch.no_grad():
            hs = [v.to(self.device) for v in vlm_hidden_states]

            if self.use_proprio and processed_state is not None:
                proprio = processed_state[: self.proprio_dim]
                proprio_tensor = torch.tensor(proprio, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                proprio_tensor = torch.zeros(1, self.proprio_dim, dtype=torch.float32, device=self.device)

            t_oft = time.time()
            action_preds = self.oft_pipeline(hs, proprio_tensor)
            oft_time = time.time() - t_oft

        # shape: (1, 1, chunk_size * action_dim)
        raw_actions = action_preds[0, 0].cpu().numpy()
        actions = raw_actions.reshape(self.chunk_size, self.action_dim)

        if verbose:
            logger.info(f"  OFT: shape={actions.shape}, {oft_time:.3f}s")

        # ---------- 后处理: 反归一化 ----------
        if self.action_denormalize_enabled and self.action_min is not None and self.action_max is not None:
            d = min(self.action_dim, len(self.action_min), len(self.action_max))
            bce_skip = set()
            if self.action_gripper_bce_mode and self.action_gripper_binarize_enabled:
                bce_skip = set(self.action_gripper_binarize_columns)
            cols = [c for c in range(d) if c not in bce_skip]
            if cols:
                ca = np.array(cols)
                actions[:, ca] = (
                    (actions[:, ca] + 1.0) / 2.0
                    * (self.action_max[ca] - self.action_min[ca])
                    + self.action_min[ca]
                )

        # ---------- 后处理: Gripper 二值化 ----------
        if self.action_gripper_binarize_enabled and self.action_gripper_binarize_columns:
            for col in self.action_gripper_binarize_columns:
                if col < self.action_dim:
                    if self.action_gripper_bce_mode:
                        prob = 1.0 / (1.0 + np.exp(-actions[:, col]))
                        actions[:, col] = np.where(prob > self.action_gripper_binarize_threshold, 1.0, 0.0)
                    else:
                        actions[:, col] = np.where(
                            actions[:, col] > self.action_gripper_binarize_threshold, 1.0, 0.0,
                        )

        # ---------- 后处理: 零填充 ----------
        if self.padding_zeros_enabled:
            cs, ad = actions.shape
            if self.padding_zeros_prepend_count > 0:
                actions = np.concatenate(
                    [np.zeros((cs, self.padding_zeros_prepend_count), dtype=np.float32), actions],
                    axis=1,
                )
            if self.padding_zeros_insert_before_last > 0:
                body = actions[:, :-1]
                tail = actions[:, -1:]
                zeros = np.zeros((cs, self.padding_zeros_insert_before_last), dtype=np.float32)
                actions = np.concatenate([body, zeros, tail], axis=1)

        # ---------- 后处理: Aggregate Sum 或 execute_horizon 截断 ----------
        if self.agg_enabled:
            h = self.agg_horizon or len(actions)
            actions = np.sum(actions[:h], axis=0, keepdims=True)  # (1, action_dim)
        elif self.execute_horizon is not None and self.execute_horizon < len(actions):
            actions = actions[: self.execute_horizon]

        total_time = time.time() - t_start

        if verbose:
            logger.info(f"  最终 actions shape={actions.shape}, 总耗时 {total_time:.3f}s")
            if self.agg_enabled:
                h = self.agg_horizon or self.chunk_size
                logger.info(f"  aggregate_sum: 前 {h} 步求和 -> 1 步动作")
            elif self.execute_horizon is not None:
                logger.info(f"  execute_horizon={self.execute_horizon} (从 chunk_size={self.chunk_size} 中截取)")
            for i in range(min(3, len(actions))):
                logger.info(f"    step {i}: {actions[i].tolist()}")
            if self.action_gripper_binarize_enabled and self.action_gripper_binarize_columns:
                final_dim = actions.shape[-1]
                pad_offset = (self.padding_zeros_prepend_count if self.padding_zeros_enabled else 0)
                for orig_col in self.action_gripper_binarize_columns:
                    col = orig_col + pad_offset
                    if col < final_dim:
                        grip_vals = actions[:, col]
                        n_close = int((grip_vals == 1.0).sum())
                        n_open = int((grip_vals == 0.0).sum())
                        first_close = int(np.argmax(grip_vals == 1.0)) if n_close > 0 else -1
                        logger.info(
                            f"  [Gripper] col={col}(原col={orig_col}): "
                            f"open={n_open}, close={n_close}/{len(grip_vals)}, "
                            f"首次闭合step={first_close if first_close >= 0 else 'N/A'}"
                        )

        # ---------- 保存 debug 快照 ----------
        self._debug = {
            "ts": time.time(),
            "request_id": self._request_count,
            "raw_image_b64": _np_to_base64_jpeg(raw_np_images[0]) if raw_np_images else None,
            "processed_image_b64": _pil_to_base64_jpeg(images[0]) if images else None,
            "processed_image2_b64": _pil_to_base64_jpeg(images[1]) if len(images) > 1 else None,
            "roi_image_b64": _np_to_base64_jpeg(roi_np) if roi_np is not None else None,
            "annotated_image_b64": _np_to_base64_jpeg(annotated_np) if annotated_np is not None else None,
            "raw_image_shape": list(raw_np_images[0].shape) if raw_np_images else None,
            "processed_image_size": list(images[0].size) if images else None,
            "processed_image2_size": list(images[1].size) if len(images) > 1 else None,
            "roi_shape": list(roi_np.shape) if roi_np is not None else None,
            "pixel_point": list(pixel_point) if pixel_point else None,
            "roi_enabled": self.roi_enabled,
            "num_images": len(images),
            "instruction": instruction,
            "world_point": world_point,
            "raw_state": [round(float(v), 6) for v in raw_state] if raw_state else None,
            "processed_state": [round(float(v), 6) for v in processed_state] if processed_state is not None else None,
            "actions": actions.tolist(),
            "action_shape": list(actions.shape),
            "timing": {
                "vlm_time": round(vlm_time, 4),
                "oft_time": round(oft_time, 4),
                "total_time": round(total_time, 4),
            },
        }

        return {
            "actions": actions.tolist(),
            "num_actions": len(actions),
            "action_dim": int(actions.shape[-1]),
        }


# ==================== FastAPI 应用 ====================

app = FastAPI(title="OFT Inference Server (robot_control compatible)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: Optional[OFTInferenceEngine] = None
_config: Dict[str, Any] = {}


@app.post("/predict")
async def predict_endpoint(payload: dict):
    if engine is None:
        return {"error": "model not loaded"}
    try:
        return engine.predict(payload)
    except Exception as e:
        logger.error(traceback.format_exc())
        return {"error": str(e)}


@app.post("/act")
async def act_endpoint(payload: dict):
    return await predict_endpoint(payload)


@app.get("/health")
def health():
    if engine is None:
        return {"status": "not_ready"}
    info = {
        "status": "ready",
        "chunk_size": engine.chunk_size,
        "execute_horizon": engine.execute_horizon,
        "action_dim": engine.action_dim,
        "proprio_dim": engine.proprio_dim,
        "device": engine.device,
        "roi_enabled": engine.roi_enabled,
        "aggregate_sum_enabled": engine.agg_enabled,
    }
    if engine.roi_enabled:
        info["roi_margins"] = engine.roi_margins
        info["roi_target_size"] = engine.roi_target_size
    return info


@app.get("/info")
def info():
    if engine is None:
        return {"status": "not_ready"}
    vlm_cfg = _config.get("vlm", {})
    return {
        "vlm_type": vlm_cfg.get("type", "unknown"),
        "vlm_model_path": vlm_cfg.get("model_path", "unknown"),
        "chunk_size": engine.chunk_size,
        "execute_horizon": engine.execute_horizon,
        "action_dim": engine.action_dim,
        "proprio_dim": engine.proprio_dim,
        "state_dim": engine.state_dim,
        "device": engine.device,
    }


# ==================== Debug 监控面板 ====================

@app.get("/debug/data")
def debug_data():
    if engine is None:
        return {"ts": 0, "message": "model not loaded"}
    if not engine._debug:
        return {"ts": 0, "message": "no prediction yet"}
    resp = dict(engine._debug)
    resp["total_requests"] = engine._request_count
    resp["uptime_s"] = round(time.time() - engine._start_time, 1)
    return resp


@app.get("/debug", response_class=HTMLResponse)
def debug_page():
    return _DEBUG_HTML


_DEBUG_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OFT Debug Monitor</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--dim:#8b949e;
  --blue:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922;--cyan:#79c0ff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'SF Mono','Cascadia Code','Consolas',monospace;background:var(--bg);color:var(--text);padding:16px;font-size:13px}
h1{color:var(--blue);font-size:20px;margin-bottom:4px}
.sub{color:var(--dim);font-size:12px;margin-bottom:14px}
.bar{padding:8px 14px;border-radius:6px;margin-bottom:14px;font-size:12px;display:flex;justify-content:space-between;align-items:center}
.bar.ok{background:#0d2818;border:1px solid #238636}
.bar.wait{background:#2d1b00;border:1px solid var(--yellow)}
.bar.err{background:#2d0a0a;border:1px solid var(--red)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px;overflow:hidden}
.card h3{color:var(--cyan);font-size:13px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}
.card h3 span{color:var(--dim);font-weight:normal;font-size:11px}
.card img{max-width:100%;height:auto;border-radius:4px;display:block}
.full-width{grid-column:1/-1}
table{width:100%;border-collapse:collapse;font-size:12px}
td,th{padding:4px 8px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--dim);font-weight:normal;white-space:nowrap}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;margin:1px 2px}
.tg{background:#0d2818;color:var(--green)}.tr{background:#2d0a0a;color:var(--red)}
.ty{background:#2d1b00;color:var(--yellow)}.tb{background:#0d1830;color:var(--blue)}
.mono{font-family:inherit}
.actions-scroll{max-height:400px;overflow-y:auto;margin-top:6px}
.actions-scroll table{font-size:11px}
.actions-scroll td{white-space:nowrap}
.timing-bar{display:flex;gap:6px;margin-top:8px;flex-wrap:wrap}
.timing-bar .seg{padding:6px 12px;border-radius:4px;text-align:center;flex:1;min-width:100px}
.timing-bar .seg .val{font-size:18px;font-weight:bold;margin-bottom:2px}
.timing-bar .seg .lbl{font-size:10px;color:var(--dim)}
.state-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(80px,1fr));gap:4px;margin-top:6px}
.state-cell{background:var(--bg);border:1px solid var(--border);border-radius:4px;padding:4px 6px;text-align:center;font-size:11px}
.state-cell .idx{color:var(--dim);font-size:9px;display:block}
.state-cell .val{color:var(--text);font-weight:bold}
</style>
</head><body>
<h1>OFT Debug Monitor</h1>
<p class="sub">VLM + OFT Pipeline Real-time Inference Viewer</p>

<div id="status_bar" class="bar wait">Waiting for first prediction...</div>

<div class="grid">
  <!-- 原始图像 -->
  <div class="card">
    <h3>Raw Image <span id="raw_dim"></span></h3>
    <img id="img_raw" alt="waiting...">
  </div>

  <!-- 处理后图像 1 -->
  <div class="card">
    <h3>Model Input Image 1 <span id="proc_dim"></span></h3>
    <img id="img_proc" alt="waiting...">
  </div>

  <!-- 处理后图像 2 (ROI / 第二张输入) -->
  <div class="card" id="card_proc2" style="display:none">
    <h3>Model Input Image 2 <span id="proc2_dim"></span></h3>
    <img id="img_proc2" alt="">
  </div>

  <!-- ROI 标注图 -->
  <div class="card" id="card_annotated" style="display:none">
    <h3>ROI Annotated <span id="roi_pixel"></span></h3>
    <img id="img_annotated" alt="">
  </div>

  <!-- ROI 裁剪图 -->
  <div class="card" id="card_roi" style="display:none">
    <h3>ROI Crop <span id="roi_dim"></span></h3>
    <img id="img_roi" alt="">
  </div>

  <!-- 耗时 -->
  <div class="card full-width">
    <h3>Timing</h3>
    <div class="timing-bar">
      <div class="seg" style="background:#0d2818;border:1px solid #238636">
        <div class="val" id="t_vlm">-</div><div class="lbl">VLM (s)</div>
      </div>
      <div class="seg" style="background:#0d1830;border:1px solid #1f6feb">
        <div class="val" id="t_oft">-</div><div class="lbl">OFT (s)</div>
      </div>
      <div class="seg" style="background:#2d1b00;border:1px solid #d29922">
        <div class="val" id="t_total">-</div><div class="lbl">Total (s)</div>
      </div>
      <div class="seg" style="background:#1a1a2e;border:1px solid #6e40c9">
        <div class="val" id="t_hz">-</div><div class="lbl">Hz</div>
      </div>
    </div>
  </div>

  <!-- Metadata -->
  <div class="card">
    <h3>Request Info</h3>
    <table>
      <tr><th>Request #</th><td id="m_reqid">-</td></tr>
      <tr><th>Total Requests</th><td id="m_total">-</td></tr>
      <tr><th>Uptime</th><td id="m_uptime">-</td></tr>
      <tr><th>Instruction</th><td id="m_instr">-</td></tr>
      <tr><th>World Point</th><td id="m_wp">-</td></tr>
      <tr><th>Images</th><td id="m_nimgs">-</td></tr>
      <tr><th>Action Shape</th><td id="m_ashape">-</td></tr>
    </table>
  </div>

  <!-- State -->
  <div class="card">
    <h3>State <span id="state_info"></span></h3>
    <div style="margin-bottom:8px">
      <b style="color:var(--yellow);font-size:11px">Raw State</b>
      <div id="raw_state" class="state-grid"></div>
    </div>
    <div>
      <b style="color:var(--green);font-size:11px">Processed State (model input)</b>
      <div id="proc_state" class="state-grid"></div>
    </div>
  </div>

  <!-- Actions -->
  <div class="card full-width">
    <h3>Action Predictions <span id="action_info"></span></h3>
    <div id="actions_table" class="actions-scroll"></div>
  </div>
</div>

<script>
let prevTs=0;
function fmt(v,d=4){return typeof v==='number'?v.toFixed(d):'-'}
function renderState(containerId,arr){
  const el=document.getElementById(containerId);
  if(!arr||!arr.length){el.innerHTML='<span style="color:var(--dim)">N/A</span>';return;}
  el.innerHTML=arr.map((v,i)=>`<div class="state-cell"><span class="idx">[${i}]</span><span class="val">${fmt(v,4)}</span></div>`).join('');
}
function renderActions(d){
  const el=document.getElementById('actions_table');
  if(!d.actions||!d.actions.length){el.innerHTML='<span style="color:var(--dim)">No actions</span>';return;}
  const acts=d.actions;
  const dim=acts[0].length;
  let hdr='<tr><th>Step</th>';
  for(let j=0;j<dim;j++) hdr+=`<th>d${j}</th>`;
  hdr+='</tr>';
  let rows='';
  for(let i=0;i<acts.length;i++){
    rows+=`<tr><td style="color:var(--dim)">${i}</td>`;
    for(let j=0;j<dim;j++){
      const v=acts[i][j];
      const abs=Math.abs(v);
      let cls='';
      if(abs>0.5)cls=' style="color:var(--yellow);font-weight:bold"';
      else if(abs<0.001)cls=' style="color:var(--dim)"';
      rows+=`<td${cls}>${v.toFixed(6)}</td>`;
    }
    rows+='</tr>';
  }
  el.innerHTML=`<table>${hdr}${rows}</table>`;
}
async function poll(){
  try{
    const r=await fetch('/debug/data');
    const d=await r.json();
    const bar=document.getElementById('status_bar');
    if(!d.ts||d.ts===0){
      bar.className='bar wait';bar.textContent=d.message||'Waiting for first prediction...';return;
    }
    if(d.ts===prevTs)return;
    prevTs=d.ts;
    const ago=((Date.now()/1000-d.ts)).toFixed(1);
    bar.className='bar ok';
    bar.innerHTML=`<span>Request #${d.request_id} — ${ago}s ago</span><span class="tag tg">LIVE</span>`;

    if(d.raw_image_b64) document.getElementById('img_raw').src='data:image/jpeg;base64,'+d.raw_image_b64;
    if(d.processed_image_b64) document.getElementById('img_proc').src='data:image/jpeg;base64,'+d.processed_image_b64;

    document.getElementById('raw_dim').textContent=d.raw_image_shape?`${d.raw_image_shape[1]}x${d.raw_image_shape[0]}`:'';
    document.getElementById('proc_dim').textContent=d.processed_image_size?`${d.processed_image_size[0]}x${d.processed_image_size[1]}`:'';

    const cProc2=document.getElementById('card_proc2');
    if(d.processed_image2_b64){
      cProc2.style.display='';
      document.getElementById('img_proc2').src='data:image/jpeg;base64,'+d.processed_image2_b64;
      document.getElementById('proc2_dim').textContent=d.processed_image2_size?`${d.processed_image2_size[0]}x${d.processed_image2_size[1]}`:'';
    }else{cProc2.style.display='none';}

    const cAnnotated=document.getElementById('card_annotated');
    const cRoi=document.getElementById('card_roi');
    if(d.annotated_image_b64){
      cAnnotated.style.display='';
      document.getElementById('img_annotated').src='data:image/jpeg;base64,'+d.annotated_image_b64;
      document.getElementById('roi_pixel').textContent=d.pixel_point?`pixel=(${d.pixel_point[0]},${d.pixel_point[1]})`:'';
    }else{cAnnotated.style.display='none';}
    if(d.roi_image_b64){
      cRoi.style.display='';
      document.getElementById('img_roi').src='data:image/jpeg;base64,'+d.roi_image_b64;
      document.getElementById('roi_dim').textContent=d.roi_shape?`${d.roi_shape[1]}x${d.roi_shape[0]}`:'';
    }else{cRoi.style.display='none';}

    if(d.timing){
      document.getElementById('t_vlm').textContent=fmt(d.timing.vlm_time,3);
      document.getElementById('t_oft').textContent=fmt(d.timing.oft_time,3);
      document.getElementById('t_total').textContent=fmt(d.timing.total_time,3);
      const hz=d.timing.total_time>0?(1.0/d.timing.total_time):0;
      document.getElementById('t_hz').textContent=fmt(hz,1);
    }

    document.getElementById('m_reqid').textContent=d.request_id||'-';
    document.getElementById('m_total').textContent=d.total_requests||'-';
    document.getElementById('m_uptime').textContent=d.uptime_s?`${d.uptime_s}s`:'?';
    document.getElementById('m_instr').textContent=d.instruction||'-';
    document.getElementById('m_wp').textContent=d.world_point?d.world_point.map(v=>v.toFixed(2)).join(', '):'N/A';
    document.getElementById('m_nimgs').textContent=d.num_images||'-';
    document.getElementById('m_ashape').textContent=d.action_shape?`[${d.action_shape.join(', ')}]`:'-';

    const rawLen=d.raw_state?d.raw_state.length:0;
    const procLen=d.processed_state?d.processed_state.length:0;
    document.getElementById('state_info').textContent=`raw=${rawLen}d → model=${procLen}d`;

    renderState('raw_state',d.raw_state);
    renderState('proc_state',d.processed_state);

    document.getElementById('action_info').textContent=d.action_shape?`${d.action_shape[0]} steps x ${d.action_shape[1]} dims`:'';
    renderActions(d);
  }catch(e){
    document.getElementById('status_bar').className='bar err';
    document.getElementById('status_bar').textContent='Error: '+e;
  }
}
setInterval(poll,1500);poll();
</script>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="OFT Inference Server (robot_control compatible)")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--host", type=str, default=None, help="监听地址 (覆盖配置文件)")
    parser.add_argument("--port", type=int, default=None, help="监听端口 (覆盖配置文件)")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="离线模式: 设置 TRANSFORMERS_OFFLINE=1, HF_HUB_OFFLINE=1, 跳过图像 resize (数据集图像直接喂给模型)",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("已启用离线模式 (跳过图像 resize，数据集图像直接喂给模型)")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.offline:
        cfg.setdefault("image_preprocess", {})["resize"] = None
        logger.info("  offline: image_preprocess.resize 已设为 None")

    host = args.host or cfg.get("host", "0.0.0.0")
    port = args.port or cfg.get("port", 8000)

    global engine, _config
    _config = cfg
    logger.info(f"加载配置: {args.config}")
    engine = OFTInferenceEngine(_config)

    final_dim = engine.action_dim
    if engine.padding_zeros_enabled:
        final_dim += engine.padding_zeros_prepend_count
        if engine.padding_zeros_insert_before_last > 0:
            final_dim += engine.padding_zeros_insert_before_last

    logger.info("=" * 60)
    logger.info(f"服务就绪: http://{host}:{port}")
    logger.info(f"  POST /predict  - 推理 (兼容 robot_control)")
    logger.info(f"  POST /act      - /predict 别名")
    logger.info(f"  GET  /health   - 健康检查")
    logger.info(f"  GET  /info     - 模型信息")
    logger.info(f"  GET  /debug    - 实时监控面板")
    logger.info(f"  chunk_size     = {engine.chunk_size}")
    logger.info(f"  execute_horizon= {engine.execute_horizon or 'all (=chunk_size)'}")
    logger.info(f"  model action_dim = {engine.action_dim}")
    logger.info(f"  final action_dim = {final_dim} (after padding)")
    logger.info(f"  proprio_dim    = {engine.proprio_dim}")
    logger.info(f"  use_proprio    = {engine.use_proprio}")
    logger.info(f"  state_dim      = {engine.state_dim}")
    logger.info(f"  image_resize   = {engine.image_resize}")
    logger.info(f"  roi_enabled    = {engine.roi_enabled}")
    if engine.roi_enabled:
        logger.info(f"  roi_margins    = {engine.roi_margins}")
        logger.info(f"  roi_target_size= {engine.roi_target_size}")
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
