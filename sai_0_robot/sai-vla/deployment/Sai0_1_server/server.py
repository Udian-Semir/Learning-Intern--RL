"""
Sai0_1 VLA 推理服务器
基于 FastAPI 的高性能推理服务，支持：
- 图像预处理：resize、翻转
- 状态预处理：零值转换、最大最小值归一化

Author: Sai0 Team
"""

import os
import sys
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import base64
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yaml

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from VLAs.Sai0_1 import (
    Sai0Inference,
    Sai0Config,
    VLMConfig,
    ActionHeadConfig,
    DataConfig,
    RealtimeInference,
)
from VLAs.Sai0_1.sai0_model import Sai0Model
from VLAs.Sai0_1.data_utils import load_normalization_stats
from deployment.Sai0_1_server.auth import get_api_key, extract_api_key_for_ratelimit
from deployment.Sai0_1_server.queue_worker import InferenceQueue

# 配置日志（同时输出到终端和文件）
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_log_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_fmt)

_file_handler = logging.FileHandler(_LOG_DIR / "server.log", encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_log_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_console_handler, _file_handler])
logger = logging.getLogger(__name__)


# ==================== 配置类 ====================

class ImagePreprocessConfig(BaseModel):
    """图像预处理配置"""
    resize: Optional[List[int]] = Field(None, description="resize 尺寸 [width, height]，None 表示不 resize")
    flip_horizontal: bool = Field(False, description="是否水平翻转")
    flip_vertical: bool = Field(False, description="是否垂直翻转")
    rotate_180: bool = Field(False, description="是否旋转 180 度")


class StateIndexConfig(BaseModel):
    """单个状态索引的配置"""
    enable_normalization: bool = Field(False, description="是否启用归一化")
    min_val: Optional[float] = Field(None, description="归一化最小值")
    max_val: Optional[float] = Field(None, description="归一化最大值")
    zero_to_minus_one: bool = Field(False, description="是否将 0 转换为 -1")


class StatePreprocessConfig(BaseModel):
    """状态预处理配置"""
    # 格式: {索引: StateIndexConfig}
    index_configs: Dict[int, StateIndexConfig] = Field(
        default_factory=dict,
        description="每个索引的配置"
    )


# ==================== 请求/响应模型 ====================

class PredictRequest(BaseModel):
    """预测请求"""
    images: List[Union[str, List]] = Field(..., description="Base64 编码的图像列表或 numpy array (JSON格式)")
    state: List[float] = Field(..., description="机器人状态向量")
    prompt: Optional[str] = Field(None, description="可选的任务指令/prompt")
    image_format: str = Field("base64", description="图像格式: 'base64' 或 'numpy'")
    
    # 图像预处理参数（可选，覆盖全局配置）
    image_resize: Optional[List[int]] = Field(None, description="图像 resize 尺寸 [width, height]")
    image_flip_horizontal: Optional[bool] = Field(None, description="是否水平翻转")
    image_flip_vertical: Optional[bool] = Field(None, description="是否垂直翻转")
    image_rotate_180: Optional[bool] = Field(None, description="是否旋转 180 度")


class TimingInfo(BaseModel):
    """时间统计"""
    preprocess_time: float = Field(..., description="预处理时间 (秒)")
    inference_time: float = Field(..., description="推理时间 (秒)")
    total_time: float = Field(..., description="总时间 (秒)")


class PredictMetadata(BaseModel):
    """预测元数据"""
    num_images: int
    state_dim: int
    action_shape: List[int]
    preprocessed_state: Optional[List[float]] = None


class PredictResponse(BaseModel):
    """预测响应"""
    actions: List  # 支持多种形状
    timing: TimingInfo
    metadata: PredictMetadata


class BatchPredictRequest(BaseModel):
    """批量预测请求"""
    batch: List[PredictRequest]


class BatchPredictResponse(BaseModel):
    """批量预测响应"""
    results: List[PredictResponse]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    pipeline_loaded: bool
    config: Dict[str, Any]


class InfoResponse(BaseModel):
    """模型信息响应"""
    vlm_type: str
    action_head_type: str
    device: str
    image_preprocess: Dict[str, Any]
    state_preprocess: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    image_preprocess: Optional[ImagePreprocessConfig] = None
    state_preprocess: Optional[Dict[int, StateIndexConfig]] = None


# ==================== 全局变量 ====================

# 推理器（默认，兼容旧端点 /predict 等）
inference_engine: Optional[RealtimeInference] = None

# 按 task suite 索引的推理引擎（/v1/act 按 task_suite 路由）
inference_engines: Dict[str, RealtimeInference] = {}

# task_suites 配置原始字典，用于区分 "未配置" 和 "权重为 null"
_task_suites_cfg: Dict[str, Any] = {}

# 配置
server_config: Dict[str, Any] = {}
image_preprocess_config: ImagePreprocessConfig = ImagePreprocessConfig()
state_preprocess_config: StatePreprocessConfig = StatePreprocessConfig()

# GPU 推理队列（用于 /v1/act 并发控制）
inference_queue: Optional[InferenceQueue] = None

# 版本信息
SERVER_VERSION = "0.1.0"
MODEL_NAME = "sai0-vla"
API_VERSION = "v1"

# ==================== 限流器 ====================

def _key_func(request: Request) -> str:
    """按 API Key 或 IP 限流"""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return get_remote_address(request)

limiter = Limiter(key_func=_key_func)


# ==================== Lifespan ====================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application: FastAPI):
    # ---- startup ----
    global inference_queue
    if server_config.get("queue", {}).get("enabled", True):
        maxsize = server_config.get("queue", {}).get("maxsize", 32)
        inference_queue = InferenceQueue(maxsize=maxsize)
        await inference_queue.start()
    yield
    # ---- shutdown ----
    v1_metrics.flush()
    logger.info("Usage stats flushed to disk")
    if inference_queue is not None:
        await inference_queue.stop()


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="Sai0_1 VLA Inference Server",
    description="""
    Sai0_1 视觉-语言-动作推理服务器
    
    功能特性：
    - 支持多种 VLM backbone (Qwen3-VL, Eagle 2.5)
    - 支持多种 Action Head (Flow Matching, OFT)
    - 图像预处理：resize, 翻转
    - 状态预处理：零值转换, 最大最小值归一化
    """,
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 预处理函数 ====================

def preprocess_image(
    image: Image.Image,
    config: ImagePreprocessConfig,
    request_overrides: Dict[str, Any] = None
) -> Image.Image:
    """
    预处理图像
    
    Args:
        image: PIL Image
        config: 图像预处理配置
        request_overrides: 请求级别的覆盖配置
    
    Returns:
        预处理后的 PIL Image
    """
    # 合并配置
    resize = request_overrides.get('resize') if request_overrides else None
    if resize is None:
        resize = config.resize
    
    flip_h = request_overrides.get('flip_horizontal') if request_overrides else None
    if flip_h is None:
        flip_h = config.flip_horizontal
    
    flip_v = request_overrides.get('flip_vertical') if request_overrides else None
    if flip_v is None:
        flip_v = config.flip_vertical
    
    rotate_180 = request_overrides.get('rotate_180') if request_overrides else None
    if rotate_180 is None:
        rotate_180 = config.rotate_180
    
    # Resize
    if resize is not None and len(resize) == 2:
        image = image.resize((resize[0], resize[1]), Image.Resampling.LANCZOS)
    
    # 翻转
    if flip_h:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    if flip_v:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    # 旋转 180 度
    if rotate_180:
        image = image.rotate(180)
    
    return image


def preprocess_state(
    state: np.ndarray,
    config: StatePreprocessConfig
) -> np.ndarray:
    """
    预处理状态向量
    
    Args:
        state: 状态向量 (state_dim,)
        config: 状态预处理配置
    
    Returns:
        预处理后的状态向量
    """
    state = state.copy()
    
    for idx, idx_config in config.index_configs.items():
        if idx >= len(state):
            continue
        
        # 零值转换为 -1
        if idx_config.zero_to_minus_one:
            if state[idx] == 0:
                state[idx] = -1
        
        # 最大最小值归一化
        if idx_config.enable_normalization:
            if idx_config.min_val is not None and idx_config.max_val is not None:
                min_val = idx_config.min_val
                max_val = idx_config.max_val
                
                # 避免除零
                if max_val != min_val:
                    # 归一化到 [0, 1]
                    state[idx] = (state[idx] - min_val) / (max_val - min_val)
                    # 归一化到 [-1, 1]
                    state[idx] = state[idx] * 2 - 1
    
    return state


# ==================== 图像编解码 ====================

def decode_image(image_str: str) -> Image.Image:
    """从 base64 解码图像"""
    if ',' in image_str:
        image_str = image_str.split(',', 1)[1]
    
    image_data = base64.b64decode(image_str)
    image = Image.open(BytesIO(image_data)).convert('RGB')
    return image


def decode_numpy_image(image_data: List) -> Image.Image:
    """从 numpy array (JSON 格式) 解码图像"""
    img_array = np.array(image_data, dtype=np.uint8)
    
    if img_array.ndim != 3:
        raise ValueError(f"Invalid image dimensions: {img_array.ndim}. Expected 3D array (H, W, C)")
    
    num_channels = img_array.shape[2]
    
    if num_channels == 3:
        image = Image.fromarray(img_array, mode='RGB')
    elif num_channels == 4:
        image = Image.fromarray(img_array, mode='RGBA').convert('RGB')
    elif num_channels == 1:
        image = Image.fromarray(img_array[:, :, 0], mode='L').convert('RGB')
    else:
        raise ValueError(f"Invalid number of channels: {num_channels}")
    
    return image


def process_images(
    image_inputs: List[Union[str, List]],
    image_format: str,
    config: ImagePreprocessConfig,
    request_overrides: Dict[str, Any] = None
) -> List[Image.Image]:
    """处理图像输入列表"""
    images = []
    
    for img_input in image_inputs:
        # 解码
        if image_format == "base64":
            if not isinstance(img_input, str):
                raise ValueError(f"Expected string for base64 format, got {type(img_input)}")
            image = decode_image(img_input)
        elif image_format == "numpy":
            if not isinstance(img_input, list):
                raise ValueError(f"Expected list for numpy format, got {type(img_input)}")
            image = decode_numpy_image(img_input)
        else:
            raise ValueError(f"Unsupported image format: {image_format}")
        
        # 预处理
        image = preprocess_image(image, config, request_overrides)
        images.append(image)
    
    return images


# ==================== 初始化函数 ====================

def _get_gpu_memory_mb(device: str = "cuda:0") -> float:
    """Return current GPU memory allocated in MB for the given device."""
    if not torch.cuda.is_available():
        return 0.0
    idx = int(device.split(":")[-1]) if ":" in device else 0
    return torch.cuda.memory_allocated(idx) / (1024 * 1024)


def _build_vlm_config(pipeline_cfg: Dict[str, Any]) -> VLMConfig:
    """从 pipeline 配置构建共享的 VLMConfig。"""
    prompt_template = pipeline_cfg.get('prompt_template', None)
    if prompt_template is None:
        if pipeline_cfg.get('add_action_prompt', False):
            prompt_template = "What action should the robot take to {instruction}?"
        else:
            prompt_template = "{instruction}"

    return VLMConfig(
        model_type=pipeline_cfg.get('vlm_type', 'qwen3_vl'),
        model_path=pipeline_cfg.get('vlm_model_path'),
        device=pipeline_cfg.get('device', 'cuda:0'),
        layers=pipeline_cfg.get('vlm_layers', [14]),
        flip_images=pipeline_cfg.get('flip_images', True),
        content_order=pipeline_cfg.get('content_order', 'images_first'),
        lowercase_instruction=pipeline_cfg.get('lowercase_instruction', True),
        add_generation_prompt=pipeline_cfg.get('add_generation_prompt', True),
        prompt_template=prompt_template,
    )


def _build_suite_engine(
    suite_name: str,
    ckpt_path: str,
    dataset_path: str,
    vlm_config: VLMConfig,
    action_head_type: str,
    ah_cfg: Dict[str, Any],
    shared_vlm,
    device: str,
    warmup_steps: int,
) -> tuple:
    """
    为单个 task suite 构建 RealtimeInference 引擎。

    返回 (engine, vlm_backbone) — vlm_backbone 用于后续 suite 共享。
    """
    action_head_config = ActionHeadConfig(
        head_type=action_head_type,
        pretrained_weights=ckpt_path,
        **ah_cfg
    )
    data_config = DataConfig(dataset_path=dataset_path) if dataset_path else DataConfig()
    sai0_config = Sai0Config(
        vlm=vlm_config,
        action_head=action_head_config,
        data=data_config,
    )

    model = Sai0Model(config=sai0_config, vlm_backbone=shared_vlm, device=device)

    if shared_vlm is None:
        mem_before_vlm = _get_gpu_memory_mb(device)
        _ = model.vlm_backbone
        shared_vlm = model._vlm_backbone
        mem_after_vlm = _get_gpu_memory_mb(device)
        vlm_mb = mem_after_vlm - mem_before_vlm
        logger.info(
            f"  VLM backbone loaded (shared) — "
            f"VRAM: {vlm_mb:.1f} MB ({vlm_mb / 1024:.2f} GB)"
        )

    mem_before_ah = _get_gpu_memory_mb(device)
    _ = model.action_head
    model.eval()
    mem_after_ah = _get_gpu_memory_mb(device)
    ah_mb = mem_after_ah - mem_before_ah
    logger.info(
        f"  [{suite_name}] Action Head loaded — "
        f"VRAM: {ah_mb:.1f} MB ({ah_mb / 1024:.2f} GB)"
    )

    normalizers = None
    if dataset_path:
        normalizers = load_normalization_stats(
            dataset_path,
            convert_quat_to_axisangle=action_head_config.convert_quat_to_axisangle,
        )

    base_inference = Sai0Inference(
        model=model, config=sai0_config, normalizers=normalizers, device=device,
    )
    engine = RealtimeInference(inference=base_inference, warmup_steps=warmup_steps)

    logger.info(f"  [{suite_name}] 引擎已创建  ckpt={ckpt_path}")
    return engine, shared_vlm


def init_inference_engine(config: Dict[str, Any]):
    """
    初始化推理引擎。

    支持两种模式：
    1. 多 task suite 模式（配置了 pipeline.task_suites）：为每个非 null 的
       suite 创建独立 Action Head + normalizers，共享 VLM backbone。
    2. 单引擎模式（未配置 task_suites）：向后兼容，使用顶层
       action_head_ckpt / dataset_path。
    """
    global inference_engine, inference_engines, _task_suites_cfg
    global server_config, image_preprocess_config, state_preprocess_config

    logger.info("=" * 60)
    logger.info("初始化 Sai0_1 VLA 推理引擎")
    logger.info("=" * 60)

    pipeline_cfg = config.get('pipeline', {})
    device = pipeline_cfg.get('device', 'cuda:0')
    action_head_type = pipeline_cfg.get('action_head_type', 'flow_matching_1')
    ah_cfg = pipeline_cfg.get('action_head') or {}
    warmup_steps = pipeline_cfg.get('warmup_steps', 3)

    vlm_config = _build_vlm_config(pipeline_cfg)

    logger.info(f"VLM Type: {vlm_config.model_type}")
    logger.info(f"VLM Model Path: {vlm_config.model_path}")
    logger.info(f"Action Head Type: {action_head_type}")
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # 多 task suite 加载
    # ------------------------------------------------------------------
    task_suites_raw = pipeline_cfg.get('task_suites') or {}
    _task_suites_cfg = task_suites_raw

    shared_vlm = None
    start_time = time.time()

    if task_suites_raw:
        logger.info(f"检测到 task_suites 配置，共 {len(task_suites_raw)} 个 suite")
        for suite_name, suite_cfg in task_suites_raw.items():
            if suite_cfg is None:
                logger.info(f"  [{suite_name}] 权重未配置 (null)，跳过")
                continue

            ckpt = suite_cfg.get('action_head_ckpt')
            ds_path = suite_cfg.get('dataset_path', '')
            if not ckpt:
                logger.warning(f"  [{suite_name}] action_head_ckpt 为空，跳过")
                continue

            engine, shared_vlm = _build_suite_engine(
                suite_name=suite_name,
                ckpt_path=ckpt,
                dataset_path=ds_path,
                vlm_config=vlm_config,
                action_head_type=action_head_type,
                ah_cfg=ah_cfg,
                shared_vlm=shared_vlm,
                device=device,
                warmup_steps=warmup_steps,
            )
            inference_engines[suite_name] = engine

        if inference_engines:
            first_key = next(iter(inference_engines))
            inference_engine = inference_engines[first_key]
            logger.info(f"默认引擎 -> {first_key}")
        else:
            logger.warning("所有 task suite 均为 null，无可用引擎")

    # ------------------------------------------------------------------
    # 单引擎回退（未配置 task_suites 或全部为 null 时）
    # ------------------------------------------------------------------
    if inference_engine is None:
        action_head_ckpt = pipeline_cfg.get('action_head_ckpt')
        if not action_head_ckpt:
            raise ValueError(
                "Missing required config: pipeline.action_head_ckpt "
                "(且未配置有效的 task_suites)"
            )
        dataset_path = pipeline_cfg.get('dataset_path', '')
        logger.info("使用单引擎模式（无 task_suites）")

        engine, _ = _build_suite_engine(
            suite_name="default",
            ckpt_path=action_head_ckpt,
            dataset_path=dataset_path,
            vlm_config=vlm_config,
            action_head_type=action_head_type,
            ah_cfg=ah_cfg,
            shared_vlm=shared_vlm,
            device=device,
            warmup_steps=warmup_steps,
        )
        inference_engine = engine

    init_time = time.time() - start_time
    total_vram = _get_gpu_memory_mb(device)
    logger.info(f"推理引擎初始化完成，耗时: {init_time:.2f}s")
    logger.info(f"可用 task suites: {list(inference_engines.keys()) if inference_engines else ['default']}")
    logger.info(f"Total GPU VRAM allocated: {total_vram:.1f} MB ({total_vram / 1024:.2f} GB)")

    # ------------------------------------------------------------------
    # 预处理配置（与 task suite 无关）
    # ------------------------------------------------------------------
    preprocess_cfg = config.get('preprocess') or {}

    img_cfg = preprocess_cfg.get('image') or {}
    image_preprocess_config = ImagePreprocessConfig(
        resize=img_cfg.get('resize'),
        flip_horizontal=img_cfg.get('flip_horizontal', False),
        flip_vertical=img_cfg.get('flip_vertical', False),
        rotate_180=img_cfg.get('rotate_180', False),
    )
    logger.info(f"图像预处理配置: {image_preprocess_config.model_dump()}")

    state_cfg = preprocess_cfg.get('state') or {}
    index_configs = {}
    for idx_str, idx_cfg in (state_cfg.get('index_configs') or {}).items():
        idx = int(idx_str)
        index_configs[idx] = StateIndexConfig(
            enable_normalization=idx_cfg.get('enable_normalization', False),
            min_val=idx_cfg.get('min_val'),
            max_val=idx_cfg.get('max_val'),
            zero_to_minus_one=idx_cfg.get('zero_to_minus_one', False),
        )
    state_preprocess_config = StatePreprocessConfig(index_configs=index_configs)
    logger.info(f"状态预处理配置: {len(index_configs)} 个索引")

    server_config = config

    logger.info("=" * 60)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ==================== Dashboard ====================

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text(encoding="utf-8")


@app.get("/dashboard", tags=["Dashboard"], include_in_schema=False)
async def dashboard():
    """Metrics dashboard UI"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<rect width="32" height="32" rx="6" fill="#6c5ce7"/>'
        '<text x="16" y="23" font-size="20" font-weight="bold" '
        'fill="white" text-anchor="middle" font-family="sans-serif">S</text>'
        '</svg>'
    )
    return Response(content=svg, media_type="image/svg+xml")


# ==================== API 路由 ====================

@app.get("/", tags=["Root"])
async def root():
    """根路径"""
    return {
        "message": "Sai0_1 VLA Inference Server",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "dashboard": "/dashboard"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """健康检查"""
    return HealthResponse(
        status='healthy' if inference_engine is not None else 'not_ready',
        pipeline_loaded=inference_engine is not None,
        config=server_config
    )


@app.get("/info", response_model=InfoResponse, tags=["Info"])
async def get_info():
    """获取服务器信息"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail='Pipeline not initialized')
    
    config = inference_engine.inference.config
    
    return InfoResponse(
        vlm_type=config.vlm.model_type,
        action_head_type=config.action_head.head_type,
        device=config.vlm.device,
        image_preprocess=image_preprocess_config.model_dump(),
        state_preprocess={
            str(k): v.model_dump() 
            for k, v in state_preprocess_config.index_configs.items()
        }
    )


@app.post("/config/update", tags=["Config"])
async def update_config(request: ConfigUpdateRequest):
    """动态更新预处理配置"""
    global image_preprocess_config, state_preprocess_config
    
    if request.image_preprocess is not None:
        image_preprocess_config = request.image_preprocess
        logger.info(f"更新图像预处理配置: {image_preprocess_config.model_dump()}")
    
    if request.state_preprocess is not None:
        state_preprocess_config = StatePreprocessConfig(
            index_configs=request.state_preprocess
        )
        logger.info(f"更新状态预处理配置: {len(request.state_preprocess)} 个索引")
    
    return {
        "status": "success",
        "image_preprocess": image_preprocess_config.model_dump(),
        "state_preprocess": {
            str(k): v.model_dump() 
            for k, v in state_preprocess_config.index_configs.items()
        }
    }


@app.get("/config/preprocess", tags=["Config"])
async def get_preprocess_config():
    """获取当前预处理配置"""
    return {
        "image_preprocess": image_preprocess_config.model_dump(),
        "state_preprocess": {
            str(k): v.model_dump() 
            for k, v in state_preprocess_config.index_configs.items()
        }
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    预测动作
    
    接收图像和状态，返回预测的动作序列。
    
    支持的图像格式：
    - base64: Base64 编码的 JPEG/PNG 字符串
    - numpy: Numpy array 的 JSON 表示 (shape: [H, W, 3], dtype: uint8)
    
    图像预处理选项（可在请求中覆盖全局配置）：
    - image_resize: resize 尺寸 [width, height]
    - image_flip_horizontal: 水平翻转
    - image_flip_vertical: 垂直翻转
    - image_rotate_180: 旋转 180 度
    
    状态预处理（按全局配置自动应用）：
    - 特定索引的零值转换为 -1
    - 特定索引的最大最小值归一化
    """
    if inference_engine is None:
        raise HTTPException(status_code=500, detail='Pipeline not initialized')
    
    try:
        total_start = time.time()
        
        # 1. 预处理
        preprocess_start = time.time()
        
        # 图像预处理覆盖配置
        image_overrides = {}
        if request.image_resize is not None:
            image_overrides['resize'] = request.image_resize
        if request.image_flip_horizontal is not None:
            image_overrides['flip_horizontal'] = request.image_flip_horizontal
        if request.image_flip_vertical is not None:
            image_overrides['flip_vertical'] = request.image_flip_vertical
        if request.image_rotate_180 is not None:
            image_overrides['rotate_180'] = request.image_rotate_180
        
        # 处理图像
        images = process_images(
            request.images,
            request.image_format,
            image_preprocess_config,
            image_overrides if image_overrides else None
        )
        
        # 处理状态
        state = np.array(request.state, dtype=np.float32)
        original_state_dim = len(state)
        
        # 应用状态预处理
        processed_state = preprocess_state(state, state_preprocess_config)
        
        preprocess_time = time.time() - preprocess_start
        
        # 2. 推理
        inference_start = time.time()
        
        instruction = request.prompt or "execute the task"
        
        actions, latency_ms = inference_engine.predict(
            images=images,
            instruction=instruction,
            state=processed_state,
            track_latency=True
        )
        
        inference_time = time.time() - inference_start
        total_time = time.time() - total_start
        
        logger.info(f"推理完成 - 预处理: {preprocess_time*1000:.1f}ms, 推理: {inference_time*1000:.1f}ms, 总计: {total_time*1000:.1f}ms")
        
        return PredictResponse(
            actions=actions.tolist(),
            timing=TimingInfo(
                preprocess_time=preprocess_time,
                inference_time=inference_time,
                total_time=total_time
            ),
            metadata=PredictMetadata(
                num_images=len(images),
                state_dim=original_state_dim,
                action_shape=list(actions.shape),
                preprocessed_state=processed_state.tolist()
            )
        )
    
    except Exception as e:
        logger.error(f"推理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """批量预测动作"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail='Pipeline not initialized')
    
    try:
        results = []
        
        for i, item in enumerate(request.batch):
            logger.info(f"处理批次项 {i+1}/{len(request.batch)}")
            
            total_start = time.time()
            
            # 预处理
            preprocess_start = time.time()
            
            image_overrides = {}
            if item.image_resize is not None:
                image_overrides['resize'] = item.image_resize
            if item.image_flip_horizontal is not None:
                image_overrides['flip_horizontal'] = item.image_flip_horizontal
            if item.image_flip_vertical is not None:
                image_overrides['flip_vertical'] = item.image_flip_vertical
            if item.image_rotate_180 is not None:
                image_overrides['rotate_180'] = item.image_rotate_180
            
            images = process_images(
                item.images,
                item.image_format,
                image_preprocess_config,
                image_overrides if image_overrides else None
            )
            
            state = np.array(item.state, dtype=np.float32)
            original_state_dim = len(state)
            processed_state = preprocess_state(state, state_preprocess_config)
            
            preprocess_time = time.time() - preprocess_start
            
            # 推理
            inference_start = time.time()
            
            instruction = item.prompt or "execute the task"
            actions, _ = inference_engine.predict(
                images=images,
                instruction=instruction,
                state=processed_state,
            )
            
            inference_time = time.time() - inference_start
            total_time = time.time() - total_start
            
            results.append(PredictResponse(
                actions=actions.tolist(),
                timing=TimingInfo(
                    preprocess_time=preprocess_time,
                    inference_time=inference_time,
                    total_time=total_time
                ),
                metadata=PredictMetadata(
                    num_images=len(images),
                    state_dim=original_state_dim,
                    action_shape=list(actions.shape),
                    preprocessed_state=processed_state.tolist()
                )
            ))
        
        return BatchPredictResponse(results=results)
    
    except Exception as e:
        logger.error(f"批量推理失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/latency_stats", tags=["Monitoring"])
async def get_latency_stats():
    """获取延迟统计"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail='Pipeline not initialized')
    
    return inference_engine.get_latency_stats()


@app.post("/latency_stats/reset", tags=["Monitoring"])
async def reset_latency_stats():
    """重置延迟统计"""
    if inference_engine is None:
        raise HTTPException(status_code=500, detail='Pipeline not initialized')
    
    inference_engine.reset_latency_history()
    return {"status": "success", "message": "Latency history reset"}


# ==================== /v1 公开 API ====================

class ActRequest(BaseModel):
    """面向外部用户的推理请求（/v1/act）"""
    images: List[str] = Field(..., description="Base64 编码的图像列表 (agentview, wrist)")
    state: List[float] = Field(..., description="机器人状态向量（原始值，服务端负责预处理）")
    instruction: str = Field(..., description="任务指令文本")
    image_format: str = Field("base64", description="图像编码格式: 'base64' 或 'numpy'")
    task_suite: Optional[str] = Field(
        None,
        description="LIBERO task suite 名称（如 libero_spatial）。"
        "不传则使用默认引擎。",
    )


class ActResponse(BaseModel):
    """面向外部用户的推理响应（/v1/act）"""
    actions: List[List[float]] = Field(..., description="动作序列 (chunk_size, action_dim)")
    action_dim: int
    chunk_size: int
    request_id: str = Field(..., description="请求追踪 ID")
    timing_ms: float = Field(..., description="总推理耗时（毫秒）")


class VersionResponse(BaseModel):
    version: str
    model: str
    api: str
    available_task_suites: List[str] = Field(
        default_factory=list,
        description="已加载的 task suite 列表",
    )


# ---------- 可观测：请求计数器 ----------

import json as _json
from collections import defaultdict
from threading import Lock as _Lock

_METRICS_FILE = _LOG_DIR / "usage_stats.json"


class _Metrics:
    """API usage tracker with per-user and per-suite breakdowns.

    Persists cumulative counts to ``_METRICS_FILE`` so data survives restarts.
    """

    def __init__(self, persist_path: Path = _METRICS_FILE):
        self._persist_path = persist_path
        self._lock = _Lock()

        self.total_requests: int = 0
        self.total_errors: int = 0
        self.total_inference_ms: float = 0.0
        self.per_user: Dict[str, int] = defaultdict(int)
        self.per_suite: Dict[str, int] = defaultdict(int)
        self.per_user_suite: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self._load()

    # ---- persistence ----

    def _load(self):
        if not self._persist_path.exists():
            return
        try:
            data = _json.loads(self._persist_path.read_text(encoding="utf-8"))
            self.total_requests = data.get("total_requests", 0)
            self.total_errors = data.get("total_errors", 0)
            self.total_inference_ms = data.get("total_inference_ms", 0.0)
            for k, v in data.get("per_user", {}).items():
                self.per_user[k] = v
            for k, v in data.get("per_suite", {}).items():
                self.per_suite[k] = v
            for user, suites in data.get("per_user_suite", {}).items():
                for s, cnt in suites.items():
                    self.per_user_suite[user][s] = cnt
            logger.info(f"Loaded usage stats from {self._persist_path}  "
                        f"(historical total_requests={self.total_requests})")
        except Exception as exc:
            logger.warning(f"Failed to load usage stats: {exc}")

    def _save(self):
        try:
            data = {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "total_inference_ms": self.total_inference_ms,
                "per_user": dict(self.per_user),
                "per_suite": dict(self.per_suite),
                "per_user_suite": {u: dict(s) for u, s in self.per_user_suite.items()},
            }
            self._persist_path.write_text(
                _json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"Failed to save usage stats: {exc}")

    # ---- recording ----

    def record(self, inference_ms: float, error: bool = False,
               user: str = "anonymous", suite: str = "default"):
        with self._lock:
            self.total_requests += 1
            self.per_user[user] += 1
            self.per_suite[suite] += 1
            self.per_user_suite[user][suite] += 1
            if error:
                self.total_errors += 1
            else:
                self.total_inference_ms += inference_ms
            if self.total_requests % 10 == 0:
                self._save()

    def flush(self):
        with self._lock:
            self._save()

    # ---- snapshot ----

    def snapshot(self) -> Dict[str, Any]:
        successful = max(self.total_requests - self.total_errors, 1)
        avg = self.total_inference_ms / successful
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "avg_inference_ms": round(avg, 2),
            "per_user": dict(self.per_user),
            "per_suite": dict(self.per_suite),
        }


v1_metrics = _Metrics()


def _get_engine_for_suite(task_suite: Optional[str]) -> RealtimeInference:
    """根据 task_suite 返回对应的推理引擎，或抛出 HTTPException。"""
    if task_suite is None:
        if inference_engine is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        return inference_engine

    if task_suite in inference_engines:
        return inference_engines[task_suite]

    if task_suite in _task_suites_cfg:
        available = list(inference_engines.keys()) or ["(无)"]
        raise HTTPException(
            status_code=404,
            detail=f"Task suite '{task_suite}' 的权重未配置 (null)。"
            f"当前可用: {available}",
        )

    available = list(inference_engines.keys()) or ["(无)"]
    raise HTTPException(
        status_code=404,
        detail=f"未知的 task suite '{task_suite}'。当前可用: {available}",
    )


def _run_inference_sync(
    engine: RealtimeInference,
    images: List[Image.Image],
    instruction: str,
    processed_state: np.ndarray,
) -> tuple:
    """同步推理函数（在线程池中被队列 worker 调用）"""
    t0 = time.time()
    actions, _ = engine.predict(
        images=images,
        instruction=instruction,
        state=processed_state,
        track_latency=True,
    )
    elapsed_ms = (time.time() - t0) * 1000
    return actions, elapsed_ms


@app.get("/version", response_model=VersionResponse, tags=["v1 - Public API"])
async def get_version():
    """返回服务版本信息"""
    return VersionResponse(
        version=SERVER_VERSION,
        model=MODEL_NAME,
        api=API_VERSION,
        available_task_suites=list(inference_engines.keys()),
    )


@app.post("/v1/act", response_model=ActResponse, tags=["v1 - Public API"])
@limiter.limit(lambda: server_config.get("rate_limit", {}).get("v1_act", "60/minute"))
async def v1_act(
    request: Request,
    body: ActRequest,
    api_key: Optional[str] = Depends(get_api_key),
):
    """
    核心推理端点（面向外部用户）

    发送观测图像 + 状态 + 任务指令，返回预测的动作序列。
    服务端自动完成图像预处理、状态归一化和动作后处理。

    可通过 ``task_suite`` 字段指定使用哪个 LIBERO task suite 的权重。
    不传则使用默认引擎。

    鉴权：需要在 Header 中传递 ``Authorization: Bearer <API_KEY>``
    （若服务端未配置 SAI0_API_KEYS 则不要求鉴权）。
    """
    engine = _get_engine_for_suite(body.task_suite)

    request_id = uuid.uuid4().hex[:12]
    total_start = time.time()

    try:
        # 1. 图像解码 + 预处理（复用现有函数）
        images = process_images(
            body.images,
            body.image_format,
            image_preprocess_config,
        )

        # 2. 状态预处理（复用现有函数）
        state = np.array(body.state, dtype=np.float32)
        processed_state = preprocess_state(state, state_preprocess_config)

        # 3. 通过队列串行推理
        if inference_queue is not None:
            actions, inference_ms = await inference_queue.submit(
                _run_inference_sync,
                args=(engine, images, body.instruction, processed_state),
                timeout=server_config.get("queue", {}).get("timeout", 60.0),
            )
        else:
            actions, inference_ms = _run_inference_sync(
                engine, images, body.instruction, processed_state,
            )

        total_ms = (time.time() - total_start) * 1000
        suite_tag = body.task_suite or "default"
        user_tag = api_key or "anonymous"
        v1_metrics.record(total_ms, user=user_tag, suite=suite_tag)

        action_list = actions.tolist()
        chunk_size = len(action_list)
        action_dim = len(action_list[0]) if chunk_size > 0 else 0

        logger.info(
            f"[v1/act] request_id={request_id} suite={suite_tag} "
            f"images={len(body.images)} state_dim={len(body.state)} "
            f"inference={inference_ms:.1f}ms total={total_ms:.1f}ms "
            f"user={api_key or 'anonymous'}"
        )

        return ActResponse(
            actions=action_list,
            action_dim=action_dim,
            chunk_size=chunk_size,
            request_id=request_id,
            timing_ms=round(total_ms, 2),
        )

    except asyncio.TimeoutError:
        v1_metrics.record(0, error=True, user=api_key or "anonymous",
                          suite=body.task_suite or "default")
        logger.warning(f"[v1/act] request_id={request_id} TIMEOUT")
        raise HTTPException(status_code=503, detail="Inference queue timeout")
    except HTTPException:
        raise
    except Exception as e:
        v1_metrics.record(0, error=True, user=api_key or "anonymous",
                          suite=body.task_suite or "default")
        logger.error(f"[v1/act] request_id={request_id} ERROR: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/metrics", tags=["v1 - Public API"])
async def v1_get_metrics(api_key: Optional[str] = Depends(get_api_key)):
    """可观测性端点：返回 /v1/act 的聚合统计"""
    gpu_info: Dict[str, Any] = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_info = {
            "gpu_utilization_pct": util.gpu,
            "gpu_memory_used_mb": round(mem.used / 1024**2),
            "gpu_memory_total_mb": round(mem.total / 1024**2),
        }
        pynvml.nvmlShutdown()
    except Exception:
        gpu_info = {"note": "pynvml not available"}

    return {
        "version": SERVER_VERSION,
        "queue_depth": inference_queue.qsize if inference_queue else 0,
        "v1_act": v1_metrics.snapshot(),
        "gpu": gpu_info,
    }


# ==================== 主入口 ====================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sai0_1 VLA 推理服务器')
    
    # 服务器配置
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    # 可选的命令行覆盖参数
    parser.add_argument('--action_head_ckpt', type=str, default=None, help='Action Head 检查点路径')
    parser.add_argument('--vlm_type', type=str, default=None, help='VLM 类型')
    parser.add_argument('--vlm_model_path', type=str, default=None, help='VLM 模型路径')
    parser.add_argument('--device', type=str, default=None, help='推理设备')
    
    args = parser.parse_args()
    
    # 加载配置文件（支持绝对路径、相对于 CWD、相对于 server.py 所在目录）
    candidate = Path(args.config)
    if candidate.exists():
        config_path = candidate.resolve()
    else:
        config_path = (Path(__file__).parent / args.config).resolve()

    if config_path.exists():
        config = load_config(str(config_path))
        logger.info(f"已加载配置文件: {config_path}")
    else:
        config = {'server': {}, 'pipeline': {}, 'preprocess': {}}
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
    
    # 命令行参数覆盖
    if args.action_head_ckpt:
        config.setdefault('pipeline', {})['action_head_ckpt'] = args.action_head_ckpt
    if args.vlm_type:
        config.setdefault('pipeline', {})['vlm_type'] = args.vlm_type
    if args.vlm_model_path:
        config.setdefault('pipeline', {})['vlm_model_path'] = args.vlm_model_path
    if args.device:
        config.setdefault('pipeline', {})['device'] = args.device
    
    # 服务器配置
    host = config.get('server', {}).get('host', args.host)
    port = config.get('server', {}).get('port', args.port)
    
    # 初始化推理引擎
    try:
        init_inference_engine(config)
    except Exception as e:
        logger.error(f"推理引擎初始化失败: {e}", exc_info=True)
        sys.exit(1)
    
    # 启动服务器
    logger.info(f"启动服务器: http://{host}:{port}")
    logger.info(f"API 文档: http://{host}:{port}/docs")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="debug" if args.debug else "info"
    )
