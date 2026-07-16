"""
API Key 鉴权模块

仅作用于 /v1/* 路由，不影响原有的 /predict 等内部端点。

合法 API Key 通过环境变量 SAI0_API_KEYS 配置（逗号分隔），例如:
    export SAI0_API_KEYS="sk-abc123,sk-xyz456"
"""

import os
import logging
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)

_valid_keys: Optional[set] = None


def _load_keys() -> set:
    global _valid_keys
    if _valid_keys is None:
        raw = os.environ.get("SAI0_API_KEYS", "")
        _valid_keys = {k.strip() for k in raw.split(",") if k.strip()}
        if _valid_keys:
            logger.info(f"已加载 {len(_valid_keys)} 个 API Key")
        else:
            logger.warning(
                "未配置 SAI0_API_KEYS 环境变量，/v1/* 端点将对所有请求开放"
            )
    return _valid_keys


def reload_keys():
    """强制重新加载 key（用于运行时热更新）"""
    global _valid_keys
    _valid_keys = None
    _load_keys()


def get_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> Optional[str]:
    """FastAPI 依赖项：校验 Bearer token 并返回 api_key 字符串。"""
    keys = _load_keys()

    if not keys:
        return None

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if credentials.credentials not in keys:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    return credentials.credentials


def extract_api_key_for_ratelimit(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> str:
    """为 slowapi 限流提取标识符（api_key 或 IP 占位）。"""
    if credentials and credentials.credentials:
        return credentials.credentials
    return "anonymous"
