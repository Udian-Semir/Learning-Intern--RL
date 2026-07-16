"""
OFT1_0 Action Head Module

Optimal Flow Transport (OFT) action head for VLA models.
"""

from .vlm2oft_pipeline import (
    VLM2OFTPipeline,
    create_vlm2oft_pipeline,
)

__all__ = [
    "VLM2OFTPipeline",
    "create_vlm2oft_pipeline",
]

