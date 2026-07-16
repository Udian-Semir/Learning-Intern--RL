"""Action Heads OFT1_0 module for GR00T Action_Heads."""

# Import action heads
from .action_heads_oft_orig import (
    DiffusionActionHead,
    L1RegressionActionHead,
    MLPResNet,
    MLPResNetBlock,
    NoisePredictionModel,
    SinusoidalPositionalEncoding,
)

# Import projectors
from .projectors_oft_orig import (
    NoisyActionProjector,
    ProprioProjector,
    ProprioProjector_Changed,
)

# Import constants from OFT1_0 root (2 levels up)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from constants import (
        ACTION_DIM,
        LLM_OUTPUT_DIM_MLP_INPUT_DIM,
        NUM_ACTIONS_CHUNK,
        NUM_DIFFUSION_STEPS,
        NUM_VLM_HIDDEN_LAYERS,
        PROPRIO_DIM,
        USE_DIFFUSION,
        USE_NOISY_ACTION_PROJECTOR,
        USE_PROPRIO_PROJECTOR,
    )
except ImportError:
    # Fallback for relative import
    try:
        from ...constants import (
            ACTION_DIM,
            LLM_OUTPUT_DIM_MLP_INPUT_DIM,
            NUM_ACTIONS_CHUNK,
            NUM_DIFFUSION_STEPS,
            NUM_VLM_HIDDEN_LAYERS,
            PROPRIO_DIM,
            USE_DIFFUSION,
            USE_NOISY_ACTION_PROJECTOR,
            USE_PROPRIO_PROJECTOR,
        )
    except ImportError:
        raise ImportError("Cannot import constants. Make sure constants.py is in OFT1_0 directory")

__all__ = [
    # Action heads
    "DiffusionActionHead",
    "L1RegressionActionHead",
    "MLPResNet",
    "MLPResNetBlock",
    "NoisePredictionModel",
    "SinusoidalPositionalEncoding",
    # Constants
    "ACTION_DIM",
    "LLM_OUTPUT_DIM_MLP_INPUT_DIM",
    "NUM_ACTIONS_CHUNK",
    "NUM_DIFFUSION_STEPS",
    "NUM_VLM_HIDDEN_LAYERS",
    "PROPRIO_DIM",
    "USE_DIFFUSION",
    "USE_NOISY_ACTION_PROJECTOR",
    "USE_PROPRIO_PROJECTOR",
    # Projectors
    "NoisyActionProjector",
    "ProprioProjector",
    "ProprioProjector_Changed",
]
