"""
Model components for bird species classification.

This module provides encoder-agnostic model components:
- Encoders: Various pretrained vision backbones (DINOv2, CLIP, etc.)
- Linear probe: Classification head for species prediction
- Model wrapper: Combines encoder + linear probe
- InferenceModel: A wrapper for easy inference.
"""

from .encoders import get_encoder, get_encoder_info
from .linear_probe import LinearProbe
from .wrappers import EncoderClassifier
from .inference import InferenceModel

__all__ = [
    'get_encoder',
    'get_encoder_info',
    'LinearProbe',
    'EncoderClassifier',
    'InferenceModel',
]
