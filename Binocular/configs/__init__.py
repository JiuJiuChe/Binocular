"""
Configuration system for bird species classifier training.

Provides dataclass-based configuration and utilities for loading/saving configs.
"""

from .base_config import TrainingConfig, load_config, save_config, merge_config_with_args

__all__ = [
    'TrainingConfig',
    'load_config',
    'save_config',
    'merge_config_with_args',
]
