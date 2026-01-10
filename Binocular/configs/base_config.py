"""
Base configuration dataclass for training experiments.

Provides a flexible configuration system using Python dataclasses and YAML files.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class TrainingConfig:
    """
    Complete configuration for training a bird species classifier.
    
    This dataclass defines all hyperparameters and settings needed for training.
    Can be loaded from YAML files or created programmatically.
    """
    
    # ===== Model Settings =====
    encoder_type: str = 'dinov2'
    encoder_name: str = 'dinov2_vitb14'
    freeze_encoder: bool = True
    classifier_type: str = 'linear'
    classifier_dropout: float = 0.0
    
    # ===== Data Settings =====
    dataset_root: str = 'Binocular/datasets/nabirds'
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    use_bbox_crop: bool = True
    pin_memory: bool = True
    
    # ===== Training Settings =====
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = 'adamw'
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_epochs: int = 5
    device: str = 'cuda'  # 'cuda', 'mps', or 'cpu'
    mixed_precision: bool = False
    seed: int = 42
    
    # ===== Scheduler Settings =====
    scheduler_type: str = 'cosine'
    min_lr: float = 1e-6
    
    # ===== Logging Settings =====
    log_dir: str = 'Binocular/artifacts/runs'
    checkpoint_dir: str = 'Binocular/artifacts/checkpoints'
    experiment_name: str = 'default_experiment'
    log_every_n_steps: int = 50
    checkpoint_every_n_epochs: int = 10
    
    # ===== Evaluation Settings =====
    val_every_n_epochs: int = 1
    save_best_only: bool = False
    
    # ===== Debug Settings =====
    debug_overfit_batch: bool = False
    debug_num_batches: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string paths to Path objects for easier handling
        self.dataset_root = str(Path(self.dataset_root))
        self.log_dir = str(Path(self.log_dir))
        self.checkpoint_dir = str(Path(self.checkpoint_dir))
        
        # Validate device
        valid_devices = ['cuda', 'mps', 'cpu']
        if self.device not in valid_devices:
            raise ValueError(f"device must be one of {valid_devices}, got {self.device}")
        
        # Validate optimizer
        valid_optimizers = ['adamw', 'sgd', 'adam']
        if self.optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}, got {self.optimizer}")
        
        # Validate scheduler
        valid_schedulers = ['cosine', 'step', 'none']
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(f"scheduler_type must be one of {valid_schedulers}, got {self.scheduler_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        save_config(self, path)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def load_config(config_path: str) -> TrainingConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        TrainingConfig instance
    
    Example:
        >>> config = load_config('configs/m2_dev.yaml')
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Flatten nested YAML structure to match dataclass fields
    flat_config = {}
    for section, values in yaml_config.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    return TrainingConfig.from_dict(flat_config)


def save_config(config: TrainingConfig, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: TrainingConfig instance
        save_path: Path where to save the YAML file
    
    Example:
        >>> save_config(config, 'artifacts/experiment_config.yaml')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    # Organize into sections for readability
    organized_config = {
        'model': {
            'encoder_type': config_dict['encoder_type'],
            'encoder_name': config_dict['encoder_name'],
            'freeze_encoder': config_dict['freeze_encoder'],
            'classifier_type': config_dict['classifier_type'],
            'classifier_dropout': config_dict['classifier_dropout'],
        },
        'data': {
            'dataset_root': config_dict['dataset_root'],
            'image_size': config_dict['image_size'],
            'batch_size': config_dict['batch_size'],
            'num_workers': config_dict['num_workers'],
            'use_bbox_crop': config_dict['use_bbox_crop'],
            'pin_memory': config_dict['pin_memory'],
        },
        'training': {
            'epochs': config_dict['epochs'],
            'learning_rate': config_dict['learning_rate'],
            'optimizer': config_dict['optimizer'],
            'weight_decay': config_dict['weight_decay'],
            'gradient_clip_norm': config_dict['gradient_clip_norm'],
            'warmup_epochs': config_dict['warmup_epochs'],
            'device': config_dict['device'],
            'mixed_precision': config_dict['mixed_precision'],
            'seed': config_dict['seed'],
        },
        'scheduler': {
            'type': config_dict['scheduler_type'],
            'min_lr': config_dict['min_lr'],
        },
        'logging': {
            'log_dir': config_dict['log_dir'],
            'checkpoint_dir': config_dict['checkpoint_dir'],
            'experiment_name': config_dict['experiment_name'],
            'log_every_n_steps': config_dict['log_every_n_steps'],
            'checkpoint_every_n_epochs': config_dict['checkpoint_every_n_epochs'],
        },
        'eval': {
            'val_every_n_epochs': config_dict['val_every_n_epochs'],
            'save_best_only': config_dict['save_best_only'],
        },
        'debug': {
            'debug_overfit_batch': config_dict['debug_overfit_batch'],
            'debug_num_batches': config_dict['debug_num_batches'],
        },
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(organized_config, f, default_flow_style=False, sort_keys=False)


def merge_config_with_args(config: TrainingConfig, args_dict: Dict[str, Any]) -> TrainingConfig:
    """
    Override config values with command-line arguments.
    
    Args:
        config: Base configuration
        args_dict: Dictionary of argument overrides (only non-None values applied)
    
    Returns:
        Updated configuration
    
    Example:
        >>> config = load_config('configs/m2_dev.yaml')
        >>> args = {'batch_size': 16, 'epochs': 30}
        >>> config = merge_config_with_args(config, args)
    """
    config_dict = config.to_dict()
    
    # Update only non-None values from args
    for key, value in args_dict.items():
        if value is not None and key in config_dict:
            config_dict[key] = value
    
    return TrainingConfig.from_dict(config_dict)
