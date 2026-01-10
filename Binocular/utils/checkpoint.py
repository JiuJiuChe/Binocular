"""
Checkpoint utilities for saving and loading model states.

Provides functions for saving/loading model, optimizer, and scheduler states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import asdict


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    config: Any,
    is_best: bool = False,
    filename: str = 'checkpoint.pth'
):
    """
    Save model checkpoint.
    
    Saves model state, optimizer state, scheduler state, metrics, and config
    to enable resuming training or loading for inference.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        epoch: Current epoch number
        metrics: Dictionary of metrics (e.g., {'val_accuracy': 0.85})
        config: Training configuration object
        is_best: Whether this is the best model so far
        filename: Name for the checkpoint file
    
    Example:
        >>> save_checkpoint(
        ...     model, optimizer, scheduler, epoch=10,
        ...     metrics={'val_accuracy': 0.85},
        ...     config=config,
        ...     is_best=True
        ... )
    """
    checkpoint_dir = Path(config.checkpoint_dir) / config.experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': asdict(config) if hasattr(config, '__dataclass_fields__') else config,
    }
    
    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as 'last.pth' for easy resuming
    if filename != 'last.pth':
        torch.save(checkpoint, checkpoint_dir / 'last.pth')
    
    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / 'best.pth'
        torch.save(checkpoint, best_path)
        print(f"💾 Saved best checkpoint (epoch {epoch}, metrics: {metrics})")
    
    # Save periodic checkpoint with epoch number
    if epoch % 10 == 0:
        epoch_path = checkpoint_dir / f'epoch_{epoch:03d}.pth'
        torch.save(checkpoint, epoch_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = 'cpu'
) -> tuple:
    """
    Load model checkpoint.
    
    Loads model state and optionally optimizer and scheduler states.
    Useful for resuming training or loading a trained model for evaluation.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional, for resuming training)
        scheduler: Scheduler to load state into (optional, for resuming training)
        device: Device to load checkpoint to ('cuda', 'mps', or 'cpu')
    
    Returns:
        epoch: Epoch number from checkpoint
        metrics: Metrics dictionary from checkpoint
    
    Example:
        >>> model = create_model(...)
        >>> epoch, metrics = load_checkpoint(
        ...     'artifacts/checkpoints/best.pth',
        ...     model,
        ...     device='mps'
        ... )
        >>> print(f"Loaded model from epoch {epoch}")
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"✅ Loaded checkpoint from epoch {epoch}")
    if metrics:
        print(f"   Metrics: {metrics}")
    
    return epoch, metrics


def find_latest_checkpoint(checkpoint_dir: str, experiment_name: str) -> Optional[Path]:
    """
    Find the latest checkpoint for an experiment.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        experiment_name: Name of the experiment
    
    Returns:
        Path to latest checkpoint or None if not found
    """
    exp_dir = Path(checkpoint_dir) / experiment_name
    
    if not exp_dir.exists():
        return None
    
    # Look for last.pth first
    last_checkpoint = exp_dir / 'last.pth'
    if last_checkpoint.exists():
        return last_checkpoint
    
    # Otherwise find highest epoch number
    checkpoints = list(exp_dir.glob('epoch_*.pth'))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))
    return checkpoints[-1]


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Get information about a checkpoint without loading the full model.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
    }
    
    return info


def save_model_for_inference(
    model: nn.Module,
    save_path: str,
    config: Any,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Save a model in a lightweight format optimized for inference.
    
    Only saves model weights and essential metadata, not optimizer/scheduler states.
    
    Args:
        model: Trained model
        save_path: Path where to save the model
        config: Training configuration
        metrics: Optional metrics to include
    
    Example:
        >>> save_model_for_inference(
        ...     model,
        ...     'artifacts/models/best_model.pth',
        ...     config,
        ...     metrics={'accuracy': 0.89}
        ... )
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    export_dict = {
        'model_state_dict': model.state_dict(),
        'config': asdict(config) if hasattr(config, '__dataclass_fields__') else config,
        'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {},
    }
    
    if metrics:
        export_dict['metrics'] = metrics
    
    torch.save(export_dict, save_path)
    print(f"💾 Saved model for inference to {save_path}")
