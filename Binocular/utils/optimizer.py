"""
Optimizer and learning rate scheduler factories.

Provides functions to create optimizers and schedulers from configuration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any


def create_optimizer(model: nn.Module, config: Any) -> optim.Optimizer:
    """
    Create optimizer based on configuration.
    
    Supports:
    - AdamW (recommended for transformers)
    - SGD with momentum
    - Adam
    
    Args:
        model: PyTorch model
        config: Training configuration with optimizer settings
    
    Returns:
        Configured optimizer
    
    Example:
        >>> optimizer = create_optimizer(model, config)
    """
    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        raise ValueError("No trainable parameters found in model!")
    
    optimizer_type = config.optimizer.lower()
    
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(
            trainable_params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(
            f"Unknown optimizer: {config.optimizer}. "
            f"Supported: ['adamw', 'adam', 'sgd']"
        )
    
    print(f"✅ Created {optimizer_type.upper()} optimizer")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Weight decay: {config.weight_decay}")
    print(f"   Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    config: Any,
    steps_per_epoch: int = 0
) -> Any:
    """
    Create learning rate scheduler based on configuration.
    
    Supports:
    - Cosine annealing (smooth decay to min_lr)
    - Step decay (decay by gamma every step_size epochs)
    - None (constant learning rate)
    
    Args:
        optimizer: Optimizer to schedule
        config: Training configuration with scheduler settings
        steps_per_epoch: Number of batches per epoch (for step-based schedulers)
    
    Returns:
        Learning rate scheduler or None
    
    Example:
        >>> scheduler = create_scheduler(optimizer, config)
        >>> # In training loop:
        >>> scheduler.step()
    """
    scheduler_type = config.scheduler_type.lower()
    
    if scheduler_type == 'none':
        print("✅ No learning rate scheduler (constant LR)")
        return None
    
    elif scheduler_type == 'cosine':
        # Cosine annealing with warmup
        if config.warmup_epochs > 0:
            # Create cosine scheduler for main training
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs - config.warmup_epochs,
                eta_min=config.min_lr
            )
            
            # Wrap with warmup
            scheduler = WarmupScheduler(
                optimizer,
                warmup_epochs=config.warmup_epochs,
                base_scheduler=cosine_scheduler,
                warmup_start_lr=config.min_lr
            )
            print(f"✅ Created Cosine scheduler with {config.warmup_epochs} warmup epochs")
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=config.min_lr
            )
            print(f"✅ Created Cosine scheduler (no warmup)")
        
        print(f"   Min LR: {config.min_lr}")
    
    elif scheduler_type == 'step':
        step_size = getattr(config, 'scheduler_step_size', 30)
        gamma = getattr(config, 'scheduler_gamma', 0.1)
        
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        print(f"✅ Created Step scheduler")
        print(f"   Step size: {step_size} epochs")
        print(f"   Gamma: {gamma}")
    
    elif scheduler_type == 'exponential':
        gamma = getattr(config, 'scheduler_gamma', 0.95)
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
        print(f"✅ Created Exponential scheduler (gamma={gamma})")
    
    else:
        raise ValueError(
            f"Unknown scheduler: {config.scheduler_type}. "
            f"Supported: ['cosine', 'step', 'exponential', 'none']"
        )
    
    return scheduler


class WarmupScheduler:
    """
    Learning rate scheduler with linear warmup followed by another scheduler.
    
    During warmup, linearly increases LR from warmup_start_lr to base LR.
    After warmup, delegates to the base scheduler.
    
    Example:
        >>> cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
        >>> scheduler = WarmupScheduler(optimizer, warmup_epochs=10, base_scheduler=cosine)
        >>> for epoch in range(100):
        ...     train(...)
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        base_scheduler: Any,
        warmup_start_lr: float = 1e-6
    ):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of warmup epochs
            base_scheduler: Scheduler to use after warmup
            warmup_start_lr: Starting learning rate for warmup
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.warmup_start_lr = warmup_start_lr
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        self.current_epoch = 0
    
    def step(self):
        """Step the scheduler."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * warmup_factor
        else:
            # Use base scheduler
            self.base_scheduler.step()
        
        self.current_epoch += 1
    
    def state_dict(self):
        """Return state dict."""
        return {
            'current_epoch': self.current_epoch,
            'base_scheduler': self.base_scheduler.state_dict() if self.base_scheduler else None,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.current_epoch = state_dict['current_epoch']
        if self.base_scheduler and state_dict.get('base_scheduler'):
            self.base_scheduler.load_state_dict(state_dict['base_scheduler'])


def get_lr(optimizer: optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer
    
    Returns:
        Current learning rate (from first param group)
    """
    return optimizer.param_groups[0]['lr']
