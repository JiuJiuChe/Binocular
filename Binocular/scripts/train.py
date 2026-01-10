"""
Training script for bird species classifier.

Implements complete training pipeline with:
- Data loading and preprocessing
- Model creation and training
- Validation and checkpointing
- TensorBoard logging
- Support for resuming from checkpoint
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Binocular.configs import load_config, TrainingConfig, merge_config_with_args
from Binocular.datasets.nabirds.dataset import NABirdsDataset
from Binocular.datasets.nabirds.transforms import get_train_transforms, get_val_transforms
from Binocular.models.wrappers import create_model
from Binocular.utils import (
    MetricTracker,
    save_checkpoint,
    load_checkpoint,
    Logger,
    create_optimizer,
    create_scheduler,
    top_k_accuracy,
)
from Binocular.utils.optimizer import get_lr


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"🌱 Random seed set to {seed}")


def create_dataloaders(config: TrainingConfig):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Training configuration
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("\n" + "=" * 80)
    print("CREATING DATALOADERS")
    print("=" * 80)
    
    # Get transforms
    train_transform = get_train_transforms(image_size=config.image_size)
    val_transform = get_val_transforms(image_size=config.image_size)
    
    # Create datasets
    train_dataset = NABirdsDataset(
        root=config.dataset_root,
        split='train',
        transform=train_transform,
        use_bbox_crop=config.use_bbox_crop
    )
    
    val_dataset = NABirdsDataset(
        root=config.dataset_root,
        split='val',
        transform=val_transform,
        use_bbox_crop=config.use_bbox_crop
    )
    
    test_dataset = NABirdsDataset(
        root=config.dataset_root,
        split='test',
        transform=val_transform,
        use_bbox_crop=config.use_bbox_crop
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    print(f"\n✅ Dataloaders created:")
    print(f"   Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} images, {len(test_loader)} batches")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Num workers: {config.num_workers}")
    print("=" * 80 + "\n")
    
    return train_loader, val_loader, test_loader, train_dataset.get_num_classes()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: TrainingConfig,
    logger: Logger
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        loader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        logger: Logger instance
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics = MetricTracker()
    
    pbar = tqdm(loader, desc=f'Epoch {epoch:3d}/{config.epochs}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # Move to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.gradient_clip_norm
            )
        
        optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()
            top5_acc = top_k_accuracy(logits, labels, k=5)
        
        metrics.update(
            loss=loss.item(),
            accuracy=acc.item(),
            top5_accuracy=top5_acc.item()
        )
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{acc.item():.3f}',
            'top5': f'{top5_acc.item():.3f}'
        })
        
        # Log to TensorBoard
        if batch_idx % config.log_every_n_steps == 0:
            logger.log_step(
                epoch=epoch,
                batch_idx=batch_idx,
                loss=loss.item(),
                accuracy=acc.item()
            )
        
        # Debug mode: stop after a few batches
        if config.debug_num_batches and batch_idx >= config.debug_num_batches:
            break
    
    return metrics.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """
    Validate model on validation set.
    
    Args:
        model: Model to validate
        loader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics = MetricTracker()
    
    for images, labels in tqdm(loader, desc='Validation', leave=False):
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        acc = (logits.argmax(dim=1) == labels).float().mean()
        top5_acc = top_k_accuracy(logits, labels, k=5)
        
        metrics.update(
            loss=loss.item(),
            accuracy=acc.item(),
            top5_accuracy=top5_acc.item()
        )
    
    return metrics.compute()


def train(config: TrainingConfig, resume_checkpoint: str = None):
    """
    Main training function.
    
    Args:
        config: Training configuration
        resume_checkpoint: Path to checkpoint to resume from (optional)
    """
    # Print configuration
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)
    print(f"Experiment: {config.experiment_name}")
    print(f"Device: {config.device}")
    print(f"Encoder: {config.encoder_name}")
    print(f"Freeze encoder: {config.freeze_encoder}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Scheduler: {config.scheduler_type}")
    print("=" * 80 + "\n")
    
    # Set random seed
    set_seed(config.seed)
    
    # Set device
    device = torch.device(config.device)
    print(f"🖥️  Using device: {device}\n")
    
    # Create dataloaders
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(config)
    
    # Create model
    print("=" * 80)
    print("CREATING MODEL")
    print("=" * 80)
    model = create_model(
        encoder_type=config.encoder_type,
        num_classes=num_classes,
        encoder_name=config.encoder_name,
        freeze_encoder=config.freeze_encoder,
        classifier_type=config.classifier_type
    )
    model = model.to(device)
    
    # Print model info
    model_info = model.get_model_info()
    print(f"\n✅ Model created:")
    print(f"   Total parameters: {model_info['total_params']:,}")
    print(f"   Trainable parameters: {model_info['trainable_params']:,}")
    print(f"   Encoder frozen: {model_info['encoder_frozen']}")
    print("=" * 80 + "\n")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Setup logging
    logger = Logger(config.log_dir, config.experiment_name)
    logger.log_model_info(model_info)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume_checkpoint:
        print(f"\n📂 Resuming from checkpoint: {resume_checkpoint}")
        start_epoch, checkpoint_metrics = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            device=config.device
        )
        best_val_acc = checkpoint_metrics.get('accuracy', 0.0)
        start_epoch += 1  # Start from next epoch
        print(f"✅ Resumed from epoch {start_epoch - 1}")
        print(f"   Best val accuracy so far: {best_val_acc:.4f}\n")
    
    # Debug mode: overfit single batch
    if config.debug_overfit_batch:
        print("🐛 DEBUG MODE: Overfitting single batch")
        print("=" * 80)
        single_batch = next(iter(train_loader))
        
        for epoch in range(config.epochs):
            model.train()
            images, labels = single_batch
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (logits.argmax(dim=1) == labels).float().mean()
            print(f"Epoch {epoch:3d}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
            
            if acc.item() > 0.99:
                print("✅ Successfully overfitted (acc > 0.99)")
                break
        
        print("=" * 80)
        return
    
    # Main training loop
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")
    
    for epoch in range(start_epoch, config.epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            logger=logger
        )
        
        # Validate
        if epoch % config.val_every_n_epochs == 0:
            val_metrics = validate(model, val_loader, criterion, device)
        else:
            val_metrics = {'loss': 0.0, 'accuracy': 0.0, 'top5_accuracy': 0.0}
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Get current learning rate
        current_lr = get_lr(optimizer)
        
        # Log epoch metrics
        logger.log_epoch(
            epoch=epoch,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            learning_rate=current_lr
        )
        
        # Check if this is the best model
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
        
        # Save checkpoint
        if epoch % config.checkpoint_every_n_epochs == 0 or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=val_metrics,
                config=config,
                is_best=is_best,
                filename=f'epoch_{epoch:03d}.pth'
            )
    
    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    
    # Load best model
    best_checkpoint_path = Path(config.checkpoint_dir) / config.experiment_name / 'best.pth'
    if best_checkpoint_path.exists():
        print(f"Loading best model from {best_checkpoint_path}")
        load_checkpoint(best_checkpoint_path, model, device=config.device)
    
    test_metrics = validate(model, test_loader, criterion, device)
    
    print(f"\n{'=' * 80}")
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Loss:         {test_metrics['loss']:.4f}")
    print(f"Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"Top-5 Acc:    {test_metrics['top5_accuracy']:.4f}")
    print("=" * 80 + "\n")
    
    logger.log_test_results(test_metrics)
    logger.close()
    
    print("✅ Training complete!")
    print(f"📊 Logs saved to: {Path(config.log_dir) / config.experiment_name}")
    print(f"💾 Checkpoints saved to: {Path(config.checkpoint_dir) / config.experiment_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train bird species classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file (e.g., configs/m2_dev.yaml)'
    )
    
    # Optional overrides
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        dest='learning_rate',
        help='Override learning rate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Override device'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override random seed'
    )
    parser.add_argument(
        '--debug-overfit-batch',
        action='store_true',
        help='Debug mode: overfit single batch'
    )
    parser.add_argument(
        '--debug-num-batches',
        type=int,
        default=None,
        help='Debug mode: limit number of batches per epoch'
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command-line arguments
    overrides = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'device': args.device,
        'seed': args.seed,
        'debug_overfit_batch': args.debug_overfit_batch,
        'debug_num_batches': args.debug_num_batches,
    }
    config = merge_config_with_args(config, overrides)
    
    # Run training
    try:
        train(config, resume_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()
