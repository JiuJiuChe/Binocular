"""
Evaluation script for trained bird species classifier.

Loads a trained model and evaluates it on the test set with comprehensive metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Binocular.configs import TrainingConfig
from Binocular.datasets.nabirds.dataset import NABirdsDataset
from Binocular.datasets.nabirds.transforms import get_val_transforms
from Binocular.models import create_model
from Binocular.utils import (
    MetricTracker,
    load_checkpoint,
    top_k_accuracy,
    per_class_accuracy,
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int
) -> dict:
    """
    Comprehensive evaluation of model.
    
    Args:
        model: Model to evaluate
        loader: Test dataloader
        device: Device to evaluate on
        num_classes: Number of classes
    
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics = MetricTracker()
    
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_targets = []
    
    print("Evaluating model...")
    for images, labels in tqdm(loader, desc='Evaluation'):
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = criterion(logits, labels)
        
        predictions = logits.argmax(dim=1)
        acc = (predictions == labels).float().mean()
        top5_acc = top_k_accuracy(logits, labels, k=5)
        top10_acc = top_k_accuracy(logits, labels, k=10)
        
        metrics.update(
            loss=loss.item(),
            accuracy=acc.item(),
            top5_accuracy=top5_acc.item(),
            top10_accuracy=top10_acc.item()
        )
        
        all_predictions.append(predictions.cpu())
        all_targets.append(labels.cpu())
    
    # Compute overall metrics
    results = metrics.compute()
    
    # Compute per-class accuracy
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    per_class_acc = per_class_accuracy(all_predictions, all_targets, num_classes)
    results['per_class_accuracy_mean'] = per_class_acc.mean().item()
    results['per_class_accuracy_std'] = per_class_acc.std().item()
    results['per_class_accuracy_min'] = per_class_acc.min().item()
    results['per_class_accuracy_max'] = per_class_acc.max().item()
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained bird species classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save evaluation results (JSON)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cuda', 'mps', 'cpu'],
        help='Device to evaluate on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for evaluation'
    )
    
    args = parser.parse_args()
    
    # Load checkpoint info
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")
    
    # Load checkpoint to get config
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint.get('config', {})
    
    # Create minimal config for evaluation
    if isinstance(config_dict, dict):
        config = TrainingConfig(**config_dict)
    else:
        # Fallback to defaults
        config = TrainingConfig()
    
    # Set device
    device = torch.device(args.device)
    print(f"🖥️  Using device: {device}\n")
    
    # Create test dataset
    print("Loading test dataset...")
    val_transform = get_val_transforms(image_size=config.image_size)
    test_dataset = NABirdsDataset(
        root=config.dataset_root,
        split='test',
        transform=val_transform,
        use_bbox_crop=config.use_bbox_crop
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    
    num_classes = test_dataset.get_num_classes()
    print(f"✅ Test dataset: {len(test_dataset)} images, {num_classes} classes\n")
    
    # Create model
    print("Creating model...")
    model = create_model(
        encoder_type=config.encoder_type,
        num_classes=num_classes,
        encoder_name=config.encoder_name,
        freeze_encoder=config.freeze_encoder,
        classifier_type=config.classifier_type
    )
    model = model.to(device)
    
    # Load checkpoint
    epoch, checkpoint_metrics = load_checkpoint(
        checkpoint_path,
        model,
        device=args.device
    )
    
    print(f"✅ Loaded model from epoch {epoch}")
    if checkpoint_metrics:
        print(f"   Checkpoint metrics: {checkpoint_metrics}\n")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, num_classes)
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Loss:                {results['loss']:.4f}")
    print(f"Accuracy:            {results['accuracy']:.4f}")
    print(f"Top-5 Accuracy:      {results['top5_accuracy']:.4f}")
    print(f"Top-10 Accuracy:     {results['top10_accuracy']:.4f}")
    print("-" * 80)
    print(f"Per-class acc mean:  {results['per_class_accuracy_mean']:.4f}")
    print(f"Per-class acc std:   {results['per_class_accuracy_std']:.4f}")
    print(f"Per-class acc min:   {results['per_class_accuracy_min']:.4f}")
    print(f"Per-class acc max:   {results['per_class_accuracy_max']:.4f}")
    print("=" * 80 + "\n")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_with_metadata = {
            'checkpoint': str(checkpoint_path),
            'epoch': epoch,
            'num_test_samples': len(test_dataset),
            'num_classes': num_classes,
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}\n")
    
    print("✅ Evaluation complete!")


if __name__ == '__main__':
    main()
