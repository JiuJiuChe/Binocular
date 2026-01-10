"""
Evaluation metrics for bird species classification.

Provides accuracy calculations, top-K metrics, per-class analysis, and metric tracking.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate overall accuracy.
    
    Args:
        predictions: Predicted class indices [batch_size]
        targets: Ground truth class indices [batch_size]
    
    Returns:
        accuracy: Scalar accuracy value
    
    Example:
        >>> preds = torch.tensor([0, 1, 2, 1])
        >>> targets = torch.tensor([0, 1, 1, 1])
        >>> acc = accuracy(preds, targets)
        >>> print(acc)  # 0.75
    """
    return (predictions == targets).float().mean()


def top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Calculate top-K accuracy.
    
    Args:
        logits: Model output logits [batch_size, num_classes]
        targets: Ground truth class indices [batch_size]
        k: Number of top predictions to consider
    
    Returns:
        top_k_acc: Scalar top-K accuracy value
    
    Example:
        >>> logits = torch.randn(4, 10)
        >>> targets = torch.tensor([0, 1, 2, 3])
        >>> acc = top_k_accuracy(logits, targets, k=5)
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))
        correct_k = correct.any(dim=1).float().sum()
        return correct_k / batch_size


def per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 555
) -> torch.Tensor:
    """
    Calculate per-class accuracy.
    
    Args:
        predictions: Predicted class indices [batch_size]
        targets: Ground truth class indices [batch_size]
        num_classes: Total number of classes
    
    Returns:
        per_class_acc: Accuracy for each class [num_classes]
                       Returns 0.0 for classes not present in targets
    
    Example:
        >>> preds = torch.tensor([0, 1, 2, 1, 0])
        >>> targets = torch.tensor([0, 1, 1, 1, 2])
        >>> acc = per_class_accuracy(preds, targets, num_classes=3)
    """
    per_class_correct = torch.zeros(num_classes, device=predictions.device)
    per_class_total = torch.zeros(num_classes, device=predictions.device)
    
    for cls in range(num_classes):
        mask = targets == cls
        if mask.sum() > 0:
            per_class_correct[cls] = (predictions[mask] == cls).sum().float()
            per_class_total[cls] = mask.sum().float()
    
    # Avoid division by zero
    per_class_acc = per_class_correct / (per_class_total + 1e-10)
    # Set accuracy to 0 for classes with no samples
    per_class_acc[per_class_total == 0] = 0.0
    
    return per_class_acc


class MetricTracker:
    """
    Track and aggregate metrics over multiple batches.
    
    Accumulates metric values and computes averages at the end of an epoch.
    
    Example:
        >>> tracker = MetricTracker()
        >>> for batch in dataloader:
        ...     loss = compute_loss(batch)
        ...     acc = compute_accuracy(batch)
        ...     tracker.update(loss=loss.item(), accuracy=acc.item())
        >>> metrics = tracker.compute()
        >>> print(metrics)  # {'loss': 0.35, 'accuracy': 0.87}
        >>> tracker.reset()
    """
    
    def __init__(self):
        """Initialize metric tracker."""
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        """
        Update metrics with new values.
        
        Args:
            **kwargs: Metric name-value pairs
        
        Example:
            >>> tracker.update(loss=0.5, accuracy=0.9, top5_accuracy=0.95)
        """
        for key, value in kwargs.items():
            # Handle both scalar tensors and Python numbers
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute average of all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to their averages
        """
        return {key: np.mean(values) for key, values in self.metrics.items()}
    
    def compute_std(self) -> Dict[str, float]:
        """
        Compute standard deviation of all tracked metrics.
        
        Returns:
            Dictionary mapping metric names to their standard deviations
        """
        return {key: np.std(values) for key, values in self.metrics.items()}
    
    def reset(self):
        """Clear all tracked metrics."""
        self.metrics.clear()
    
    def get_current_values(self, metric_name: str) -> List[float]:
        """
        Get all recorded values for a specific metric.
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            List of all recorded values
        """
        return self.metrics.get(metric_name, [])
    
    def __len__(self) -> int:
        """Return number of updates recorded."""
        if not self.metrics:
            return 0
        return len(next(iter(self.metrics.values())))
    
    def __repr__(self) -> str:
        """String representation of current metrics."""
        if not self.metrics:
            return "MetricTracker(empty)"
        computed = self.compute()
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in computed.items())
        return f"MetricTracker({metrics_str})"


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 555
) -> torch.Tensor:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices [N]
        targets: Ground truth class indices [N]
        num_classes: Total number of classes
    
    Returns:
        confusion_matrix: [num_classes, num_classes] where entry (i, j) is
                         the number of samples with true label i predicted as j
    """
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    
    for t, p in zip(targets, predictions):
        confusion_matrix[t.long(), p.long()] += 1
    
    return confusion_matrix


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int = 555
) -> torch.Tensor:
    """
    Compute class weights for handling imbalanced datasets.
    
    Uses inverse frequency weighting: weight = total_samples / (n_classes * class_count)
    
    Args:
        targets: Ground truth class indices [N]
        num_classes: Total number of classes
    
    Returns:
        class_weights: Weight for each class [num_classes]
    """
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    total_samples = len(targets)
    
    # Avoid division by zero for classes with no samples
    class_weights = torch.zeros(num_classes)
    mask = class_counts > 0
    class_weights[mask] = total_samples / (num_classes * class_counts[mask])
    
    return class_weights
