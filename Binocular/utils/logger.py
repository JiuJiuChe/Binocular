"""
Logging utilities for tracking training progress.

Provides TensorBoard logging and console output formatting.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class Logger:
    """
    Logger for tracking training progress with TensorBoard and console output.
    
    Handles:
    - Step-level metrics (loss, accuracy per batch)
    - Epoch-level metrics (average loss, accuracy per epoch)
    - Test results
    - Hyperparameter tracking
    
    Example:
        >>> logger = Logger('artifacts/runs', 'my_experiment')
        >>> logger.log_step(epoch=0, batch_idx=10, loss=0.5, accuracy=0.8)
        >>> logger.log_epoch(epoch=0, train_metrics={'loss': 0.4}, val_metrics={'accuracy': 0.85})
        >>> logger.close()
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        console_log: bool = True
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Base directory for logs
            experiment_name: Name of the experiment
            console_log: Whether to also print to console
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.console_log = console_log
        self.step = 0
        
        # Create log directory
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.experiment_dir)
            print(f"📊 TensorBoard logging to: {self.experiment_dir}")
            print(f"   View with: tensorboard --logdir {self.log_dir}")
        else:
            self.writer = None
            print(f"⚠️  TensorBoard not available, logging to console only")
        
        # Create log file for text output
        self.log_file = self.experiment_dir / 'training.log'
        self._log_to_file(f"Experiment: {experiment_name}")
        self._log_to_file(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log_to_file("-" * 80)
    
    def _log_to_file(self, message: str):
        """Write message to log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_step(
        self,
        epoch: int,
        batch_idx: int,
        loss: float,
        accuracy: float,
        **kwargs
    ):
        """
        Log training step metrics.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            loss: Training loss
            accuracy: Training accuracy
            **kwargs: Additional metrics to log
        """
        if self.writer:
            self.writer.add_scalar('train/loss_step', loss, self.step)
            self.writer.add_scalar('train/accuracy_step', accuracy, self.step)
            
            for key, value in kwargs.items():
                self.writer.add_scalar(f'train/{key}_step', value, self.step)
        
        self.step += 1
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: Optional[float] = None
    ):
        """
        Log epoch-level metrics.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            learning_rate: Current learning rate (optional)
        """
        # Log to TensorBoard
        if self.writer:
            # Training metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}_epoch', value, epoch)
            
            # Validation metrics
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)
            
            # Learning rate
            if learning_rate is not None:
                self.writer.add_scalar('train/learning_rate', learning_rate, epoch)
        
        # Console and file output
        msg = (
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
            f"Train Acc: {train_metrics.get('accuracy', 0):.4f} | "
            f"Val Loss: {val_metrics.get('loss', 0):.4f} | "
            f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
        )
        
        if 'top5_accuracy' in val_metrics:
            msg += f" | Val Top-5: {val_metrics['top5_accuracy']:.4f}"
        
        if learning_rate is not None:
            msg += f" | LR: {learning_rate:.6f}"
        
        if self.console_log:
            print(msg)
        
        self._log_to_file(msg)
    
    def log_test_results(self, test_metrics: Dict[str, float]):
        """
        Log final test results.
        
        Args:
            test_metrics: Test metrics dictionary
        """
        if self.writer:
            for key, value in test_metrics.items():
                self.writer.add_scalar(f'test/{key}', value, 0)
        
        # Console and file output
        msg = "=" * 80 + "\n"
        msg += "FINAL TEST RESULTS\n"
        msg += "=" * 80 + "\n"
        for key, value in test_metrics.items():
            msg += f"{key:20s}: {value:.4f}\n"
        msg += "=" * 80
        
        if self.console_log:
            print(msg)
        
        self._log_to_file(msg)
    
    def log_config(self, config: Any):
        """
        Log configuration as hyperparameters.
        
        Args:
            config: Configuration object or dictionary
        """
        if not self.writer:
            return
        
        # Convert config to dictionary if needed
        if hasattr(config, '__dataclass_fields__'):
            from dataclasses import asdict
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            return
        
        # Flatten nested dicts for TensorBoard
        flat_config = {}
        for key, value in config_dict.items():
            if isinstance(value, (int, float, str, bool)):
                flat_config[key] = value
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float, str, bool)):
                        flat_config[f"{key}/{subkey}"] = subvalue
        
        try:
            self.writer.add_hparams(flat_config, {})
        except Exception as e:
            print(f"Warning: Could not log hyperparameters: {e}")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """
        Log model information.
        
        Args:
            model_info: Dictionary with model metadata
        """
        msg = "\n" + "=" * 80 + "\n"
        msg += "MODEL INFORMATION\n"
        msg += "=" * 80 + "\n"
        for key, value in model_info.items():
            msg += f"{key:20s}: {value}\n"
        msg += "=" * 80 + "\n"
        
        if self.console_log:
            print(msg)
        
        self._log_to_file(msg)
    
    def log_images(self, tag: str, images, step: int):
        """
        Log images to TensorBoard.
        
        Args:
            tag: Image tag/name
            images: Images tensor or grid
            step: Global step
        """
        if self.writer:
            self.writer.add_images(tag, images, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """
        Log histogram to TensorBoard.
        
        Args:
            tag: Histogram tag/name
            values: Values to histogram
            step: Global step
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def close(self):
        """Close the logger and flush all pending writes."""
        if self.writer:
            self.writer.close()
        
        self._log_to_file(f"\nEnded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.console_log:
            print(f"✅ Logging complete. Logs saved to: {self.experiment_dir}")


def setup_logger(config: Any) -> Logger:
    """
    Factory function to create a logger from config.
    
    Args:
        config: Training configuration
    
    Returns:
        Logger instance
    """
    logger = Logger(
        log_dir=config.log_dir,
        experiment_name=config.experiment_name,
        console_log=True
    )
    
    # Log configuration
    logger.log_config(config)
    
    return logger
