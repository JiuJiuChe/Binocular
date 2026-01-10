"""
Model wrapper that combines encoder and classification head.

This module provides a unified interface for the complete classification pipeline:
encoder (frozen/unfrozen) → features → classification head → logits
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any


class EncoderClassifier(nn.Module):
    """
    Complete classification model combining encoder and classification head.
    
    This wrapper combines:
    1. A pretrained vision encoder (e.g., DINOv2, CLIP)
    2. A classification head (linear probe or MLP)
    
    The encoder can be frozen (linear probing) or unfrozen (fine-tuning).
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        freeze_encoder: bool = True
    ):
        """
        Initialize encoder-classifier model.
        
        Args:
            encoder: Pretrained vision encoder
            classifier: Classification head (LinearProbe or MLPProbe)
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        
        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and classifier.
        
        Args:
            x: Input images [batch_size, 3, H, W]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Ensure encoder is in eval mode if frozen
        if self.freeze_encoder:
            self.encoder.eval()
        
        # Extract features from encoder
        with torch.set_grad_enabled(not self.freeze_encoder):
            features = self.encoder(x)
        
        # Classify features
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification (useful for analysis).
        
        Args:
            x: Input images [batch_size, 3, H, W]
        
        Returns:
            features: Image features [batch_size, feature_dim]
        """
        if self.freeze_encoder:
            self.encoder.eval()
        
        with torch.set_grad_enabled(not self.freeze_encoder):
            features = self.encoder(x)
        
        return features
    
    def predict(self, x: torch.Tensor, top_k: int = 5) -> tuple:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input images [batch_size, 3, H, W]
            top_k: Number of top predictions to return
        
        Returns:
            top_probs: Top-k probabilities [batch_size, top_k]
            top_classes: Top-k class indices [batch_size, top_k]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_classes = torch.topk(probs, k=top_k, dim=-1)
        
        return top_probs, top_classes
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights for fine-tuning."""
        self.freeze_encoder = False
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder.train()
    
    def freeze_encoder_weights(self):
        """Freeze encoder weights (for linear probing)."""
        self.freeze_encoder = True
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    def get_trainable_params(self) -> int:
        """Count number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Count total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get summary information about the model."""
        return {
            'encoder_frozen': self.freeze_encoder,
            'total_params': self.get_total_params(),
            'trainable_params': self.get_trainable_params(),
            'encoder_type': type(self.encoder).__name__,
            'classifier_type': type(self.classifier).__name__,
        }


def create_model(
    encoder_type: str,
    num_classes: int,
    encoder_name: Optional[str] = None,
    freeze_encoder: bool = True,
    classifier_type: str = 'linear',
    classifier_kwargs: Optional[Dict[str, Any]] = None
) -> EncoderClassifier:
    """
    Factory function to create a complete encoder-classifier model.
    
    Args:
        encoder_type: Type of encoder ('dinov2', 'rope_vit', 'clip', 'openclip')
        num_classes: Number of output classes (555 for NABirds)
        encoder_name: Specific encoder variant (optional)
        freeze_encoder: Whether to freeze encoder weights
        classifier_type: Type of classifier ('linear' or 'mlp')
        classifier_kwargs: Additional arguments for classifier
    
    Returns:
        model: Complete EncoderClassifier model
    
    Examples:
        >>> # Create DINOv2 base with linear probe
        >>> model = create_model('dinov2', num_classes=555, encoder_name='dinov2_vitb14')
        >>> 
        >>> # Create with MLP head
        >>> model = create_model(
        ...     'dinov2', 
        ...     num_classes=555,
        ...     classifier_type='mlp',
        ...     classifier_kwargs={'hidden_dims': [512], 'dropout': 0.1}
        ... )
    """
    from .encoders import get_encoder, get_encoder_info
    from .linear_probe import LinearProbe, MLPProbe
    
    # Get encoder info to determine feature dimension
    info = get_encoder_info(encoder_type, encoder_name)
    feature_dim = info.feature_dim
    
    # Create encoder
    encoder = get_encoder(encoder_type, encoder_name, freeze=freeze_encoder)
    
    # Create classifier
    classifier_kwargs = classifier_kwargs or {}
    if classifier_type == 'linear':
        classifier = LinearProbe(feature_dim, num_classes, **classifier_kwargs)
    elif classifier_type == 'mlp':
        classifier = MLPProbe(feature_dim, num_classes, **classifier_kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    # Combine into complete model
    model = EncoderClassifier(encoder, classifier, freeze_encoder=freeze_encoder)
    
    return model
