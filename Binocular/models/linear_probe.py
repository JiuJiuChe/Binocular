"""
Linear probe classification head for frozen encoder features.

A linear probe is a simple linear layer that maps frozen encoder features
to class predictions. It's commonly used to evaluate the quality of pretrained
representations.
"""

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Linear classification head for species prediction.
    
    This is a simple linear layer that maps encoder features to class logits.
    When used with a frozen encoder, this is known as "linear probing" - a
    technique to evaluate pretrained representations without fine-tuning.
    """
    
    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0):
        """
        Initialize linear probe.
        
        Args:
            feature_dim: Dimension of input features from encoder
            num_classes: Number of output classes (555 for NABirds)
            dropout: Dropout probability (default: 0.0)
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.dropout = dropout
        
        layers = []
        
        # Optional dropout for regularization
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        
        # Linear classification layer
        layers.append(nn.Linear(feature_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Map features to class logits.
        
        Args:
            features: Encoder features [batch_size, feature_dim]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.classifier(features)
    
    def get_num_classes(self) -> int:
        """Return the number of output classes."""
        return self.num_classes


class MLPProbe(nn.Module):
    """
    MLP classification head with hidden layers.
    
    This is a more complex probe with one or more hidden layers. Can be used
    if simple linear probing is not sufficient.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dims: list = [512],
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        Initialize MLP probe.
        
        Args:
            feature_dim: Dimension of input features from encoder
            num_classes: Number of output classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu', 'silu')
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        
        # Select activation function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU
        elif activation.lower() == 'silu':
            act_fn = nn.SiLU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build MLP layers
        layers = []
        in_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Map features to class logits through MLP.
        
        Args:
            features: Encoder features [batch_size, feature_dim]
        
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        return self.classifier(features)
    
    def get_num_classes(self) -> int:
        """Return the number of output classes."""
        return self.num_classes
