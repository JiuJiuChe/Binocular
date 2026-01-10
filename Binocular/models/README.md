# Models Module

Encoder-agnostic model components for bird species classification.

## Overview

This module provides a flexible, encoder-agnostic architecture for image classification:

```
Image → Encoder (frozen/unfrozen) → Features → Classifier → Logits
```

## Quick Start

```python
from models import create_model

# Create a DINOv2-base model with frozen encoder
model = create_model(
    encoder_type='dinov2',
    num_classes=555,
    encoder_name='dinov2_vitb14',
    freeze_encoder=True,
    classifier_type='linear'
)

# Make predictions
import torch
images = torch.randn(4, 3, 224, 224)
model.eval()

with torch.no_grad():
    logits = model(images)  # [4, 555]
    top_probs, top_classes = model.predict(images, top_k=5)
```

## Components

### Encoders ([`encoders.py`](encoders.py:1))

Vision encoders that extract features from images.

**Implemented:**
- **DINOv2** (Meta AI): 4 variants (ViT-S/B/L/G)

**Planned:**
- RoPE ViT
- CLIP
- OpenCLIP

#### Usage

```python
from models import get_encoder, get_encoder_info

# Get encoder info without loading
info = get_encoder_info('dinov2', 'dinov2_vitb14')
print(f"Feature dim: {info.feature_dim}")  # 768

# Load encoder
encoder = get_encoder('dinov2', 'dinov2_vitb14', freeze=True)
features = encoder(images)  # [batch_size, 768]
```

### Classification Heads ([`linear_probe.py`](linear_probe.py:1))

Map encoder features to class predictions.

**LinearProbe**: Simple linear layer (for linear probing)
**MLPProbe**: Multi-layer perceptron (for more capacity)

#### Usage

```python
from models import LinearProbe

# Create linear probe
probe = LinearProbe(
    feature_dim=768,      # DINOv2-ViT-B
    num_classes=555,      # NABirds species
    dropout=0.0
)

logits = probe(features)  # [batch_size, 555]
```

### Model Wrapper ([`wrappers.py`](wrappers.py:1))

Combines encoder and classifier into a complete model.

**EncoderClassifier**: Full model with encoder + classifier
**create_model()**: Factory function for easy creation

#### Usage

```python
from models.wrappers import EncoderClassifier, create_model

# Method 1: Factory function (recommended)
model = create_model(
    encoder_type='dinov2',
    num_classes=555,
    encoder_name='dinov2_vitb14',
    freeze_encoder=True,
    classifier_type='linear'
)

# Method 2: Manual construction
from models import get_encoder, LinearProbe

encoder = get_encoder('dinov2', 'dinov2_vitb14', freeze=True)
classifier = LinearProbe(768, 555)
model = EncoderClassifier(encoder, classifier, freeze_encoder=True)

# Forward pass
logits = model(images)

# Get features
features = model.get_features(images)

# Top-K predictions
top_probs, top_classes = model.predict(images, top_k=5)

# Model info
info = model.get_model_info()
print(info)
```

## DINOv2 Models

| Model | Feature Dim | Params | Size |
|-------|-------------|--------|------|
| dinov2_vits14 | 384 | 22M | ~85MB |
| dinov2_vitb14 | 768 | 86M | ~330MB |
| dinov2_vitl14 | 1024 | 304M | ~1.1GB |
| dinov2_vitg14 | 1536 | 1.1B | ~4.2GB |

**Recommendation**: `dinov2_vitb14` offers the best balance of performance and speed.

## Freeze vs. Unfreeze

### Linear Probing (Frozen Encoder)

Only trains the classification head. Fast and effective for transfer learning.

```python
model = create_model('dinov2', num_classes=555, freeze_encoder=True)
print(model.get_trainable_params())  # ~427K (classifier only)
```

### Fine-Tuning (Unfrozen Encoder)

Trains the entire model. Slower but potentially better performance.

```python
model = create_model('dinov2', num_classes=555, freeze_encoder=False)
print(model.get_trainable_params())  # ~86M (all parameters)

# Or toggle dynamically
model.freeze_encoder_weights()  # Freeze
model.unfreeze_encoder()        # Unfreeze
```

## Integration with Dataset

```python
from datasets.nabirds import NABirdsDataset, get_train_transforms
from models import create_model
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Create dataset and loader
dataset = NABirdsDataset(
    root='Binocular/datasets/nabirds',
    split='train',
    transform=get_train_transforms(image_size=224),
    use_bbox_crop=True
)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Create model
model = create_model('dinov2', num_classes=555, freeze_encoder=True)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Training loop
model.train()
for images, labels in loader:
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
```

## Testing

Run the test suite:

```bash
cd Lucy
uv run examples/test_model.py
```

See [`../examples/test_model.py`](../examples/test_model.py:1) for comprehensive tests.

## API Reference

### `get_encoder(encoder_type, model_name=None, freeze=True)`

Load a vision encoder.

**Args:**
- `encoder_type` (str): Type of encoder ('dinov2', 'rope_vit', 'clip', 'openclip')
- `model_name` (str, optional): Specific model variant
- `freeze` (bool): Whether to freeze weights

**Returns:**
- `encoder` (nn.Module): Encoder module

### `get_encoder_info(encoder_type, model_name=None)`

Get encoder configuration without loading.

**Args:**
- `encoder_type` (str): Type of encoder
- `model_name` (str, optional): Specific model variant

**Returns:**
- `info` (EncoderInfo): Configuration with `feature_dim` and `image_size`

### `create_model(encoder_type, num_classes, **kwargs)`

Create complete model with encoder + classifier.

**Args:**
- `encoder_type` (str): Type of encoder
- `num_classes` (int): Number of output classes
- `encoder_name` (str, optional): Specific encoder variant
- `freeze_encoder` (bool): Whether to freeze encoder (default: True)
- `classifier_type` (str): Type of classifier ('linear' or 'mlp')
- `classifier_kwargs` (dict, optional): Additional classifier arguments

**Returns:**
- `model` (EncoderClassifier): Complete model

## Design Philosophy

1. **Encoder-Agnostic**: Easy to swap between different backbones
2. **PyTorch Native**: Standard `nn.Module` conventions
3. **Frozen-First**: Optimized for linear probing, fine-tuning as option
4. **Factory Functions**: Simple model creation
5. **Type Hints**: Better IDE support and documentation
6. **Well-Tested**: Comprehensive test coverage

## Adding New Encoders

To add a new encoder:

1. Create an encoder class in [`encoders.py`](encoders.py:1) following the pattern
2. Add configuration to `ENCODER_CONFIGS`
3. Update `get_encoder()` factory function
4. Update `get_encoder_info()` function
5. Add tests to [`../examples/test_model.py`](../examples/test_model.py:1)

Example template:

```python
class MyEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        # Load pretrained model
        self.model = load_pretrained(model_name)
        self.feature_dim = get_feature_dim(model_name)
        
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_feature_dim(self) -> int:
        return self.feature_dim
```

## See Also

- [`../examples/test_model.py`](../examples/test_model.py:1) - Test suite
- [`../examples/README.md`](../examples/README.md:1) - Usage examples
- [`../../plans/PHASE_1_SUMMARY.md`](../../plans/PHASE_1_SUMMARY.md:1) - Phase 1 summary
