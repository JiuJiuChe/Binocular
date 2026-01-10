# Examples

This directory contains example scripts demonstrating how to use the dataset and model implementations.

## Test Dataset

The [`test_dataset.py`](test_dataset.py:1) script verifies that the dataset loading and preprocessing works correctly.

### Running the Test

Make sure you've run the preprocessing script first to generate the splits:

```bash
cd Lucy
uv run scripts/preprocess_nabirds.py
```

Then run the test script:

```bash
cd Lucy
uv run examples/test_dataset.py
```

### What the Test Does

1. **Dataset Loading**: Tests loading train, validation, and test splits
2. **DataLoader**: Tests creating a PyTorch DataLoader with batching
3. **Bounding Box Cropping**: Compares dataset with and without bbox cropping
4. **Class Mapping**: Verifies the class ID to species name mapping

### Expected Output

The test should print information about:
- Number of images and classes in each split
- Sample shapes after transforms (should be `[3, 224, 224]`)
- Batch processing with DataLoader
- First 10 species names

All tests should pass with a "✓ All tests passed!" message.

## Test Model

The [`test_model.py`](test_model.py:1) script verifies that the model components work correctly.

### Running the Test

```bash
cd Lucy
uv run examples/test_model.py
```

### What the Test Does

1. **Encoder Info**: Tests getting encoder configurations without loading models
2. **Encoder Loading**: Tests loading DINOv2 encoder and extracting features
3. **Linear Probe**: Tests creating classification head
4. **Complete Model**: Tests creating encoder + classifier with factory function
5. **Freeze/Unfreeze**: Tests toggling encoder training mode

### Expected Output

The test will:
- Load the DINOv2-ViT-S encoder (smallest variant for quick testing)
- Create a linear probe classifier
- Test forward pass with dummy data
- Verify parameter counts and shapes
- Test freeze/unfreeze functionality

All tests should pass with "ALL TESTS PASSED ✓" message.

## Usage Example - Dataset

```python
from datasets.nabirds import NABirdsDataset, get_train_transforms
from torch.utils.data import DataLoader

# Create dataset
dataset = NABirdsDataset(
    root='Binocular/datasets/nabirds',
    split='train',
    transform=get_train_transforms(image_size=224),
    use_bbox_crop=True
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Iterate
for images, labels in dataloader:
    # images: [batch_size, 3, 224, 224]
    # labels: [batch_size]
    pass
```

## Usage Example - Model

```python
from models import create_model
import torch

# Create DINOv2 model with linear probe
model = create_model(
    encoder_type='dinov2',
    num_classes=555,
    encoder_name='dinov2_vitb14',
    freeze_encoder=True,
    classifier_type='linear'
)

# Make predictions
model.eval()
images = torch.randn(4, 3, 224, 224)  # Batch of 4 images

with torch.no_grad():
    # Get logits
    logits = model(images)  # [4, 555]
    
    # Get top-5 predictions
    top_probs, top_classes = model.predict(images, top_k=5)
    # top_probs: [4, 5], top_classes: [4, 5]
```
