# NABirds Dataset Module

This module provides a clean PyTorch Dataset implementation for the NABirds species classification dataset.

## Features

- ✅ **Species-level classification** with 555 bird species
- ✅ **Stratified train/validation/test splits** for reproducibility
- ✅ **Bounding box cropping** to focus on the bird
- ✅ **Standard transforms** including ImageNet normalization
- ✅ **Class ID to species name mapping**

## Dataset Statistics

After preprocessing:
- **Train**: 21,730 images
- **Validation**: 2,199 images  
- **Test**: 24,633 images
- **Classes**: 555 species (stratified across all splits)

## Files

- [`dataset.py`](dataset.py:1) - PyTorch Dataset implementation
- [`transforms.py`](transforms.py:1) - Image transforms including bbox cropping
- [`utils.py`](utils.py:1) - Utility functions for loading metadata
- [`nabirds.py`](nabirds.py:1) - Original NABirds loader (Python 2, for reference)

## Quick Start

### 1. Generate Splits

First, run the preprocessing script to create train/val/test splits:

```bash
cd Lucy
uv run scripts/preprocess_nabirds.py
```

This creates three files in `Binocular/datasets/nabirds/splits/`:
- `train.txt` - Training image IDs
- `val.txt` - Validation image IDs
- `test.txt` - Test image IDs

### 2. Use the Dataset

```python
from datasets.nabirds import NABirdsDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader

# Training dataset
train_dataset = NABirdsDataset(
    root='Binocular/datasets/nabirds',
    split='train',
    transform=get_train_transforms(image_size=224),
    use_bbox_crop=True
)

# Validation dataset
val_dataset = NABirdsDataset(
    root='Binocular/datasets/nabirds',
    split='val',
    transform=get_val_transforms(image_size=224),
    use_bbox_crop=True
)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Get class information
num_classes = train_dataset.get_num_classes()  # 555
class_names = train_dataset.get_class_names_list()
```

## Transforms

### Training Transforms
- Resize to target size (default 224x224)
- Random horizontal flip (p=0.5)
- Color jitter (brightness, contrast, saturation, hue)
- ImageNet normalization

### Validation/Test Transforms
- Resize to target size (default 224x224)
- ImageNet normalization (no augmentation)

### Bounding Box Cropping

All images are cropped to their bounding boxes before applying other transforms. This focuses the model on the bird rather than background. You can disable this with `use_bbox_crop=False`.

## Dataset Format

The dataset returns:
- **image**: Tensor of shape `[3, H, W]` (after transforms)
- **label**: Integer in range `[0, num_classes-1]`

To convert labels to species names:
```python
label = 42
species_name = dataset.get_class_name(label)
```

## Testing

Run the test script to verify everything works:

```bash
cd Lucy
uv run examples/test_dataset.py
```

See [`examples/README.md`](../../examples/README.md:1) for more details.

## Implementation Details

### Stratified Splitting
The validation set is created by taking a 10% stratified sample from the official training set, ensuring all species are represented in all splits.

### Class Mapping
The dataset maintains two mappings:
- `class_id_to_idx`: Maps original NABirds class IDs (strings) to PyTorch indices (0 to N-1)
- `idx_to_class_id`: Reverse mapping for getting species names

This ensures compatibility with PyTorch loss functions while preserving the original species information.

### Reproducibility
All splits use a fixed random seed (42) for reproducibility. The same splits will be generated every time.
