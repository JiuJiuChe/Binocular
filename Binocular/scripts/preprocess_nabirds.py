"""
Preprocessing script for NABirds dataset.
Creates train/validation/test splits with stratification by species.
"""

import os
import sys
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets.nabirds.utils import (
    load_train_test_split,
    load_image_labels,
    load_class_names
)


def create_stratified_splits(dataset_path, val_ratio=0.1, random_seed=42):
    """
    Create stratified train/validation/test splits.
    
    Args:
        dataset_path: Path to NABirds dataset
        val_ratio: Ratio of training data to use for validation
        random_seed: Random seed for reproducibility
    """
    import random
    random.seed(random_seed)
    
    print("Loading dataset metadata...")
    # Load the official train/test split
    train_images, test_images = load_train_test_split(dataset_path)
    
    # Load image labels for stratification
    image_labels = load_image_labels(dataset_path)
    
    # Load class names for verification
    class_names = load_class_names(dataset_path)
    
    print(f"Total training images: {len(train_images)}")
    print(f"Total test images: {len(test_images)}")
    print(f"Total classes: {len(class_names)}")
    
    # Group training images by class for stratified sampling
    class_to_images = defaultdict(list)
    for img_id in train_images:
        class_id = image_labels[img_id]
        class_to_images[class_id].append(img_id)
    
    # Stratified split: sample validation set from each class
    train_split = []
    val_split = []
    
    for class_id, images in class_to_images.items():
        # Shuffle images for this class
        shuffled = images.copy()
        random.shuffle(shuffled)
        
        # Calculate validation size for this class
        n_val = max(1, int(len(shuffled) * val_ratio))
        
        # Split
        val_split.extend(shuffled[:n_val])
        train_split.extend(shuffled[n_val:])
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_split)}")
    print(f"  Validation: {len(val_split)}")
    print(f"  Test: {len(test_images)}")
    
    # Save splits to files
    splits_dir = os.path.join(dataset_path, 'splits')
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(os.path.join(splits_dir, 'train.txt'), 'w') as f:
        for img_id in sorted(train_split):
            f.write(f"{img_id}\n")
    
    with open(os.path.join(splits_dir, 'val.txt'), 'w') as f:
        for img_id in sorted(val_split):
            f.write(f"{img_id}\n")
    
    with open(os.path.join(splits_dir, 'test.txt'), 'w') as f:
        for img_id in sorted(test_images):
            f.write(f"{img_id}\n")
    
    print(f"\nSplits saved to: {splits_dir}")
    
    # Verify stratification
    print("\nVerifying stratification...")
    train_classes = set(image_labels[img_id] for img_id in train_split)
    val_classes = set(image_labels[img_id] for img_id in val_split)
    test_classes = set(image_labels[img_id] for img_id in test_images)
    
    print(f"  Classes in train: {len(train_classes)}")
    print(f"  Classes in val: {len(val_classes)}")
    print(f"  Classes in test: {len(test_classes)}")
    print(f"  Classes in all splits: {len(train_classes | val_classes | test_classes)}")


if __name__ == '__main__':
    # Dataset path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, '..', 'datasets', 'nabirds', 'data')
    
    print(f"Dataset path: {dataset_path}")
    create_stratified_splits(dataset_path)
