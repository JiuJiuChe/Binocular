"""
Test script for NABirds dataset loading.
Verifies that the dataset can be loaded and that transforms work correctly.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datasets.nabirds import NABirdsDataset, get_train_transforms, get_val_transforms
from torch.utils.data import DataLoader


def test_dataset_loading():
    """Test basic dataset loading."""
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)
    
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'nabirds', 'data')
    
    # Test loading all splits
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} Split:")
        print("-" * 40)
        
        # Get appropriate transforms
        if split == 'train':
            transform = get_train_transforms(image_size=224)
        else:
            transform = get_val_transforms(image_size=224)
        
        # Load dataset
        dataset = NABirdsDataset(
            root=dataset_path,
            split=split,
            transform=transform,
            use_bbox_crop=True
        )
        
        print(f"  Number of images: {len(dataset)}")
        print(f"  Number of classes: {dataset.get_num_classes()}")
        
        # Test loading first sample
        image, label = dataset[0]
        print(f"  First sample shape: {image.shape}")
        print(f"  First sample label: {label}")
        print(f"  First sample class: {dataset.get_class_name(label)}")


def test_dataloader():
    """Test DataLoader with the dataset."""
    print("\n" + "=" * 60)
    print("Testing DataLoader")
    print("=" * 60)
    
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'nabirds', 'data')
    
    # Create dataset
    dataset = NABirdsDataset(
        root=dataset_path,
        split='train',
        transform=get_train_transforms(image_size=224),
        use_bbox_crop=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # Use 0 for testing
    )
    
    print(f"\nDataLoader batch size: {dataloader.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get one batch
    images, labels = next(iter(dataloader))
    print(f"\nBatch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Batch labels: {labels.tolist()}")


def test_bbox_cropping():
    """Test with and without bbox cropping."""
    print("\n" + "=" * 60)
    print("Testing Bounding Box Cropping")
    print("=" * 60)
    
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'nabirds', 'data')
    
    # Without bbox crop
    dataset_no_crop = NABirdsDataset(
        root=dataset_path,
        split='train',
        transform=get_train_transforms(image_size=224),
        use_bbox_crop=False
    )
    
    # With bbox crop
    dataset_with_crop = NABirdsDataset(
        root=dataset_path,
        split='train',
        transform=get_train_transforms(image_size=224),
        use_bbox_crop=True
    )
    
    print(f"\nWithout bbox crop - image shape: {dataset_no_crop[0][0].shape}")
    print(f"With bbox crop - image shape: {dataset_with_crop[0][0].shape}")
    print("\nBoth should be (3, 224, 224) after transforms")


def test_class_mapping():
    """Test class ID to name mapping."""
    print("\n" + "=" * 60)
    print("Testing Class Mapping")
    print("=" * 60)
    
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'nabirds', 'data')
    
    dataset = NABirdsDataset(
        root=dataset_path,
        split='train',
        transform=None,
        use_bbox_crop=False
    )
    
    print(f"\nTotal classes: {dataset.get_num_classes()}")
    print("\nFirst 10 class names:")
    for i in range(min(10, dataset.get_num_classes())):
        print(f"  {i}: {dataset.get_class_name(i)}")


if __name__ == '__main__':
    try:
        test_dataset_loading()
        test_dataloader()
        test_bbox_cropping()
        test_class_mapping()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
