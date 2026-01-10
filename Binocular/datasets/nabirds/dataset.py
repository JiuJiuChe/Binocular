"""
PyTorch Dataset implementation for NABirds species classification.
"""

import os
from PIL import Image
from torch.utils.data import Dataset

from .utils import (
    load_class_names,
    load_image_labels,
    load_image_paths,
    load_bounding_box_annotations
)
from .transforms import CropBoundingBox


class NABirdsDataset(Dataset):
    """
    NABirds dataset for species-level bird classification.
    
    Features:
    - Uses pre-generated train/val/test splits
    - Crops images to bounding boxes
    - Returns species-level labels
    - Supports custom transforms
    """
    
    def __init__(self, root, split='train', transform=None, use_bbox_crop=True):
        """
        Args:
            root: Path to NABirds dataset directory
            split: One of 'train', 'val', or 'test'
            transform: torchvision transforms to apply after bbox crop
            use_bbox_crop: Whether to crop images to bounding boxes
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.use_bbox_crop = use_bbox_crop
        
        # Load dataset metadata
        self.class_names = load_class_names(root)
        self.image_labels = load_image_labels(root)
        self.image_paths = load_image_paths(root, path_prefix=os.path.join(root, 'images'))
        
        if use_bbox_crop:
            self.bboxes = load_bounding_box_annotations(root)
            self.bbox_crop = CropBoundingBox()
        else:
            self.bboxes = None
            self.bbox_crop = None
        
        # Load split file
        split_file = os.path.join(root, 'splits', f'{split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Please run preprocessing script first: python Binocular/scripts/preprocess_nabirds.py"
            )
        
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f if line.strip()]
        
        # Create class_id to integer mapping for PyTorch
        unique_class_ids = sorted(set(self.image_labels.values()))
        self.class_id_to_idx = {class_id: idx for idx, class_id in enumerate(unique_class_ids)}
        self.idx_to_class_id = {idx: class_id for class_id, idx in self.class_id_to_idx.items()}
        
        print(f"Loaded {split} split: {len(self.image_ids)} images, {len(unique_class_ids)} classes")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor of shape (C, H, W)
            label: Integer class label (0 to num_classes-1)
        """
        image_id = self.image_ids[idx]
        
        # Load image
        image_path = self.image_paths[image_id]
        image = Image.open(image_path).convert('RGB')
        
        # Crop to bounding box if enabled
        if self.use_bbox_crop and self.bboxes is not None:
            bbox = self.bboxes[image_id]
            image = self.bbox_crop(image, bbox)
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get label
        class_id = self.image_labels[image_id]
        label = self.class_id_to_idx[class_id]
        
        return image, label
    
    def get_class_name(self, label):
        """
        Get species name from integer label.
        
        Args:
            label: Integer class label (0 to num_classes-1)
        
        Returns:
            Species name string
        """
        class_id = self.idx_to_class_id[label]
        return self.class_names[class_id]
    
    def get_num_classes(self):
        """Returns the number of species classes in this split."""
        return len(self.class_id_to_idx)
    
    def get_class_names_list(self):
        """
        Returns list of all class names in label order.
        
        Returns:
            List of species names, indexed by label
        """
        return [self.get_class_name(i) for i in range(self.get_num_classes())]
