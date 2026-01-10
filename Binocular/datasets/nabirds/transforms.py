"""
Custom transforms for NABirds dataset preprocessing.
Includes bounding box cropping and standard augmentations.
"""

from PIL import Image
from torchvision import transforms


class CropBoundingBox:
    """
    Crop image to bounding box coordinates.
    
    Bounding box format: (x, y, width, height)
    where (x, y) is the top-left corner.
    """
    
    def __call__(self, image, bbox):
        """
        Args:
            image: PIL Image
            bbox: tuple of (x, y, width, height)
        
        Returns:
            Cropped PIL Image
        """
        x, y, width, height = bbox
        return image.crop((x, y, x + width, y + height))


def get_train_transforms(image_size=224, normalize=True):
    """
    Get training transforms including bounding box crop and augmentations.
    
    Args:
        image_size: Target image size (assumes square)
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        transforms.Compose object (note: bbox crop must be applied separately)
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ]
    
    if normalize:
        # ImageNet normalization
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_list)


def get_val_transforms(image_size=224, normalize=True):
    """
    Get validation/test transforms (no augmentation).
    
    Args:
        image_size: Target image size (assumes square)
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        transforms.Compose object (note: bbox crop must be applied separately)
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        # ImageNet normalization
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        )
    
    return transforms.Compose(transform_list)
