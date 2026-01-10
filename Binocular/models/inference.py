"""
Inference interface for the NABirds Species Classifier.

This module provides a high-level `InferenceModel` class to easily load a
trained model from a checkpoint and run predictions on new images.

Key Features:
- Load model and configuration from a single checkpoint file.
- Handle necessary image preprocessing (resize, crop, normalize).
- Predict on image file paths or PIL Image objects.
- Return top-K predictions with class names and confidence scores.
"""

from typing import Union, List, Tuple
from pathlib import Path
import torch
from PIL import Image

from Binocular.models.wrappers import create_model
from Binocular.datasets.nabirds.transforms import get_val_transforms
from huggingface_hub import hf_hub_download
 
from Binocular.configs import TrainingConfig


class InferenceModel:
    """
    A wrapper for a trained bird species classifier for easy inference.
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = 'cpu'):
        """
        Initialize the inference model.

        Args:
            checkpoint_path (Union[str, Path]): Path to the model checkpoint (.pth file).
            device (str): The device to run inference on ('cuda', 'mps', 'cpu').
        """
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found at {self.checkpoint_path}")

        # Load checkpoint and configuration
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        self.config = TrainingConfig(**checkpoint['config'])
        
        # Load class names from the checkpoint
        if 'class_names' not in checkpoint:
            raise ValueError("Checkpoint must contain 'class_names' for inference.")
        self.class_names = checkpoint['class_names']
        self.idx_to_class_name = {i: name for i, name in enumerate(self.class_names)}
        num_classes = len(self.class_names)

        # Create model and load weights
        self.model = self._load_model(checkpoint, num_classes)
        self.model.to(self.device)
        self.model.eval()

        # Get image transforms
        self.transforms = get_val_transforms(image_size=self.config.image_size)
        
        print("✅ Inference model loaded successfully.")
        print(f"   - Model: {self.config.encoder_name}")
        print(f"   - Trained for {checkpoint.get('epoch', 'N/A')} epochs.")
        print(f"   - Device: {self.device}")


    @classmethod
    def from_pretrained(cls, repo_id: str, filename: str, device: str = 'cpu'):
        """
        Load a pretrained model from a Hugging Face Hub repository.

        Args:
            repo_id (str): The ID of the repository to load from (e.g., 'your-username/your-repo').
            filename (str): The name of the checkpoint file in the repository.
            device (str): The device to run inference on.

        Returns:
            InferenceModel: An instance of the model.
        """
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return cls(checkpoint_path, device=device)


    def _load_model(self, checkpoint: dict, num_classes: int):
        """Creates model from config and loads state dict from checkpoint."""
        model = create_model(
            encoder_type=self.config.encoder_type,
            encoder_name=self.config.encoder_name,
            num_classes=num_classes,
            freeze_encoder=self.config.freeze_encoder,
            classifier_type=self.config.classifier_type,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Run inference on a single image.

        Args:
            image (Union[str, Path, Image.Image]): Path to the image file or a PIL Image object.
            top_k (int): The number of top predictions to return.

        Returns:
            List[Tuple[str, float]]: A list of (class_name, probability) tuples.
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError("Input 'image' must be a file path or a PIL.Image object.")

        # Apply transformations
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)

        # Get model output
        logits = self.model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)

        # Get top K predictions
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

        # Format results
        top_probs = top_probs.squeeze().cpu().numpy()
        top_indices = top_indices.squeeze().cpu().numpy()

        results = []
        for idx, prob in zip(top_indices, top_probs):
            class_name = self.idx_to_class_name[idx]
            results.append((class_name, float(prob)))

        return results
