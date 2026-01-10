"""
Utilities for downloading model checkpoints from a central location.

This module provides a mechanism similar to Hugging Face's `from_pretrained`
to download model checkpoints from a URL and cache them locally.
"""

import os
import torch
from pathlib import Path
from torch.hub import download_url_to_file

# --- Model Registry ---
# Maps a model name to its public URL.
#
# TODO: Upload your checkpoint and replace the placeholder URL.
# A good place to host it is in a GitHub Release for your repository.
# ----------------------
PRETRAINED_MODELS = {
    'dinov2_vitb14_nabirds_v1': 'https://example.com/path/to/your/best_with_classes.pth',
    # Add other models here in the future
}


def get_cache_dir() -> Path:
    """
    Returns the cache directory for storing downloaded models.
    
    Defaults to ~/.cache/Binocular_classifier, creating it if it doesn't exist.
    """
    cache_dir = Path(os.path.expanduser('~')) / '.cache' / 'Binocular_classifier'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_checkpoint(model_name: str) -> str:
    """
    Downloads a checkpoint from the registry if not already cached.

    Args:
        model_name (str): The name of the model to download (must be in PRETRAINED_MODELS).

    Returns:
        str: The local file path to the cached checkpoint.
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(PRETRAINED_MODELS.keys())}")

    url = PRETRAINED_MODELS[model_name]
    cache_dir = get_cache_dir()
    
    # Use the filename from the URL for the cached file
    filename = Path(url).name
    cached_path = cache_dir / filename

    if not cached_path.exists():
        print(f"Downloading '{model_name}' from {url}...")
        try:
            download_url_to_file(url, str(cached_path), progress=True)
            print(f"✅ Download complete. Saved to {cached_path}")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            # Clean up partial download if it exists
            if cached_path.exists():
                os.remove(cached_path)
            raise
    else:
        print(f"✅ Using cached model from {cached_path}")
        
    return str(cached_path)
