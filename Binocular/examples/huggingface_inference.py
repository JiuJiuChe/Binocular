"""
Example of loading and using a model from the Hugging Face Hub.
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Binocular.models.inference import InferenceModel


def run_huggingface_inference(repo_id: str, image_path: str):
    """
    Loads a model from the Hugging Face Hub and runs inference.

    Args:
        repo_id (str): The Hugging Face repository ID (e.g., 'jiujiuche/binocular').
        image_path (str): Path to the image to classify.
    """
    print(f"Loading model from Hugging Face Hub repo: {repo_id}")
    
    # This will download the model from the Hub and cache it locally
    model = InferenceModel.from_pretrained(
        repo_id=repo_id,
        filename='dinov2_vitb14_nabirds.pth',  # Assuming this is the name of your file on the Hub
        device='cpu'  # Or 'cuda', 'mps'
    )
    
    print("\nRunning prediction...")
    predictions = model.predict(image_path, top_k=5)
    
    print(f"\nTop 5 predictions for '{image_path}':")
    for class_name, probability in predictions:
        print(f"  - {class_name:<30} | Confidence: {probability:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a model from the Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help="The Hugging Face Hub repository ID (e.g., 'jiujiuche/binocular')."
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help="Path to the input image file."
    )
    args = parser.parse_args()
    
    run_huggingface_inference(args.repo_id, args.image)


if __name__ == '__main__':
    main()
