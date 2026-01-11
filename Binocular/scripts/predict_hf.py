"""
Command-line script for running inference on a single image using a model 
from the Hugging Face Hub.
"""

import argparse
from pathlib import Path
import sys

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Binocular.models.inference import InferenceModel

def main():
    """Main function to handle command-line arguments and run prediction."""
    parser = argparse.ArgumentParser(
        description="Run bird species classification on a single image from a Hugging Face model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        default='jiujiuche/binocular',
        help="The Hugging Face repository ID to download the model from."
    )
    parser.add_argument(
        '--filename',
        type=str,
        default='artifacts/dinov2_vitb14_nabirds.pth',
        help="The name of the model file in the Hugging Face repository."
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help="Number of top predictions to display."
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cuda', 'mps', 'cpu'],
        help="Device to use for inference."
    )

    args = parser.parse_args()

    # --- 1. Initialize the Inference Model from Hugging Face ---
    try:
        print(f"Loading model from Hugging Face repository: {args.repo_id}/{args.filename}...")
        model = InferenceModel.from_pretrained(repo_id=args.repo_id, filename=args.filename, device=args.device)
    except Exception as e:
        print(f"An error occurred during model loading: {e}")
        sys.exit(1)

    # --- 2. Run Prediction ---
    try:
        print(f"\nRunning prediction on image: {args.image}...")
        predictions = model.predict(image=args.image, top_k=args.top_k)
    except FileNotFoundError:
        print(f"Error: Image file not found at {args.image}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        sys.exit(1)

    # --- 3. Display Results ---
    print(f"\nTop {args.top_k} Predictions:")
    print("-" * 30)
    for i, (class_name, probability) in enumerate(predictions):
        print(f"{i+1}. {class_name:<20} | Confidence: {probability:.2%}")
    print("-" * 30)


if __name__ == "__main__":
    main()
