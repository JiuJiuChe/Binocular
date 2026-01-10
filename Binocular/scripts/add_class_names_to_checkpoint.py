"""
Script to add class names to an existing model checkpoint.

This is useful for making checkpoints self-contained for inference,
so they don't have a dependency on the dataset code to retrieve
class names.

Example usage:
    python Binocular/scripts/add_class_names_to_checkpoint.py \
        --checkpoint artifacts/checkpoints/dinov2_vitb14_rtx4090/best.pth \
        --output artifacts/checkpoints/dinov2_vitb14_rtx4090/best_with_classes.pth
"""

import argparse
from pathlib import Path
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Binocular.datasets.nabirds.dataset import NABirdsDataset


def add_class_names(checkpoint_path: str, output_path: str, dataset_root: str):
    """
    Adds class names from NABirdsDataset to a checkpoint file.

    Args:
        checkpoint_path (str): Path to the input checkpoint.
        output_path (str): Path to save the modified checkpoint.
        dataset_root (str): Path to the root of the NABirds dataset.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'class_names' in checkpoint:
        print("✅ Class names already exist in the checkpoint. No action needed.")
        return

    print("📚 Loading class names from NABirds dataset...")
    # We only need the dataset to extract the class names.
    # Split and other parameters don't matter for this purpose.
    dataset = NABirdsDataset(root=dataset_root, split='train')
    class_names = [dataset.get_class_name(i) for i in range(dataset.get_num_classes())]

    # Add class names to the checkpoint
    checkpoint['class_names'] = class_names

    print(f"💾 Saving updated checkpoint to {output_path}...")
    torch.save(checkpoint, output_path)
    print("✅ Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Add class names to a model checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to the input checkpoint file."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Path to save the updated checkpoint file."
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='Binocular/datasets/nabirds/data/',
        help="Root directory of the NABirds dataset."
    )
    args = parser.parse_args()

    try:
        add_class_names(args.checkpoint, args.output, args.dataset_root)
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == '__main__':
    main()
