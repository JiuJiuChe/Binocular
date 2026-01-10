"""
Script to upload a model checkpoint to the Hugging Face Hub.

This script simplifies the process of uploading a large model file using Git LFS.

Prerequisites:
1. Make sure you have `git-lfs` installed.
   - On macOS: `brew install git-lfs`
   - On Debian/Ubuntu: `sudo apt-get install git-lfs`
   - Then run `git lfs install` in your terminal.
2. Log in to your Hugging Face account.
   - Run `huggingface-cli login` in your terminal and provide your token.

Example usage:
    python Binocular/scripts/upload_to_hf.py \
        --repo-id "jiujiuche/binocular" \
        --checkpoint "path/to/your/best_with_classes.pth" \
        --commit-message "Upload initial model checkpoint"
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, get_token
from shutil import copyfile

def upload_to_hub(repo_id: str, checkpoint_path: str, path_in_repo: str, commit_message: str):
    """
    Clones a Hugging Face repository, adds a file, and pushes it with Git LFS.

    Args:
        repo_id (str): The ID of the repository (e.g., 'jiujiuche/binocular').
        checkpoint_path (str): The local path to the model checkpoint to upload.
        commit_message (str): The commit message for the upload.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")

    # Get token from cache
    token = get_token()
    if token is None:
        raise EnvironmentError(
            "Hugging Face token not found. Please log in using `huggingface-cli login`."
        )

    print(f"Uploading {checkpoint_path.name} to repository: {repo_id}")

    # Use HfApi to upload the file. This handles git-lfs automatically.
    api = HfApi()
    
    api.upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )

    print("\n✅ Upload complete!")
    print(f"Check your model at: https://huggingface.co/{repo_id}/tree/main")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a model checkpoint to the Hugging Face Hub.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help="The Hugging Face Hub repository ID (e.g., 'jiujiuche/binocular')."
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Path to the model checkpoint file to upload."
    )
    parser.add_argument(
        '--commit-message',
        type=str,
        default="Upload model checkpoint",
        help="The commit message for the upload."
    )
    parser.add_argument(
        '--path-in-repo',
        type=str,
        default=None,
        help="The destination path of the file in the repository. If not set, defaults to the filename."
    )
    args = parser.parse_args()

    try:
        # Determine the destination path
        path_in_repo = args.path_in_repo if args.path_in_repo else Path(args.checkpoint).name
        upload_to_hub(args.repo_id, args.checkpoint, path_in_repo, args.commit_message)
    except Exception as e:
        print(f"❌ An error occurred during upload: {e}")


if __name__ == '__main__':
    main()
