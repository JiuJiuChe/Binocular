# NABirds Species Classifier

<img src="./Binocular/assets/binacular.png" width="100" alt="Binocular">

This project provides a pretrained, species-level bird classifier for North American birds, built on the [NABirds dataset](https://dl.allaboutbirds.org/nabirds). It uses a powerful DINOv2 Vision Transformer (ViT) backbone with a linear probe, allowing for accurate and efficient classification.

The model can be easily loaded and used for inference directly from the [Hugging Face Hub](https://huggingface.co/jiujiuche/Binocular).

## Quick Start: Inference with the Published Package

Get predictions from the pretrained model in just a few lines of code.

### 1. Installation

Install the package directly from PyPI:

```bash
pip install binocular-birds
```

### 2. Run Inference from the Command Line

After installation, you can use the `predict_hf` command-line tool to run inference on an image. The tool will automatically download the pretrained model from the Hugging Face Hub.

```bash
predict_hf --image "path/to/your/bird_image.jpg"
```

The output will be a list of the top 5 predicted bird species and their confidence scores.

### 3. Run Inference in Python

You can also use the `InferenceModel` in your own Python scripts:

```python
from Binocular.models.inference import InferenceModel
from PIL import Image

# Initialize the model from Hugging Face
model = InferenceModel.from_pretrained(
    repo_id="jiujiuche/artifacts",
    filename="dinov2_vitb14_nabirds.pth"
)

# Open an image
img = Image.open("path/to/your/bird_image.jpg")

# Get predictions
predictions = model.predict(img, top_k=5)

# Print the results
for species, confidence in predictions:
    print(f"{species}: {confidence:.2%}")
```

## Training (Optional)

This repository also contains the full pipeline for training the model from scratch.

### Dataset Setup

1.  Download the NABirds dataset from [https://dl.allaboutbirds.org/nabirds](https://dl.allaboutbirds.org/nabirds).
2.  Extract the dataset files into the `Binocular/datasets/nabirds/` directory.

### Training Command

The training process is managed through a main script that uses configuration files to define the training parameters.

```bash
# Example for training on an M2 Mac
python Binocular/scripts/train.py --config Binocular/configs/m2_dev.yaml

# Example for training on a CUDA-enabled GPU
python Binocular/scripts/train.py --config Binocular/configs/rtx4090.yaml
```

Training configurations can be found in the `Binocular/configs/` directory.

## Project Structure

```
Binocular/
├── assets/           # Project icons
├── datasets/         # Data loaders and dataset files
├── models/           # Model implementations and inference logic
├── configs/          # Configuration files for experiments
├── scripts/          # Training, evaluation, and utility scripts
└── examples/         # Example scripts for using the models
```

## Dependencies

- **PyTorch**
- **torchvision**
- **huggingface-hub**
- **Pillow**, **tqdm**, **numpy**, etc.

See [`pyproject.toml`](./pyproject.toml) for the complete list.

## License

The code in this repository is provided under the MIT License. The NABirds dataset has its own [license terms](https://dl.allaboutbirds.org/nabirds), which should be reviewed before use.
