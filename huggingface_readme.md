---
license: mit
language: en
tags:
- dinov2
- vit
- image-classification
- bird-classification
- nabirds
---

# Binocular: A NABirds Species Classifier

<img src="https://raw.githubusercontent.com/JiuJiuChe/Binocular/main/Binocular/assets/binacular.png" width="100" alt="Binocular">

This project provides a pretrained, species-level bird classifier for North American birds, built on the [NABirds dataset](https://dl.allaboutbirds.org/nabirds). It uses a powerful DINOv2 Vision Transformer (ViT) backbone with a linear probe, allowing for accurate and efficient classification.

The model can be easily loaded and used for inference directly from the [Hugging Face Hub](https://huggingface.co/jiujiuche/binocular).

## Model Details

- **Model Type:** Vision Transformer (ViT) with a linear classification head.
- **Backbone:** DINOv2 (ViT-B/14).
- **Dataset:** [NABirds](https://dl.allaboutbirds.org/nabirds) - A large-scale dataset of North American birds with over 48,000 annotated images of 555 species.
- **Resolution:** Images are resized to 224x224 pixels.

## How to Get Started with the Model

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

### 3. Run Inference in Python

You can also use the `InferenceModel` in your own Python scripts:

```python
from Binocular.models.inference import InferenceModel
from PIL import Image

# This will download the model from the hub automatically
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

## Uses

This model is intended for classifying bird species in images. It can be used for:
- Academic research in ornithology and computer vision.
- Hobbyist applications for bird identification.
- As a baseline for more complex models.

## Risks, Limitations and Biases

- The model is trained on the NABirds dataset and will perform best on images of North American birds. Its performance on out-of-distribution images (e.g., birds from other continents, low-quality images, or synthetic images) may be degraded.
- The dataset may contain biases related to geography, lighting conditions, and image quality, which could affect model performance.

## Training

The full pipeline for training the model from scratch is available in the [GitHub repository](https://github.com/JiuJiuChe/Binocular).

### Dataset Setup

1.  Download the NABirds dataset from [https://dl.allaboutbirds.org/nabirds](https://dl.allaboutbirds.org/nabirds).
2.  Extract the dataset files into the `Binocular/datasets/nabirds/` directory within the cloned repository.

### Training Command

The training process is managed through a main script that uses configuration files to define the training parameters.

```bash
# Example for training on an M2 Mac
python Binocular/scripts/train.py --config Binocular/configs/m2_dev.yaml
```

Training configurations for different hardware can be found in the `Binocular/configs/` directory.

## Citation

If you use this model in your research, please consider citing the original DINOv2 and NABirds papers, as well as this repository.

```bibtex
@misc{Binocular,
  author = {JiuJiuChe},
  title = {Binocular: A NABirds Species Classifier},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JiuJiuChe/Binocular}}
}
```
