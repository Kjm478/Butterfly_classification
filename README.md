# Image Classification Experiments

This repository contains Python scripts for experimenting with image classification using both deep learning (ResNet) and traditional machine learning (PCA + k-NN) techniques. The code includes implementations for training a ResNet model on a custom dataset, applying PCA for dimensionality reduction followed by k-NN classification, and visualizing model performance with tools like Grad-CAM.

The dataset used in these scripts is assumed to be an image folder structure (e.g., `./data/train`, `./data/valid`, `./data/test`) with subdirectories for each class, such as a butterfly species dataset. Adjust the paths and parameters as needed for your specific use case.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Script 1: ResNet Classification](#script-1-resnet-classification)
  - [Script 2: PCA + k-NN Classification](#script-2-pca--k-nn-classification)
  - [Script 3: Enhanced ResNet with Grad-CAM](#script-3-enhanced-resnet-with-grad-cam)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Deep Learning:** Trains a ResNet model with residual blocks for image classification.
- **Traditional ML:** Implements PCA for dimensionality reduction and k-NN for classification.
- **Visualization:** Includes loss/accuracy plots, confusion matrices, and Grad-CAM heatmaps for interpretability.
- **Data Preprocessing:** Applies image transformations like resizing, normalization, and augmentation.

## Requirements

- Python 3.8+
- Libraries:
  - `torch` (PyTorch)
  - `torchvision`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tqdm`
  - `Pillow` (PIL)
  - `opencv-python` (cv2)
  - `torchcam` (for Grad-CAM visualization)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
