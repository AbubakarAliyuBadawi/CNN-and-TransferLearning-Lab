# CNN and Transfer Learning Project

This repository contains an implementation of Convolutional Neural Networks (CNNs) and transfer learning to perform image classification tasks. The project is structured as a hands-on learning experience to explore the fundamentals of CNNs and the practical benefits of leveraging pre-trained models for specific datasets.

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Building a CNN from Scratch](#1-building-a-cnn-from-scratch)
  - [2. Transfer Learning with Pre-trained Models](#2-transfer-learning-with-pre-trained-models)
  - [3. Fine-Tuning Pre-trained Models](#3-fine-tuning-pre-trained-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [How to Run](#how-to-run)
- [References](#references)

## Introduction
In this project, we explore the development of CNNs for image classification, starting from simple architectures to leveraging pre-trained models for improved performance. The tasks include training on the MNIST dataset and applying transfer learning for fish species classification using the Fish4Knowledge dataset.

## Objectives
- Develop a custom CNN model and understand its architecture.
- Train and evaluate the CNN on the MNIST dataset.
- Use transfer learning with pre-trained models like ResNet18, AlexNet, VGG16, and MobileNetV2.
- Fine-tune pre-trained models to adapt them to a specific dataset.

## Dataset

### MNIST
- **Description**: A dataset of handwritten digits (0-9) in grayscale.
- **Usage**: To build and train a basic CNN model.

### Fish4Knowledge
- **Description**: A collection of underwater fish species images.
- **Usage**: Used for transfer learning experiments.

## Methodology

### 1. Building a CNN from Scratch
- **Architecture**:
  - Input: Grayscale images (28x28 for MNIST).
  - Layers:
    - 2 Convolutional layers with ReLU and MaxPooling.
    - Fully connected layers for classification.
- **Framework**: PyTorch.
- **Training**:
  - Optimizer: SGD.
  - Loss Function: CrossEntropyLoss.
  - Training on MNIST for classification accuracy.

### 2. Transfer Learning with Pre-trained Models
- **Models Used**:
  - ResNet18
  - AlexNet
  - VGG16
  - MobileNetV2
- **Modifications**:
  - Replaced the final fully connected layer to match the Fish4Knowledge dataset's class count.
  - Frozen all layers except the final classifier during initial training.

### 3. Fine-Tuning Pre-trained Models
- Fine-tuned the entire network after achieving stable results by training only the classifier.
- Improved accuracy and generalization.

## Evaluation Metrics
- **Loss**: CrossEntropyLoss.
- **Accuracy**: Percentage of correctly classified images.

## Results

| Model          | Accuracy (Transfer Learning) | Accuracy (Fine-Tuning) |
|----------------|-------------------------------|-------------------------|
| ResNet18       | 98.0%                         | 98.7%                   |
| AlexNet        | 95.4%                         | 97.5%                   |
| VGG16          | 95.3%                         | 97.2%                   |
| MobileNetV2    | 89.7%                         | 94.5%                   |

## How to Run

### Requirements
- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- matplotlib

### Steps
1. Clone the repository.
```bash
$ git clone https://github.com/yourusername/cnn_transfer_learning
$ cd cnn_transfer_learning
```
2. Install dependencies.
```bash
$ pip install -r requirements.txt
```
3. Run the main script.
```bash
$ python cnn_transfer_learning_lab.py
```

## References
1. [PyTorch Documentation](https://pytorch.org/docs/)
2. [Fish4Knowledge Dataset](https://homepages.inf.ed.ac.uk/rbf/Fish4Knowledge/)
3. [ImageNet Classes](https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)

---
Feel free to contribute to this project or raise issues in the repository!

