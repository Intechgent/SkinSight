"""Data loading and preparation for SkinSight (DermaMNIST)."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import DermaMNIST


# ImageNet normalisation stats - required because we use a model pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(image_size: int = 224):
    """Transform pipeline for a pretrained ResNet: resize, to-tensor, ImageNet-normalise."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_loaders(batch_size: int = 32, image_size: int = 224):
    """Build train and validation DataLoaders for DermaMNIST.

    Returns (train_loader, val_loader, train_dataset). The dataset is
    returned too so callers can compute class weights from its labels.
    """
    transform = get_transform(image_size)

    train_data = DermaMNIST(split="train", download=True, size=28, transform=transform)
    val_data = DermaMNIST(split="val", download=True, size=28, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, train_data


def compute_class_weights(dataset):
    """Inverse-frequency class weights (normalised to average ~1) for imbalanced training.

    Rare classes get larger weights so their errors cost more in the loss.
    Returns a float32 tensor suitable for nn.CrossEntropyLoss(weight=...).
    """
    counts = np.unique(dataset.labels.flatten(), return_counts=True)[1]
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)