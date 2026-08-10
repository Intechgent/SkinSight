"""Model definitions for SkinSight skin-lesion classification."""

import torch.nn as nn
from torchvision import models


class TinyNet(nn.Module):
    """A small from-scratch baseline: two fully-connected layers.

    Takes flattened RGB 28x28 images (3*28*28 = 2352 inputs) and outputs
    scores for `num_classes` classes. Used as the scratch baseline to
    compare against transfer learning.
    """

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.fc1 = nn.Linear(2352, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)      # flatten each image, keep batch dim
        x = nn.functional.relu(self.fc1(x))
        return self.fc2(x)                # raw logits (no softmax; the loss applies it)


def build_resnet(num_classes: int = 7, freeze_body: bool = True):
    """Build a ResNet-18 pretrained on ImageNet, adapted for `num_classes`.

    The final layer is replaced with a fresh classifier. If `freeze_body`
    is True, all pretrained layers are frozen and only the new head is
    trained (fast, CPU-friendly transfer learning).
    """
    model = models.resnet18(weights="IMAGENET1K_V1")

    if freeze_body:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Linear(512, num_classes)   # fresh head, trainable by default
    return model