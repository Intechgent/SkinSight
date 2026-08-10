"""Training and evaluation functions for SkinSight."""

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """Fix all random sources for reproducible runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, train_loader, class_weights=None, epochs: int = 3, lr: float = 1e-3):
    """Train a model with (optionally weighted) cross-entropy.

    Only parameters with requires_grad=True are updated, so a frozen-body
    ResNet trains only its head. Returns the trained model.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for images, labels in train_loader:
            labels = labels.squeeze()
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch}  train loss {running / len(train_loader):.4f}")

    return model


@torch.no_grad()
def evaluate(model, val_loader, num_classes: int = 7):
    """Evaluate on validation data. Returns (accuracy, per_class_recall list)."""
    model.eval()
    correct = total = 0
    per_class_correct = [0] * num_classes
    per_class_total = [0] * num_classes

    for images, labels in val_loader:
        labels = labels.squeeze()
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for c in range(num_classes):
            mask = labels == c
            per_class_total[c] += mask.sum().item()
            per_class_correct[c] += (preds[mask] == c).sum().item()

    accuracy = 100 * correct / total
    per_class_recall = [
        100 * per_class_correct[c] / per_class_total[c] if per_class_total[c] else 0
        for c in range(num_classes)
    ]
    return accuracy, per_class_recall