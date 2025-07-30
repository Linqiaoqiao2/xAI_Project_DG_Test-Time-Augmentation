# src/tent/tent.py

"""
Original TENT (Test-Time Entropy Minimization)
- Applies entropy minimization on test batches.
- Updates only BatchNorm affine parameters (γ, β).
- No augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from copy import deepcopy


def softmax_entropy(logits):
    return -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(1).mean()


def configure_model_for_tent(model):
    """
    Enable adaptation of only BatchNorm affine parameters (gamma/beta).
    """
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False  # disable running stats
    return model


def collect_bn_params(model):
    return [p for m in model.modules() if isinstance(m, nn.BatchNorm2d)
            for p in [m.weight, m.bias] if p is not None]


def run_tent_adaptation(model, test_loader, config, device="cuda"):
    """
    Performs TENT adaptation on test set.

    Args:
        model: torch model (will be modified in-place)
        test_loader: DataLoader
        config: dictionary containing TENT settings
        device: cuda or cpu

    Returns:
        accuracy: float
    """
    steps = config.get("steps_per_batch", 1)
    lr = config.get("lr", 1e-3)

    model = configure_model_for_tent(model)
    params = collect_bn_params(model)
    optimizer = SGD(params, lr=lr)

    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        for _ in range(steps):
            optimizer.zero_grad()
            logits = model(images)
            loss = softmax_entropy(logits)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


# === Optional: reset model to clean state ===
def copy_model(model):
    return deepcopy(model).to(next(model.parameters()).device)


def get_adaptation_method(config):
    """
    Unified entry point for TENT, to be called from run_tent.py

    Returns:
        adapt_fn: callable (model, dataloader, device) -> accuracy
    """
    def adapt_fn(model, dataloader, device):
        return run_tent_adaptation(model, dataloader, config["tent"], device)
    return adapt_fn
