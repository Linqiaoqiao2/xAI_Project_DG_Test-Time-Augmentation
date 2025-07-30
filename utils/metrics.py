# utils/metrics.py

import torch

def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        preds: Tensor of shape [B, C]
        labels: Tensor of shape [B]

    Returns:
        accuracy in percentage
    """
    correct = preds.argmax(dim=1).eq(labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / total


def per_class_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> dict:
    """
    Compute per-class accuracy.

    Args:
        preds: Tensor of shape [B, C]
        labels: Tensor of shape [B]
        num_classes: int

    Returns:
        Dictionary: {class_idx: accuracy}
    """
    preds_cls = preds.argmax(dim=1)
    per_class = {}
    for cls in range(num_classes):
        mask = labels == cls
        total = mask.sum().item()
        correct = (preds_cls[mask] == labels[mask]).sum().item()
        acc = 100.0 * correct / total if total > 0 else 0.0
        per_class[cls] = acc
    return per_class
