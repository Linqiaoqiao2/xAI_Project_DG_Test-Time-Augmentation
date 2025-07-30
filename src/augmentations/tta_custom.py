# src/augmentations/tta_custom.py

import torch
import torch.nn.functional as F
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def _get_tta_transforms():
    transforms_list = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    ]

    def apply(img):
        return torch.stack([t(img) for t in transforms_list])  # [3, 3, 224, 224]

    return apply

def _fuse_logits(logits_list):
    """
    logits_list: list of [B, C] logits (e.g., 3 views)
    Returns: [B, C] after averaging and softmax
    """
    avg_logits = torch.stack(logits_list).mean(dim=0)
    probs = F.softmax(avg_logits, dim=1)
    return probs

def get_tta_strategy():
    """
    Returns:
        transform_fn: Callable that takes PIL.Image and returns [num_views, 3, H, W]
        fuse_fn: Callable that fuses [B, C] logits list → [B, C] probs
    """
    return _get_tta_transforms(), _fuse_logits
