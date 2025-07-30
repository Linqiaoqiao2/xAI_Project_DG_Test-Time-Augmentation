# src/augmentations/tta_mvdg.py

import torch
import torch.nn.functional as F
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def _get_tta_transforms(t=32):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    def apply(img):
        return torch.stack([transform(img) for _ in range(t)])  # [t, 3, 224, 224]

    return apply

def _fuse_logits(logits_list):
    """
    logits_list: list of length t, each element shape [B, C]
    Returns: [B, C] tensor after averaging and softmax
    """
    logits_sum = torch.stack(logits_list, dim=0).sum(dim=0)  # [B, C]
    probs = F.softmax(logits_sum / len(logits_list), dim=1)
    return probs

def get_tta_strategy():
    """
    Returns:
        transform_fn: Callable that takes PIL.Image and returns [num_views, 3, H, W]
        fuse_fn: Callable that takes list of logits [B, C] and returns [B, C]
    """
    return _get_tta_transforms(), _fuse_logits
