# src/augmentations/tta_tencrop.py

import torch
import torchvision.transforms as transforms
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def _get_tta_transforms():
    resize_crop = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),  # Returns tuple of 10 PIL images
    ])

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    def apply(img):
        crops = resize_crop(img)  # 10 crops
        return torch.stack([normalize(crop) for crop in crops])  # shape [10, 3, 224, 224]

    return apply

def _fuse_logits(logits_batch):
    """
    logits_batch: Tensor of shape [B * 10, num_classes]
    Returns: Tensor of shape [B, num_classes] after averaging 10 views
    """
    bs = logits_batch.shape[0] // 10
    num_classes = logits_batch.shape[1]
    logits = logits_batch.view(bs, 10, num_classes)
    return logits.mean(dim=1)

def get_tta_strategy():
    """
    Returns:
        transform_fn: Callable that takes PIL.Image and returns [num_views, 3, H, W]
        fuse_fn: Callable that fuses [B*num_views, C] -> [B, C]
    """
    return _get_tta_transforms(), _fuse_logits
