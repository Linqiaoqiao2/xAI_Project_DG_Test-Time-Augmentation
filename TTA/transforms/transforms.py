# transforms.py
import torch
from torchvision import transforms
from typing import List
from PIL import Image

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_tta_transforms(mode: str = "basic") -> List[transforms.Compose]:
    """
    Returns a list of torchvision transform pipelines for TTA.
    
    Args:
        mode: str, one of ['basic', 'flip', 'rotate', 'five']
    
    Returns:
        List of torchvision transforms.Compose objects
    """
    if mode == "basic":
        return [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        ]
    
    elif mode == "flip":
        return [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        ]

    elif mode == "rotate":
        return [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]) for _ in range(3)
        ]

    elif mode == "flip+rotate":
        return [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]) for _ in range(5)
        ]
    
    else:
        raise ValueError(f"Unsupported TTA mode: {mode}")


def apply_tta(image: Image.Image, mode: str = "flip") -> List[torch.Tensor]:
    """
    Apply TTA transforms to a PIL image and return a list of augmented tensors.
    
    Args:
        image: PIL.Image
        mode: one of the modes in get_tta_transforms

    Returns:
        List of torch.Tensor, each of shape (C, H, W)
    """
    transforms_list = get_tta_transforms(mode)
    augmented = []
    for tf in transforms_list:
        aug_img = tf(image)
        if isinstance(aug_img, list):  # in case of FiveCrop
            augmented.extend(aug_img)
        else:
            augmented.append(aug_img)
    return augmented
