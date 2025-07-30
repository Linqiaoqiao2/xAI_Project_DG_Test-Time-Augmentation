# src/model_loader.py

import os
import torch
import torchvision.models as models
from torch import nn

def get_model(config, target_domain=None, seed=None):
    """
    Create and return a model based on config.
    
    Args:
        config (dict): The entire config loaded from YAML.
        target_domain (str): Target domain name, used for checkpoint filename.
        seed (int): Random seed, used for checkpoint filename.
    
    Returns:
        model (torch.nn.Module)
    """
    model_name = config["model"]["name"]
    pretrained = config["model"]["pretrained"]
    num_classes = config["model"]["num_classes"]
    checkpoint_dir = config["model"].get("checkpoint_dir", None)

    # ===== Instantiate model backbone =====
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    # ===== Replace final layer =====
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # ===== Load checkpoint if available =====
    if checkpoint_dir and target_domain is not None and seed is not None:
        ckpt_name = f"{model_name}_seed{seed}_target{target_domain}.pth"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f" Loaded checkpoint from: {ckpt_path}")
        else:
            print(f" Warning: Checkpoint not found at {ckpt_path}, using randomly initialized weights.")
    else:
        print(" No checkpoint loaded (either directory or target/seed not provided).")

    return model
