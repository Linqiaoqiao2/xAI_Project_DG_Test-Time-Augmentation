# src/tent/tent_custom_tta.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy

# ====== Custom TTA Transform List ======
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_custom_tta_transforms():
    return [
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

# ====== Core TENT Functions ======
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()

def collect_bn_params(model):
    params = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.affine:
                if module.weight is not None:
                    params.append(module.weight)
                if module.bias is not None:
                    params.append(module.bias)
    return params

def configure_model(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            if module.affine:
                module.weight.requires_grad = True
                module.bias.requires_grad = True

def tent_adapt(model, image, optimizer, steps=1):
    model = model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(image)
        loss = softmax_entropy(outputs)
        loss.backward()
        optimizer.step()
    return model(image)  # Return final logits

# ====== Evaluation Entry Point ======
def run_tent_custom_tta(model, test_loader, config, device="cuda"):
    steps = config.get("tent", {}).get("steps_per_batch", 1)
    lr = config.get("tent", {}).get("lr", 1e-3)
    momentum = config.get("tent", {}).get("momentum", 0.9)

    transforms_list = get_custom_tta_transforms()
    model.eval()

    correct, total = 0, 0
    for image, label in test_loader:
        label = label.to(device)
        image = image[0] if isinstance(image, (list, tuple)) else image  # unwrap if needed
        image_pil = transforms.ToPILImage()(image.squeeze(0).cpu())

        logits_list = []
        for transform in transforms_list:
            aug_img = transform(image_pil).unsqueeze(0).to(device)
            model_copy = deepcopy(model)
            configure_model(model_copy)
            params = collect_bn_params(model_copy)
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
            logits = tent_adapt(model_copy, aug_img, optimizer, steps=steps)
            logits_list.append(logits.squeeze(0))

        logits_mean = torch.stack(logits_list).mean(dim=0)
        pred = logits_mean.argmax().item()
        correct += (pred == label.item())
        total += 1

    return correct / total


# ====== Standardized Entry Point ======
def get_adaptation_method(config):
    """
    Entry function for `run_tent_custom_tta.py`

    Returns:
        adapt_fn: Callable(model, dataloader, device) -> accuracy
    """
    def adapt_fn(model, dataloader, device):
        return run_tent_custom_tta(model, dataloader, config, device)
    return adapt_fn
