# src/train_aug/train_with_mvdg.py

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import yaml

from src.dataset.pacs_loader import PACSFromSplit, SimpleImageDataset
from src.model_loader import get_model
from src.train_aug.utils_train import train_one_epoch, eval_one_epoch


def run_train_mvdg_from_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg.get("experiment_name", "train_mvdg")
    seeds = cfg["seed"]
    device = torch.device(cfg.get("device", "cuda"))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    aug_cfg = cfg.get("augmentation", {})

    # === Transforms ===
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    mvdg_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    SAVE_ROOT = os.path.join("/home/proj25gr5/xAI_neu/train_with_mvdg", experiment_name)
    os.makedirs(SAVE_ROOT, exist_ok=True)

    for model_name in ["resnet18", "resnet50"]:
        if model_cfg["name"] != model_name:
            continue

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)

            for target_domain in ['art_painting', 'cartoon', 'photo', 'sketch']:
                if target_domain != data_cfg["target_domain"]:
                    continue

                print(f"\n==== Training {model_name.upper()} with MVDG Aug | Seed {seed} | Target: {target_domain} ====")
                split_file = f"data/splits/split_seed{seed}_target_{target_domain}.json"
                train_data = PACSFromSplit(split_file, split_type="train")
                val_data = PACSFromSplit(split_file, split_type="val_source")
                test_data = PACSFromSplit(split_file, split_type="val")

                train_set = SimpleImageDataset([s["img"] for s in train_data.samples],
                                               [s["label"] for s in train_data.samples],
                                               transform=mvdg_transform)
                val_set = SimpleImageDataset([s["img"] for s in val_data.samples],
                                             [s["label"] for s in val_data.samples],
                                             transform=val_transform)
                test_set = SimpleImageDataset([s["img"] for s in test_data.samples],
                                              [s["label"] for s in test_data.samples],
                                              transform=val_transform)

                loader_args = {
                    'batch_size': data_cfg.get("batch_size", 32),
                    'num_workers': data_cfg.get("num_workers", 4)
                }
                train_loader = DataLoader(train_set, shuffle=True, **loader_args)
                val_loader = DataLoader(val_set, shuffle=False, **loader_args)
                test_loader = DataLoader(test_set, shuffle=False, **loader_args)

                model = get_model(config=cfg, target_domain=target_domain, seed=seed).to(device)
                criterion = nn.CrossEntropyLoss()

                outer_opt_cfg = train_cfg["outer_optimizer"]
                inner_opt_cfg = train_cfg["inner_optimizer"]
                outer_optimizer = getattr(optim, outer_opt_cfg["type"])(model.parameters(),
                                                                          lr=outer_opt_cfg["lr"],
                                                                          weight_decay=outer_opt_cfg["weight_decay"])
                inner_optimizer = getattr(optim, inner_opt_cfg["type"])(model.parameters(),
                                                                          lr=inner_opt_cfg["lr"],
                                                                          weight_decay=inner_opt_cfg["weight_decay"])

                best_val_acc = 0.0
                patience_counter = 0
                model_path = os.path.join(SAVE_ROOT, f"{model_name}_seed{seed}_target_{target_domain}.pth")

                for epoch in range(train_cfg["epochs"]):
                    if outer_opt_cfg["apply_epochs"][0] <= epoch <= outer_opt_cfg["apply_epochs"][1]:
                        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, outer_optimizer, device)
                    else:
                        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, inner_optimizer, device)

                    val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
                    print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        torch.save(model.state_dict(), model_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= train_cfg.get("early_stopping_patience", 5):
                            print(">> Early stopping.")
                            break

                model.load_state_dict(torch.load(model_path))
                test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
                print(f"[✓] Final Test Acc on {target_domain} = {test_acc:.3f}")
