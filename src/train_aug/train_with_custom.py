import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset.pacs_loader import get_pacs_dataloaders
from src.model_loader import get_model
from src.train_aug.utils_train import train_one_epoch, eval_one_epoch



def run_train_custom_from_config(config_path: str):
    # ===== Load YAML Config =====
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    seeds = cfg["seed"] if isinstance(cfg["seed"], list) else [cfg["seed"]]
    output_root = cfg["output_dir"]
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    experiment_name = cfg["experiment_name"]
    categories = data_cfg["categories"]
    dataset_root = data_cfg["dataset_root"]
    target_domains = data_cfg["target_domains"]
    source_domains = data_cfg["source_domains"]

    # ===== Transforms =====
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    custom_transforms = [
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
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    ]

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # ===== Random wrapper for custom aug =====
    class RandomCustomAug:
        def __init__(self, transforms_list):
            self.transforms = transforms_list

        def __call__(self, x):
            t = np.random.choice(self.transforms)
            return t(x)

    custom_transform = RandomCustomAug(custom_transforms)

    for target_domain in target_domains:
        output_dir = os.path.join(output_root, target_domain)
        os.makedirs(output_dir, exist_ok=True)

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            # === Get dataloaders ===
            train_loader, val_loader, test_loader = get_pacs_dataloaders(
                dataset_root=dataset_root,
                source_domains=[d for d in source_domains if d != target_domain],
                target_domain=target_domain,
                categories=categories,
                val_split=data_cfg["val_split"],
                batch_size=data_cfg["batch_size"],
                num_workers=data_cfg["num_workers"],
                train_transform=custom_transform,
                test_transform=val_transform,
                seed=seed
            )

            # === Model ===
            config = {"model": model_cfg}
            model = get_model(config=config, target_domain=target_domain, seed=seed).to(device)
            optimizer = optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            patience_counter = 0
            model_path = os.path.join(output_dir, f"{experiment_name}_seed{seed}.pth")

            for epoch in range(train_cfg["epochs"]):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
                print(f"[{experiment_name} | Target {target_domain} | Seed {seed}] "
                      f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= train_cfg["early_stopping_patience"]:
                        print(">> Early stopping.")
                        break

            # Final test
            model.load_state_dict(torch.load(model_path))
            test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
            print(f"[✓] Final Test Acc on {target_domain} (Seed {seed}) = {test_acc:.3f}")
