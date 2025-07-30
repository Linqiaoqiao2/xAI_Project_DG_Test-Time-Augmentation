# src/train_aug/baseline_train.py

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset.pacs_loader import PACSFromSplit, SimpleImageDataset
from src.model_loader import get_model
from src.train_aug.utils_train import train_one_epoch, eval_one_epoch

def run_baseline_from_config(config_path: str):
    """Run baseline training based on official PACS split file."""
    # ==== Load Config ====
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    seeds = cfg["seed"] if isinstance(cfg["seed"], list) else [cfg["seed"]]
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ==== Transforms ====
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(seed)

        target = data_cfg["target_domain"]
        split_file = f"data/splits/split_seed{seed}_target_{target}.json"

        # ==== Load Datasets ====
        train_data = PACSFromSplit(split_file, split_type="train")
        val_data   = PACSFromSplit(split_file, split_type="val_source")
        test_data  = PACSFromSplit(split_file, split_type="val")

        train_set = SimpleImageDataset([s["img"] for s in train_data.samples],
                                       [s["label"] for s in train_data.samples],
                                       transform=transform)
        val_set = SimpleImageDataset([s["img"] for s in val_data.samples],
                                     [s["label"] for s in val_data.samples],
                                     transform=transform)
        test_set = SimpleImageDataset([s["img"] for s in test_data.samples],
                                      [s["label"] for s in test_data.samples],
                                      transform=transform)

        train_loader = DataLoader(train_set, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"])
        val_loader   = DataLoader(val_set, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])
        test_loader  = DataLoader(test_set, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"])

        # ==== Model ====
        config = {"model": model_cfg}
        model = get_model(config=config, target_domain=target, seed=seed).to(device)
        criterion = nn.CrossEntropyLoss()

        lr = float(train_cfg["lr"])
        weight_decay = float(train_cfg["weight_decay"])

        if train_cfg["optimizer"].lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError("Only Adam optimizer is supported currently.")

        # ==== Training ====
        patience = train_cfg["early_stopping_patience"]
        best_val_acc = 0.0
        patience_counter = 0
        ckpt_path = os.path.join(output_dir, f"{cfg['experiment_name']}_seed{seed}_target{target}.pth")

        for epoch in range(train_cfg["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)
            print(f"[{cfg['experiment_name']} - Seed {seed}] Epoch {epoch+1} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(">> Early stopping.")
                    break

        # ==== Final Evaluation ====
        model.load_state_dict(torch.load(ckpt_path))
        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
        print(f"[✓] Final Test Acc on {target} (Seed {seed}): {test_acc:.3f}")
