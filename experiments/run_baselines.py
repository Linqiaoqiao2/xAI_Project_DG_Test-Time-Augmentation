import os
import argparse
import yaml
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.logger import init_logger
from utils.seed import set_random_seed
from utils.results_io import save_json
from utils.metrics import accuracy

from src.dataset.pacs_loader import SimpleImageDataset
from src.model_loader import get_model
from src.train_aug.utils_train import train_one_epoch, eval_one_epoch
from src.train_aug.baseline_train import run_baseline_from_config

from glob import glob
from copy import deepcopy

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def collect_image_paths(dataset_root, domains, categories):
    data = []
    for domain in domains:
        for label in categories:
            img_paths = glob(os.path.join(dataset_root, domain, label, "*.jpg"))
            for p in img_paths:
                data.append((p, categories.index(label)))
    return data

def run_manual_glob_baseline(config):
    logger = init_logger()
    categories = config["data"]["categories"]
    dataset_root = config["data"]["dataset_root"]
    if not os.path.isabs(dataset_root):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_root = os.path.join(project_root, dataset_root)
    target = config["data"]["target_domain"]
    sources = config["data"]["source_domains"]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for seed in config["seed"]:
        set_random_seed(seed)
        logger.info(f"Running Baseline (glob) | Target: {target} | Seed: {seed}")

        source_data = collect_image_paths(dataset_root, sources, categories)
        image_paths, labels = zip(*source_data)

        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            test_size=float(config["data"]["val_split"]),
            stratify=labels, random_state=seed
        )

        train_set = SimpleImageDataset(train_paths, train_labels, transform)
        val_set = SimpleImageDataset(val_paths, val_labels, transform)

        train_loader = DataLoader(train_set, batch_size=int(config["data"]["batch_size"]), shuffle=True, num_workers=int(config["data"]["num_workers"]))
        val_loader = DataLoader(val_set, batch_size=int(config["data"]["batch_size"]), shuffle=False, num_workers=int(config["data"]["num_workers"]))

        model = get_model(config).to(config["device"])
        criterion = torch.nn.CrossEntropyLoss()

        opt_type = config["train"]["optimizer"].lower()
        lr = float(config["train"]["lr"])
        weight_decay = float(config["train"]["weight_decay"])

        if opt_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_type == "sgd":
            momentum = float(config["train"]["momentum"])
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

        best_acc = 0.0
        patience = int(config["train"].get("early_stopping_patience", 5))
        patience_counter = 0

        for epoch in range(config["train"]["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config["device"])
            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, config["device"])

            logger.info(f"[Epoch {epoch+1}] Train Acc: {train_acc:.2f} | Val Acc: {val_acc:.2f}")

            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                ckpt_name = f"{config['model']['name']}_seed{seed}_target{target}.pth"
                ckpt_path = os.path.join(config["output_dir"], ckpt_name)
                os.makedirs(config["output_dir"], exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"[✓] Saved best model: {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"[✗] Early stopping at epoch {epoch+1}")
                    break

        result = {"target": target, "seed": seed, "val_acc": best_acc}
        result_path = os.path.join(config["output_dir"], f"result_seed{seed}_{target}.json")
        save_json(result, result_path)
        logger.info(f"[✓] Result saved to {result_path}")

def main(config_path):
    base_config = load_config(config_path)
    all_domains = base_config["data"].get("target_domains")

    if not all_domains or not isinstance(all_domains, list):
        raise ValueError("Please define 'data.target_domains' as a list in your config.")

    for target in all_domains:
        config = deepcopy(base_config)
        config["data"]["target_domain"] = target
        config["data"]["source_domains"] = [d for d in all_domains if d != target]
        config["experiment_name"] = f"{base_config['experiment_name']}_{target}"
        config["output_dir"] = os.path.join(base_config["output_dir"], target)

        if config["data"].get("use_split", False):
            run_baseline_from_config(config)
        else:
            run_manual_glob_baseline(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
