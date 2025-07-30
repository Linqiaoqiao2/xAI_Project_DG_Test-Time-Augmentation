# experiments/run_test_tta.py

import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from utils.seed import set_random_seed
from utils.logger import init_logger
from utils.results_io import save_json
from utils.metrics import accuracy

from src.model_loader import get_model
from src.dataset.pacs_loader import PACSFromSplit, SimpleImageDataset

# === Dynamic import of TTA strategy ===
def load_tta_strategy(name):
    if name == "tencrop":
        from src.augmentations.tta_tencrop import get_tta_transforms, fuse_logits
    elif name == "mvdg":
        from src.augmentations.tta_mvdg import get_tta_transforms, fuse_logits
    elif name == "custom":
        from src.augmentations.tta_custom import get_tta_transforms, fuse_logits
    else:
        raise ValueError(f"Unknown TTA strategy: {name}")
    return get_tta_transforms, fuse_logits

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger = init_logger()

    tta_name = config["tta_strategy"]
    get_tta_transforms, fuse_fn = load_tta_strategy(tta_name)

    for seed in config["seeds"]:
        set_random_seed(seed)
        target_domain = config["data"]["target_domain"]
        logger.info(f"Evaluating TTA | Strategy: {tta_name} | Target: {target_domain} | Seed: {seed}")

        # === Load model ===
        model = get_model(config=config, target_domain=target_domain, seed=seed).to(device)
        model.eval()

        # === Load test set ===
        split_path = config["data"]["split_path_template"].format(seed=seed, domain=target_domain)
        test_data = PACSFromSplit(split_path, split_type="val")

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        to_tensor = transforms.Compose([
            transforms.Resize((224, 224)),  # safety
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        all_preds, all_labels = [], []

        for sample in test_data.samples:
            img_path, label = sample["img"], sample["label"]
            image = Image.open(img_path).convert("RGB")
            tta_views = get_tta_transforms()
            logits_list = []

            for t in tta_views:
                input_tensor = t(image).unsqueeze(0).to(device)  # [1, 3, H, W]
                logits = model(input_tensor)  # [1, num_classes]
                logits_list.append(logits.squeeze(0))  # [C]

            pred = fuse_fn(logits_list)
            all_preds.append(pred.item())
            all_labels.append(label)

        acc = accuracy(torch.tensor(all_preds), torch.tensor(all_labels))
        logger.info(f"[✓] Final TTA Accuracy: {acc:.2f}")

        # === Save result ===
        result = {
            "strategy": tta_name,
            "target": target_domain,
            "seed": seed,
            "accuracy": acc
        }
        out_path = os.path.join(config["output"]["save_dir"], f"{tta_name}_seed{seed}_{target_domain}.json")
        save_json(result, out_path)
        logger.info(f"Saved result to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
