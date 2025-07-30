# experiments/run_tent_custom_tta.py

import os
import yaml
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from src.dataset.pacs_loader import PACSFromSplit
from src.model_loader import get_model
from src.tent.tent import tent_adaptation
from src.augmentations.tta_custom import get_tta_transforms, fuse_logits

from utils.logger import init_logger
from utils.seed import set_random_seed
from utils.results_io import save_json
from utils.metrics import accuracy


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def identity_transform(x):
    return x


def main(config_path):
    cfg = load_config(config_path)
    logger = init_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seeds = cfg["seeds"]
    domains = cfg["data"]["domains"]
    data_root = cfg["data"]["data_root"]
    batch_size = cfg["data"]["batch_size"]
    output_dir = cfg["output"]["save_dir"]
    os.makedirs(output_dir, exist_ok=True)

    tta_transforms = get_tta_transforms()

    for seed in seeds:
        set_random_seed(seed)

        for target_domain in domains:
            logger.info(f">>> TENT + Custom TTA | Target: {target_domain} | Seed: {seed}")
            
            split_path = os.path.join("data/splits", f"split_seed{seed}_target_{target_domain}.json")
            val_data = PACSFromSplit(split_path, split_type="val", transform=identity_transform)
            val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=cfg["data"]["num_workers"])

            model_cfg = {
                "model": {
                    "name": cfg["model"]["architecture"],
                    "pretrained": cfg["model"]["pretrained"],
                    "checkpoint_path": cfg["model"]["checkpoint_path"],
                    "num_classes": cfg["model"]["num_classes"]
                }
            }

            model = get_model(model_cfg, target_domain, seed).to(device)
            model.train()  # Important for TENT
            model = tent_adaptation(model, cfg["tent"], logger)

            preds, labels = [], []

            for sample in val_loader:
                img, label = sample
                label = label.to(device)

                pil_img = transforms.ToPILImage()(img.squeeze(0).cpu())
                tta_imgs = torch.stack([t(pil_img) for t in tta_transforms]).to(device)

                with torch.no_grad():
                    logits = model(tta_imgs)  # shape: [num_views, C]
                fused_logits = fuse_logits(logits.unsqueeze(0))  # shape: [1, C]
                pred = fused_logits.argmax(dim=1)

                preds.append(pred.item())
                labels.append(label.item())

            acc = accuracy(torch.tensor(preds), torch.tensor(labels))
            logger.info(f"[✓] Final Accuracy on {target_domain} | Seed {seed} = {acc:.2f}")

            # === Save JSON ===
            result = {
                "target_domain": target_domain,
                "seed": seed,
                "accuracy": acc
            }
            save_path = os.path.join(output_dir, f"{cfg['experiment_name']}_seed{seed}_{target_domain}.json")
            save_json(result, save_path)
            logger.info(f"Saved result to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
