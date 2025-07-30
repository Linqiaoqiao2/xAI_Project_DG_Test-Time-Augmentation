import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from utils.seed import set_random_seed
from utils.logger import init_logger
from utils.results_io import save_json
from utils.metrics import accuracy
from src.model_loader import get_model
from src.dataset.pacs_loader import PACSFromSplit

# === Dynamic import of TTA strategy ===
def load_tta_strategy(name):
    if name == "mvdg":
        from src.augmentations.tta_mvdg import get_tta_transforms, fuse_logits
    elif name == "custom":
        from src.augmentations.tta_custom import get_tta_transforms, fuse_logits
    else:
        raise ValueError(f"Unsupported TTA strategy: {name}")
    return get_tta_transforms, fuse_logits

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger = init_logger()

    # === TTA strategy ===
    tta_name = config["tta_strategy"]
    get_transforms, fuse_logits_fn = load_tta_strategy(tta_name)
    transform_list = get_transforms(config["tta"].get("views", 32))

    # === Config values ===
    checkpoint_dir = config["model"]["checkpoint_dir"]
    target_domains = config["data"]["target_domains"]
    seeds = config["seed"] if isinstance(config["seed"], list) else [config["seed"]]
    batch_size = config["data"].get("batch_size", 1)
    num_workers = config["data"].get("num_workers", 4)

    for seed in seeds:
        set_random_seed(seed)

        for target in target_domains:
            logger.info(f"[TTA ON TRAINED] Strategy: {tta_name} | Target: {target} | Seed: {seed}")

            # === Load model from checkpoint ===
            model = get_model(config, target, seed).to(device)
            ckpt_path = os.path.join(checkpoint_dir, f"{config['experiment_name']}_seed{seed}_target_{target}.pth")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            # === Load test dataset ===
            split_file = f"data/splits/split_seed{seed}_target_{target}.json"
            dataset = PACSFromSplit(split_file, split_type="val", transform=None)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

            all_preds, all_labels = [], []
            to_pil = transforms.ToPILImage()

            for image, label in tqdm(dataloader, desc=f"TTA {target}", leave=False):
                image_pil = to_pil(image.squeeze(0))
                views = torch.stack([t(image_pil) for t in transform_list]).to(device)  # [T, 3, H, W]
                logits = model(views)  # [T, C]
                fused_logits = fuse_logits_fn([logits])  # → [1, C]
                pred = fused_logits.argmax(dim=1).item()

                all_preds.append(pred)
                all_labels.append(label.item())

            acc = accuracy(torch.tensor(all_preds), torch.tensor(all_labels))
            logger.info(f"[✓] Final TTA Accuracy on {target} (Seed {seed}) = {acc:.2f}")

            # === Save result ===
            result = {
                "experiment": config["experiment_name"],
                "strategy": tta_name,
                "target": target,
                "seed": seed,
                "accuracy": acc
            }

            os.makedirs("results/tta_on_trained", exist_ok=True)
            out_path = os.path.join("results/tta_on_trained", f"{config['experiment_name']}_seed{seed}_{target}.json")
            save_json(result, out_path)
            logger.info(f"Saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    main(args.config)
