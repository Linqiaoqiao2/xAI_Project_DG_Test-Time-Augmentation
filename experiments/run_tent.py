import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.seed import set_random_seed
from utils.logger import init_logger
from utils.results_io import save_json
from utils.metrics import accuracy

from src.model_loader import get_model
from src.dataset.pacs_loader import PACSFromSplit
from src.tent.tent import enable_tent_adaptation

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(config_path):
    config = load_config(config_path)
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger = init_logger()

    seeds = config["seed"] if isinstance(config["seed"], list) else [config["seed"]]
    target_domain = config["data"]["target_domain"]
    batch_size = config["data"].get("batch_size", 16)
    num_workers = config["data"].get("num_workers", 4)

    for seed in seeds:
        set_random_seed(seed)
        logger.info(f"[TENT] Target: {target_domain} | Seed: {seed}")

        # === Load test split ===
        split_file = f"data/splits/split_seed{seed}_target_{target_domain}.json"
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        test_set = PACSFromSplit(split_file, split_type="val", transform=transform)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # === Load model and enable TENT adaptation ===
        model = get_model(config, target_domain, seed).to(device)
        model_path = os.path.join(config["model"]["checkpoint_dir"], f"{config['experiment_name']}_seed{seed}_target_{target_domain}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.train()  # Required by Tent
        model = enable_tent_adaptation(model)

        # === True online TENT: Adapt on batch(t), eval on batch(t+1) ===
        all_preds, all_labels = [], []
        prev_inputs, prev_labels = None, None

        for inputs, labels in tqdm(test_loader, desc=f"TENT {target_domain}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            if prev_inputs is not None:
                with torch.no_grad():
                    outputs = model(prev_inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(prev_labels.cpu().tolist())

            # Adapt on current batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, outputs.softmax(dim=1), reduction='none').mean()
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()

            prev_inputs, prev_labels = inputs, labels

        # === Evaluate last batch ===
        if prev_inputs is not None:
            with torch.no_grad():
                outputs = model(prev_inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(prev_labels.cpu().tolist())

        acc = accuracy(torch.tensor(all_preds), torch.tensor(all_labels))
        logger.info(f"[✓] TENT Final Accuracy on {target_domain} (Seed {seed}) = {acc:.2f}")

        # === Save result ===
        result = {
            "experiment": config["experiment_name"],
            "target": target_domain,
            "seed": seed,
            "accuracy": acc
        }
        os.makedirs("results/tent", exist_ok=True)
        out_path = os.path.join("results/tent", f"{config['experiment_name']}_seed{seed}_{target_domain}.json")
        save_json(result, out_path)
        logger.info(f"Saved result to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
