import os
import json
import random
import yaml
from glob import glob
from sklearn.model_selection import train_test_split
import argparse

# Define PACS domains and class categories
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

def preprocess_pacs_image_paths(root_dir):
    """Collect image paths with associated domain and label."""
    all_data = []
    for domain in PACS_DOMAINS:
        for label in PACS_CATEGORIES:
            image_paths = glob(os.path.join(root_dir, domain, label, "*.jpg"))
            for path in image_paths:
                all_data.append({
                    "img": path,
                    "domain": domain,
                    "label": PACS_CATEGORIES.index(label)
                })
    return all_data

def generate_lodo_splits(all_data, output_dir, seeds):
    """Generate leave-one-domain-out (LODO) splits using given seeds."""
    os.makedirs(output_dir, exist_ok=True)

    for seed in seeds:
        for target_domain in PACS_DOMAINS:
            source_data = [x for x in all_data if x["domain"] != target_domain]
            target_data = [x for x in all_data if x["domain"] == target_domain]

            image_paths = [x["img"] for x in source_data]
            labels = [x["label"] for x in source_data]

            train_imgs, val_imgs, train_labels, val_labels = train_test_split(
                image_paths, labels, test_size=0.15, stratify=labels, random_state=seed
            )

            train_samples = [x for x in source_data if x["img"] in train_imgs]
            val_samples = [x for x in source_data if x["img"] in val_imgs]

            split_data = {
                "train": train_samples,
                "val": target_data,       # use full target domain as test set
                "val_source": val_samples # validation split from source domains
            }

            filename = f"split_seed{seed}_target_{target_domain}.json"
            with open(os.path.join(output_dir, filename), "w") as f:
                json.dump(split_data, f, indent=4)

            print(f"Saved {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Read paths and seeds from config
    root_dir = config["data"]["dataset_path"]  # e.g., "/path/to/PACS"
    output_dir = config["data"]["split_output_dir"]  # e.g., "data/splits"
    seeds = config.get("seed", [42])  # fallback to [42] if not defined

    data = preprocess_pacs_image_paths(root_dir)
    generate_lodo_splits(data, output_dir, seeds)
