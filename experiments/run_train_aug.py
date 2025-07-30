# experiments/run_train_aug.py

import argparse
import yaml
import os
import copy

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_temp_config(config_dict, tmp_path):
    with open(tmp_path, "w") as f:
        yaml.dump(config_dict, f)

def main(config_path):
    config = load_config(config_path)
    strategy = config.get("train_aug", "").lower()

    if "target_domains" in config["data"]:
        target_domains = config["data"]["target_domains"]
    else:
        target_domains = [config["data"]["target_domain"]]

    for target in target_domains:
        # deep copy config to isolate each run
        cfg = copy.deepcopy(config)
        cfg["data"]["target_domain"] = target
        cfg["data"].pop("target_domains", None)

        # dynamic experiment name & output folder
        cfg["experiment_name"] = f"{config['experiment_name']}_target_{target}"
        cfg["output_dir"] = os.path.join(config["output_dir"], f"{target}")
        os.makedirs(cfg["output_dir"], exist_ok=True)

        # save a temporary config file
        tmp_path = f"tmp_train_aug_{target}.yaml"
        save_temp_config(cfg, tmp_path)

        if strategy == "custom":
            from src.train_aug.train_with_custom import run_train_custom_from_config
            run_train_custom_from_config(tmp_path)
        elif strategy == "mvdg":
            from src.train_aug.train_with_mvdg import run_train_mvdg_from_config
            run_train_mvdg_from_config(tmp_path)
        else:
            raise ValueError(f"Unsupported training augmentation strategy: '{strategy}'")

        os.remove(tmp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with training-time augmentation.")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file.")
    args = parser.parse_args()
    main(args.config)
