#!/usr/bin/env python
# coding: utf-8
"""
PACS TTA Evaluation Script — ResNet18 with AlexNet, MVDG, and Custom TTA
"""

import torch
import sys
sys.path.append("/home/proj25gr5/xAI")
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from baseline_resnet18 import PACS_Dataset, PACS_CATEGORIES, BASE_PACS_PATH, DEVICE

def get_model(num_classes, pretrained=True, arch='resnet18'):
    if arch == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# === 三个 TTA 策略 ===（保持不变，略）

# === TTA collate function ===
def tta_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels

# === Evaluation ===
seeds = [42, 123, 2024]
domains = ["art_painting", "cartoon", "photo", "sketch"]
strategies = ['alexnet', 'mvdg', 'custom']
model_dir = "/home/proj25gr5/xAI_neu/models_resnet18_pretrained"
save_root = "/home/proj25gr5/xAI_neu"

results, avg_results = {}, {}
print(f"Using device: {DEVICE}")

for domain in domains:
    results[domain], avg_results[domain] = {}, {}
    for strategy in strategies:
        accs = []
        for seed in seeds:
            print(f"\n📊 Domain: {domain} | Strategy: {strategy} | Seed: {seed}")
            model_path = os.path.join(model_dir, f"resnet18_seed_{seed}_best_model_target_{domain}.pth")
            model = get_model(len(PACS_CATEGORIES), pretrained=True, arch='resnet18')
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            test_dataset = PACS_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES, transform=None)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=tta_collate_fn)

            predictor = {
                'alexnet': AlexNetTTAPredictor,
                'mvdg': MVDGTTAPredictor,
                'custom': CustomTTAPredictor
            }[strategy](model)

            correct, total = 0, 0
            for images, labels in tqdm(test_loader):
                labels = labels.to(DEVICE)
                preds = predictor.predict_batch(images)
                correct += (preds == labels).sum().item()
                total += len(labels)

            acc = correct / total
            print(f"   ➜ Accuracy: {acc:.4f}")
            results[domain].setdefault(f"seed_{seed}", {})[strategy] = acc
            accs.append(acc)

        avg_results[domain][strategy] = sum(accs) / len(accs)
        print(f"✅ AVG Accuracy ({strategy}, {domain}): {avg_results[domain][strategy]:.4f}")

# === 保存结果 ===
now = datetime.now().strftime("%Y%m%d_%H%M")
save_dir = os.path.join(save_root, f"TTA_resnet18_results_{now}")
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "tta_results_resnet18.json"), "w") as f:
    json.dump({"per_seed": results, "averaged": avg_results}, f, indent=4)
print(f"\n✅ Results saved to {save_dir}")

# === 绘图 ===
for strategy in strategies:
    accs = [avg_results[d][strategy] for d in domains]
    plt.bar(domains, accs, label=strategy, alpha=0.7)

plt.ylabel("Average Accuracy")
plt.title("TTA Strategies (ResNet18)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "tta_plot_resnet18.png"))
plt.show()
