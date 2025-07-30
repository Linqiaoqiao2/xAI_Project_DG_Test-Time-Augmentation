#!/usr/bin/env python
# coding: utf-8

"""
PACS TTA Evaluation Script — ResNet50 with AlexNet, MVDG, and Custom TTA
"""

import torch
import sys
sys.path.append("/home/proj25gr5/xAI")  # baseline_resnet18.py 所在路径
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

from baseline_resnet18 import PACS_Dataset, PACS_CATEGORIES, BASE_PACS_PATH, DEVICE

# ======= get_model for ResNet50 (pretrained) =======
def get_model(num_classes, pretrained=True, arch='resnet50'):
    if arch == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# ======= TTA Predictors =======
class AlexNetTTAPredictor:
    def __init__(self, model):
        self.model = model.eval().to(DEVICE)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])(
                    transforms.ToTensor()(crop)) for crop in crops
            ]))
        ])

    def predict_batch(self, batch):
        bs = len(batch)
        crops = torch.stack([self.transform(img) for img in batch])  # [bs, 10, 3, 224, 224]
        crops = crops.view(-1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            logits = self.model(crops)
            logits = logits.view(bs, 10, -1).mean(1)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(1)
        return preds


class MVDGTTAPredictor:
    def __init__(self, model, t=32):
        self.model = model.eval().to(DEVICE)
        self.t = t
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def predict_batch(self, batch):
        preds = []
        with torch.no_grad():
            for img in batch:
                logits_sum = torch.zeros(self.model.fc.out_features, device=DEVICE)
                for _ in range(self.t):
                    aug_img = self.transform(img).unsqueeze(0).to(DEVICE)
                    logits_sum += self.model(aug_img).squeeze(0)
                probs = F.softmax(logits_sum / self.t, dim=0)
                preds.append(torch.argmax(probs).item())
        return torch.tensor(preds, device=DEVICE)


class CustomTTAPredictor:
    def __init__(self, model):
        self.model = model.eval().to(DEVICE)
        self.transforms = [
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        ]

    def predict_batch(self, batch):
        all_outputs = []
        with torch.no_grad():
            for transform in self.transforms:
                images_aug = torch.stack([transform(img) for img in batch]).to(DEVICE)
                outputs = self.model(images_aug)
                all_outputs.append(outputs)
        avg_logits = torch.stack(all_outputs).mean(dim=0)
        probs = F.softmax(avg_logits, dim=1)
        preds = probs.argmax(1)
        return preds

# ======= Collate & Config =======
def tta_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels

seeds = [42, 123, 2024]
domains = ["art_painting", "cartoon", "photo", "sketch"]
strategies = ['alexnet', 'mvdg', 'custom']
model_dir = "/home/proj25gr5/xAI_neu/models_resnet50_pretrained"
save_root = "/home/proj25gr5/xAI_neu"

results, avg_results = {}, {}

# ======= Evaluation =======
print(f"Using device: {DEVICE}")
for domain in domains:
    results[domain], avg_results[domain] = {}, {}
    for strategy in strategies:
        accs = []
        for seed in seeds:
            print(f"\n📊 Domain: {domain} | Strategy: {strategy} | Seed: {seed}")
            model_path = os.path.join(model_dir, f"resnet50_seed_{seed}_best_model_target_{domain}.pth")
            model = get_model(len(PACS_CATEGORIES), pretrained=True, arch='resnet50')
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

            test_dataset = PACS_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES, transform=None)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=tta_collate_fn)

            if strategy == 'alexnet':
                predictor = AlexNetTTAPredictor(model)
            elif strategy == 'mvdg':
                predictor = MVDGTTAPredictor(model)
            elif strategy == 'custom':
                predictor = CustomTTAPredictor(model)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

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

# ======= Save Results =======
now = datetime.now().strftime("%Y%m%d_%H%M")
save_dir = os.path.join(save_root, f"TTA_resnet50_results_{now}")
os.makedirs(save_dir, exist_ok=True)

result_file = os.path.join(save_dir, "tta_results_resnet50.json")
plot_file = os.path.join(save_dir, "tta_plot_resnet50.png")

with open(result_file, "w") as f:
    json.dump({"per_seed": results, "averaged": avg_results}, f, indent=4)
print(f"\n✅ Results saved to {result_file}")

# ======= Plot =======
for strategy in strategies:
    accs = [avg_results[dom][strategy] for dom in domains]
    plt.bar(domains, accs, label=strategy, alpha=0.7)

plt.ylabel("Average Accuracy")
plt.title("TTA Strategies (ResNet50)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(plot_file)
plt.show()
print(f"✅ Plot saved to {plot_file}")
