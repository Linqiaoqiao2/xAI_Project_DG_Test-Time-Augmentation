# -*- coding: utf-8 -*-
"""
Evaluate ResNet50 (MVDG-train-only models) on PACS with:
1. No Test-Time Augmentation
2. MVDG-style 32x Weak Augmentation
"""

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

# ======= Config =======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACS_DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
PACS_CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
SEEDS = [42, 123, 2024]
MODEL_DIR = "/home/proj25gr5/xAI_neu/models_mvdg_train_only_resnet50"
BASE_PACS_PATH = "/home/proj25gr5/xAI/kfold"
SAVE_ROOT = f"/home/proj25gr5/xAI_neu/TTA_results_mvdg_eval_resnet50_{datetime.now().strftime('%Y%m%d_%H%M')}"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ======= Dataset =======
class PACS_Dataset(Dataset):
    def __init__(self, base_path, domain, categories, transform=None):
        self.image_paths, self.labels = [], []
        for idx, cat in enumerate(categories):
            folder = os.path.join(base_path, domain, cat)
            if os.path.isdir(folder):
                for file in os.listdir(folder):
                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(os.path.join(folder, file))
                        self.labels.append(idx)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class PACS_ImageList(Dataset):
    def __init__(self, image_paths, labels):
        self.images = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)

# ======= Predictors =======
class NoAugPredictor:
    def __init__(self, model):
        self.model = model.eval().to(DEVICE)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict_batch(self, images):
        with torch.no_grad():
            inputs = torch.stack([self.transform(img) for img in images]).to(DEVICE)
            logits = self.model(inputs)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        return preds

class MVDGTTAPredictor:
    def __init__(self, model, t=32):
        self.model = model.eval().to(DEVICE)
        self.t = t
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def predict_batch(self, images):
        preds = []
        with torch.no_grad():
            for img in images:
                logits_sum = torch.zeros(self.model.fc.out_features).to(DEVICE)
                for _ in range(self.t):
                    aug_img = self.transform(img).unsqueeze(0).to(DEVICE)
                    logits_sum += self.model(aug_img).squeeze(0)
                probs = F.softmax(logits_sum / self.t, dim=0)
                preds.append(torch.argmax(probs).item())
        return torch.tensor(preds).to(DEVICE)

# ======= Main Evaluation =======
results = {}
strategies = ["no_aug", "mvdg"]

for domain in PACS_DOMAINS:
    test_set = PACS_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES, transform=None)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

    results[domain] = {}

    for seed in SEEDS:
        model_filename = f"resnet50_seed{seed}_target{domain}.pth"
        model_path = os.path.join(MODEL_DIR, model_filename)

        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(PACS_CATEGORIES))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        print(f"\nEvaluating: Domain={domain}, Seed={seed}")

        for strategy in strategies:
            if strategy == "no_aug":
                predictor = NoAugPredictor(model)
            else:
                predictor = MVDGTTAPredictor(model)

            correct, total = 0, 0
            for images, labels in tqdm(test_loader):
                labels = labels.to(DEVICE)
                preds = predictor.predict_batch(images)
                correct += (preds == labels).sum().item()
                total += len(labels)

            acc = correct / total
            print(f"  [{strategy}] Accuracy: {acc:.4f}")
            results[domain].setdefault(strategy, {})[f"seed_{seed}"] = acc

# ======= Average & Save =======
avg_results = {}
for domain in PACS_DOMAINS:
    avg_results[domain] = {
        strategy: np.mean(list(results[domain][strategy].values()))
        for strategy in strategies
    }

with open(os.path.join(SAVE_ROOT, "tta_vs_noaug_results_resnet50.json"), "w") as f:
    json.dump({"per_seed": results, "avg": avg_results}, f, indent=4)

print(f"\n✅ Results saved to {SAVE_ROOT}/tta_vs_noaug_results_resnet50.json")

# ======= Plot =======
x = np.arange(len(PACS_DOMAINS))
bar_width = 0.35

fig, ax = plt.subplots()
for i, strategy in enumerate(strategies):
    accs = [avg_results[d][strategy] for d in PACS_DOMAINS]
    ax.bar(x + i * bar_width, accs, width=bar_width, label=strategy)

ax.set_ylabel('Accuracy')
ax.set_title('TTA vs No Augmentation (MVDG-trained ResNet50 Models)')
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(PACS_DOMAINS)
ax.set_ylim(0, 1.0)
ax.legend()
plt.tight_layout()
plot_path = os.path.join(SAVE_ROOT, "tta_vs_noaug_plot_resnet50.png")
plt.savefig(plot_path)
plt.show()

print(f"✅ Plot saved to {plot_path}")
