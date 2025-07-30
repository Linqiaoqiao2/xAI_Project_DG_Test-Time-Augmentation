# -*- coding: utf-8 -*-
"""
Evaluate ResNet50 models (custom training augmentation) with Custom TTA (testing)
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

# ======= Model Loader =======
def get_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# ======= Custom TTA Only =======
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

# ======= Data Loader =======
def tta_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return images, labels

# ======= Config =======
seeds = [42, 123, 2024]
domains = ["art_painting", "cartoon", "photo", "sketch"]
model_base_dir = "/home/proj25gr5/xAI_neu/pretrained_resnet50_custom_aug"

results = {}
avg_results = {}

# ======= Evaluation Loop =======
print(f"Using device: {DEVICE}")
for domain in domains:
    results[domain] = {}
    accs = []

    for seed in seeds:
        print(f"\n🎯 Domain: {domain} | Seed: {seed}")
        model_path = os.path.join(model_base_dir, f"custom_seed_{seed}", f"resnet50_custom_target_{domain}.pth")

        model = get_model(num_classes=len(PACS_CATEGORIES))
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        test_dataset = PACS_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=tta_collate_fn)

        predictor = CustomTTAPredictor(model)

        correct, total = 0, 0
        for images, labels in tqdm(test_loader):
            labels = labels.to(DEVICE)
            preds = predictor.predict_batch(images)
            correct += (preds == labels).sum().item()
            total += len(labels)

        acc = correct / total
        print(f"   ➜ Accuracy: {acc:.4f}")
        results[domain][f"seed_{seed}"] = acc
        accs.append(acc)

    avg_results[domain] = sum(accs) / len(accs)
    print(f"✅ AVG Accuracy ({domain}): {avg_results[domain]:.4f}")

# ======= Save Results =======
now = datetime.now().strftime("%Y%m%d_%H%M")
save_dir = f"/home/proj25gr5/xAI_neu/TTA_resnet50_custom_only_{now}"
os.makedirs(save_dir, exist_ok=True)

result_file = os.path.join(save_dir, "tta_results_resnet50_custom_only.json")
plot_file = os.path.join(save_dir, "tta_plot_resnet50_custom_only.png")

with open(result_file, "w") as f:
    json.dump({"per_seed": results, "averaged": avg_results}, f, indent=4)
print(f"\n✅ Results saved to {result_file}")

# ======= Plot =======
accs = [avg_results[dom] for dom in domains]
plt.bar(domains, accs, label="Custom TTA", color='skyblue')
plt.ylabel("Average Accuracy")
plt.title("Custom TTA on ResNet50 (Custom Training)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(plot_file)
plt.show()
print(f"✅ Plot saved to {plot_file}")
