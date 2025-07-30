# -*- coding: utf-8 -*-
"""
Train ResNet-50 on PACS with LODO using Custom Augmentation (Train Only)
All models and results stored in /home/proj25gr5/xAI_neu
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ====== Settings ======
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
PRETRAINED = True
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
PATIENCE = 5
SEEDS = [42, 123, 2024]
BASE_PACS_PATH = "/home/proj25gr5/xAI/kfold"
SAVE_ROOT = "/home/proj25gr5/xAI_neu/pretrained_resnet50_custom_aug"
os.makedirs(SAVE_ROOT, exist_ok=True)

# ====== Dataset Classes ======
class PACS_Dataset(Dataset):
    def __init__(self, base_path, domain_name, categories, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label_idx, category in enumerate(categories):
            category_path = os.path.join(base_path, domain_name, category)
            if not os.path.isdir(category_path):
                continue
            for img_file in os.listdir(category_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(category_path, img_file))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class SimpleImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ====== Augmentation ======
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

custom_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ====== Model for ResNet-50 ======
def get_model(num_classes=len(PACS_CATEGORIES), pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ====== Training & Evaluation ======
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(dataloader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def eval_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# ====== Main Training Loop ======
if __name__ == "__main__":
    results = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed_all(seed)

        results[f"seed_{seed}"] = {}

        for target_domain in PACS_DOMAINS:
            print(f"\n==== Seed {seed} | Target Domain: {target_domain} ====")
            source_domains = [d for d in PACS_DOMAINS if d != target_domain]
            all_image_paths, all_labels = [], []

            for domain in source_domains:
                ds = PACS_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES)
                all_image_paths.extend(ds.image_paths)
                all_labels.extend(ds.labels)

            train_paths, val_paths, train_labels, val_labels = train_test_split(
                all_image_paths, all_labels,
                test_size=0.15, shuffle=True,
                stratify=all_labels, random_state=seed
            )

            train_dataset = SimpleImageDataset(train_paths, train_labels, transform=get_custom_train_transform())
            val_dataset = SimpleImageDataset(val_paths, val_labels, transform=val_transform)
            test_dataset = PACS_Dataset(BASE_PACS_PATH, target_domain, PACS_CATEGORIES, transform=val_transform)

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            model = get_model(pretrained=PRETRAINED).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

            best_val_acc = 0.0
            patience_counter = 0
            save_dir = os.path.join(SAVE_ROOT, f"custom_seed_{seed}")
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"resnet50_custom_target_{target_domain}.pth")

            for epoch in range(NUM_EPOCHS):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
                val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, DEVICE)
                print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered!")
                    break

            # Final evaluation
            model.load_state_dict(torch.load(model_path))
            test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, DEVICE)
            print(f"\n🎯 Final Test Acc on [{target_domain}] | Seed {seed}: {test_acc:.3f}")
            results[f"seed_{seed}"][target_domain] = float(test_acc)

    # Average results
    domain_avg = {d: [] for d in PACS_DOMAINS}
    for seed_key in results:
        for d in PACS_DOMAINS:
            domain_avg[d].append(results[seed_key][d])
    domain_avg_acc = {d: float(np.mean(domain_avg[d])) for d in PACS_DOMAINS}
    overall_avg_acc = float(np.mean([acc for accs in domain_avg.values() for acc in accs]))

    results["domain_avg_acc"] = domain_avg_acc
    results["overall_avg_acc"] = overall_avg_acc

    with open(os.path.join(SAVE_ROOT, "results_summary_custom.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\n✅ All training done. Summary saved.")
