# -*- coding: utf-8 -*-
"""
MVDG-style Meta-Learning Pipeline (ResNet-18) for PACS Dataset
Implements outer loop (SGD) and inner loop (Adam) optimization.
Train-only phase with no test-time augmentation.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

# ==== Device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Paths and Configs ====
BASE_PACS_PATH = "/home/proj25gr5/xAI/kfold"
SAVE_ROOT = "/home/proj25gr5/xAI_neu/models_mvdg_train_only_resnet18"
os.makedirs(SAVE_ROOT, exist_ok=True)

PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

SEEDS = [42, 2024, 123]
BATCH_SIZE = 64
NUM_EPOCHS = 30
IMG_SIZE = 224

# ==== Transforms ====
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==== Dataset ====
class PACSDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"[ERROR] Cannot open image: {path} | {e}")
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# ==== Model ====
def get_resnet18(num_classes=7):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ==== Train & Eval ====
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Training", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

# ==== Main ====
if __name__ == "__main__":
    print(">>> Script started")
    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for target_domain in PACS_DOMAINS:
            source_domains = [d for d in PACS_DOMAINS if d != target_domain]
            print(f"\n[Seed {seed}] Target: {target_domain} | Sources: {source_domains}")

            image_paths, labels = [], []
            for domain in source_domains:
                for label_idx, cat in enumerate(PACS_CATEGORIES):
                    cat_path = os.path.join(BASE_PACS_PATH, domain, cat)
                    for f in os.listdir(cat_path):
                        if f.lower().endswith(('jpg', 'jpeg', 'png')):
                            image_paths.append(os.path.join(cat_path, f))
                            labels.append(label_idx)

            print("Total images:", len(image_paths))
            print("Label distribution:", Counter(labels))

            train_paths, val_paths, train_labels, val_labels = train_test_split(
                image_paths, labels, test_size=0.15, stratify=labels, random_state=seed
            )

            print("Train stats:", Counter(train_labels))
            print("Val stats:", Counter(val_labels))

            train_set = PACSDataset(train_paths, train_labels, transform=train_transform)
            val_set = PACSDataset(val_paths, val_labels, transform=test_transform)
            test_set = PACSDataset(
                [os.path.join(BASE_PACS_PATH, target_domain, cat, f)
                 for cat in PACS_CATEGORIES for f in os.listdir(os.path.join(BASE_PACS_PATH, target_domain, cat))
                 if f.lower().endswith(('jpg', 'jpeg', 'png'))],
                [PACS_CATEGORIES.index(cat) for cat in PACS_CATEGORIES for f in os.listdir(os.path.join(BASE_PACS_PATH, target_domain, cat))
                 if f.lower().endswith(('jpg', 'jpeg', 'png'))],
                transform=test_transform
            )

            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            model = get_resnet18().to(device)

            outer_optimizer = optim.SGD(model.parameters(), lr=0.05, weight_decay=5e-4)
            inner_optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=5e-4)
            criterion = nn.CrossEntropyLoss()

            best_val_acc = 0.0
            for epoch in range(NUM_EPOCHS):
                if epoch < 24:
                    train_loss, train_acc = train(model, train_loader, outer_optimizer, criterion)
                else:
                    for param_group in outer_optimizer.param_groups:
                        param_group['lr'] = 5e-3
                    for param_group in inner_optimizer.param_groups:
                        param_group['lr'] = 1e-4
                    train_loss, train_acc = train(model, train_loader, inner_optimizer, criterion)

                val_loss, val_acc = evaluate(model, val_loader, criterion)
                print(f"Epoch {epoch+1:02d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = os.path.join(SAVE_ROOT, f"resnet18_seed{seed}_target{target_domain}.pth")
                    torch.save(model.state_dict(), model_path)

            model.load_state_dict(torch.load(model_path))
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            print(f"✅ Final Test Acc on [{target_domain}] = {test_acc:.4f}")
