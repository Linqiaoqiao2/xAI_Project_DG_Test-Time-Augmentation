# -*- coding: utf-8 -*-
"""
Baseline ResNet-18 with TTA-style Augmentations in Training,
Multi-Seed, Logging, Resume (Checkpoint), and WandB Support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import numpy as np
import os
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ✅ wandb 配置
use_wandb = True
if use_wandb:
    import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

MODEL_CHOICE = 'resnet18'
PRETRAINED = True
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
PATIENCE = 5
SEEDS = [42, 2024, 123]

BASE_PACS_PATH = "/home/proj25gr5/xAI/kfold"
SAVE_ROOT = "/home/proj25gr5/xAI/trained_models_augmented"
os.makedirs(SAVE_ROOT, exist_ok=True)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def get_train_transform(strategy):
    if strategy == 'default':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation([0, 360]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif strategy == 'alexnet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif strategy == 'light':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

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

def get_model(num_classes=len(PACS_CATEGORIES), pretrained=True):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

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

if __name__ == "__main__":
    strategies = ['default', 'alexnet', 'light']

    for strategy in strategies:
        print(f"\n==================\n Training with Augmentation Strategy: {strategy}\n==================")
        train_transform = get_train_transform(strategy)

        for seed in SEEDS:
            print(f"\n===== Seed: {seed} =====")
            torch.manual_seed(seed)
            np.random.seed(seed)
            if DEVICE == 'cuda':
                torch.cuda.manual_seed_all(seed)

            for target_domain in PACS_DOMAINS:
                print(f"\n--- Target Domain: {target_domain} ---")
                source_domains = [d for d in PACS_DOMAINS if d != target_domain]

                train_datasets = [PACS_Dataset(BASE_PACS_PATH, d, PACS_CATEGORIES, transform=train_transform) for d in source_domains]
                train_dataset = ConcatDataset(train_datasets)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

                test_dataset = PACS_Dataset(BASE_PACS_PATH, target_domain, PACS_CATEGORIES, transform=val_test_transform)
                test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

                model = get_model(len(PACS_CATEGORIES), PRETRAINED).to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

                best_val_acc = 0.0
                patience_counter = 0
                epoch_log = []

                run_name = f"{strategy}_seed{seed}_target_{target_domain}"
                save_subdir = os.path.join(SAVE_ROOT, strategy)
                os.makedirs(save_subdir, exist_ok=True)
                log_path = os.path.join(save_subdir, f"log_{run_name}.json")
                checkpoint_path = os.path.join(save_subdir, f"ckpt_{run_name}.pth")
                best_model_path = os.path.join(save_subdir, f"best_model_{run_name}.pth")

                # ✅ Resume if checkpoint exists
                if os.path.exists(checkpoint_path):
                    print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    best_val_acc = checkpoint["best_val_acc"]
                    patience_counter = checkpoint["patience_counter"]
                    epoch_log = checkpoint["epoch_log"]

                if use_wandb:
                    wandb.init(project="PACS_TTA_Training", name=run_name, config={
                        "strategy": strategy,
                        "epochs": NUM_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "seed": seed,
                        "target_domain": target_domain,
                    }, reinit=True)

                for epoch in range(len(epoch_log), NUM_EPOCHS):
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
                    val_loss, val_acc = eval_one_epoch(model, test_loader, criterion, DEVICE)
                    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

                    epoch_log.append({
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })

                    if use_wandb:
                        wandb.log({
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc
                        })

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        patience_counter = 0
                        torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1

                    # ✅ Save checkpoint
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_acc": best_val_acc,
                        "patience_counter": patience_counter,
                        "epoch_log": epoch_log
                    }, checkpoint_path)

                    with open(log_path, "w") as f:
                        json.dump({
                            "strategy": strategy,
                            "seed": seed,
                            "target_domain": target_domain,
                            "epochs": epoch_log,
                            "best_val_acc": best_val_acc
                        }, f, indent=4)

                    if patience_counter >= PATIENCE:
                        print("Early stopping triggered!")
                        break

                # ✅ Remove checkpoint after finishing
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)

                if use_wandb:
                    wandb.finish()

    print("\n✅ Finished all strategy + seed combinations. Saved models and logs.")
