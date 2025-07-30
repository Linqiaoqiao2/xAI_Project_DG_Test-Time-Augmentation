# -*- coding: utf-8 -*-
"""
Baseline ResNet-50 with Strict Leave-One-Domain-Out (LODO), stratified random split on source domains,
Early Stopping, Multi-Seed, Logging, Resume, and WandB Support.
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

# ✅ wandb config
use_wandb = True
if use_wandb:
    import wandb

# ✅ device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ✅ PACS config
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

# ✅ hyperparameters
MODEL_CHOICE = 'resnet50'
PRETRAINED = True
NUM_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 3e-5
PATIENCE = 5
SEEDS = [42, 2024, 123]

# ✅ file paths
BASE_PACS_PATH = "/home/proj25gr5/xAI/kfold"
SAVE_ROOT = "/home/proj25gr5/xAI_neu/trained_models_resnet50"
RESULTS_DIR = SAVE_ROOT
os.makedirs(SAVE_ROOT, exist_ok=True)

# ✅ dataset classes
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

# ✅ transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def get_model(num_classes=len(PACS_CATEGORIES), pretrained=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ✅ training and evaluation loops
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
        for inputs, labels in tqdm(dataloader, desc="Val/Test", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# ✅ main training loop
if __name__ == "__main__":
    all_seed_results = []

    for seed in SEEDS:
        print(f"\n==========================\n Running with Seed: {seed}\n==========================")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if DEVICE == 'cuda':
            torch.cuda.manual_seed_all(seed)

        seed_accuracies = []

        for target_domain in PACS_DOMAINS:
            print(f"\n===== Target Domain: {target_domain} =====")
            source_domains = [d for d in PACS_DOMAINS if d != target_domain]

            all_image_paths, all_labels = [], []
            for d in source_domains:
                ds = PACS_Dataset(BASE_PACS_PATH, d, PACS_CATEGORIES, transform=None)
                all_image_paths.extend(ds.image_paths)
                all_labels.extend(ds.labels)

            train_paths, val_paths, train_labels, val_labels = train_test_split(
                all_image_paths,
                all_labels,
                test_size=0.15,
                random_state=seed,
                shuffle=True,
                stratify=all_labels
            )

            train_dataset = SimpleImageDataset(train_paths, train_labels, transform=train_transform)
            val_dataset = SimpleImageDataset(val_paths, val_labels, transform=val_test_transform)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
            test_dataset = PACS_Dataset(BASE_PACS_PATH, target_domain, PACS_CATEGORIES, transform=val_test_transform)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

            model = get_model(len(PACS_CATEGORIES), PRETRAINED).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

            best_val_acc = 0.0
            patience_counter = 0
            epoch_log = []

            log_path = os.path.join(SAVE_ROOT, f"log_resnet50_seed_{seed}_target_{target_domain}.json")
            checkpoint_path = os.path.join(SAVE_ROOT, f"checkpoint_resnet50_seed_{seed}_target_{target_domain}.pth")
            best_model_path = os.path.join(SAVE_ROOT, f"resnet50_seed_{seed}_best_model_target_{target_domain}.pth")

            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log_data = json.load(f)
                if log_data.get("finished", False):
                    print(f"✅ Log found and finished for seed {seed}, domain {target_domain}, skipping...")
                    seed_accuracies.append(log_data["best_val_acc"])
                    continue

            if use_wandb:
                wandb_run_name = f"ResNet50_Seed{seed}_Domain_{target_domain}"
                wandb.init(project="PACS_ResNet50_Baseline", name=wandb_run_name, config={
                    "epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "model": MODEL_CHOICE,
                    "seed": seed,
                    "target_domain": target_domain,
                }, reinit=True)

            for epoch in range(NUM_EPOCHS):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
                val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, DEVICE)
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

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "patience_counter": patience_counter,
                    "epoch_log": epoch_log
                }
                torch.save(checkpoint, checkpoint_path)

                log_data = {
                    "seed": seed,
                    "target_domain": target_domain,
                    "epochs": epoch_log,
                    "best_val_acc": best_val_acc,
                    "finished": False
                }
                with open(log_path, "w") as f:
                    json.dump(log_data, f, indent=4)

                if patience_counter >= PATIENCE:
                    print("Early stopping triggered!")
                    break

            log_data["finished"] = True
            with open(log_path, "w") as f:
                json.dump(log_data, f, indent=4)

            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            if use_wandb:
                wandb.finish()

            model.load_state_dict(torch.load(best_model_path))
            test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, DEVICE)
            print(f"🎯 Final Evaluation on Target Domain [{target_domain}] - Acc: {test_acc:.3f}")
            seed_accuracies.append(test_acc)

        all_seed_results.append(seed_accuracies)

    mean_accuracies = np.mean(all_seed_results, axis=0).tolist()
    final_results = {
        "seeds": SEEDS,
        "domains": PACS_DOMAINS,
        "accuracies_per_seed": all_seed_results,
        "mean_accuracies": mean_accuracies,
        "average_overall": float(np.mean(mean_accuracies)),
        "worst_case_overall": float(np.min(mean_accuracies))
    }
    with open(os.path.join(RESULTS_DIR, "baseline_resnet50_results_multi_seed.json"), "w") as f:
        json.dump(final_results, f, indent=4)

    print("\n✅ Finished! Saved all models and results.")
