# -*- coding: utf-8 -*-
"""
TENT + Custom Test Time Augmentation Comparison Script
-----------------------------------------------------

This script compares two strategies:
1. TENT Only (baseline) - Original TENT without TTA
2. TENT + Custom TTA - TENT with 3-transform Custom augmentation strategy

Custom TTA Strategy:
- Transform 1: Original (Resize + Normalize)
- Transform 2: Horizontal Flip
- Transform 3: Random Rotation (15 degrees)
- Fusion: Mean logits → Softmax → Argmax
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import random

# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PACS_DOMAINS = ['art_painting', 'cartoon', 'photo', 'sketch']
PACS_CATEGORIES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
SEEDS = [42, 2024, 123]
BATCH_SIZE = 32

BASE_PACS_PATH = "/home/proj25gr5/xAI/kfold"
MODEL_ROOT = "/home/proj25gr5/xAI_neu/pretrained_resnet50_models_and_results"
SAVE_ROOT = "/home/proj25gr5/xAI_neu/tent_tta_comparison_resnet50"
os.makedirs(SAVE_ROOT, exist_ok=True)

# 为两种策略创建子文件夹
TENT_ONLY_DIR = os.path.join(SAVE_ROOT, "tent_only")
TENT_TTA_DIR = os.path.join(SAVE_ROOT, "tent_custom_tta")
os.makedirs(TENT_ONLY_DIR, exist_ok=True)
os.makedirs(TENT_TTA_DIR, exist_ok=True)

# ==== Seed Fix ====
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ==== Transform Definitions ====
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Base transform for TENT Only
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Custom TTA transforms
custom_tta_transforms = [
    # Transform 1: Original
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    # Transform 2: Horizontal Flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    # Transform 3: Random Rotation
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
]

# ==== Dataset Classes ====
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

class PACS_TTA_Dataset(Dataset):
    """Dataset for Custom TTA - returns multiple transformed versions"""
    def __init__(self, base_path, domain_name, categories, tta_transforms):
        self.tta_transforms = tta_transforms
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
        
        # Apply all TTA transforms
        transformed_images = []
        for transform in self.tta_transforms:
            transformed_images.append(transform(img))
        
        # Stack into a single tensor: [num_transforms, C, H, W]
        tta_batch = torch.stack(transformed_images, dim=0)
        return tta_batch, label

# ==== Model Functions ====
def get_model(num_classes=len(PACS_CATEGORIES)):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def configure_model_for_tent(model):
    """配置模型用于TENT适应"""
    for p in model.parameters():
        p.requires_grad = False
    
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.track_running_stats = False
            m.momentum = None
            if m.weight is not None:
                m.weight.requires_grad = True
            if m.bias is not None:
                m.bias.requires_grad = True
    
    return model

# ==== TENT Functions ====
def tent_adapt_single_batch(model, batch_images, optimizer):
    """在单个batch上执行TENT适应"""
    model.train()
    batch_images = batch_images.to(DEVICE)
    
    outputs = model(batch_images)
    probs = torch.softmax(outputs, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    
    optimizer.zero_grad(set_to_none=True)
    entropy.backward()
    optimizer.step()
    
    return entropy.item()

def tent_adapt_single_batch_tta(model, tta_batch, optimizer):
    """在TTA batch上执行TENT适应 - 使用融合后的logits计算entropy"""
    model.train()
    tta_batch = tta_batch.to(DEVICE)  # [batch_size, num_transforms, C, H, W]
    
    batch_size, num_transforms = tta_batch.shape[:2]
    
    # Reshape to process all transforms at once
    tta_batch_flat = tta_batch.view(-1, *tta_batch.shape[2:])  # [batch_size * num_transforms, C, H, W]
    
    # Forward pass on all transformed versions
    outputs_flat = model(tta_batch_flat)  # [batch_size * num_transforms, num_classes]
    
    # Reshape back and average logits
    outputs_reshaped = outputs_flat.view(batch_size, num_transforms, -1)  # [batch_size, num_transforms, num_classes]
    fused_logits = outputs_reshaped.mean(dim=1)  # [batch_size, num_classes]
    
    # Calculate entropy on fused predictions
    probs = torch.softmax(fused_logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    entropy.backward()
    optimizer.step()
    
    return entropy.item()

# ==== Evaluation Functions ====
def evaluate_batch_standard(model, images, labels):
    """标准评估（无TTA）"""
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(DEVICE))
        preds = outputs.argmax(dim=1)
        correct = (preds == labels.to(DEVICE)).sum().item()
        total = labels.size(0)
    return correct, total, preds

def evaluate_batch_tta(model, tta_batch, labels):
    """TTA评估 - 使用Custom融合策略"""
    model.eval()
    with torch.no_grad():
        tta_batch = tta_batch.to(DEVICE)  # [batch_size, num_transforms, C, H, W]
        batch_size, num_transforms = tta_batch.shape[:2]
        
        # Reshape and forward pass
        tta_batch_flat = tta_batch.view(-1, *tta_batch.shape[2:])
        outputs_flat = model(tta_batch_flat)
        
        # Reshape and fuse logits (mean fusion)
        outputs_reshaped = outputs_flat.view(batch_size, num_transforms, -1)
        fused_logits = outputs_reshaped.mean(dim=1)  # Mean over transforms
        
        # Final prediction
        preds = fused_logits.argmax(dim=1)
        correct = (preds == labels.to(DEVICE)).sum().item()
        total = labels.size(0)
    
    return correct, total, preds

def evaluate_full(model, loader, use_tta=False):
    """评估整个数据集"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            if use_tta:
                images, labels = batch
                c, t, _ = evaluate_batch_tta(model, images, labels)
            else:
                images, labels = batch
                c, t, _ = evaluate_batch_standard(model, images, labels)
            correct += c
            total += t
    return correct / total if total > 0 else 0.0

# ==== 运行单种策略的实验 ====
def run_tent_only_experiment():
    """运行TENT Only实验"""
    print(f"\n{'='*60}")
    print(f"🔬 运行 TENT Only 实验")
    print(f"{'='*60}")
    
    results = {}
    log_path = os.path.join(TENT_ONLY_DIR, "tent_only_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    
    with open(log_path, "a") as log_f:
        for seed in SEEDS:
            set_seed(seed)
            results[str(seed)] = {}

            for domain in PACS_DOMAINS:
                print(f"\n🌍 Domain: {domain} | 🔢 Seed: {seed} | 🏷️ Strategy: TENT Only")

                # 创建标准数据集
                dataset = PACS_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES, transform=base_transform)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                
                # 加载和配置模型
                model = get_model()
                model_path = os.path.join(MODEL_ROOT, f"resnet50_seed_{seed}_best_model_target_{domain}.pth")
                if not os.path.exists(model_path):
                    print(f"❌ Model not found: {model_path}")
                    continue
                    
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)

                baseline_acc = evaluate_full(model, loader, use_tta=False)
                print(f"📏 Baseline Accuracy: {baseline_acc:.4f}")

                model = configure_model_for_tent(model)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

                # True Online TENT
                data_iter = iter(loader)
                online_correct = 0
                online_total = 0
                batch_count = 0
                
                try:
                    prev_images, prev_labels = next(data_iter)
                    batch_count += 1
                except StopIteration:
                    continue

                for curr_images, curr_labels in tqdm(data_iter, desc=f"TENT Only [{domain}]"):
                    batch_count += 1
                    
                    # 评估当前batch
                    correct, total, preds = evaluate_batch_standard(model, curr_images, curr_labels)
                    online_correct += correct
                    online_total += total
                    current_acc = online_correct / online_total
                    
                    # 在上一个batch上适应
                    entropy_val = tent_adapt_single_batch(model, prev_images, optimizer)
                    
                    # 记录日志
                    log_entry = {
                        "strategy": "tent_only",
                        "seed": seed,
                        "domain": domain,
                        "batch_id": batch_count,
                        "correct": correct,
                        "batch_size": total,
                        "entropy": round(entropy_val, 6),
                        "running_accuracy": round(current_acc, 6)
                    }
                    log_f.write(json.dumps(log_entry) + "\n")
                    
                    prev_images, prev_labels = curr_images, curr_labels

                final_acc = online_correct / online_total if online_total > 0 else 0.0
                improvement = final_acc - baseline_acc
                print(f"✅ Final Accuracy: {final_acc:.4f} (Δ{improvement:+.4f})")

                # 保存模型和结果
                save_name = f"seed_{seed}_adapted_{domain}_tent_only.pth"
                torch.save(model.state_dict(), os.path.join(TENT_ONLY_DIR, save_name))
                
                results[str(seed)][domain] = {
                    "final_accuracy": final_acc,
                    "baseline_accuracy": baseline_acc,
                    "improvement": improvement
                }
                
                del model
                torch.cuda.empty_cache()

    # 保存结果
    results_json = os.path.join(TENT_ONLY_DIR, "tent_only_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=4)
    
    return results

def run_tent_tta_experiment():
    """运行TENT + Custom TTA实验"""
    print(f"\n{'='*60}")
    print(f"🔬 运行 TENT + Custom TTA 实验")
    print(f"{'='*60}")
    
    results = {}
    log_path = os.path.join(TENT_TTA_DIR, "tent_tta_log.jsonl")
    if os.path.exists(log_path):
        os.remove(log_path)
    
    with open(log_path, "a") as log_f:
        for seed in SEEDS:
            set_seed(seed)
            results[str(seed)] = {}

            for domain in PACS_DOMAINS:
                print(f"\n🌍 Domain: {domain} | 🔢 Seed: {seed} | 🏷️ Strategy: TENT + Custom TTA")

                # 创建TTA数据集
                tta_dataset = PACS_TTA_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES, custom_tta_transforms)
                tta_loader = DataLoader(tta_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                
                # 为baseline评估创建标准数据集
                standard_dataset = PACS_Dataset(BASE_PACS_PATH, domain, PACS_CATEGORIES, transform=base_transform)
                standard_loader = DataLoader(standard_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                
                # 加载和配置模型
                model = get_model()
                model_path = os.path.join(MODEL_ROOT, f"resnet50_seed_{seed}_best_model_target_{domain}.pth")
                if not os.path.exists(model_path):
                    print(f"❌ Model not found: {model_path}")
                    continue
                    
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)

                # Baseline with TTA
                baseline_acc = evaluate_full(model, tta_loader, use_tta=True)
                print(f"📏 Baseline Accuracy (with TTA): {baseline_acc:.4f}")

                model = configure_model_for_tent(model)
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

                # True Online TENT with TTA
                data_iter = iter(tta_loader)
                online_correct = 0
                online_total = 0
                batch_count = 0
                
                try:
                    prev_tta_images, prev_labels = next(data_iter)
                    batch_count += 1
                except StopIteration:
                    continue

                for curr_tta_images, curr_labels in tqdm(data_iter, desc=f"TENT+TTA [{domain}]"):
                    batch_count += 1
                    
                    # 使用TTA评估当前batch
                    correct, total, preds = evaluate_batch_tta(model, curr_tta_images, curr_labels)
                    online_correct += correct
                    online_total += total
                    current_acc = online_correct / online_total
                    
                    # 在上一个TTA batch上适应
                    entropy_val = tent_adapt_single_batch_tta(model, prev_tta_images, optimizer)
                    
                    # 记录日志
                    log_entry = {
                        "strategy": "tent_tta",
                        "seed": seed,
                        "domain": domain,
                        "batch_id": batch_count,
                        "correct": correct,
                        "batch_size": total,
                        "entropy": round(entropy_val, 6),
                        "running_accuracy": round(current_acc, 6)
                    }
                    log_f.write(json.dumps(log_entry) + "\n")
                    
                    prev_tta_images, prev_labels = curr_tta_images, curr_labels

                final_acc = online_correct / online_total if online_total > 0 else 0.0
                improvement = final_acc - baseline_acc
                print(f"✅ Final Accuracy: {final_acc:.4f} (Δ{improvement:+.4f})")

                # 保存模型和结果
                save_name = f"seed_{seed}_adapted_{domain}_tent_tta.pth"
                torch.save(model.state_dict(), os.path.join(TENT_TTA_DIR, save_name))
                
                results[str(seed)][domain] = {
                    "final_accuracy": final_acc,
                    "baseline_accuracy": baseline_acc,
                    "improvement": improvement
                }
                
                del model
                torch.cuda.empty_cache()

    # 保存结果
    results_json = os.path.join(TENT_TTA_DIR, "tent_tta_results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=4)
    
    return results

# ==== Main Execution ====
print(f"🚀 开始 TENT vs TENT+Custom TTA 对比实验")
print(f"📝 批次大小: {BATCH_SIZE}")
print(f"🎲 测试种子: {SEEDS}")
print(f"🌍 测试域: {PACS_DOMAINS}")
print(f"🔧 Custom TTA: 3 transforms (原图 + 水平翻转 + 旋转15度)")

# 运行两个实验
tent_only_results = run_tent_only_experiment()
tent_tta_results = run_tent_tta_experiment()

# ==== 对比分析 ====
print(f"\n{'='*80}")
print(f"📊 对比分析结果")
print(f"{'='*80}")

comparison_data = {
    "tent_only": tent_only_results,
    "tent_tta": tent_tta_results
}

# 计算统计信息
strategy_stats = {}
for strategy_name, results in comparison_data.items():
    strategy_stats[strategy_name] = {}
    for domain in PACS_DOMAINS:
        accs = []
        improvements = []
        for seed in SEEDS:
            if str(seed) in results and domain in results[str(seed)]:
                accs.append(results[str(seed)][domain]["final_accuracy"])
                improvements.append(results[str(seed)][domain]["improvement"])
        
        if accs:
            strategy_stats[strategy_name][domain] = {
                "mean_accuracy": np.mean(accs),
                "std_accuracy": np.std(accs),
                "mean_improvement": np.mean(improvements),
                "std_improvement": np.std(improvements)
            }

# 打印对比结果
print("\n📈 策略对比摘要:")
print("-" * 80)
print(f"{'Domain':<12} {'TENT Only':<20} {'TENT + TTA':<20} {'TTA Gain':<15}")
print("-" * 80)

tta_gains = []
for domain in PACS_DOMAINS:
    tent_only_acc = strategy_stats["tent_only"][domain]["mean_accuracy"]
    tent_tta_acc = strategy_stats["tent_tta"][domain]["mean_accuracy"]
    tta_gain = tent_tta_acc - tent_only_acc
    tta_gains.append(tta_gain)
    
    print(f"{domain:<12} {tent_only_acc:.4f} ± {strategy_stats['tent_only'][domain]['std_accuracy']:.4f} "
          f"{tent_tta_acc:.4f} ± {strategy_stats['tent_tta'][domain]['std_accuracy']:.4f} "
          f"{tta_gain:+.4f}")

print("-" * 80)
overall_tent_only = np.mean([strategy_stats["tent_only"][d]["mean_accuracy"] for d in PACS_DOMAINS])
overall_tent_tta = np.mean([strategy_stats["tent_tta"][d]["mean_accuracy"] for d in PACS_DOMAINS])
overall_gain = overall_tent_tta - overall_tent_only

print(f"{'Overall':<12} {overall_tent_only:.4f}{' '*13} {overall_tent_tta:.4f}{' '*13} {overall_gain:+.4f}")

# ==== 生成对比图表 ====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 子图1: 准确率对比
x = np.arange(len(PACS_DOMAINS))
width = 0.35

tent_only_means = [strategy_stats["tent_only"][d]["mean_accuracy"] for d in PACS_DOMAINS]
tent_only_stds = [strategy_stats["tent_only"][d]["std_accuracy"] for d in PACS_DOMAINS]
tent_tta_means = [strategy_stats["tent_tta"][d]["mean_accuracy"] for d in PACS_DOMAINS]
tent_tta_stds = [strategy_stats["tent_tta"][d]["std_accuracy"] for d in PACS_DOMAINS]

bars1 = ax1.bar(x - width/2, tent_only_means, width, yerr=tent_only_stds, 
                label='TENT Only', alpha=0.8, color='lightcoral', capsize=5)
bars2 = ax1.bar(x + width/2, tent_tta_means, width, yerr=tent_tta_stds, 
                label='TENT + Custom TTA', alpha=0.8, color='skyblue', capsize=5)

# 添加数值标签
for i, (bar1, bar2, acc1, acc2) in enumerate(zip(bars1, bars2, tent_only_means, tent_tta_means)):
    ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
             f'{acc1:.3f}', ha='center', va='bottom', fontsize=9)
    ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
             f'{acc2:.3f}', ha='center', va='bottom', fontsize=9)

ax1.set_xlabel('Target Domain')
ax1.set_ylabel('Accuracy')
ax1.set_title('TENT vs TENT + Custom TTA Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(PACS_DOMAINS)
ax1.legend()
ax1.grid(axis='y', linestyle='--', alpha=0.3)
ax1.set_ylim(0, 1.0)

# 子图2: TTA增益
colors = ['green' if gain > 0 else 'red' for gain in tta_gains]
bars3 = ax2.bar(PACS_DOMAINS, tta_gains, color=colors, alpha=0.7)

for bar, gain in zip(bars3, tta_gains):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001 if gain > 0 else bar.get_height() - 0.003,
             f'{gain:+.3f}', ha='center', va='bottom' if gain > 0 else 'top', fontsize=10, fontweight='bold')

ax2.set_xlabel('Target Domain')
ax2.set_ylabel('TTA Gain (Accuracy)')
ax2.set_title('Custom TTA Performance Gain')
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.grid(axis='y', linestyle='--', alpha=0.3)

plt.tight_layout()
comparison_plot_path = os.path.join(SAVE_ROOT, "tent_vs_tent_tta_comparison.png")
plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# ==== 保存完整对比结果 ====
complete_comparison = {
    "metadata": {
        "batch_size": BATCH_SIZE,
        "seeds": SEEDS,
        "domains": PACS_DOMAINS,
        "custom_tta_transforms": ["resize+normalize", "horizontal_flip", "rotation_15deg"]
    },
    "results": comparison_data,
    "statistics": strategy_stats,
    "summary": {
        "overall_tent_only": overall_tent_only,
        "overall_tent_tta": overall_tent_tta,
        "overall_tta_gain": overall_gain,
        "domain_tta_gains": dict(zip(PACS_DOMAINS, tta_gains))
    }
}

complete_results_path = os.path.join(SAVE_ROOT, "complete_tent_tta_comparison.json")
with open(complete_results_path, "w") as f:
    json.dump(complete_comparison, f, indent=4)

print(f"\n✅ 对比实验完成！")
print(f"📊 对比图表: {comparison_plot_path}")
print(f"📄 完整结果: {complete_results_path}")
print(f"📁 所有文件保存在: {SAVE_ROOT}")

# ==== 最终结论 ====
print(f"\n🎯 实验结论:")
if overall_gain > 0:
    print(f"   ✅ Custom TTA平均提升准确率 {overall_gain:.4f} ({overall_gain*100:.2f}%)")
    best_domain = PACS_DOMAINS[np.argmax(tta_gains)]
    best_gain = max(tta_gains)
    print(f"   🏆 最大提升在 {best_domain} domain: +{best_gain:.4f}")
else:
    print(f"   ❌ Custom TTA平均降低准确率 {abs(overall_gain):.4f}")
    
if min(tta_gains) < 0:
    worst_domain = PACS_DOMAINS[np.argmin(tta_gains)]
    worst_gain = min(tta_gains)
    print(f"   ⚠️  {worst_domain} domain 性能下降: {worst_gain:.4f}")