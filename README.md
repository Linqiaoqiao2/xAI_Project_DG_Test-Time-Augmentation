# Domain Generalization with Test-Time Techniques

This repository presents a unified evaluation framework for evaluating test-time and training-time strategies for domain generalization in image classification, using the PACS dataset. We systematically evaluate:

- **Baseline Performance**:
  - ResNet-18 and ResNet-50 (both pretrained and non-pretrained)

- **Test-Time Augmentation (TTA)** strategies applied on baseline models:
  - TenCrop
  - MVDG
  - Custom (horizontal flip + rotation)

- **Training-Time Augmentation**:
  - MVDG-style (ColorJitter + MixStyle)
  - Custom (horizontal flip + rotation)

- **TTA on Training-Time Augmented Models**:
  - Test-Time MVDG and Custom strategies applied on MVDG-trained and Custom-trained models

- **Test-Time Adaptation (TTAp)**:
  - TENT (Test-Time Entropy Minimization) applied on pretrained ResNet-50
  - TENT + Custom TTA applied on pretrained ResNet-50

All experiments follow the **Leave-One-Domain-Out (LODO)** protocol on the PACS dataset.

---

## 📂 Project Structure

```
experiments/         # Entry points to run baselines, TTA, TENT, etc.
src/
├── augmentations/   # TTA strategies
├── configs/         # All YAML configuration files
├── dataset/         # PACS loader
├── tent/            # TENT and TENT+TTA logic
├── train_aug/       # Baseline and training-time augmentation logic
data/                # PACS preprocessing script
scripts/             # Shell scripts for batch execution
utils/               # Logging, metrics, result saving
results_summary.json # Summary of all experiments
```

---

## 🚀 Getting Started

### 1. Set Up the Environment

```bash
conda env create -f environment.yaml
conda activate dg-tta
```

### 2. Prepare the PACS Dataset

Preprocessed PACS split files should be located under:

```
/path/to/kfold/{art_painting, cartoon, photo, sketch}
```

You can use `data/pacs_preprocessing.py` to convert PACS into this structure.

---

## 💻 Example Usage

### Run Baseline:

```bash
python experiments/run_baselines.py --config src/configs/baseline_resnet18_pt.yaml
```

### Run TTA (e.g., TenCrop):

```bash
python experiments/run_test_tta.py --config src/configs/test_tta_tencrop_resnet18_pt.yaml
```

### Run TENT:

```bash
python experiments/run_tent.py --config src/configs/tent_resnet50.yaml
```

### Run TENT + Custom TTA:

```bash
python experiments/run_tent_custom_tta.py --config src/configs/tent_custom_tta_resnet50.yaml
```

---

## 📊 Results

All results are saved in `results/` and summarized in `results_summary.json`. Each script logs domain-wise accuracy and optionally saves confusion matrices and charts.

---

## 📁 Scripts for Batch Execution

```bash
bash scripts/all_baselines.sh
bash scripts/all_tta_eval.sh
bash scripts/all_tent.sh
bash scripts/all_tent_tta.sh
bash scripts/all_train_aug.sh
bash scripts/all_tta_on_train_aug.sh
```

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

