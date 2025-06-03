
# 📄 evaluate_tta.py 使用说明 / Documentation for `evaluate_tta.py`

## 📁 项目结构要求 / Project Structure Requirements

请将以下文件与文件夹放置在 `DomainBed` 根目录中：  
Place the following files and folders in the root directory of `DomainBed`:

```
DomainBed/
├── domainbed/
│   ├── datasets.py         🔁 替换为你的版本 / Replace with your version
│
├── evaluate_tta.py         ✅ 主评估脚本 / Main evaluation script
├── transforms/             ✅ 自定义 TTA 变换模块 / Custom TTA transforms
│   ├── __init__.py
│   └── transforms.py
│
├── train_output/
│   └── model.pkl           ✅ 已训练好的模型 / Trained model
```

---

## 🛠️ 步骤 1：替换 datasets.py / Step 1: Replace `datasets.py`

将你的自定义 `dataset.py` 替换原始的 DomainBed `datasets.py` 文件：  
Replace the original DomainBed `datasets.py` with your modified version:

```bash
cp /your_path/dataset.py DomainBed/domainbed/datasets.py
```

---

## 🛠️ 步骤 2：放置 transforms 文件夹 / Step 2: Add `transforms` folder

将整个 `transforms/` 文件夹放置在 `DomainBed/` 根目录下：  
Place the entire `transforms/` folder under the root of `DomainBed`.

---

## 🛠️ 步骤 3：运行 evaluate_tta.py / Step 3: Run `evaluate_tta.py`

```bash
cd DomainBed

python -m evaluate_tta \
  --model_path train_output/model.pkl \
  --data_dir /path/to/data \
  --dataset PACS \
  --test_env 3 \
  --tta_mode flip+rotate
```

---

### 参数说明 / Argument Description

| 参数 / Argument   | 描述 / Description |
|------------------|--------------------|
| `--model_path`   | 训练好的模型路径 / Path to trained model |
| `--data_dir`     | 数据集根目录 / Root data directory |
| `--dataset`      | 数据集名称，如 PACS / Dataset name, e.g., PACS |
| `--test_env`     | 测试 Domain ID / Test domain index |
| `--tta_mode`     | TTA 模式：basic, flip, rotate, flip+rotate / TTA mode |

---

## ✅ 附加建议 / Additional Tips

- **确保你在 DomainBed 根目录中运行脚本 / Run from DomainBed root**  
  否则可能会遇到 `ModuleNotFoundError: No module named 'domainbed'` 错误。

- **加入 PYTHONPATH / Add to PYTHONPATH if needed**：

```bash
export PYTHONPATH=$PYTHONPATH:/your_path/DomainBed
```

- **依赖安装 / Install dependencies**：

```bash
pip install torch torchvision tqdm Pillow
```

---

如需添加更多 TTA 模式，请在 `transforms/transforms.py` 中扩展 `get_tta_transforms()`。  
To add more TTA modes, extend `get_tta_transforms()` in `transforms/transforms.py`.
