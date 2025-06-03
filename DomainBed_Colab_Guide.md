# 📚 DomainBed 使用说明 | Usage Instructions

本项目通过 `domainbed_baseline.ipynb` 在 Google Colab 上运行 DomainBed 的训练流程，并为 PACS 数据集复现论文实验结果。

This project provides a Google Colab-based pipeline (`domainbed_baseline.ipynb`) to run DomainBed training and reproduce experimental results on the PACS dataset.

---

## ✅ 使用步骤 | Step-by-Step Guide

### 1. 依序执行 Notebook 中的所有程序单元格  
**请按照从上到下的顺序运行 `domainbed_baseline.ipynb` 中的每一个代码单元格。**  

This notebook is designed to be run sequentially. Simply execute each cell from top to bottom.

---

### 2. 上传 `kaggle.json` 文件以下载 PACS 数据集  
当 Notebook 提示你上传文件时，请上传你的 `kaggle.json` 认证文件。  
你可以在 Kaggle 网站中生成该文件：

- 打开 https://www.kaggle.com/account
- 滚动至 **API** 部分
- 点击 **Create New API Token** 以下载 `kaggle.json`

📁 上传后将自动开始下载所需数据。

When prompted to upload a file, please upload your `kaggle.json`.  
To generate one:

- Go to https://www.kaggle.com/account
- Scroll to the **API** section
- Click **Create New API Token** to download `kaggle.json`

---

### 3. 覆盖两个脚本文件以将 worker 数设置为 2  
**在运行最后一个训练单元格之前**，请手动用当前分支下的修改版本替换以下两个文件：

- `domainbed/scripts/train.py`
- `domainbed/lib/fast_data_loader.py`

这些文件已将 `num_workers` 参数硬编码为 `2`，以确保在 Colab 中正常运行，避免 multiprocessing 报错。

⚙️ 替换方法（在 Colab 文件浏览器中）：
- 打开左侧目录树，找到 `domainbed/scripts/` 和 `domainbed/lib/` 文件夹
- 找到 `train.py` 和 `fast_data_loader.py`
- 右键 → `Upload` → 上传你本地修改好的两个文件

**务必在运行最后的训练步骤前完成此操作！**

Before running the final training cell, overwrite the following two files with the custom versions from this branch:

- `domainbed/scripts/train.py`
- `domainbed/lib/fast_data_loader.py`

These versions hardcode `num_workers = 2` to avoid multiprocessing issues on Colab.

Use the Colab file browser:
- Navigate to `domainbed/scripts/` and `domainbed/lib/`
- Right-click on `train.py` and `fast_data_loader.py`
- Select `Upload` to overwrite the original files

---

### 4. 启动训练 | Start Training  
确保上述步骤完成后，即可运行最后一个程序单元格，启动模型训练。

Once everything is set, run the final code cell to start model training.

---

📌 如需使用其他算法（如 IRM、CORAL 等），请修改 `algorithm` 参数。  
You can switch between ERM, IRM, CORAL etc. by changing the `--algorithm` parameter.
