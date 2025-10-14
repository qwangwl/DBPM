# Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition

[English](./README.md) | [简体中文](./README_zh.md)

本仓库是论文《Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition》的官方 PyTorch 实现。

## 文件结构

```
.
├── configs/
│   └── dbpm.yaml           # 模型与训练的超参数配置文件
├── datasets/
│   └── seed_feature.py     # SEED数据集加载
│   └── seediv_feature.py   # SEED-IV数据集加载
├── loss_funcs/
│   └── transfer_losses.py  # 项目中使用损失函数
├── models/
│   └── DBPM.py             # DBPM 模型的定义
├── trainers/
│   └── DBPMTrainer.py      # 模型的训练、验证和测试逻辑
├── utils/
|   └── _build_loader.py    # dataloader的辅助函数和DBSCAN-Based Cluster捕获
│   └── _get_datasets.py    # 数据处理相关的辅助函数
│   └── _get_models.py      # 模型定义相关的辅助函数
|   └── utils.py			# 其他的辅助函数
├── cross_subject.py        # 跨被试者实验的程序入口
├── cross_dataset.py        # 跨数据集实验的程序入口
├── analysis.ipynb          # 实验结果的可视化与分析
├── requirements.txt		# 环境
└── README.md
```

## 环境要求

在开始之前，请确保您已安装以下环境和依赖项：

- Python (>= 3.12)
- PyTorch (>= 2.8.0)
- Scikit-learn
- NumPy

您可以使用 pip 安装所有必需的依赖项（可能包含一些不必要的包）：

```base
pip install -r requirements.txt
```

## 使用说明

### 1. 数据准备

本项目支持 SEED 和 SEED-IV 数据集。请先下载数据集，然后在配置文件 `configs/dbpm.yaml` 中修改 `seed4_path` 和 `seed3_path` 两个参数，使其指向您本机存储数据的正确路径。

示例 `configs/dbpm.yaml` 内容:

```
# ...其他参数...
seed4_path: "/path/to/your/SEED-IV/eeg_raw_data"
seed3_path: "/path/to/your/SEED/Preprocessed_EEG"
# ...其他参数...
```

### 2. 运行实验

#### 跨被试者实验

要运行跨被试者实验，请执行 `cross_subject.py` 脚本。该脚本将根据配置文件进行模型的训练和评估。您需要通过命令行参数指定所使用的数据集名称 (`dataset_name`) 和会话 (`session`)。

```base
python cross_subject.py --dataset_name <dataset_name> --session <session>
```

其中:

- `<dataset_name>`: `seed3` 或 `seed4`。
- `<session>`: 实验对应的会话，通常为 `1`, `2` 或 `3`。

例如，在 SEED-IV 数据集的第 1 个会话上进行实验：

```
python cross_subject.py --dataset_name seed4 --session 1
```

#### 跨数据集实验

要运行跨数据集实验，请直接执行 `cross_dataset.py` 脚本。通过命令行给参数指定源域数据集和目标域数据集

```base
python cross_dataset.py --source_dataset <source_dataset> --target_dataset <target_dataset>
```

其中:

- `<source_name>`: 源域数据集的名称 (例如: `seed3`, `seed4`)
- `<target_name>`: 目标域数据集的名称 (例如: `seed3`, `seed4`)

例如，使用 SEED-IV 作为源域，SEED 作为目标域进行实验：

```base
python cross_dataset.py --source_dataset seed4 --target_dataset seed3
```

#### 一键运行所有实验

大多数情况下，我们会需要一键运行所有的实验来获得最终的实验结果。您可以参考 `runs/` 目录下的脚本来执行此操作。

**重要提示**： 在运行批量脚本之前，您需要对 `cross_subject.py` 文件进行一项关键修改。请找到并**注释掉**其中关于 `tmp_saved_path` 的定义，以保证每个实验的结果都能被正确地、独立地存储，防止结果被意外覆盖。

如，在 `cross_subject.py` 中找到类似下面的代码并将其注释：

```
    tmp_saved_path = f"logs\\{args.dataset_name}_{args.session}_{args.emotion}\\{args.ablation}"
    setattr(args, "tmp_saved_path", tmp_saved_path)
```

## 如何引用

如果您在您的研究中使用了本仓库的代码或方法，请引用我们的论文：

```
@article{your_citation_key,
  title={Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition},
  author={作者1, 作者2, and 作者3},
  journal={期刊名},
  year={年份},
  volume={卷号},
  pages={页码}
}
```

## 联系方式

如果您有任何问题，欢迎通过 GitHub Issues 提出，或直接联系我们。
