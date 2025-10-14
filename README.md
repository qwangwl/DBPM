

# Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition

[English](./README.md) | [简体中文](./README_zh.md)

This repository is the official PyTorch implementation of the paper "Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition".

## File Structure

```
.
├── configs/
│   └── dbpm.yaml           # Hyperparameter configuration file for the model and training
├── datasets/
│   └── seed_feature.py     # SEED dataset loader
│   └── seediv_feature.py   # SEED-IV dataset loader
├── loss_funcs/
│   └── transfer_losses.py  # Loss functions used in the project
├── models/
│   └── DBPM.py             # Definition of the DBPM model
├── trainers/
│   └── DBPMTrainer.py      # Logic for model training, validation, and testing
├── utils/
|   └── _build_loader.py    # Helper functions for dataloader and DBSCAN-Based Cluster capturing
│   └── _get_datasets.py    # Helper functions for data processing
│   └── _get_models.py      # Helper functions for model definition
|   └── utils.py            # Other utility functions
├── cross_subject.py        # Entry point for cross-subject experiments
├── cross_dataset.py        # Entry point for cross-dataset experiments
├── analysis.ipynb          # Visualization and analysis of experimental results
├── requirements.txt        # Environment dependencies
└── README.md
```

## Prerequisites

Before you begin, please ensure you have the following environment and dependencies installed:

- Python (>= 3.12)
- PyTorch (>= 2.8.0)
- Scikit-learn
- NumPy

You can install all required dependencies using pip (this may include some unnecessary packages):

```
pip install -r requirements.txt
```

## How to Use

### 1. Data Preparation

This project supports the SEED and SEED-IV datasets. Please download the datasets first, then modify the `seed4_path` and `seed3_path` parameters in the configuration file `configs/dbpm.yaml` to point to the correct data paths on your local machine.

Example `configs/dbpm.yaml` content:

```
# ...other parameters...
seed4_path: "/path/to/your/SEED-IV/eeg_raw_data"
seed3_path: "/path/to/your/SEED/Preprocessed_EEG"
# ...other parameters...
```

### 2. Running Experiments

#### Cross-Subject Experiments

To run cross-subject experiments, execute the `cross_subject.py` script. This script will train and evaluate the model according to the configuration file. You need to specify the dataset name (`dataset_name`) and session (`session`) via command-line arguments.

```
python cross_subject.py --dataset_name <dataset_name> --session <session>
```

- `<dataset_name>`: `seed3` or `seed4`.
- `<session>`: The session for the experiment, typically `1`, `2`, or `3`.

For example, to run an experiment on the first session of the SEED dataset:

```
python cross_subject.py --dataset_name seed3 --session 1
```

#### Cross-Dataset Experiments

To run cross-dataset experiments, execute the `cross_dataset.py` script directly. Specify the source and target datasets via command-line arguments.

```
python cross_dataset.py --source_dataset <source_dataset> --target_dataset <target_dataset>
```

- `<source_dataset>`: The name of the source domain dataset (e.g., `seed3`, `seed4`).
- `<target_dataset>`: The name of the target domain dataset (e.g., `seed3`, `seed4`).

For example, to run an experiment using SEED-IV as the source domain and SEED as the target domain:

```
python cross_dataset.py --source_dataset seed4 --target_dataset seed3
```

#### One-Click Script for All Experiments

In many cases, you may want to run all experiments at once to obtain the final results. You can refer to the scripts in the `runs/` directory to do this.

**Important Note**: Before running the batch scripts, you need to make a critical modification to the `cross_subject.py` file. Please find and **comment out** the definition of `tmp_saved_path` to ensure that the results of each experiment are saved correctly and independently, preventing them from being accidentally overwritten.

For example, find the following code in `cross_subject.py` and comment it out:

```
tmp_saved_path = f"logs\\{args.dataset_name}_{args.session}_{args.emotion}\\{args.ablation}"
setattr(args, "tmp_saved_path", tmp_saved_path)
```

## How to Cite

If you use the code or methods from this repository in your research, please cite our paper:

```
@article{your_citation_key,
  title={Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition},
  author={Author 1, Author 2, and Author 3},
  journal={Journal Name},
  year={Year},
  volume={Volume},
  pages={Pages}
}
```

## Contact

If you have any questions, feel free to open an issue on GitHub or contact us directly.
