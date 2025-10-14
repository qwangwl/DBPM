# Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition

This repository contains a **PyTorch implementation** of the paper:

*Fine-Grained Label Propagation via Density-Based Prototype Matching for Cross-Subject EEG Emotion Recognition*.

It provides a full pipeline for EEG emotion recognition, including **cross-subject and cross-dataset experiments**. 

## Code Organization

- **`dataloaders/`** — EEG data preprocessing and DataLoader construction.  
- **`datasets/`** — Functions to read SEED, SEED-IV, and other EEG datasets.  
- **`dataset_utils.py`** — Utilities for dataset handling, clustering, and label mapping.  
- **`models/`** — Network.  
- **`loss_funcs/`** — Loss functions.  
- **`trainers/`** — Training.  
- **`runs/`** — Example scripts to reproduce experiments on different datasets.  
- **`main.py`** — Entry point for training.  
- **`a_draft/`** — Exploratory code for hyperparameter search and preliminary experiments.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/qwangwl/DBPM
cd DBPM
````

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Default Execution

Running `python main.py` directly will execute an experiment based on the default parameters set in `config.py` (which are typically configured for the Session 1 of the SEED dataset):

```bash
python main.py
```

### Running More Datasets and Specifying Parameters

If you wish to run different datasets or specify experimental parameters, please refer to the specialized run scripts located in the `runs` directory. To run experiments for a specific dataset, please check and use the corresponding `runs/run_{dataset_name}.py` file.

For example, if you want to run experiments on a dataset, you would likely execute a command similar to the following (refer to the respective run_*.py file for the exact command):

```bash
python runs/run_seed.py  # Example for running the SEED dataset
python runs/run_deap.py  # Example for running the DEAP dataset
```

### Warning

**Please note:** When running different experimental scenarios (e.g., different datasets or cross-dataset evaluations), certain revisions to the core code may be required:

1.  **For Different Datasets:** When running a dataset like **DEAP**, you will need to revise the description (or definition) of the **number of targets** within `main.py` to match the dataset's characteristics.
2.  **For Cross-Dataset Experiments:** When performing cross-dataset analysis, the **entry point** of the program within `main.py` will likely need to be revised to accommodate the loading and processing of multiple datasets.

## Citation

If you use this code in your research, please cite the original paper:
```bibtex
@article{Wang2025An,
  title={An Emotion Recognition Framework via Cross-modal Alignment of EEG and Eye Movement Data},
  author={Qi Wang and others},
  journal={arXiv preprint arXiv:2509.04938},
  year={2025}
}
```

-----

*This README was generated with the assistance of Google Gemini.*
