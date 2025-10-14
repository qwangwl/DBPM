# -*- encoding: utf-8 -*-
'''
file       :cross_dataset.py
Date       :2025/10/13 20:29:10
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "20"
import time
import torch
import random
import numpy as np

from utils.utils import setup_seed, create_dir_if_not_exists
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset
from sklearn.preprocessing import MinMaxScaler
from config import get_parser
from utils._build_loaders import process_source_domain_with_mix, process_source_domains, process_target_domain
from utils._get_models import get_model_utils

def get_seed_data(args, dataset_name):
    if dataset_name.lower() == "seed3":
        num_classes = 3
        dataset = SEEDFeatureDataset(args.seed3_path, sessions=[1, 2, 3]).get_dataset()
        data, int_labels, Group = dataset["data"], dataset["labels"], dataset["groups"]
        data = data.reshape(-1, 310)
        int_labels += 1
        sGroup = (Group[:, 2] - 1) * 15 + Group[:, 0]

    elif dataset_name.lower() == "seed4":
        num_classes = 4
        dataset = SEEDIVFeatureDataset(args.seed4_path, sessions=[1, 2, 3]).get_dataset()
        data, int_labels, Group = dataset["data"], dataset["labels"], dataset["groups"]
        data = data.reshape(-1, 310)
        lookup = np.array([1, 0, 0, 2])  # 0→1, 1→0, 2→0, 3→2
        int_labels = lookup[int_labels]
        sGroup = (Group[:, 2] - 1) * 15 + Group[:, 0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    scaler = MinMaxScaler(feature_range=(-1, 1))
    for sid in np.unique(sGroup):
        data[sGroup == sid] = scaler.fit_transform(data[sGroup == sid])

    one_hot_labels = np.eye(num_classes)[int_labels].astype("float32")
    return data, one_hot_labels, Group

def get_cross_data_pm_loader(args, source_dataset, target_dataset, target_subject=1, source_lists=None):

    x_source, y_source, Group_s = get_seed_data(args, source_dataset)
    s_source = (Group_s[:, 2] - 1) * 15 + Group_s[:, 0] 

    x_target, y_target, Group_t = get_seed_data(args, target_dataset)
    s_target = (Group_t[:, 2] - 1) * 15 + Group_t[:, 0]

    if source_lists is None:
        source_lists = list(np.unique(s_source))

    source_pretrain_loader = process_source_domain_with_mix(args, x_source, y_source, s_source, source_lists)
    source_train_loaders = process_source_domains(args, x_source, y_source, s_source)

    source_loaders = {
        "pretrain": source_pretrain_loader,
        "train": source_train_loaders
    }

    target_mask = s_target == target_subject
    target_loader = process_target_domain(args, x_target[target_mask], y_target[target_mask])

    return source_loaders, target_loader


def train_for_cross_dataset(target, args):
    """
    Train the model for a specific target.
    """
    # Set random seed for each subject
    setup_seed(args.seed)

    source_loader, target_loader = get_cross_data_pm_loader(args, 
                                                            source_dataset = args.source_dataset, 
                                                            target_dataset = args.target_dataset, 
                                                            target_subject = target)
    trainer = get_model_utils(args)
    best_acc, np_log = trainer.train(source_loader, target_loader)    

    # Save model
    if args.saved_model:
        torch.save(trainer.get_best_model_state(), os.path.join(args.tmp_saved_path, f"t{target}_best.pth"))
    np.savetxt(os.path.join(args.tmp_saved_path, f"t{target}.csv"), np_log, delimiter=",", fmt='%.4f')
    return best_acc


def main_for_cross_dataset(args):
    """
    Main function to run the training process.
    """
    setup_seed(args.seed)
    create_dir_if_not_exists(args.tmp_saved_path)
    
    best_acc_mat = []
    for target in range(1, 46):
        best_acc = train_for_cross_dataset(target, args)
        best_acc_mat.append(best_acc)
        print(f"target: {target}, best_acc: {best_acc}")
    
    mean = np.mean(best_acc_mat)
    std = np.std(best_acc_mat)

    with open(os.path.join(args.tmp_saved_path, f"mean_acc.txt"), 'w', encoding="utf-8") as f:
        for target, best_acc in enumerate(best_acc_mat):
            output_line = f"target: {target+1}, best_acc: {best_acc:.6f}"
            print(output_line)  
            f.write(output_line + '\n')  

        all_best_acc_line = f"all_best_acc: {mean:.4f} / {std:.4f}"
        print(all_best_acc_line)  
        f.write(all_best_acc_line + '\n') 

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    setattr(args, "num_classes", 3)
    setattr(args, "num_of_subjects", 45) # three sessions
    setattr(args, "feature_dim", 310)

    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    tmp_saved_path = f"logs\\cross_dataset\\{args.source_dataset}_to_{args.target_dataset}\\"
    setattr(args, "tmp_saved_path", tmp_saved_path)
    
    main_for_cross_dataset(args)
