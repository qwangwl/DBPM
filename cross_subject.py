# -*- encoding: utf-8 -*-
'''
file       :cross_subject.py
Date       :2025/10/13 19:16:20
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "20"
import time
import torch
import random
import numpy as np

from config import get_parser

from utils.utils import setup_seed, create_dir_if_not_exists
from utils._get_datasets import get_dataset
from utils._build_loaders import process_source_domain_with_mix, process_source_domains, process_target_domain
from utils._get_models import get_model_utils

def get_pm_laoder(args, dataset, target, source_lists = None):
    data, one_hot_mat, subject_ids = dataset["data"], dataset["one_hot_mat"], dataset["groups"][:, 0]

    if source_lists is None:
        source_lists = list(range(1, args.num_of_subjects + 1))
        source_lists.remove(target)
    source_lists = np.array(source_lists)

    loader_target = process_target_domain(args, data[subject_ids==target], one_hot_mat[subject_ids==target])

    pre_source_loader = process_source_domain_with_mix(
        args,
        data, 
        one_hot_mat, 
        subject_ids, 
        source_lists, )
    with_ssts = False if args.ablation == "WithOutSSTS" else True
    loader_source = process_source_domains(
        args, 
        data[np.isin(subject_ids, source_lists)], 
        one_hot_mat[np.isin(subject_ids, source_lists)], 
        subject_ids[np.isin(subject_ids, source_lists)],
        with_ssts=with_ssts)

    loaders_source = {
        "pretrain": pre_source_loader,
        "train": loader_source
    }
    return loaders_source, loader_target

def train(dataset,  target, args):
    """
    Train the model for a specific target.
    """
    # Set random seed for each subject
    setup_seed(args.seed)

    source_loader, target_loader = get_pm_laoder(args, dataset, target)
    trainer = get_model_utils(args)
    best_acc, np_log = trainer.train(source_loader, target_loader)    

    # Save model
    if args.saved_model:
        torch.save(trainer.get_best_model_state(), os.path.join(args.tmp_saved_path, f"t{target}_best.pth"))
    np.savetxt(os.path.join(args.tmp_saved_path, f"t{target}.csv"), np_log, delimiter=",", fmt='%.4f')
    return best_acc


def main(args):
    """
    Main function to run the training process.
    """
    setup_seed(args.seed)
    create_dir_if_not_exists(args.tmp_saved_path)
    
    # Load the dataset first in LOSO to avoid repeated loading for each train
    dataset = get_dataset(args)
    
    best_acc_mat = []

    for target in range(1, args.num_of_subjects + 1):
        best_acc = train(dataset, target, args)
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
    args = get_parser().parse_args()
    tmp_saved_path = f"logs\\{args.dataset_name}_{args.session}_{args.emotion}\\{args.ablation}"
    setattr(args, "tmp_saved_path", tmp_saved_path)
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.dataset_name == "seed3":
        setattr(args, "num_classes", 3)
        setattr(args, "num_of_subjects", 15)
        setattr(args, "feature_dim", 310)
    elif args.dataset_name == "seed4":
        setattr(args, "num_classes", 4)
        setattr(args, "num_of_subjects", 15)
        setattr(args, "feature_dim", 310)
    main(args)
        