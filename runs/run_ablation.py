# -*- encoding: utf-8 -*-
'''
file       :run_ablation.py
Date       :2025/10/13 21:39:26
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import os

dataset_name = "seed3"  
ablation_styles = ["WithOutPM", "WithOutPLLoss", "WithOutTransferLoss", "WithOutPretrain", "WithOutSSTS"]
for ablation in ablation_styles:
    for sess in [1, 2, 3]:
        tmp_saved_path = f"logs\\{dataset_name}_{sess}_none\\{ablation}"
        command = (
            f"python cross_subject.py "
            f"--config configs/dbpm.yaml "
            f"--tmp_saved_path {tmp_saved_path} "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--ablation {ablation} "
        )
        os.system(command)


dataset_name = "seed4"  
ablation_styles = ["WithOutPM", "WithOutPLLoss", "WithOutTransferLoss", "WithOutPretrain", "WithOutSSTS"]
for ablation in ablation_styles:
    for sess in [1, 2, 3]:
        tmp_saved_path = f"logs\\{dataset_name}_{sess}_none\\{ablation}"
        command = (
            f"python cross_subject.py "
            f"--config configs/dbpm.yaml "
            f"--tmp_saved_path {tmp_saved_path} "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--lr 1e-3 "
            f"--ablation {ablation} "
        )
        os.system(command)