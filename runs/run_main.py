# -*- encoding: utf-8 -*-
'''
file       :run_seed.py
Date       :2025/09/17 11:26:22
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os

dataset_name = "seed3"  
for sess in [1, 2, 3]:
    tmp_saved_path = f"logs\\{dataset_name}_{sess}_none\\DBPM"
    command = (
        f"python cross_subject.py "
        f"--config configs/dbpm.yaml "
        f"--tmp_saved_path {tmp_saved_path} "
        f"--dataset_name {dataset_name} "
        f"--session {sess} "
    )
    os.system(command)

dataset_name = "seed4"  
for sess in [1, 2, 3]:
    tmp_saved_path = f"logs\\{dataset_name}_{sess}_none\\DBPM"
    command = (
        f"python cross_subject.py "
        f"--config configs/dbpm.yaml "
        f"--tmp_saved_path {tmp_saved_path} "
        f"--dataset_name {dataset_name} "
        f"--lr 1e-3 "
        f"--session {sess} "
    )
    os.system(command)
