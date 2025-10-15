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
        f"--tmp_saved_path {tmp_saved_path} "
        f"--dataset_name {dataset_name} "
        f"--session {sess} "
    )
    os.system(command)

dataset_name = "seed4"
# seed4为什么使用了lr=1e-3来着，可能是为了追更高的sota.... 实际上lr=5e-5的准确率仅相差1%左右，为了保持与论文的一致性，还是使用1e-3吧
lr = 1e-3
for sess in [1, 2, 3]:
    tmp_saved_path = f"logs\\{dataset_name}_{sess}_none\\DBPM"
    command = (
        f"python cross_subject.py "
        f"--tmp_saved_path {tmp_saved_path} "
        f"--dataset_name {dataset_name} "
        f"--session {sess} "
        f"--lr {lr} " 
    )
    os.system(command)
