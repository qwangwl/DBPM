# -*- encoding: utf-8 -*-
'''
file       :run_params_for_seed.py
Date       :2025/05/09 15:24:41
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import configargparse
import os
config_path = "configs\\transfer.yaml"
seed = 20

epochs = 1000
log_interval = 1
early_stop = 0
model_type = "PMEEG"

dataset_name = "seed3"
num_of_class = 3

# Default parameters
pl_weight = 0.011
pl_weight_str = f"{pl_weight:.4f}".replace(".", "")

batch_size = 96
lr = 5e-5
session = 1
#TODO 0. pre epochs
for noisy_level in [0.7]:
    noisy_level_str = f"{noisy_level:.2f}".replace(".", "")
    tmp_saved_path = f"E:\\EEG\\logs\\0509-MixDBPM\\{dataset_name}_{session}\\noisy\\{noisy_level_str}"
    command = (
        f"python main.py "
        f"--seed {seed} "
        f"--dataset_name {dataset_name} "
        f"--config {config_path} "
        f"--n_epochs {epochs} "
        f"--log_interval {log_interval} "
        f"--tmp_saved_path {tmp_saved_path} "
        f"--early_stop {early_stop} "
        f"--transfer_loss_type dann "
        f"--saved_model False "
        f"--pl_weight {pl_weight} "
        f"--lr {lr} "
        f"--noisy_level {noisy_level} "
    )
    os.system(command)
