# -*- encoding: utf-8 -*-
'''
file       :run_seediv.py
Date       :2025/05/12 18:22:26
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

dataset_name = "seed4"
num_of_class = 4

# Default parameters
pl_weight = 0.011
pl_weight_str = f"{pl_weight:.4f}".replace(".", "")

batch_size = 96
lr = 1e-3

for session in [1]:
    tmp_saved_path = f"E:\\EEG\\logs\\0818-MixDBPM\\{dataset_name}_{session}\\main_trial\\"
    command = (
        f"python main.py "
        f"--seed {seed} "
        f"--config {config_path} "
        f"--dataset_name {dataset_name} "
        f"--session {session} "
        f"--n_epochs {epochs} "
        f"--log_interval {log_interval} "
        f"--early_stop {early_stop} "
        f"--tmp_saved_path {tmp_saved_path} "
        f"--transfer_loss_type dann "
        f"--saved_model True "
        f"--batch_size {batch_size} "
        f"--pl_weight {pl_weight} "
        f"--lr {lr} "
        )
    os.system(command)

