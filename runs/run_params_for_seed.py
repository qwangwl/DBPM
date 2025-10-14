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
for pre in [6, 22, 26, 56]:
    tmp_saved_path = f"E:\\EEG\\logs\\0509-MixDBPM\\{dataset_name}_{session}\\params\\pre\\{pre}"
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
        f"--pre_epochs {pre}"
    )
    os.system(command)


# session = 1
# # TODO 1. test beta
# # [ 0.012, 0.007, 0.008, 0.009, 0.010, 0.011]
# # [ 0.013, 0.014, 0.015, 0.016, 0.017, 0.018]
# for pl_weight in [ 0.006]:
#     pl_weight_str = f"{pl_weight:.4f}".replace(".", "")
#     tmp_saved_path = f"E:\\EEG\\logs\\0509-MixDBPM\\{dataset_name}_{session}\\params\\beta\\{pl_weight_str}"
#     command = (
#         f"python main.py "
#         f"--seed {seed} "
#         f"--dataset_name {dataset_name} "
#         f"--config {config_path} "
#         f"--n_epochs {epochs} "
#         f"--log_interval {log_interval} "
#         f"--tmp_saved_path {tmp_saved_path} "
#         f"--early_stop {early_stop} "
#         f"--transfer_loss_type dann "
#         f"--saved_model False "
#         f"--pl_weight {pl_weight} "
#         f"--lr {lr}"
#     )
#     os.system(command)

#TODO 2. test batch_size
# pl_weight = 0.011
# pl_weight_str = f"{pl_weight:.4f}".replace(".", "")
# for batch_size in [ 32, 48, 64, 72, 96, 128]:
#     tmp_saved_path = f"E:\\EEG\\logs\\0509-MixDBPM\\{dataset_name}_{session}\\params\\bs\\{batch_size}"
#     command = (
#         f"python main.py "
#         f"--seed {seed} "
#         f"--dataset_name {dataset_name} "
#         f"--config {config_path} "
#         f"--n_epochs {epochs} "
#         f"--log_interval {log_interval} "
#         f"--tmp_saved_path {tmp_saved_path} "
#         f"--early_stop {early_stop} "
#         f"--transfer_loss_type dann "
#         f"--saved_model False "
#         f"--pl_weight {pl_weight} "
#         f"--batch_size {batch_size} "
#     )
#     os.system(command)

# TODO 3. test learning_rate
# for lr in [ 5e-5 ]:
#     tmp_saved_path = f"E:\\EEG\\logs\\0509-MixDBPM\\{dataset_name}\\params\\dropout\\true"
#     command = (
#         f"python main.py "
#         f"--seed {seed} "
#         f"--dataset_name {dataset_name} "
#         f"--config {config_path} "
#         f"--n_epochs {epochs} "
#         f"--log_interval {log_interval} "
#         f"--tmp_saved_path {tmp_saved_path} "
#         f"--early_stop {early_stop} "
#         f"--transfer_loss_type dann "
#         f"--saved_model False "
#         f"--pl_weight {pl_weight} "
#         f"--lr {lr} "
#     )
#     os.system(command)