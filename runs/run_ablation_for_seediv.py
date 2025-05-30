# -*- encoding: utf-8 -*-
'''
file       :run_params_for_seed.py
Date       :2025/05/09 15:24:41
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import configargparse
import os
# os.environ["LOKY_MAX_CPU_COUNT"] = "20"  # 替换为你的实际物理核心数
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
session = 3

# ablation_styles = ["WithOutPM", "WithOutSSTS", "WithOutPLLoss", "WithOutTransferLoss"]
# for ablation in ablation_styles:
#     tmp_saved_path = f"E:\\EEG\\logs\\0509-MixDBPM\\{dataset_name}_{session}\\ablation\\{ablation}"
#     command = (
#         f"python main.py "
#         f"--seed {seed} "
#         f"--dataset_name {dataset_name} "
#         f"--session {session} "
#         f"--config {config_path} "
#         f"--n_epochs {epochs} "
#         f"--log_interval {log_interval} "
#         f"--tmp_saved_path {tmp_saved_path} "
#         f"--early_stop {early_stop} "
#         f"--transfer_loss_type dann "
#         f"--saved_model False "
#         f"--pl_weight {pl_weight} "
#         f"--batch_size {batch_size} "
#         f"--ablation {ablation} "
#         f"--lr {lr} "
#     )
#     os.system(command)

# # WOPretrain
for session in [1, 2, 3]:
    ablation = "WithOutMixSource"
    tmp_saved_path = f"E:\\EEG\\logs\\0509-MixDBPM\\{dataset_name}_{session}\\ablation\\{ablation}"
    command = (
        f"python main.py "
        f"--seed {seed} "
        f"--pre_epochs 14 "
        f"--dataset_name {dataset_name} "
        f"--session {session} "
        f"--config {config_path} "
        f"--n_epochs {epochs} "
        f"--log_interval {log_interval} "
        f"--tmp_saved_path {tmp_saved_path} "
        f"--early_stop {early_stop} "
        f"--transfer_loss_type dann "
        f"--saved_model False "
        f"--pl_weight {pl_weight} "
        f"--batch_size {batch_size} "
        f"--lr {lr} "
        f"--ablation {ablation} "
    )
    os.system(command)