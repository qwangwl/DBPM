# -*- encoding: utf-8 -*-
'''
file       :run_params_for_seed.py
Date       :2025/05/09 15:24:41
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import configargparse
import os
# os.environ["LOKY_MAX_CPU_COUNT"] = "20"  
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
# ablation_styles = ["PMEEG", "WithOutPM", "WithOutSSTS", "WithOutPLLoss", "WithOutTransferLoss", "WithOutMixSource"]
# for ablation in ablation_styles:
#     for session in [1, 2, 3]:
#         tmp_saved_path = f"E:\\EEG\\logs\\0818-MixDBPM\\{dataset_name}_{session}\\ablation\\{ablation}"
#         command = (
#             f"python main.py "
#             f"--seed {seed} "
#             f"--dataset_name {dataset_name} "
#             f"--session {session} "
#             f"--config {config_path} "
#             f"--n_epochs {epochs} "
#             f"--log_interval {log_interval} "
#             f"--tmp_saved_path {tmp_saved_path} "
#             f"--early_stop {early_stop} "
#             f"--transfer_loss_type dann "
#             f"--saved_model False "
#             f"--pl_weight {pl_weight} "
#             f"--batch_size {batch_size} "
#             f"--ablation {ablation} "
#             f"--lr {lr} "
#         )
#         os.system(command)

# ablation_styles = ["WithOutPretrain"]
# for ablation in ablation_styles:
#     for session in [1, 2, 3]:
#         tmp_saved_path = f"E:\\EEG\\logs\\0818-MixDBPM\\{dataset_name}_{session}\\ablation\\{ablation}"
#         command = (
#             f"python main.py "
#             f"--seed {seed} "
#             f"--dataset_name {dataset_name} "
#             f"--session {session} "
#             f"--config {config_path} "
#             f"--n_epochs {epochs} "
#             f"--log_interval {log_interval} "
#             f"--tmp_saved_path {tmp_saved_path} "
#             f"--early_stop {early_stop} "
#             f"--transfer_loss_type dann "
#             f"--saved_model False "
#             f"--pl_weight {pl_weight} "
#             f"--batch_size {batch_size} "
#             f"--ablation PMEEG "
#             f"--pre_epochs 0 "
#             f"--lr {lr} "
#         )
#         os.system(command)
