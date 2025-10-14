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
early_stop = 500
model_type = "PMEEG"

dataset_name = "deap"
# Default parameters
pl_weight = 0.011
pl_weight_str = f"{pl_weight:.4f}".replace(".", "")

batch_size = 96
lr = 1e-3

emotion = "valence"

tmp_saved_path = f"E:\\EEG\\logs\\1004-MixDBPM\\deap\\base\\"
command = (
    f"python main_for_deap.py "
    f"--seed {seed} "
    f"--config {config_path} "
    f"--dataset_name {dataset_name} "
    f"--n_epochs {epochs} "
    f"--emotion {emotion} "
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

# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap\\wo_pm\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model True "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--ablation WithOutPM "
#     f"--lr {lr} "
#     )
# os.system(command)

# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap\\wo_ssts\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model True "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--ablation WithOutSSTS "
#     f"--lr {lr} "
#     )
# os.system(command)

emotion = "arousal"

# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap_aro\\base\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--pre_epochs 1000 "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model True "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--lr {lr} "
#     )
# os.system(command)

# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap_aro\\main\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model True "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--lr {lr} "
#     )
# os.system(command)

# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap_aro\\wo_pm\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model True "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--ablation WithOutPM "
#     f"--lr {lr} "
#     )
# os.system(command)
 
# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap_aro\\wo_ssts\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model True "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--ablation WithOutSSTS "
#     f"--lr {lr} "
#     )
# os.system(command)
emotion = "arousal"

# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap_aro\\wopre\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--pre_epochs 0 "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model False "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--lr {lr} "
# )
# os.system(command)

# ablation_styles = ["WithOutPLLoss", "WithOutTransferLoss"]
# for ablation in ablation_styles:
#     tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap_aro\\{ablation}"
#     command = (
#         f"python main_for_deap.py "
#         f"--seed {seed} "
#         f"--config {config_path} "
#         f"--dataset_name {dataset_name} "
#         f"--n_epochs {epochs} "
#         f"--emotion {emotion} "
#         f"--log_interval {log_interval} "
#         f"--early_stop {early_stop} "
#         f"--tmp_saved_path {tmp_saved_path} "
#         f"--transfer_loss_type dann "
#         f"--saved_model False "
#         f"--batch_size {batch_size} "
#         f"--pl_weight {pl_weight} "
#         f"--ablation {ablation} "
#         f"--lr {lr} "
#     )
# os.system(command)

# emotion = "valence"
# tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap\\wopre\\"
# command = (
#     f"python main_for_deap.py "
#     f"--seed {seed} "
#     f"--config {config_path} "
#     f"--dataset_name {dataset_name} "
#     f"--n_epochs {epochs} "
#     f"--pre_epochs 0 "
#     f"--emotion {emotion} "
#     f"--log_interval {log_interval} "
#     f"--early_stop {early_stop} "
#     f"--tmp_saved_path {tmp_saved_path} "
#     f"--transfer_loss_type dann "
#     f"--saved_model False "
#     f"--batch_size {batch_size} "
#     f"--pl_weight {pl_weight} "
#     f"--lr {lr} "
# )
# os.system(command)

# ablation_styles = ["WithOutPLLoss", "WithOutTransferLoss"]
# for ablation in ablation_styles:
#     tmp_saved_path = f"E:\\EEG\\logs\\0904-MixDBPM\\deap\\{ablation}"
#     command = (
#         f"python main_for_deap.py "
#         f"--seed {seed} "
#         f"--config {config_path} "
#         f"--dataset_name {dataset_name} "
#         f"--n_epochs {epochs} "
#         f"--emotion {emotion} "
#         f"--log_interval {log_interval} "
#         f"--early_stop {early_stop} "
#         f"--tmp_saved_path {tmp_saved_path} "
#         f"--transfer_loss_type dann "
#         f"--saved_model False "
#         f"--batch_size {batch_size} "
#         f"--pl_weight {pl_weight} "
#         f"--ablation {ablation} "
#         f"--lr {lr} "
#     )
# os.system(command)