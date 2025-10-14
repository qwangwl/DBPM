# -*- encoding: utf-8 -*-
'''
file       :main.py
Date       :2024/10/20 14:30:28
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os
import numpy as np
import torch
from torch import nn
import random
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

from config import get_parser
from models import PMEEG, PMEEGWithOutPM
from trainers import  SSTSPMTrainer, PMTrainer

from dataset_utils import getDataLoader, getCrossDataLoader
import time

os.environ["LOKY_MAX_CPU_COUNT"] = "20"  

def setup_seed(seed):  ## setup the random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def weight_init(m):  ## model parameter intialization
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        m.bias.data.zero_()

def get_model_utils(args):
    # 模型
    base_params = {
        "input_dim" : 310,
        "num_of_class" : args.num_of_class,
    }
    pm_params = {
        "transfer_loss_type" : args.transfer_loss_type,
        "max_iter" : args.max_iter,
        "num_of_s_clusters": args.num_of_s_clusters,
        "num_of_t_clusters": args.num_of_t_clusters,
    }

    combined_params = {**base_params, **pm_params}
    if args.ablation == "WithOutPM":
        model = PMEEGWithOutPM(**combined_params).cuda()
    else:
        model = PMEEG(**combined_params).cuda()

    params = model.get_parameters()
    optimizer = torch.optim.RMSprop(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    else:
        scheduler = None

    trainer_params = {
        "lr_scheduler" : scheduler,
        "batch_size" : args.batch_size,
        "n_epochs" : args.n_epochs,
        "transfer_loss_weight" : args.transfer_loss_weight,
        "early_stop" : args.early_stop,
        "tmp_saved_path" : args.tmp_saved_path,
        "log_interval" : args.log_interval,
        "pre_epochs" : args.pre_epochs,
        "pl_weight" : args.pl_weight,
    }

    if args.ablation == "WithOutSSTS":
        trainer = PMTrainer(
            model, 
            optimizer, 
            **trainer_params
        )
    else:
        ssts_params = {
            "ablation": args.ablation
        }
        combined_params = {**trainer_params, **ssts_params}
        trainer = SSTSPMTrainer(
            model, 
            optimizer, 
            **combined_params
        )
    return trainer

def train(target, args):

    setup_seed(args.seed)
    
    create_dir_if_not_exists(args.tmp_saved_path)
    source_lists = list(range(1, 46))
    # source_lists.remove(target)

    # source_loader, target_loader = getDataLoader(args, target, source_lists, noisy_level=args.noisy_level)
    source_loader, target_loader = getCrossDataLoader(args, target, source_lists, noisy_level=args.noisy_level)
    
    setattr(args, "max_iter", 1000) 

    trainer = get_model_utils(args)
    
    best_acc, np_log = trainer.train(source_loader, target_loader)    
    
    if args.saved_model:
        cur_target_saved_path = os.path.join(args.tmp_saved_path, str(target))
        create_dir_if_not_exists(cur_target_saved_path)
        torch.save(trainer.get_model_state(), os.path.join(cur_target_saved_path, f"many_last.pth"))
        torch.save(trainer.get_best_model_state(), os.path.join(cur_target_saved_path, f"many_best.pth"))
    np.savetxt(os.path.join(args.tmp_saved_path, f"t{target}.csv"), np_log, delimiter=",",  fmt='%.4f')
    return best_acc

def main(args):

    setup_seed(args.seed)

    if args.dataset_name == "seed3":
        setattr(args, "num_of_class", 3)
        setattr(args, "path", args.seed3_path)
    elif args.dataset_name == "seed4":
        setattr(args, "num_of_class", 4)
        setattr(args, "path", args.seed4_path)

    best_acc_mat = []
    for target in range(1, 46):
        start_time = time.time()
        best_acc = train(target, args)
        elapsed_time = time.time() - start_time
        best_acc_mat.append(best_acc)
        print(f"target: {target}, best_acc: {best_acc}, time: {elapsed_time:.2f}s")
    
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

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    main(args)

