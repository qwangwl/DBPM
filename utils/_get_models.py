# -*- encoding: utf-8 -*-
'''
file       :_get_models.py
Date       :2025/10/13 19:16:35
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''
import torch
from models import DBPM
from trainers import DBPMTrainer

# Non-core code, not added to init.py
from models.DBPM import DBPMWithOutPM
from trainers.DBPMTrainer import DBPMTrainerAblation

def get_model_utils(args):
    """
    Get model, optimizer, scheduler, and trainer.
    """
    if args.ablation == "WithOutPM":
        model = DBPMWithOutPM(
            input_dim=args.feature_dim,
            num_classes=args.num_classes,
            max_iter = args.max_iter,
            num_src_clusters = args.num_src_clusters,
            num_tgt_clusters = args.num_tgt_clusters
        ).cuda()
    else:
        model = DBPM(
            input_dim=args.feature_dim,
            num_classes=args.num_classes,
            max_iter = args.max_iter,
            num_src_clusters = args.num_src_clusters,
            num_tgt_clusters = args.num_tgt_clusters
        ).cuda()

    # Optimizer
    optimizer = torch.optim.RMSprop(
        model.get_parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
        )
    
    # Trainer
    Trainer_params = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "n_epochs": args.n_epochs,
        "log_interval": args.log_interval,
        "early_stop": args.early_stop,
        "transfer_loss_weight": args.transfer_loss_weight,
        "device": args.device,
        "pre_epochs": args.pre_epochs,
        "pl_weight": args.pl_weight,
    }
    if "WithOut" in args.ablation and args.ablation != "WithOutPM":
        trainer = DBPMTrainerAblation(ablation = args.ablation, **Trainer_params)
    else:
        trainer = DBPMTrainer(**Trainer_params)
    return trainer