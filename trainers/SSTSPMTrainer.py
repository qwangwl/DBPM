# -*- encoding: utf-8 -*-
'''
file       :BaseTrainer.py
Date       :2024/08/06 18:04:46
Author     :qwangwl
'''

import torch
import numpy as np
import copy
import utils as utils
import random

from trainers.PMTrainer import PMTrainer
import time


class SSTSPMTrainer(PMTrainer):
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler = None,
                 n_epochs: int = 100, 
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 transfer_loss_weight: int = 1,
                 pl_weight: float = 0.01,
                 device: str = "cuda:0", 
                 **kwargs):
        super(SSTSPMTrainer, self).__init__(
            model=model,
            optimizer=optimizer,
            n_epochs=n_epochs,
            log_interval=log_interval,
            early_stop=early_stop,
            transfer_loss_weight=transfer_loss_weight,
            pl_weight=pl_weight,
            lr_scheduler=lr_scheduler,
            device=device,
            **kwargs
        )
        self.ablation = kwargs["ablation"]
        self.best_model_state = None

    def train_one_epoch(self, source_loader,  target_loader, epoch):
    
        self.model.train()
        
        len_source_loader = len(source_loader)
        len_target_loader = len(target_loader)

        n_batch = min(len_source_loader, len_target_loader) - 1

        loss_clf = utils.AverageMeter()
        loss_transfer = utils.AverageMeter()
        loss_pll = utils.AverageMeter()
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for _ in range(n_batch):
            try:
                src_data, src_label, src_cluster = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                src_data, src_label, src_cluster = next(source_iter)
            try:
                tgt_data, _, tgt_cluster = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_data, _, tgt_cluster = next(target_iter)

            src_data, src_label = src_data.to(
                self.device), src_label.to(self.device)
            tgt_data = tgt_data.to(self.device)

            cls_loss, transfer_loss, pll_loss = self.model(src_data, tgt_data, src_label, tgt_cluster, src_cluster)
            
            loss = 0
            if self.ablation == "WithOutTransferLoss":
                if epoch < self.pre_epochs:
                   loss = cls_loss 
                else:
                    loss = cls_loss + self.pl_weight * pll_loss
            elif self.ablation == "WithOutPLLoss":
                loss = cls_loss + self.transfer_loss_weight * transfer_loss
            elif self.ablation == "WithOutPretrain":
                loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.pl_weight * pll_loss
            else:
                if epoch < self.pre_epochs:
                    loss = cls_loss + self.transfer_loss_weight * transfer_loss
                else:
                    loss = cls_loss + self.transfer_loss_weight * transfer_loss + self.pl_weight * pll_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            loss_clf.update(cls_loss.detach().item())
            loss_transfer.update(transfer_loss.detach().item())
            loss_pll.update(pll_loss.detach().item())
            
        return loss_clf.avg, loss_transfer.avg, loss_pll.avg

    def train(self, source_loaders, target_loader):

        stop = 0
        best_acc = 0.0
        log = []
        for epoch in range(self.n_epochs):
            self.model.train()
            
            if epoch < self.pre_epochs:
                if self.ablation == "WithOutMixSource":
                    current_source = (epoch) % len(source_loaders["pretrain"])
                    source_loader = source_loaders["pretrain"][current_source]
                else:
                    source_loader = source_loaders["pretrain"]
            else:
                current_source = (epoch - self.pre_epochs) % len(source_loaders["train"])
                source_loader = source_loaders["train"][current_source]   
            
            loss_clf, loss_transfer, loss_pll \
                = self.train_one_epoch(source_loader, target_loader, epoch)
            self.post_epoch_processing(source_loader, target_loader, epoch)

            self.model.eval()
            with torch.no_grad():
                source_acc = self.test(source_loader)
                target_acc = self.test(target_loader)

            log.append([loss_clf, loss_transfer, loss_pll, source_acc, target_acc, best_acc])

            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0

            info = (
                f'Epoch: [{epoch + 1:2d}/{self.n_epochs}], '
                f'loss_clf: {loss_clf:.4f}, '
                f'loss_transfer: {loss_transfer:.4f}, '
                f'loss_pll: {loss_pll:.4f}, '
                f'source_acc: {source_acc:.4f}, '
                f'target_acc: {target_acc:.4f}, '
                f'best_acc: {best_acc:.4f} '
            )

            if ( self.early_stop > 0 and stop >= self.early_stop ):
                print(info)
                break
            
            if (epoch + 1) % self.log_interval == 0 or epoch == 0:
                print(info)


        np_log = np.array(log, dtype=float)
        return best_acc, np_log
    

