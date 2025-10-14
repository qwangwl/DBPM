# -*- encoding: utf-8 -*-
'''
file       :DBPMTrainer.py
Date       :2025/10/13 19:22:14
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import numpy as np
import copy
import utils.utils as utils

class DBPMTrainer(object):
    # The Pytorch implementation of Denisity-Based Prototype Matching(DBPM) Trainer
    def __init__(self, 
                 model, 
                 optimizer, 
                 lr_scheduler = None,
                 n_epochs: int = 1000, 
                 log_interval: int = 1, 
                 early_stop: int = 0,
                 transfer_loss_weight: int = 1,
                 device: str = "cuda:0", 
                 pre_epochs: int = 14,
                 pl_weight: float = 0.011,
                 **kwargs):
        """
        Initializes the BaseTrainer.

        Parameters:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            n_epochs (int, optional): Number of epochs to train. Defaults to 1000.
            log_interval (int, optional): Interval for logging training progress. Defaults to 1.
            early_stop (int, optional): Number of epochs without improvement before stopping. Defaults to 0.
            transfer_loss_weight (float, optional): Weight for the transfer loss. Defaults to 1.0.
            lr_scheduler (callable, optional): Learning rate scheduler. Defaults to None.
            device (str, optional): Device to use for training. Defaults to "cuda:0".
        """
        super(DBPMTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.early_stop = early_stop
        self.transfer_loss_weight = transfer_loss_weight
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.pl_weight = pl_weight
        self.pre_epochs = pre_epochs
    
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
            
            loss = self.compute_loss(cls_loss, transfer_loss, pll_loss, epoch)

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

            source_loader = self.get_source_loader(source_loaders, epoch)

            loss_clf, loss_transfer, loss_pll \
                = self.train_one_epoch( source_loader, target_loader,  epoch)

            self.post_epoch_processing(source_loader, target_loader, epoch)

            self.model.eval()
            with torch.no_grad():
                source_acc = self.test(source_loader)
                target_acc = self.test(target_loader)

            stop += 1
            if target_acc > best_acc:
                best_acc = target_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                stop = 0

            log.append([loss_clf, loss_transfer, loss_pll, source_acc, target_acc, best_acc])

            info = (
                f'Epoch: [{epoch + 1:2d}/{self.n_epochs}], '
                f'loss_clf: {loss_clf:.4f}, '
                f'loss_transfer: {loss_transfer:.4f}, '
                f'loss_pll: {loss_pll:.4f}, '
                f'source_acc: {source_acc:.4f}, '
                f'target_acc: {target_acc:.4f}, '
                f'best_acc: {best_acc:.4f}'
            )

            if (self.early_stop > 0 and stop >= self.early_stop):
                print(info)
                break

            if (epoch+1) % self.log_interval == 0 or epoch == 0:
                print(info)
        
        np_log = np.array(log, dtype=float)
        return best_acc, np_log
    
    def get_source_loader(self, source_loaders, epoch):
        """DBPM包含两个阶段，第一个阶段需要Mixed-Source数据，第二个阶段需要顺序使用各个Source数据"""
        if epoch < self.pre_epochs:
            source_loader = source_loaders["pretrain"]
        else:
            current_source = (epoch - self.pre_epochs) % len(source_loaders["train"])
            source_loader = source_loaders["train"][current_source]   
        return source_loader
    
    def compute_loss(self, cls_loss, transfer_loss, pll_loss, epoch):
        """DBPM包含两个阶段，第一个阶段不进行PM，不求解pll_loss"""
        if epoch < self.pre_epochs:
            return cls_loss + self.transfer_loss_weight * transfer_loss
        else:
            return cls_loss + self.transfer_loss_weight * transfer_loss + self.pl_weight * pll_loss

    def test(self, dataloader):
        feature, labels, _ = dataloader.dataset.get_data()
        labels = np.argmax(labels.numpy(), axis=1)
        y_preds = self.model.predict(feature.to(self.device))
        acc = np.sum(y_preds == labels) / len(labels)
        return acc * 100.

    def post_epoch_processing(self, source_loader,  target_loader, epoch):
        if epoch >= self.pre_epochs:
            tgt_x, _, tgt_cluster = target_loader.dataset.get_data()
            src_x, src_y, src_cluster = source_loader.dataset.get_data()
            self.model.epoch_end_hook(src_x.to(self.device), tgt_x.to(self.device), src_y.to(self.device), src_cluster, tgt_cluster)


class DBPMTrainerAblation(DBPMTrainer):
    def __init__(self, 
                 model, 
                 optimizer, 
                 ablation: str = None,
                 **kwargs):
        super(DBPMTrainerAblation, self).__init__(model, optimizer, **kwargs)
        self.ablation = ablation

    def get_source_loader(self, source_loaders, epoch):
        if self.ablation == "WithOutPretrain":
            current_source = epoch % len(source_loaders["train"])
            source_loader = source_loaders["train"][current_source]
            return source_loader
        elif self.ablation == "WithOutSSTS":
            return source_loaders["pretrain"]
        else:   
            return super().get_source_loader(source_loaders, epoch)
        
    def compute_loss(self, cls_loss, transfer_loss, pll_loss, epoch):
        if self.ablation == "WithOutTransferLoss":
            return cls_loss if epoch < self.pre_epochs else cls_loss + self.pl_weight * pll_loss
        elif self.ablation == "WithOutPLLoss":
            return cls_loss + self.transfer_loss_weight * transfer_loss
        elif self.ablation == "WithOutPretrain":
            return cls_loss + self.transfer_loss_weight * transfer_loss + self.pl_weight * pll_loss
        else:
            return super().compute_loss(cls_loss, transfer_loss, pll_loss, epoch)
        
    def post_epoch_processing(self, source_loader,  target_loader, epoch):
        if epoch >= self.pre_epochs or self.ablation == "WithOutPretrain":
            tgt_x, _, tgt_cluster = target_loader.dataset.get_data()
            src_x, src_y, src_cluster = source_loader.dataset.get_data()
            self.model.epoch_end_hook(src_x.to(self.device), tgt_x.to(self.device), src_y.to(self.device), src_cluster, tgt_cluster)
