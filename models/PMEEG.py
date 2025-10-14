# -*- encoding: utf-8 -*-
'''
file       :PMEEG.py
Date       :2024/11/20 15:36:34
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loss_funcs import TransferLoss


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=310, hidden_1=64, hidden_2=64, dropout_prob=0.5):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params
    
    
class LabelClassifier(nn.Module):
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dim: int = 32,
                 num_of_class: int = 3,
                 ):
        super(LabelClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, num_of_class)

    def forward(self, feature):
        f1 = self.fc1(feature)
        f2 = self.fc2(f1)
        return self.fc3(f2)
    
    def predict(self, feature):
        with torch.no_grad():
            logits = F.softmax(self.forward(feature), dim=1)
            y_preds = np.argmax(logits.cpu().numpy(), axis=1)
        return y_preds
    

    def get_parameters(self):
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
            {"params": self.fc3.parameters(), "lr_mult": 1}
        ]
        return params


class PMEEG(nn.Module):
    def __init__(self, 
                 input_dim: int= 310, 
                 num_of_class: int=3, 
                 max_iter: int=1000, 
                 transfer_loss_type: str="dann", 
                 num_of_s_clusters: int=15,
                 num_of_t_clusters: int=15,
                 topk: int = 1,
                 **kwargs):
        super(PMEEG, self).__init__()

        self.feature_extractor = FeatureExtractor(input_dim=input_dim)
        self.classifier = LabelClassifier(input_dim=64, num_of_class=num_of_class)
        
        # 定义基本的操作
        self.max_iter = max_iter
        self.num_of_class = num_of_class
        self.num_of_s_clusters = num_of_s_clusters
        self.num_of_t_clusters = num_of_t_clusters
        self.transfer_loss_type = transfer_loss_type
        
        self.criterion = nn.CrossEntropyLoss()

        self.source_cluster_label = torch.zeros(num_of_s_clusters).type(torch.long)
        self.target_cluster_label = torch.zeros(num_of_t_clusters).type(torch.long)
        self.source_cluster_P = torch.randn(num_of_s_clusters, 64) # 目标域的原型表征
        self.pll_criterion = nn.CrossEntropyLoss()

        # Prototype Matching TopK
        self.topk = topk

        transfer_loss_args = {
            "loss_type" : self.transfer_loss_type,
            "max_iter" : self.max_iter,
            "num_class" : self.num_of_class,
            **kwargs
        }

        self.adv_criterion = TransferLoss(**transfer_loss_args)

    def forward(self, source, target, source_label, target_cluster, source_cluster=None):
        source_feature = self.feature_extractor(source)
        target_feature = self.feature_extractor(target)
        source_clf = self.classifier(source_feature)
        target_clf = self.classifier(target_feature)

        cls_loss = self.criterion(source_clf, source_label)
        # target_cluster_preds = self.predict_by_sample_wise(source_feature, source_label, target_feature).to(target_clf.device)
        target_cluster_preds = torch.tensor([self.target_cluster_label[int(cluster.item())] for cluster in target_cluster], dtype=torch.long).to(target_clf.device)
        
        pll_loss = self.pll_criterion(target_clf, target_cluster_preds)
    
        trans_loss = self.adv_criterion(source_feature, target_feature)
        return cls_loss, trans_loss, pll_loss
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.feature_extractor(x)
            preds = self.classifier.predict(feature)
        return preds
    
    def predict_by_sample_wise(self, source_feature, source_label, target_feature):
        """
        Generate pseudo-labels for target features using sample-wise similarity to source features.
        """
        self.eval()
        with torch.no_grad():
            source_feature_norm = F.normalize(source_feature, p=2, dim=1)
            target_feature_norm = F.normalize(target_feature, p=2, dim=1)

            sim_matrix = torch.matmul(target_feature_norm, source_feature_norm.T)  # [target_num, source_num]

            max_indices = torch.argmax(sim_matrix, dim=1).cpu()
            target_pred = source_label[max_indices]
        return target_pred

    def predict_prob(self, x):
        self.eval()
        with torch.no_grad():
            feature = self.feature_extractor(x)
            output = self.classifier(feature)
            logits = F.softmax(output, dim=1)
        return logits

    def get_parameters(self):
        params = [
            *self.feature_extractor.get_parameters(),
            *self.classifier.get_parameters(),
        ]
        params.append(
            {"params": self.adv_criterion.loss_func.domain_classifier.parameters(), "lr_mult":1}
        )
        return params
    

    def update_target_cluster_label(self, tgt_x, target_cluster):
        """
        Updates the cluster label for the target data.
        Args:
            tgt_x (torch.Tensor): The input tensor for the target data.
            target_cluster (torch.Tensor): The cluster indices for the target data.
        Returns:
            None
        """

        self.eval()
        with torch.no_grad():
            target_feature = self.feature_extractor(tgt_x)

            # 计算目标域的原型表征
            target_cluster_one_hot = F.one_hot(
                target_cluster.to(torch.long), num_classes=self.num_of_t_clusters
            ).float()
            target_cluster_P = torch.matmul(
                torch.inverse(torch.diag(target_cluster_one_hot.sum(axis=0))) + torch.eye(self.num_of_t_clusters),
                torch.matmul(target_cluster_one_hot.T, target_feature.cpu())
            )

            target_cluster_P = F.normalize(target_cluster_P, p=2, dim=1)
            source_cluster_P = F.normalize(self.source_cluster_P, p=2, dim=1)
            consine_sim_matrix = torch.matmul(target_cluster_P, source_cluster_P.T)

            # Topk
            top_indices = torch.topk(consine_sim_matrix, k=self.topk).indices
            topk_labels = self.source_cluster_label[top_indices]
            for i in range(topk_labels.size(0)):
                labels, counts = torch.unique(topk_labels[i], return_counts=True)
                majority_label = labels[torch.argmax(counts)]  
                self.target_cluster_label[i] = majority_label

    def update_source_cluster_P(self, source_x, source_cluster):
        self.eval()
        with torch.no_grad():
            source_feature = self.feature_extractor(source_x)
            source_cluster_one_hot = F.one_hot(
                source_cluster.to(torch.long), num_classes=self.num_of_s_clusters
            ).float()
            self.source_cluster_P = torch.matmul(
                torch.inverse(torch.diag(source_cluster_one_hot.sum(axis=0))) + torch.eye(self.num_of_s_clusters),
                torch.matmul(source_cluster_one_hot.T, source_feature.cpu())
            )
    
    def reset_source_cluster_label(self, source_label, source_cluster):
        # for Single Source Phase Training Every Epoch need to rest source cluster label
        source_label = torch.argmax(source_label, dim=-1).cpu().detach().numpy()
        unique_clusters = torch.unique(source_cluster)
        # print(unique_clusters)
        for cluster in unique_clusters:
            samples_in_cluster_index = np.where(source_cluster == cluster)[0]
            # print(source_label)
            preds_for_samples = source_label[samples_in_cluster_index]
            if len(preds_for_samples) == 0:
                self.source_cluster_label[int(cluster.item())] = 0
            else:
                label_for_current_cluster = np.argmax(np.bincount(preds_for_samples))
                self.source_cluster_label[int(cluster.item())] = label_for_current_cluster 

    def epoch_end_hook(self, source_x, target_x, source_label, source_cluster, target_cluster):
        self.reset_source_cluster_label(source_label, source_cluster)
        self.update_source_cluster_P(source_x, source_cluster)
        self.update_target_cluster_label(target_x, target_cluster)


class PMEEGWithOutPM(PMEEG):
    # W/O PM
    def __init__(self, **kwargs):
        super(PMEEGWithOutPM, self).__init__(**kwargs)
    
    def update_target_cluster_label(self, tgt_x, target_cluster):
        self.eval()
        with torch.no_grad():

            target_feature = self.feature_extractor(tgt_x)
            target_logits = self.classifier(target_feature)

            target_preds = torch.argmax(target_logits, dim=-1).cpu().detach().numpy()
            unique_clusters = torch.unique(target_cluster)
            for cluster in unique_clusters:
                samples_in_cluster_index = np.where(target_cluster == cluster)[0]
                preds_for_samples = target_preds[samples_in_cluster_index]
                if len(preds_for_samples) == 0:
                    self.target_cluster_label[int(cluster.item())] = 0
                else:
                    label_for_current_cluster = np.argmax(np.bincount(preds_for_samples))
                    self.target_cluster_label[int(cluster.item())] = label_for_current_cluster

if __name__ == "__main__":
    print("test")