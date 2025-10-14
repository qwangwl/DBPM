# -*- encoding: utf-8 -*-
'''
file       :deap_utils.py
Date       :2025/09/02 23:03:27
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import numpy as np
import torch
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset, DEAPDataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing

from sklearn.cluster import DBSCAN
from collections import Counter

def perform_clustering(args, X, y):
    clustering = DBSCAN(eps=1, min_samples=3).fit(X)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1 
    return num_clusters, X[mask], y[mask], labels[mask]

class CustomDataset(TensorDataset):
    def __init__(self, d1, d2, d3):
        super(CustomDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx], self.d3[idx]
    
    def data(self):
        return self.d1
    def label(self):
        return self.d2
    def cluster(self):
        return self.d3

def get_deap(args):
    num_of_class = 2
    num_of_channels = 32

    params = {
        "feature_name" : args.feature_name,
        "window_sec" : args.window_sec, 
        "step_sec" : args.step_sec,
        "labels" : args.emotion
    }

    DEAP = DEAPDataset(args.deap_path, **params)
    feature_dim = num_of_channels * DEAP.get_feature_dim()
    dataset = DEAP.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    data = data.reshape(-1, feature_dim)
    Label = Label.reshape(-1)
    data = avg_moving(data, Group)
    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_of_class", num_of_class)
    Label = (Label > 5).astype(int)

    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")

    return data, one_hot_mat, Group

def process_target_domain(args, EEG, sGroup, target, one_hot_mat, tGroup=0):
    target_mask = sGroup == target
    # print(EEG[target_mask].shape)
    # print(target, sGroup)
    n, x, y, c = perform_clustering(args, EEG[target_mask], one_hot_mat[target_mask])
    # n, x, y, c = len(np.unique(tGroup[target_mask])), EEG[target_mask], one_hot_mat[target_mask], tGroup[target_mask] - 1
    # print(n, x, y, c)
    setattr(args, "num_of_t_clusters", n)
    target_features = torch.from_numpy(x).type(torch.Tensor)
    target_labels = torch.from_numpy(y)
    target_cluster = torch.from_numpy(c)
    target_dataset = CustomDataset(target_features, target_labels, target_cluster)
    # print(target_features.shape)
    target_loader = DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    return target_loader


def process_source_domain(args, EEG, sGroup, source_lists, one_hot_mat, noisy_level=0, tGroup=None):
    cluster_results = {}
    min_clusters = float('inf')
    
    for source in source_lists:
        mask = sGroup == source
        n, x, y, c = perform_clustering(args, EEG[mask], one_hot_mat[mask])
        # n, x, y, c = len(np.unique(tGroup[mask])), EEG[mask], one_hot_mat[mask], tGroup[mask] - 1
        cluster_results[source] = (n, x, y, c)
        # print(n)
        min_clusters = min(min_clusters, n)

    # print(min_clusters)
    setattr(args, "num_of_s_clusters", min_clusters)
    source_datasets = []

    for source, (n, x, y, c) in cluster_results.items():
        cluster_counts = Counter(c)
        top_clusters = [cluster for cluster, _ in cluster_counts.most_common(min_clusters)]
        mask = np.isin(c, top_clusters)

        source_features = torch.from_numpy(x[mask]).type(torch.Tensor)
        
        source_labels = torch.from_numpy(y[mask]).type(torch.Tensor)

        source_cluster = c[mask]
        unique_clusters = np.unique(source_cluster)
        cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
        source_cluster = torch.tensor([cluster_mapping[label] for label in source_cluster], dtype=torch.long)

        source_datasets.append(CustomDataset(source_features, source_labels, source_cluster))
    
    source_loaders = [
        DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        ) for dataset in source_datasets
    ]
    return source_loaders


def avg_moving(data,  groups, window_size=20):
    normalized_data = []

    def avg_moving_func(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='same')
    for subject_id in np.unique(groups[:, 0]):
        subject_mask = groups[:, 0] == subject_id
        subject_data = data[subject_mask]
        subject_trials = groups[subject_mask][:, 1]
        unique_trials = np.unique(subject_trials)
        for trial in unique_trials:
            trial_mask = subject_trials == trial
            trial_data = subject_data[trial_mask]
            trial_data_normalized = np.apply_along_axis(avg_moving_func, 0, trial_data)
            
            normalized_data.append(trial_data_normalized)
    data = np.vstack(normalized_data)
    return data


def process_source_domain_without_SSTS(args, EEG, sGroup, source_lists, one_hot_mat):
    source_mask = np.isin(sGroup, source_lists)
    n, x, y, c = perform_clustering(args, EEG[source_mask], one_hot_mat[source_mask])
    setattr(args, "num_of_s_clusters", n)
    
    source_features = torch.from_numpy(x).type(torch.Tensor)
    source_labels = torch.from_numpy(y)
    source_cluster = torch.from_numpy(c)
    source_dataset = CustomDataset(source_features, source_labels, source_cluster)

    source_loader = DataLoader(
        dataset=source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    return source_loader

def getDataLoader(args, target=1, source_lists=None):
    EEG, one_hot_mat, Group = get_deap(args)
    sGroup = Group[:, 0]  # subject ID
    # print(EEG.shape, one_hot_mat.shape, np.unique(sGroup))
    target_loader = process_target_domain(args, EEG, sGroup, target, one_hot_mat, tGroup=Group[:, 1])

    if source_lists is None:
        source_lists = list(range(1, 33))
        source_lists.remove(target)

    pretrain_loader = process_source_domain_without_SSTS(args, EEG, sGroup, source_lists, one_hot_mat)

    if args.ablation == "WithOutSSTS":
        train_loader = process_source_domain_without_SSTS(args, EEG, sGroup, source_lists, one_hot_mat)
    else:
        train_loader = process_source_domain(args, EEG, sGroup, source_lists, one_hot_mat)

    source_loaders = {
        "pretrain": pretrain_loader,
        "train": train_loader
    }
    return source_loaders, target_loader