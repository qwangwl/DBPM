# -*- encoding: utf-8 -*-
'''
file       :load_data.py
Date       :2024/11/20 09:17:43
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

def perform_clustering(args, X, y):
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(X)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1 
    return num_clusters, X[mask], y[mask], labels[mask]

# for cross_dataset
# def load_data_from_file(path, session, dataset_name="seed3"):
#     if dataset_name == "seed3":
#         EEG, Label, Group = SEEDFeatureDataset(path, session=session).data()
#         Label += 1 # begin from 0
#         EEG = EEG.reshape(-1, 310)
#         sGroup = (Group[:, 0] - 1) * 15 + Group[:, 1] # subject ID 
#         # print(EEG.shape, Label.shape, Group.shape)

#     elif dataset_name == "seed4":
#         EEG, Label, Group = SEEDIVFeatureDataset(path, session=session).data()
#         EEG = EEG.reshape(-1, 310)
#         sGroup = (Group[:, 0] - 1) * 15 + Group[:, 1] # subject ID 
#         lookup = np.array([1, 0, 0, 2])  # 对应 0→1, 1→0, 2→0, 3→2
#         Label = lookup[Label]

#     return EEG, Label, sGroup

def load_data_from_file(path, session, dataset_name="seed3", emotion=None):
    if dataset_name == "seed3":
        EEG, Label, Group = SEEDFeatureDataset(path, session=session).data()
        Label += 1 # begin from 0
        EEG = EEG.reshape(-1, 310)
        sGroup = Group[:, 1] # subject ID 
        tGroup = Group[:, 2]
        
    elif dataset_name == "seed4":
        EEG, Label, Group = SEEDIVFeatureDataset(path, session=session).data()
        EEG = EEG.reshape(-1, 310)
        sGroup = Group[:, 1] # subject ID 
    
    elif dataset_name == "deap":
        EEG, Label, Group = DEAPDataset(path, labels=emotion).data()
        EEG = EEG.reshape(-1, 8064)
        sGroup = Group[:, 1]
        tGroup = Group[:, 2]

    return EEG, Label, sGroup

def load_and_preprocess_data(path, session, dataset_name, num_of_class):
    EEG, Label, sGroup = load_data_from_file(path, session, dataset_name)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(sGroup):
        EEG[sGroup == i] = min_max_scaler.fit_transform(EEG[sGroup == i])
    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")
    return EEG, one_hot_mat, sGroup

def process_target_domain(args, EEG, sGroup, target, one_hot_mat):
    target_mask = sGroup == target
    # print(EEG[target_mask].shape)
    # print(target, sGroup)
    n, x, y, c = perform_clustering(args, EEG[target_mask], one_hot_mat[target_mask])
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


def add_noise_to_labels(labels, noise_level=0.1):
    """
    Add noise to one-hot encoded labels by randomly flipping them.
    
    Args:
    - labels (np.ndarray): One-hot encoded labels.
    - noise_level (float): The probability of flipping each label.
    
    Returns:
    - np.ndarray: Noisy labels.
    """
    noisy_labels = labels.copy()
    num_classes = labels.shape[1]
    for i in range(labels.shape[0]):
        if np.random.rand() < noise_level:
            # Randomly select a new class different from the current one
            current_class = np.argmax(labels[i])
            new_class = np.random.choice([c for c in range(num_classes) if c != current_class])
            noisy_labels[i] = np.eye(num_classes)[new_class]
    return noisy_labels

def process_source_domain(args, EEG, sGroup, source_lists, one_hot_mat, noisy_level=0):
    cluster_results = {}
    min_clusters = float('inf')
    
    for source in source_lists:
        mask = sGroup == source
        n, x, y, c = perform_clustering(args, EEG[mask], one_hot_mat[mask])
        cluster_results[source] = (n, x, y, c)
        # (n)
        min_clusters = min(min_clusters, n)

    
    setattr(args, "num_of_s_clusters", min_clusters)
    source_datasets = []

    for source, (n, x, y, c) in cluster_results.items():
        cluster_counts = Counter(c)
        top_clusters = [cluster for cluster, _ in cluster_counts.most_common(min_clusters)]
        mask = np.isin(c, top_clusters)

        source_features = torch.from_numpy(x[mask]).type(torch.Tensor)
        
        source_labels = torch.from_numpy(add_noise_to_labels(y[mask], noisy_level)).type(torch.Tensor)

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

def getDataLoader(args, target=1, source_lists=None, noisy_level=0):
    EEG, one_hot_mat, sGroup = load_and_preprocess_data(args.path, args.session, args.dataset_name, args.num_of_class)

    target_loader = process_target_domain(args, EEG, sGroup, target, one_hot_mat)

    if source_lists is None:
        source_lists = list(range(1, 16))
        source_lists.remove(target)

    if args.ablation == "WithOutMixSource":
        pretrain_loader = process_source_domain(args, EEG, sGroup, source_lists, one_hot_mat, noisy_level)
    else:
        pretrain_loader = process_source_domain_without_SSTS(args, EEG, sGroup, source_lists, one_hot_mat)

    if args.ablation == "WithOutSSTS":
        train_loader = process_source_domain_without_SSTS(args, EEG, sGroup, source_lists, one_hot_mat)
    else:
        train_loader = process_source_domain(args, EEG, sGroup, source_lists, one_hot_mat, noisy_level)

    if args.ablation == "TestStartup":
      n = args.startup - 1 
      train_loader = [train_loader[n]] + train_loader[:n] + train_loader[n+1:]


    source_loaders = {
        "pretrain": pretrain_loader,
        "train": train_loader
    }
    return source_loaders, target_loader


def getCrossDataLoader(args, target=1, source_lists=None, noisy_level=0):
    seed3_path = args.seed3_path
    seed4_path = args.seed4_path
    EEG, one_hot_mat, sGroup = load_and_preprocess_data(seed3_path, [1, 2, 3], "seed3", 3)
    if source_lists is None:
        source_lists = list(range(1, 46))

    pretrain_loader = process_source_domain_without_SSTS(args, EEG, sGroup, source_lists, one_hot_mat)
    train_loader = process_source_domain(args, EEG, sGroup, source_lists, one_hot_mat, noisy_level)

    source_loaders = {
        "pretrain": pretrain_loader,
        "train": train_loader
    }

    EEG, one_hot_mat, sGroup = load_and_preprocess_data(seed4_path, [1, 2, 3], "seed4", 3)
    
    target_loader = process_target_domain(args, EEG, sGroup, target, one_hot_mat)
    return source_loaders, target_loader