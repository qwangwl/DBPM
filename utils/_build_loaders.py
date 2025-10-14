# -*- encoding: utf-8 -*-
'''
file       :_build_loaders.py
Date       :2025/10/13 19:19:35
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import torch
import numpy as np
from sklearn.cluster import DBSCAN
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter
from utils._get_datasets import *

class PMDataset(TensorDataset):
    def __init__(self, d1, d2, d3):
        super(PMDataset, self).__init__()
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3

    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, idx):
        return self.d1[idx], self.d2[idx], self.d3[idx]
    
    def get_data(self):
        return self.d1, self.d2, self.d3

def perform_clustering(args, X, y):
    clustering = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit(X)
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    mask = labels != -1
    return num_clusters, X[mask], y[mask], labels[mask]

def process_target_domain(args, x_target, y_target):
    n, x, y, c = perform_clustering(args, x_target, y_target)
    setattr(args, "num_tgt_clusters", n)
    target_features = torch.from_numpy(x).type(torch.Tensor)
    target_labels = torch.from_numpy(y)
    target_cluster = torch.from_numpy(c)

    target_dataset = PMDataset(target_features, target_labels, target_cluster)
    target_loader = DataLoader(
        dataset=target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    return target_loader

def process_source_domains(args, x_source, y_source, s_source, with_ssts=True):
    cluster_results = {}
    min_clusters = float('inf')

    for subject_id in np.unique(s_source):
        mask = s_source == subject_id
        n, x, y, c = perform_clustering(args, x_source[mask], y_source[mask])
        cluster_results[subject_id] = (n, x, y, c)
        min_clusters = min(min_clusters, n)

    if with_ssts:
        setattr(args, "num_src_clusters", min_clusters)
    source_datasets = []

    for subject_id, (n, x, y, c) in cluster_results.items():
        # 1. Filters the data to ensure each subject has the same number of clusters.
        cluster_counts = Counter(c)
        top_clusters = [cluster for cluster, _ in cluster_counts.most_common(min_clusters)]
        mask = np.isin(c, top_clusters)

        source_features = torch.from_numpy(x[mask]).type(torch.Tensor)
        source_labels = torch.from_numpy(y[mask]) 
        # 2. Reorders cluster IDs to ensure consistency.
        source_cluster = c[mask]
        unique_clusters = np.unique(source_cluster)
        cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_clusters)}
        source_cluster = torch.tensor([cluster_mapping[label] for label in source_cluster], dtype=torch.long)
        source_datasets.append(PMDataset(source_features, source_labels, source_cluster))

    source_loaders = [
        DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        ) for dataset in source_datasets
    ]

    return source_loaders

def process_source_domain_with_mix(args, x_source, y_source, s_source, source_lists):
    source_mask = np.isin(s_source, source_lists)
    n, x, y, c = perform_clustering(args, x_source[source_mask], y_source[source_mask])
    setattr(args, "num_src_clusters", n)
    source_features = torch.from_numpy(x).type(torch.Tensor)
    source_labels = torch.from_numpy(y)
    source_cluster = torch.from_numpy(c)
    source_dataset = PMDataset(source_features, source_labels, source_cluster)

    source_loader = DataLoader(
        dataset=source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    return source_loader