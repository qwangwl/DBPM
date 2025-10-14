# -*- encoding: utf-8 -*-
'''
file       :_get_dataset.py
Date       :2025/02/17 20:56:28
Email      :qiang.wang@stu.xidian.edu.cn
Author     :qwangxdu
'''

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datasets import SEEDFeatureDataset, SEEDIVFeatureDataset

def get_dataset(args):
    if args.dataset_name == "seed3":
        data, one_hot_mat, Group = get_seed(args)
    elif args.dataset_name == "seed4":
        data, one_hot_mat, Group = get_seediv(args)
    return {
        "data": data,
        "one_hot_mat": one_hot_mat, 
        "groups": Group
    }


def get_seed(args):

    num_of_class = 3
    num_of_channels = 62

    SEED = SEEDFeatureDataset(args.seed3_path, sessions=args.session)
    feature_dim = num_of_channels * SEED.get_feature_dim()
    dataset = SEED.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    Label += 1 # begin from 0
    data = data.reshape(-1, feature_dim)
    subject_ids = Group[:, 0] # subject ID 

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_classes", num_of_class)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(subject_ids):
        data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])

    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")
    return data, one_hot_mat, Group

def get_seediv(args):
    num_of_class = 4
    num_of_channels = 62

    SEEDIV = SEEDIVFeatureDataset(args.seed4_path, sessions=args.session)
    feature_dim = num_of_channels * SEEDIV.get_feature_dim()
    dataset = SEEDIV.get_dataset()

    data, Label, Group = dataset["data"], dataset["labels"], dataset["groups"]
    data = data.reshape(-1, feature_dim)
    subject_ids = Group[:, 0] # subject ID 

    setattr(args, "feature_dim", feature_dim)
    setattr(args, "num_classes", num_of_class)

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    for i in np.unique(subject_ids):
        data[subject_ids==i] = min_max_scaler.fit_transform(data[subject_ids == i])

    one_hot_mat = np.eye(len(Label), num_of_class)[Label].astype("float32")
    return data, one_hot_mat, Group
