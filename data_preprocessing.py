import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import torch
from torch.utils.data import random_split
import tqdm
import numpy as np
import pickle


def get_ds(ds_path):
    """
    Reads the meta-dataset, severs any positive rows besides the first positive one and assigns label appropriately.
    """
    data = []
    for file in tqdm.tqdm(glob.glob(ds_path + '/*'), desc='Creating ds'):
        table = pd.read_csv(file, sep='|')
        sepsis = table['SepsisLabel'].array
        itemindex = np.where(sepsis == 1)[0]
        if len(itemindex) >= 1:
            table.drop(itemindex[1:], inplace=True)
        label = table['SepsisLabel'].sum()
        features = table.drop('SepsisLabel', axis=1)
        data.append([features, label])
    return data


def clean_table(df: pd.DataFrame):
    """
    A basic preprocessing scheme
    """
    cpy = df.copy()#.fillna(0)
    mea = cpy.mean(axis=0, skipna=True)
    mea = mea.fillna(0)
    #flat = torch.tensor(mea).float()
    return mea
# Custom DataLoader Class


# def under_sampler(squashed):
#     features = [l[0] for l in squashed]
#     labels = [l[1] for l in squashed]
#     return RandomUnderSampler(sampling_strategy=1/3, random_state=42).fit_resample(features, labels)


class CustomMetaDataset(Dataset):
    def __init__(self, ds_paths, transform=None):
        self.features = pickle.load(open(ds_paths[0], 'rb'))
        self.labels = pickle.load(open(ds_paths[1], 'rb'))
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature_vec = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature_vec = self.transform(feature_vec).float()
        return feature_vec, label


class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        table, label = self.subset[idx]
        if self.transform:
            table = self.transform(table)
        return table, label

    def __len__(self):
        return len(self.subset)


def create_dataloader(ds_path, transform, bs, shuffle):
    cds = CustomMetaDataset(ds_paths=ds_path, transform=transform)
    return DataLoader(cds, batch_size=bs, shuffle=shuffle, num_workers=1, pin_memory=True), cds


def train_test_split(torch_dataset, batch_size, shuffle=False, train_ratio=0.7):
    trainset, valset = random_split(torch_dataset,
                                    [int(len(torch_dataset) * train_ratio),
                                     len(torch_dataset) - int(len(torch_dataset) * train_ratio)],
                                    generator=torch.Generator().manual_seed(42))
    trainset, valset= SubsetDataset(trainset, None), SubsetDataset(valset, None)
    return DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True), trainset, \
           DataLoader(valset, batch_size=batch_size, shuffle=shuffle, num_workers=1, pin_memory=True), valset



