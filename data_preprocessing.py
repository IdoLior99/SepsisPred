import pandas as pd
from torch.utils.data import Dataset, DataLoader
import glob
import torch
from torch.utils.data import random_split
import tqdm
import numpy as np
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer


def impute(meta_df: pd.DataFrame):
    # Split df by col categories
    lab_cols = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate',
                'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct',
                'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']
    vit_cols = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']
    dem_cols = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS']

    vit_df = meta_df[vit_cols]
    lab_df = meta_df[lab_cols]
    dem_df = meta_df[dem_cols]

    # Vital sign imputations
    vit_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    vit_imp_df = pd.DataFrame(vit_imp.fit_transform(vit_df.to_numpy()), columns=vit_cols)

    # Demographics imputation
    dem_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dem_imp_df = pd.DataFrame(dem_imp.fit_transform(dem_df.to_numpy()), columns=dem_cols)

    # Laboratory values imputation
    lab_imp = IterativeImputer(max_iter=10, random_state=0)
    lab_imp_df = pd.DataFrame(np.round(lab_imp.fit_transform(lab_df.to_numpy())), columns=lab_cols)

    meta_df = pd.concat([meta_df["Patient"], vit_imp_df, lab_imp_df, dem_imp_df, meta_df["SepsisLabel"]], axis=1)
    return meta_df


def get_ds_adv(ds_path, cols, topickle=True, name=''):
    """
    Reads the meta-dataset, severs any positive rows besides the first positive one and assigns label appropriately.
    """
    data = []
    csv_files = glob.glob(ds_path + "/*.psv")
    first = True
    if topickle:
        for i, file in tqdm.tqdm(enumerate(csv_files), desc='Creating mega ds'):
            if first:
                mega_table = pd.read_csv(file, sep='|')
                mega_table['Patient'] = [i]*len(mega_table.index)
                first = False
            else:
                table = pd.read_csv(file, sep='|')
                table['Patient'] = [i] * len(table.index)
                mega_table = mega_table.append(table, ignore_index=True)

        pickle.dump(mega_table, open(f'mega_table_{name}.p', 'wb'))
    else:
        print("Loading previously configured mega-table")
        mega_table = pickle.load(open(f'mega_table_{name}.p', 'rb'))
    mega_table = impute(mega_table)  # imputation
    if len(cols) > 1:
        mega_table = mega_table[['Patient']+cols+['SepsisLabel']]  # feature selection
    for idx, table in tqdm.tqdm(mega_table.groupby(['Patient'])):
        table = table.reset_index()
        sepsis = table['SepsisLabel'].array
        itemindex = list(np.where(sepsis == 1)[0])
        if len(itemindex) >= 1:
            table.drop(itemindex[1:], inplace=True)
        label = table['SepsisLabel'].sum()
        features = table.drop(['Patient', 'SepsisLabel', 'index'], axis=1)
        data.append([features, label])
    return data


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


def w_mean(cdf):
  df = cdf.copy()
  num_rows = len(df.index)
  sum = 0
  for i, row in enumerate(df.index):
    print (1/(num_rows-i))
    df.iloc[i:i+1, :] = (1/(num_rows-i)) * df.iloc[i:i+1, :]
    sum += (1/(num_rows-i))
  df = df.apply(lambda x: round(x/sum, 3))
  return df


def clean_table(df: pd.DataFrame, tensorize=True):
    """
    A basic preprocessing scheme
    """
    cpy = df.copy()#.fillna(0)
    mea = cpy.mean(axis=0, skipna=True)
    mea = mea.fillna(0)
    if tensorize:
        mea = torch.tensor(mea).float()
    return mea


class CustomMetaDataset(Dataset):
    def __init__(self, ds_paths, transform=None):
        if len(ds_paths) > 1:
            self.features = pickle.load(open(ds_paths[0], 'rb'))
            self.labels = pickle.load(open(ds_paths[1], 'rb'))
        else:
            base = pickle.load(open(ds_paths[0], 'rb'))
            self.features = [samp[0] for samp in base]
            self.labels = [samp[1] for samp in base]
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



