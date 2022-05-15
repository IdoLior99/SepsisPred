from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import pickle
import torch
import pandas as pd
import data_preprocessing as dp
import tqdm
import p3_vizzes
from sklearn.model_selection import train_test_split


# A downsampling / train test splitting script.

def under_sampler(squashed):
    features = [l[0] for l in squashed]
    labels = [l[1] for l in squashed]
    return RandomUnderSampler(sampling_strategy=1 / 4, random_state=42).fit_resample(features, labels)


def downsample_dataset(base, name):
    squashed = [(dp.clean_table(samp[0], tensorize=False), samp[1]) for samp in tqdm.tqdm(base, desc='cleaning tables')]
    features, labels = under_sampler(squashed)
    pickle.dump(features, open(f'{name}_features_undersampled.p', 'wb'))
    pickle.dump(labels, open(f'{name}_labels_undersampled.p', 'wb'))


def sample_and_split():
    base = pickle.load(open('train_processed_adv.p', 'rb'))
    features = [f[0] for f in base]
    labels = [f[1] for f in base]
    feat_train, feat_val, lab_train, lab_val = train_test_split(features, labels)
    new_train = [[f, l] for f, l in zip(feat_train, lab_train)]
    new_test = [[f, l] for f, l in zip(feat_val, lab_val)]
    pickle.dump(new_test, open('val_adv.p', 'wb'))
    p3_vizzes.class_imbalance_plot(lab_train)
    downsample_dataset(base, name='adv_train')  # For the final models
    downsample_dataset(new_train, name='adv_part_train')  # For validation/model selection
    wee = pickle.load(open('adv_part_train_labels_undersampled.p', 'rb'))
    p3_vizzes.class_imbalance_plot(wee)
    print("Done")
