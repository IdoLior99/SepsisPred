from runs_and_eval import eval_csv
import matplotlib.pyplot as plt
import seaborn as sns;
import numpy as np
import pickle
import pandas as pd

sns.set()


def class_imbalance_plot(labels):
    counts = [len(labels) - sum(labels), sum(labels)]
    counts = pd.DataFrame({'Healthy/Sick': ['Healthy', 'Sick'], 'Count': counts})
    ax = sns.barplot(x="Healthy/Sick", y="Count", data=counts, palette="deep")
    ax.bar_label(ax.containers[0])
    plt.show()


def print_roc(dsets):
    train_df, train_auc = eval_csv(dsets[0], 'roc')
    test_df, test_auc = eval_csv(dsets[1], 'roc')
    fig, ax = plt.subplots()
    sns.lineplot(data=train_df, x='fpr', y='tpr', ax=ax, label='train')
    sns.lineplot(data=test_df, x='fpr', y='tpr', ax=ax, label='test').set(
        title=f'ROC curves, train auc = {train_auc}, test auc= {test_auc}')
    plt.legend(loc="lower right")
    plt.show()


def line_graph(model_1_p, model_2_p, model_3_p, col, epochs=5):
    mod1 = pickle.load(open(model_1_p, 'rb'))[col]
    mod2 = pickle.load(open(model_2_p, 'rb'))[col]
    mod3 = pickle.load(open(model_3_p, 'rb'))[col][:5]
    res_df = pd.DataFrame({'model1': mod1, 'model2': mod2, 'model3': mod3, 'epochs': np.arange(1, epochs + 1)})
    fig, ax = plt.subplots()
    sns.lineplot(data=res_df, x='epochs', y='model1', ax=ax, label='net1')
    sns.lineplot(data=res_df, x='epochs', y='model2', ax=ax, label='net2')
    sns.lineplot(data=res_df, x='epochs', y='model3', ax=ax, label='net3').set(title="3 Networks comparison")
    plt.legend(loc="lower right")
    plt.show()


def traintest_line_graph(train_stats_p, test_stats_p, col):
    train_stats = pickle.load(open(train_stats_p, 'rb'))[col]
    test_stats = pickle.load(open(test_stats_p, 'rb'))[col]
    epochs = len(train_stats)
    if col == 'loss':  # Tensors
        train_stats = [ts.item() for ts in train_stats]
        test_stats = [ts.item() for ts in test_stats]
    res_df = pd.DataFrame({'train': train_stats, 'test': test_stats, 'epochs': np.arange(1, epochs + 1)})
    fig, ax = plt.subplots()
    sns.lineplot(data=res_df, x='epochs', y='train', ax=ax, label='train')
    sns.lineplot(data=res_df, x='epochs', y='test', ax=ax, label='test').set(
        title=f'{col} values per epoch', ylabel=f'{col}')
    plt.legend(loc="lower right")
    plt.show()
