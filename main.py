import data_preprocessing as dp
import pickle
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()


def class_imbalance_plot(labels):
    ax = sns.histplot(labels)
    ax.bar_label(ax.containers[0])
    plt.show()


if __name__ == '__main__':
    # Generate pickle files:
    """
    train_ds = dp.get_ds('data/train')
    pickle.dump(train_ds, open('train_processed.p', 'wb'))
    test_ds = dp.get_ds('data/test')
    pickle.dump(train_ds, open('test_processed.p', 'wb'))
    """

    # Class imbalance plots:
    """
    test_ds = pickle.load(open('test_processed.p', 'rb'))
    class_imbalance_plot([samp[1] for samp in test_ds])
    train_ds = pickle.load(open('train_processed.p', 'rb'))
    class_imbalance_plot([samp[1] for samp in train_ds])
    """

    train_dataloader, train_set = dp.create_dataloader('train_processed.p', dp.clean_table, 256, False)
    test_dataloader, test_set = dp.create_dataloader('test_processed.p', dp.clean_table, 256, False)

    print("hi")
