import data_preprocessing as dp
import runs_and_eval as run
import pickle
import matplotlib.pyplot as plt
import seaborn as sns;
import torch

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
    pickle.dump(test_ds, open('test_processed.p', 'wb'))
    """

    # Class imbalance plots:
    """
    test_ds = pickle.load(open('test_processed.p', 'rb'))
    class_imbalance_plot([samp[1] for samp in test_ds])
    train_ds = pickle.load(open('train_processed.p', 'rb'))
    class_imbalance_plot([samp[1] for samp in train_ds])
    """

    # Basic Net training:
    train_dataloader, train_set = dp.create_dataloader(['basic_train_features_undersampled.p',
                                                        'basic_train_labels_undersampled.p'], torch.tensor, 256, True)
    test_dataloader, test_set = dp.create_dataloader(['basic_test_features_undersampled.p',
                                                      'basic_test_labels_undersampled.p'], torch.tensor, 256, True)
    basic_net_params = {'in_dim': 40, 'mlp_hidden_dims': [1024, 1024, 512, 512, 256, 256, 128, 64, 64],
                        'output_dim': 2,
                        'activation_type': 'relu', 'final_activation_type': 'logsoftmax', 'dropout': 0.5}
    run.run_eval_classifier(num_epochs=5, batch_size=256, learning_rate=0.001, network_params=basic_net_params,
                            train_set=train_set,
                            model_name='baseline', train_loader=train_dataloader, test_loader=test_dataloader)

    print("hi")
