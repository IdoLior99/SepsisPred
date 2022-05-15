import data_preprocessing as dp
import runs_and_eval as run
import pickle
import torch
import numpy as np
from runs_and_eval import eval_csv, grid_search
import p3_vizzes
import sampler

def run_final_models(base=False):
    if base:
        # (Final) Basic Net training:
        train_dataloader, train_set = dp.create_dataloader(['basic_train_features_undersampled.p',
                                                            'basic_train_labels_undersampled.p'], torch.tensor, 256, True)
        test_dataloader, test_set = dp.create_dataloader(['test_processed.p'], dp.clean_table, 256, True)
        basic_net_params = {'in_dim': 40, 'mlp_hidden_dims': [1024, 1024, 512, 512, 256, 256, 128, 64, 64],
                            'output_dim': 2,
                            'activation_type': 'relu', 'final_activation_type': 'logsoftmax', 'dropout': 0.5}
        run.run_eval_classifier(num_epochs=200, batch_size=256, learning_rate=0.001, network_params=basic_net_params,
                                train_set=train_set,
                                model_name='baseline', train_loader=train_dataloader, test_loader=test_dataloader)

    else:
        # (Final) Advanced Net training:
        train_dataloader, train_set = dp.create_dataloader(['adv_train_features_undersampled.p',
                                                            'adv_train_labels_undersampled.p'], torch.tensor, 256, True)

        test_dataloader, test_set = dp.create_dataloader(['test_processed_adv.p'], dp.clean_table, 256, True)
        adv_net_params = {'in_dim': 40, 'mlp_hidden_dims': [2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64],
                          'output_dim': 2,
                          'activation_type': 'relu', 'final_activation_type': 'logsoftmax', 'dropout': 0.5}

        run.run_eval_classifier(num_epochs=300, batch_size=256, learning_rate=0.0005, network_params=adv_net_params,
                                train_set=train_set,
                                model_name='advanced', train_loader=train_dataloader, test_loader=test_dataloader)

        # Generating plots:
        """
        vizzes.traintest_line_graph('train_stats_advanced.p', 'test_stats_advanced.p', 'accruacy')
        vizzes.traintest_line_graph('train_stats_advanced.p', 'test_stats_advanced.p', 'loss')
        vizzes.traintest_line_graph('train_stats_advanced.p', 'test_stats_advanced.p', 'f1')
        """


def run_model_selection():
    train_dataloader, train_set = dp.create_dataloader(['adv_part_train_features_undersampled.p',
                                                        'adv_part_train_labels_undersampled.p'], torch.tensor, 256,
                                                       True)
    test_dataloader, test_set = dp.create_dataloader(['val_adv.p'], dp.clean_table, 256, True)
    cat_vals = [[40], [[2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64],
                       [1024, 1024, 512, 512, 256, 256, 128, 64, 64],
                       [256, 256, 128, 64, 64]], [2], ['relu', 'tanh'], ['logsoftmax'], [0.5], [30, 50], [0.001, 0.005]]
    best_params = grid_search(cat_vals, train_dataloader, train_set, test_dataloader, 256)
    return best_params


if __name__ == '__main__':
    # Generate pickle files:
    """
    train_ds = dp.get_ds('data/train')
    pickle.dump(train_ds, open('train_processed.p', 'wb'))
    test_ds = dp.get_ds('data/test')
    pickle.dump(test_ds, open('test_processed.p', 'wb'))
    
    train_ds = dp.get_ds_adv('data/train', [], topickle=True, name='train')
    pickle.dump(train_ds, open('train_processed_adv.p', 'wb'))
    test_ds = dp.get_ds_adv('data/test', [], topickle=True, name='train')
    pickle.dump(test_ds, open('test_processed_adv.p', 'wb'))
    """
    # Train-test split and Downsampling:
    """
    sampler.sample_and_split()
    """
    # Run model selection:
    """
    run_model_selection()
    """
    # Run final models:
    """
    run_final_models(True)  # Base model
    run_final_models(False)  # Advanced/final model
    """
