from data_preprocessing import train_test_split
import torch
from torch import nn
from sklearn.metrics import f1_score
import pickle
from models import MLP
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import itertools
import glob


def eval_csv(path_to_csv, type='f1'):
    true_labels = []
    csv_files = glob.glob('data/test' + "/*.psv")
    for f in csv_files:
        table = pd.read_csv(f, sep='|')
        seps = table['SepsisLabel'].sum()
        true_labels.append(1 if seps > 0 else 0)
    lolz = sum(true_labels)
    pred_df = pd.read_csv(path_to_csv, header=None)
    # pred_df = pred_df.sort_values(pred_df.columns[0])
    preds = pred_df.iloc[:, 1].tolist()
    if type == 'f1':
        return f1_score(true_labels, preds)
    else:
        fpr, tpr, _ = roc_curve(true_labels, preds)
        roc_df = pd.DataFrame.from_dict(dict(fpr=fpr, tpr=tpr))
        auc = roc_auc_score(true_labels, preds)
        return roc_df, auc


def train_and_eval(clf, train_dataloader, train_dataset, test_dataloader, test_dataset, num_epochs, batch_size,
                   learning_rate, model_name, test_epoch_check=False, save=False, checkpoint=True):
    """
    Train and evaluate a classifier. Keep data necessary for visualizations.
    """
    save_epoch = 0
    print_every = 10
    train_scores = []
    test_scores = []
    train_accs = []
    test_accs = []
    train_f1s = []
    test_f1s = []
    train_losses = []
    test_losses = []
    print(clf.structure)
    if torch.cuda.is_available():
        print("cuda!!")
        clf.cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=learning_rate)
    data_length = len(train_dataloader.dataset)
    for epoch in range(num_epochs):
        print(f"---------- EPOCH {epoch + 1} ----------")

        # Training the model
        epoch_train_scores = []
        epoch_train_preds = []
        epoch_train_labels = []
        epoch_train_correct = 0
        for i, (flats, labels) in enumerate(train_dataloader):
            if torch.cuda.is_available():
                # print("cuda??")
                flats = flats.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = clf(flats)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total = labels.size(0)
            train_correct = (preds == labels).sum()
            batch_acc = float(train_correct) / train_total

            epoch_train_scores.extend(outputs.tolist())
            epoch_train_preds.extend(preds.tolist())
            epoch_train_labels.extend([label.item() for label in labels])
            if (i + 1) % print_every == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Train Acc: %.4f'
                      % (epoch + 1, num_epochs, i + 1,
                         len(train_dataset) // batch_size, loss.data, batch_acc))

        epoch_train_correct = sum(
            [1 if epoch_pred == epoch_label else 0 for epoch_pred, epoch_label in
             zip(epoch_train_preds, epoch_train_labels)])
        epoch_train_acc = float(epoch_train_correct) / data_length

        epoch_loss = criterion(torch.tensor(epoch_train_scores), torch.tensor(epoch_train_labels)).data
        epoch_f1 = f1_score(epoch_train_labels, epoch_train_preds)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)
        train_f1s.append(epoch_f1)

        if test_epoch_check or epoch == num_epochs - 1:
            # Testing Model Performance:

            print("Testing post-epoch model on test set:")
            epoch_test_f1, epoch_test_loss, epoch_test_acc = eval_clf(clf, test_dataloader, criterion)
            test_f1s.append(epoch_test_f1)
            if checkpoint and max(test_f1s) == epoch_test_f1:
                print("... Updated best model params ...")
                torch.save(clf.state_dict(), f'model{model_name}_checkpoint.pkl')
                save_epoch = epoch
            test_losses.append(epoch_test_loss)
            test_accs.append(epoch_test_acc)
            print('-------- Epoch [%d/%d] Final stats: --------\n '
                  'Loss: Train %.4f  Test %.4f \n'
                  'Train Acc: Train %.4f  Test %.4f \n'
                  'F1: Train %.4f  Test %.4f \n'
                  % (epoch + 1, num_epochs,
                     epoch_loss, epoch_test_loss, epoch_train_acc, epoch_test_acc, epoch_f1, epoch_test_f1))
        else:
            print('-------- Epoch [%d/%d] Final stats: --------\n '
                  'Loss: Train %.4f \n'
                  'Train Acc: Train %.4f\n'
                  'F1: Train %.4f \n'
                  % (epoch + 1, num_epochs,
                     epoch_loss, epoch_train_acc, epoch_f1))
    if save:
        torch.save(clf.state_dict(), f'model{model_name}.pkl')
        pickle.dump({'f1': train_f1s, 'accruacy': train_accs, 'loss': train_losses},
                    open(f"train_stats_{model_name}.p", "wb"))
        pickle.dump({'f1': test_f1s, 'accruacy': test_accs, 'loss': test_losses},
                    open(f"test_stats_{model_name}.p", "wb"))
    print(save_epoch)
    print("...Training Done...")
    return (train_losses, test_losses), (train_f1s, test_f1s), (train_accs, test_accs)


def eval_clf(clf, test_dataloader, criterion):
    """
    Test a model on the test dataset
    """
    clf.eval()
    test_scores = []
    test_labels = []
    for test_images, t_labels in test_dataloader:
        with torch.no_grad():
            if torch.cuda.is_available():
                # print("cuda??")
                test_images = test_images.cuda()
                t_labels = t_labels.cuda()
            test_outputs = clf(test_images)
        test_scores.extend(test_outputs.tolist())
        test_labels.extend([t_label.item() for t_label in t_labels])
    test_preds = torch.argmax(torch.tensor(test_scores), 1)
    test_loss = criterion(torch.tensor(test_scores), torch.tensor(test_labels)).data
    test_f1 = f1_score(test_labels, test_preds)
    test_correct = sum([1 if epoch_pred == epoch_label else 0 for epoch_pred, epoch_label in
                        zip(test_preds, test_labels)])
    test_acc = float(test_correct) / len(test_labels)
    clf.train()
    return test_f1, test_loss, test_acc


def run_eval_classifier(num_epochs, batch_size, learning_rate, network_params, train_set, model_name,
                        test_e_check=False, save=False, train_loader=None, test_loader=None):
    clf = MLP(in_dim=network_params["in_dim"],
              mlp_hidden_dims=network_params["mlp_hidden_dims"], output_dim=network_params["output_dim"],
              activation_type=network_params["activation_type"],
              final_activation_type=network_params["final_activation_type"],
              dropout=network_params["dropout"])

    # clf.load_state_dict(torch.load('modeladvanced_checkpoint.pkl', map_location=torch.device('cpu')))

    if test_loader and train_loader:
        train_and_eval(clf, train_loader, train_set, test_loader, None, num_epochs,
                       batch_size, learning_rate, test_epoch_check=True, save=True, model_name=model_name)
    else:
        subtrain_loader, subtrain_set, subtest_loader, subtest_set = train_test_split(train_set, batch_size)
        train_and_eval(clf, subtrain_loader, subtrain_set, subtest_loader, subtest_set, num_epochs,
                       batch_size, learning_rate, test_epoch_check=test_e_check, save=save, model_name=model_name)


def grid_search(cat_vals, train_dataloader, train_set, test_dataloader, batch_size):
    all_perms = list(itertools.product(*cat_vals))
    best_f1 = 0
    best_params = []
    for perm in all_perms:
        clf = MLP(in_dim=perm[0],
                  mlp_hidden_dims=perm[1], output_dim=perm[2],
                  activation_type=perm[3],
                  final_activation_type=perm[4],
                  dropout=perm[5])
        _, f1s, _ = train_and_eval(clf, train_dataloader, train_set, test_dataloader, None, num_epochs=perm[6],
                                   batch_size=batch_size, learning_rate=perm[7], test_epoch_check=False, save=False, checkpoint=False,
                                   model_name='')
        if f1s[1][0] > best_f1:
            best_f1 = f1s[1][0]
            best_params = perm
    return best_params
