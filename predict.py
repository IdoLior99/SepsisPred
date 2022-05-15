import os
import argparse
import numpy as np
import pandas as pd
from models import MLP
import torch
import tqdm
import data_preprocessing as dp
# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()

# Reading input folder
files = os.listdir(args.input_folder)


# Load model
network_params = {'in_dim': 40, 'mlp_hidden_dims': [2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128, 64],
                      'output_dim': 2,
                      'activation_type': 'relu', 'final_activation_type': 'logsoftmax', 'dropout': 0.5}

clf = MLP(in_dim=network_params["in_dim"],
          mlp_hidden_dims=network_params["mlp_hidden_dims"], output_dim=network_params["output_dim"],
          activation_type=network_params["activation_type"],
          final_activation_type=network_params["final_activation_type"],
          dropout=network_params["dropout"])

clf.load_state_dict(torch.load('modeladvanced_59.pkl', map_location=torch.device('cpu')))

patient_ids = []
preds = []
scores = []
clf.eval()
data = dp.get_ds_adv(args.input_folder, cols=[], topickle=True, name='predict') #TODO true in submission
for i, (table, label) in tqdm.tqdm(enumerate(data), desc='predicting'):
    patient_ids.append(i)
    with torch.no_grad():
        inp = dp.clean_table(table)
        score = clf(torch.unsqueeze(inp, 0))
        preds.append(torch.argmax(score, 1).item())
        scores.append(score[0][1].item()) #TODO: For ROC curve


prediction_df = pd.DataFrame([*zip(patient_ids, preds)])
score_df = pd.DataFrame([*zip(patient_ids, scores)])

prediction_df.to_csv("prediction.csv", index=False, header=False)
score_df.to_csv("scores.csv", index=False, header=False)

from runs_and_eval import eval_csv
print(eval_csv('prediction.csv'))
