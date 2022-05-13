from torch import nn
import torch

POOLINGS = {"avg": nn.AvgPool2d(2), "max": nn.MaxPool2d(2)}
ACTIVATIONS = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), 'logsoftmax': nn.LogSoftmax(dim=1)}


class MLP(nn.Module):

    def __init__(self, in_dim,
                 mlp_hidden_dims, output_dim,
                 activation_type, final_activation_type, dropout=0,):
        """
        A flexible MLP Class
        """
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.structure = ""
        activation = ACTIVATIONS[activation_type]
        final_activation = ACTIVATIONS[final_activation_type]
        mlp_layers = []
        self.structure += "--------- MLP --------- \n"
        if dropout:
            mlp_layers.append(nn.Dropout(dropout))
            self.structure += f"Dropout({dropout}) \n"
        if (len(mlp_hidden_dims)) > 0:
            mlp_layers.extend([nn.Linear(in_dim, mlp_hidden_dims[0]), activation])
            self.structure += f"Linear({in_dim},{mlp_hidden_dims[0]}), {activation_type} \n"
            for i in range(len(mlp_hidden_dims) - 1):
                mlp_layers.extend([nn.Linear(mlp_hidden_dims[i], mlp_hidden_dims[i + 1]), activation])
                self.structure += f"Linear({mlp_hidden_dims[i]},{mlp_hidden_dims[i + 1]}), {activation_type} \n"
                if dropout:
                    mlp_layers.append(nn.Dropout(dropout))
                    self.structure += f"Dropout({dropout}) \n"
            mlp_layers.extend([nn.Linear(mlp_hidden_dims[-1], output_dim), final_activation])
            self.structure += f"Linear({mlp_hidden_dims[-1]},{output_dim}), {final_activation_type} \n"
        else:
            mlp_layers.extend([nn.Linear(in_dim, output_dim), final_activation])
            self.structure += f"Linear({in_dim},{output_dim}), {final_activation_type} \n"
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(self, x):
        out = self.mlp(x)
        return out
