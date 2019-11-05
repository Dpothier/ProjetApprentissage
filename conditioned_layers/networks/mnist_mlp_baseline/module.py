import torch
import torch.nn as nn
import math


class BaselineMnistMLP(nn.Module):
    def __init__(self, dropout_rate, state_size, depth):
        super().__init__()
        corrected_layers = []

        if depth == 1:
            self.last_layer = nn.Linear(in_features=784, out_features=10)
        else:
            self.last_layer = nn.Linear(in_features=state_size, out_features=10)
            corrected_layers.append(nn.Linear(in_features=784, out_features=state_size))
            mid_layers = depth - 2
            for i in range(mid_layers):
                corrected_layers.append(nn.Linear(in_features=state_size, out_features=state_size))



        self.corrected_layers = nn.ModuleList(corrected_layers)

        self.dropout = nn.Dropout(dropout_rate)
        nn.init.kaiming_normal_(self.last_layer.weight.data)
        for layer in self.corrected_layers:
            nn.init.kaiming_normal_(layer.weight.data)


    def __call__(self, input):
        output = input.view(-1, 784)
        for layer in self.corrected_layers:
            output = self.dropout(torch.relu(layer(output)))

        output = self.last_layer(output)
        return output