import torch
import torch.nn as nn
import math


class BaselineMnistMLP(nn.Module):
    def __init__(self, state_size, depth):
        super().__init__()

        self.first_layer = nn.Linear(in_features=784, out_features=state_size)
        self.last_layer = nn.Linear(in_features=state_size, out_features=10)
        mid_layers = []

        for i in range(depth):
            mid_layers.append(nn.Linear(in_features=state_size, out_features=state_size))

        nn.init.kaiming_normal_(self.first_layer.weight.data)
        nn.init.kaiming_normal_(self.last_layer.weight.data)
        for layer in mid_layers:
            nn.init.kaiming_normal_(layer.weight.data)

        self.mid_layers = nn.ModuleList(mid_layers)


    def __call__(self, input):
        output = input.view(-1, 784)
        output = torch.relu(self.first_layer(output))
        for layer in self.mid_layers:
            output = torch.relu(layer(output))

        output = self.last_layer(output)
        return output