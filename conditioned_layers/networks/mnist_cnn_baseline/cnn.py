import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self, state_channels, number_of_conv_layers, number_of_classes, in_channels=1):
        super().__init__()
        self.initial_layer = nn.Conv2d(in_channels=in_channels, out_channels=state_channels, kernel_size=1, padding=0)
        self.classification_layer = nn.Linear(in_features=state_channels, out_features=number_of_classes)

        mid_layers = []
        for i in range(number_of_conv_layers):
            mid_layers.append(nn.Conv2d(in_channels=state_channels, out_channels=state_channels, kernel_size=3, padding=1))

        self.mid_layers = nn.ModuleList(mid_layers)


    def __call__(self, input: Tensor):
        output = torch.relu(self.initial_layer(input))
        for layer in self.mid_layers:
            output = torch.relu(output + layer(output))

        output = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(3).squeeze(2)
        return self.classification_layer(output)
