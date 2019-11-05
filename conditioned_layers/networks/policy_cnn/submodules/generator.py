import torch
import torch.nn as nn
from networks.policy_cnn.submodules.ConvolutionLayers import PolicyGeneratedResidualConvolutionLayer

class ConvLayerGenerator(nn.Module):
    def __init__(self, kernel_size, state_channels, seed_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.state_channels = state_channels
        self.seed_size = seed_size

        self.generator = nn.Linear(in_features=seed_size, out_features=state_channels*kernel_size*kernel_size)


    def __call__(self, seed):
        batch_size = seed.size()[0]

        layer = self.generator(seed)
        layer = torch.tanh(layer)
        layer = layer.view(batch_size, self.state_channels, self.state_channels, self.kernel_size, self.kernel_size)
        return PolicyGeneratedResidualConvolutionLayer(weight=layer, padding=self.kernel_size//2)
