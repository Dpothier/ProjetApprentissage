import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.policy_mlp.submodules.LinearLayers import Multi_Linear_layer

# The function to convert a state to a set of queries. The number of queries must be equal to the number of primitives
class Parametrized_SeedToQueryFct(nn.Module):
    # layers is a list of layers that will be applied successively. Should be of dimension number_of_layer x output_size x input_size
    # The first layer should have input_size = state_size and output_size = query_size. Every other layer should have input_size = output_size = query_size
    # activation is the activation function to use between every layer
    def __init__(self, transformations_per_layer, layers, activation):
        super().__init__()
        self.transformations_per_layer = transformations_per_layer
        self.layers = nn.ModuleList(layers)
        self.activation_fct = activation

    # Input is of sizes batch x seed
    # All n layers are applied to the seed to produce n attention vectors
    def __call__(self, input: torch.Tensor):
        output = input.expand(self.transformations_per_layer, -1, -1)

        max_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            output = layer(output)
            if index < max_index:
                output = self.activation_fct(output)

        return output.transpose(0,1)

class ReluSeedToQueryFct(Parametrized_SeedToQueryFct):
    def __init__(self, number_of_layers, transformations_per_layer, seed_size, query_size, layer_constructor):
        seedToQueryLayers = []
        for i in range (number_of_layers - 1):
            seedToQueryLayers.append(layer_constructor(transformations_per_layer, seed_size, seed_size))

        seedToQueryLayers.append(layer_constructor(transformations_per_layer, seed_size, query_size))
        super().__init__(transformations_per_layer, seedToQueryLayers, F.relu)