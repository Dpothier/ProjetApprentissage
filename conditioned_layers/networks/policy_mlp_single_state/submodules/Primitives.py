import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from networks.policy_mlp.submodules.LinearLayers import PolicyGeneratedLinearLayer


class Parametrized_MLP_Primitives(nn.Module):
    def __init__(self, primitives_weight, primitives_bias, layer_constructor, learn_primitives=True):
        super().__init__()
        self.primitives_size = primitives_weight.size()[0]
        self.number_of_primitives = primitives_weight.size()[1]
        self.layer_constructor = layer_constructor


        self.primitives_weight = Parameter(primitives_weight, requires_grad=learn_primitives)
        self.primitives_bias = Parameter(primitives_bias, requires_grad=learn_primitives)


    def __call__(self, attention):
        # attention dim are number of output x number of primitives
        # For each output to be produced, we have a different attention vector
        batch_size = attention.size()[0]
        number_of_outputs = attention.size()[1]
        # The attention vector do not have a state_size dimension, but the attention weights must be applied on every state feature, so we expand the vector
        weight_attention = attention.expand(self.primitives_size,-1, -1,  -1)
        weight_attention = weight_attention.transpose(0, 1).transpose(1, 2)

        # The primitives are the same for every output, so we expand the weights and biases
        prepared_primitives = self.primitives_weight.expand(number_of_outputs, -1, -1).expand(batch_size, -1, -1, -1)
        layer_weight = (prepared_primitives * weight_attention).sum(3)

        layer_bias = (self.primitives_bias.expand(number_of_outputs, -1).expand(batch_size, -1, -1) * attention).sum(2)

        return self.layer_constructor(layer_weight, layer_bias)

class MLP_Primitives(Parametrized_MLP_Primitives):
    def __init__(self, primitives_size, number_of_primitives, layer_constructor):
        primitives_weight = torch.zeros((primitives_size, number_of_primitives))
        primitives_bias =  torch.zeros((number_of_primitives))

        nn.init.kaiming_normal_(primitives_weight)
        nn.init.normal_(primitives_bias)

        super().__init__(primitives_weight, primitives_bias, layer_constructor, True)

# This primitive set uses preset primitives. It's parameters should be removed from optimisation
class MLP_PrimitivesPreSet(Parametrized_MLP_Primitives):
    def __init__(self, number_of_primitives, layer_constructor):
        primitives_weight = torch.zeros((number_of_primitives, number_of_primitives))
        primitives_bias = torch.zeros((number_of_primitives))

        nn.init.kaiming_normal_(primitives_weight)
        nn.init.normal_(primitives_bias)

        super().__init__(primitives_weight, primitives_bias, layer_constructor, False)


class MLP_NoPrimitives(nn.Module):
    def __init__(self, layer_constructor):
        super().__init__()
        self.layer_contructor = layer_constructor

    def __call__(self, input):
        return self.layer_contructor(input)
