import torch
import torch.nn as nn
import torch.nn.functional as F

class Parametrized_Linear_Layer(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, input):
        return F.linear(input, self.weight, self.bias)


class MLP_Primitives(nn.Module):
    def __init__(self, primitives_size, number_of_primitives, primitives_weight=None, primitives_bias=None):
        super().__init__()
        self.primitives_size = primitives_size
        self.number_of_primitives = number_of_primitives

        if primitives_weight is not None:
            self.primitives_weight = primitives_weight
        else:
            self.primitives_weight = nn.init.kaiming_normal_(torch.zeros([primitives_size, number_of_primitives]))

        if primitives_bias is not None:
            self.primitives_bias = primitives_bias
        else:
            self.primitives_bias = torch.zeros([number_of_primitives])
            nn.init.normal_(self.primitives_bias)


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

        return Parametrized_Linear_Layer(layer_weight, layer_bias)



class policyMLP(nn.Module):

    def __init__(self, primitive_size, number_of_primitives, number_of_steps, input_size):
        super().__init__()
        self.start_layer = nn.Linear(input_size, primitive_size)





