import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# A Linear layer where the parameters are injected instead of locally instantiated and learned
class Parametrized_Linear_Layer(nn.Module):
    # weights are output_size x input_size
    # bias are output_size
    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.weight = Parameter(weight, requires_grad=True)
        self.bias = Parameter(bias, requires_grad=True)

    def __call__(self, input: Tensor):
        return F.linear(input, self.weight, self.bias)

# A Parametrized layer where multiple Linear layers are applied at once on the same input data
class Parametrized_Multi_Linear_Layer(nn.Module):
    # weights are layer_number x output_size x input_size
    # bias are layer_number x output_size x 1 (should probably be unsqueezed(2)
    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.weight = Parameter(weight, requires_grad=True)
        self.bias = Parameter(bias, requires_grad=True)

    def __call__(self, input: Tensor):
        batch_size = input.size()[1]
        output = input.matmul(self.weight)
        bias = self.bias.unsqueeze(1)
        bias = bias.expand(-1, batch_size, -1)
        return  output + bias

# A Linear layer where multiple layers are applied at once at the same data. The weights are instantiated and learned internally
class Multi_Linear_layer(Parametrized_Multi_Linear_Layer):
    # weights are layer_number x output_size x input_size
    # bias are layer_number x output_size x 1 (should probably be unsqueezed(2)
    def __init__(self, number_of_layers, in_feature, out_feature):
        weight = torch.zeros(number_of_layers, in_feature, out_feature)
        bias = torch.zeros(number_of_layers, out_feature)

        torch.nn.init.kaiming_normal_(weight)
        torch.nn.init.kaiming_normal_(bias)

        super().__init__(weight, bias)


# A Parametrized layer where multiple Linear layers are applied at once on the same input data
class Parametrized_Multi_Linear_Residual_Layer(nn.Module):
    # weights are layer_number x output_size x input_size
    # bias are layer_number x output_size x 1 (should probably be unsqueezed(2)
    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.weight = Parameter(weight, requires_grad=True)
        self.bias = Parameter(bias, requires_grad=True)

    def __call__(self, input: Tensor):
        batch_size = input.size()[1]
        residual = input.matmul(self.weight)
        bias = self.bias.unsqueeze(1)
        bias = bias.expand(-1, batch_size, -1)
        residual = residual + bias

        return  input + residual

# A Linear layer where multiple layers are applied at once at the same data. The weights are instantiated and learned internally
class Multi_Linear_Residual_layer(Parametrized_Multi_Linear_Residual_Layer):
    # weights are layer_number x output_size x input_size
    # bias are layer_number x output_size x 1 (should probably be unsqueezed(2)
    def __init__(self, number_of_layers, in_feature, out_feature):
        weight = torch.zeros(number_of_layers, in_feature, out_feature)
        bias = torch.zeros(number_of_layers, out_feature)

        torch.nn.init.kaiming_normal_(weight)
        torch.nn.init.kaiming_normal_(bias)

        super().__init__(weight, bias)

class PolicyGeneratedLinearLayer(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor):
        super().__init__()
        self.weight = Parameter(weight, requires_grad=True)
        self.bias = Parameter(bias, requires_grad=True)


    def __call__(self, input: Tensor):
        input = input.unsqueeze(1)
        output = input.matmul(self.weight).squeeze(1)
        output += self.bias
        return output


class PolicyGeneratedResidualLinearLayer(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor = None):
        super().__init__()
        self.weight = weight
        if bias is None:
            self.bias = None
        else:
            self.bias = bias


    def __call__(self, input: Tensor):
        residual = input.unsqueeze(1)
        residual = residual.matmul(self.weight).squeeze(1)
        if self.bias is not None:
            residual += self.bias
        return input + residual


class PolicyGeneratedResidualConvolutionLayer(nn.Module):
    def __init__(self, weight: Tensor, bias: Tensor = None, padding=0):
        super().__init__()
        self.padding = padding
        self.weight = weight
        self.bias = bias

    # input est batch x state_channels x width x length
    def __call__(self, input):

        residual = torch.stack([
            F.conv2d(input[i, :, :, :].unsqueeze(0), self.weight[i,:,:,:,:], padding=self.padding) for i, x_i in enumerate(torch.unbind(input, dim=0), 0)
        ], dim=0).squeeze(1)
        return input + residual


