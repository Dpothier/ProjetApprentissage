import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# A policy MLP receive a batch of input vectors. Batch size is the size of the batch, input size the size of the input vector
# The output of each layer is the state of size state_size. At t=0, the state != input and is blank vector
# Every layer should have state_size as in_size and out_size. If state_size != input_size, a special layer is trained to transform the input in state
# The output is given to a step-wise RNN. This rnn output a seed of size batch_size x seed_size
# The seed is fed to the SeedToQueryFunction. The seed is batch size x seed size, the query tensor is batch_size x out_size x  query size.
# For each query size vector in the query tensor, the distance with a key vector of the key tensor is calculated. The key tensor is primitives_no x query size
# The distance function output a tensor of size batch x out x primitives_no. The primitives_no dimension is softmaxed.
# The final output of the seed to attention function is batch x out x primitives_no
# The PrimitivesToLayer function maintains a primitive tensor of size primitives_no x in
# The PrimitivesToLayer function use the ponderation produced by the softmax to combine linearly every in vector in its tensor.
# The PrimitivesToLayer function outputs a layer tensor of size batch x out x in
# The layer is applied to the current state of size batch x out. Since out = in, the state is also of size batch x in
# The layer produce a new state, and the process restart
# The last state goes through a classification layer and is softmaxed for output to the loss function.


# Implements a GRU, but computes a single step at a time. Parameter are injected
class Paramerized_StepwiseGRU(nn.Module):
    def __init__(self, update_mem, update_input, reset_mem, reset_input, candidate_mem, candidate_input):
        super().__init__()
        self.update_mem = update_mem
        self.update_input = update_input
        self.reset_mem = reset_mem
        self.reset_input = reset_input
        self.candidate_mem = candidate_mem
        self.candidate_input = candidate_input

    # x is batch x state_size, h is batch x seed_size
    # Output is batch x seed_size
    def __call__(self, x, h):
        z = F.sigmoid(self.update_input(x) + self.update_mem(h))
        r = F.sigmoid(self.reset_input(x) + self.reset_mem(h))
        candidate = F.tanh(self.candidate_input(x) + r * self.candidate_mem(h))
        output_1 = z * h
        output_2 = (1 - z) * candidate
        output = output_1 + output_2
        return output


class StepwiseGRU(Paramerized_StepwiseGRU):
    def __init__(self, x_size, h_size):
        update_mem = nn.Linear(h_size, h_size)
        update_input = nn.Linear(x_size, h_size)
        reset_mem = nn.Linear(h_size, h_size)
        reset_input = nn.Linear(x_size, h_size)
        candidate_mem = nn.Linear(h_size, h_size)
        candidate_input = nn.Linear(x_size, h_size)
        super().__init__(update_mem, update_input, reset_mem, reset_input, candidate_mem, candidate_input)





# A Linear layer where the parameters are injected instead of locally instantiated and learned
class Parametrized_Linear_Layer(nn.Module):
    # weights are output_size x input_size
    # bias are output_size
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, input):
        return F.linear(input, self.weight, self.bias)

# A Parametrized layer where multiple Linear layers are applied at once on the same input data
class Parametrized_Multi_Linear_Layer(nn.Module):
    # weights are layer_number x output_size x input_size
    # bias are layer_number x output_size x 1 (should probably be unsqueezed(2)
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

    def __call__(self, input):
        return input.matmul(self.weight) + self.bias

# A Linear layer where multiple layers are applied at once at the same data. The weights are instantiated and learned internally
class Multi_Linear_layer(Parametrized_Multi_Linear_Layer):
    # weights are layer_number x output_size x input_size
    # bias are layer_number x output_size x 1 (should probably be unsqueezed(2)
    def __init__(self, number_of_layers, in_feature, out_feature):
        weight = torch.nn.init.kaiming_normal(torch.Tensor(number_of_layers, out_feature, in_feature))
        bias = torch.nn.init.kaiming_normal(torch.Tensor(number_of_layers, out_feature, 1))

        super().__init__(weight, bias)





class Parametrized_MLP_Primitives(nn.Module):
    def __init__(self, primitives_size, number_of_primitives, primitives_weight=None, primitives_bias=None):
        super().__init__()
        self.primitives_size = primitives_size
        self.number_of_primitives = number_of_primitives

        if primitives_weight is not None:
            self.primitives_weight = primitives_weight
        else:
            self.primitives_weight = nn.init.kaiming_normal_(torch.Tensor((primitives_size, number_of_primitives)))

        if primitives_bias is not None:
            self.primitives_bias = primitives_bias
        else:
            self.primitives_bias = torch.Tensor((number_of_primitives))
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

class MLP_Primitives(Parametrized_MLP_Primitives):
    def __init__(self, primitives_size, number_of_primitives):
        primitives_weight = nn.init.kaiming_normal_(torch.Tensor((primitives_size, number_of_primitives)))
        primitives_bias =  torch.Tensor((number_of_primitives))
        nn.init.normal_(primitives_bias)

        super().__init__(primitives_weight, primitives_bias)



# The function to convert a state to a set of queries. The number of queries must be equal to the number of primitives
class Parametrized_SeedToQueryFct(nn.Module):
    # layers is a list of layers that will be applied successively. Should be of dimension number_of_layer x output_size x input_size
    # The first layer should have input_size = state_size and output_size = query_size. Every other layer should have input_size = output_size = query_size
    # activation is the activation function to use between every layer
    def __init__(self, layers, activation):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.activation_fct = activation

    # Input is of sizes batch x seed
    # All n layers are applied to the seed to produce n attention vectors
    def __call__(self, input):
        output = input

        max_index = len(self.layers) - 1
        for index, layer in enumerate(self.layers):
            output = layer(output)
            if index < max_index:
                output = self.activation_fct(output)

        return output.transpose(0,1)

class ReluSeedToQueryFct(Parametrized_SeedToQueryFct):
    def __init__(self, depth, number_of_layers, seed_size, number_of_primitives):
        seedToQueryLayers = []
        for i in range (depth-1):
            seedToQueryLayers.append(Multi_Linear_layer(number_of_layers, seed_size, seed_size))

        seedToQueryLayers.append(Multi_Linear_layer(number_of_layers, seed_size, seed_size))
        super().__init__(seedToQueryLayers, F.relu)

class InjectedQueryToAttentionFct(nn.Module):
    # Keys are query_size x no_primitives
    def __init__(self, keys):
        super().__init__()
        self.keys = keys

    # Query is batch x output x query_size
    def __call__(self, query):
        batch_size = query.size()[0]
        output_size = query.size()[1]

        working_keys = self.keys.expand(batch_size, -1, -1) # output x query_size x no_primitives

        output = query.matmul(working_keys)
        output = torch.softmax(output, dim=2)
        return output

class QueryToAttentionFct(InjectedQueryToAttentionFct):
    def __init__(self, query_size, no_primitives):
        keys = torch.Tensor((query_size, no_primitives))
        keys = torch.nn.init.kaiming_normal(keys)
        super().__init__(keys)




# The attention function used over primitives in a policy MLP
class PolicyMLPAttention(nn.Module):
    def __init__(self, state_size, query_key_size, state_to_query_fct, keys):
        super().__init__()
        self.state_size = state_size
        self.query_key_size = query_key_size

        self.state_to_query_fct = state_to_query_fct
        self.keys = keys


    # The state should be of sizes batch_size x seed size
    def __call__(self, state):
        # The queries should be of size batch_size x seed size x state size
        queries = self.state_to_query_fct(state)






class PolicyMLP(nn.Module):
    # T is the number of recurrent computation steps
    # InitialLayer is a Linear layer of size batch x state x input
    # StepwiseRNN is an RNN computing one time step at a time have inputs of size batch x state and outputs of size batch x seed
    def __init__(self, T, initial_seed, initialLayer, stepwiseRNN, seedToQueryFct, queryToAttentionFct, classification_layer, primitives_set, primitive_size):
        super().__init__()
        self.initial_seed = Parameter(initial_seed)
        self.initialLayer = initialLayer
        self.stepwiseRNN = stepwiseRNN
        self.seedToQueryFct = seedToQueryFct
        self.queryToAttentionFct = queryToAttentionFct
        self.primitives_set = primitives_set
        self.classification_layer = classification_layer

    # X is size batch x input
    def __call__(self, X):
        batch_size = X.size()[0]
        state = self.initialLayer(X)
        seed = self.initial_seed.expand(batch_size, -1)
        for i in range(self.T):
            seed = self.stepwiseRNN(seed, state)
            query_vectors = self.seedToQueryFct(seed)
            attention_vectors = self.queryToAttentionFct(query_vectors)
            layer = self.primitives_set(attention_vectors)
            state = layer(state)

        output = self.classification_layer(state)
        return output


def BuildPolicyMLP(T, input_size, state_size, seed_size, number_of_primitives, output_size):
    initial_seed = torch.Zero((seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU(state_size, seed_size)
    seedToQueryFunction = ReluSeedToQueryFct(2, state_size, seed_size, number_of_primitives)
    queryToAttentionFunction = QueryToAttentionFct(seed_size, number_of_primitives)
    primitiveSets = MLP_Primitives(state_size, number_of_primitives)
    classification_layer = nn.Linear(state_size, output_size)

    return PolicyMLP(T, initial_seed, initial_layer, stepwiseRNN, seedToQueryFunction, queryToAttentionFunction, primitiveSets, classification_layer)









