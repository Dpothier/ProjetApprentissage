import torch
import torch.nn as nn
from networks.policy_mlp.submodules.Primitives import MLP_Primitives
from networks.policy_mlp.submodules.Primitives import MLP_PrimitivesPreSet
from networks.policy_mlp.submodules.Primitives import MLP_NoPrimitives
from networks.policy_mlp.submodules.stepwiseGRU import StepwiseGRU
from networks.policy_mlp.submodules.stepwiseGRU import StepwiseGRU_layer_generator
from networks.policy_mlp.submodules.SeedToQueryFct import ReluSeedToQueryFct
from networks.policy_mlp.submodules.QueryToAttention import QueryToAttentionFct
from networks.policy_mlp.submodules.QueryToAttention import NoAttentionFct
from networks.policy_mlp.submodules.LinearLayers import PolicyGeneratedLinearLayer
from networks.policy_mlp.submodules.LinearLayers import PolicyGeneratedResidualLinearLayer
from networks.policy_mlp.submodules.LinearLayers import Multi_Linear_layer
from networks.policy_mlp.submodules.LinearLayers import Multi_Linear_Residual_layer
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


class SeedToAttentionPolicyMLP(nn.Module):
    # T is the number of recurrent computation steps
    # InitialLayer is a Linear layer of size batch x state x input
    # StepwiseRNN is an RNN computing one time step at a time have inputs of size batch x state and outputs of size batch x seed
    def __init__(self, T, initial_seed, initialLayer, stepwiseRNN, seedToQueryFct, queryToAttentionFct,
                 classification_layer, primitives_set):
        super().__init__()
        self.T = T
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
        X = X.view(-1, 784)
        state = torch.relu(self.initialLayer(X))
        seed = self.initial_seed.expand(batch_size, -1)
        for i in range(self.T):
            seed = self.stepwiseRNN(seed, state)
            query_vectors = self.seedToQueryFct(seed)
            attention_vectors = self.queryToAttentionFct(query_vectors)
            layer = self.primitives_set(attention_vectors)
            state = torch.relu(layer(state))

        output = self.classification_layer(state)
        return output





class PolicyMLP(nn.Module):
    # T is the number of recurrent computation steps
    # InitialLayer is a Linear layer of size batch x state x input
    # StepwiseRNN is an RNN computing one time step at a time have inputs of size batch x state and outputs of size batch x seed
    def __init__(self, T, initial_seed, initialLayer, stepwiseRNN, seedToQueryFct, queryToAttentionFct, classification_layer, primitives_set):
        super().__init__()
        self.T = T
        self.initial_seed = Parameter(initial_seed, requires_grad=True)
        self.initialLayer = initialLayer
        self.stepwiseRNN = stepwiseRNN
        self.seedToQueryFct = seedToQueryFct
        self.queryToAttentionFct = queryToAttentionFct
        self.primitives_set = primitives_set
        self.classification_layer = classification_layer


        self.add_module("initialLayer", initialLayer)
        self.add_module("stepwiseRNN", stepwiseRNN)
        self.add_module("seedToQueryFct", seedToQueryFct)
        self.add_module("queryToAttentionFct", queryToAttentionFct)
        self.add_module("primitives_set", primitives_set)
        self.add_module("classification_layer", classification_layer)

        self.layer = None

    # X is size batch x input
    def __call__(self, X):
        batch_size = X.size()[0]
        X = X.view(-1, 784)
        self.state0 = torch.relu(self.initialLayer(X))
        self.seed0 = self.initial_seed.expand(batch_size, -1)
        for i in range(self.T):
            self.seed1 = self.stepwiseRNN(self.state0, self.seed0)
            self.query_vectors = self.seedToQueryFct(self.seed1)
            self.attention_vectors = self.queryToAttentionFct(self.query_vectors)
            self.layer = self.primitives_set(self.attention_vectors)
            self.state1 = torch.relu(self.layer(self.state0))

        output = self.classification_layer(self.state1)
        return output

class PolicyMLPWithOnlyRNN(nn.Module):
    # T is the number of recurrent computation steps
    # InitialLayer is a Linear layer of size batch x state x input
    # StepwiseRNN is an RNN computing one time step at a time have inputs of size batch x state and outputs of size batch x seed
    def __init__(self, T, initial_seed, initialLayer, stepwiseRNN, classification_layer, layer_constructor):
        super().__init__()
        self.T = T
        self.initial_seed = Parameter(initial_seed, requires_grad=True)
        self.initialLayer = initialLayer
        self.stepwiseRNN = stepwiseRNN
        self.classification_layer = classification_layer
        self.layer_constructor = layer_constructor

    # X is size batch x input
    def __call__(self, X):
        batch_size = X.size()[0]
        X = X.view(-1, 784)
        state = torch.relu(self.initialLayer(X))
        seed = self.initial_seed.expand(batch_size, -1, -1)
        for i in range(self.T):
            seed = self.stepwiseRNN(state, seed)
            layer = self.layer_constructor(torch.tanh(seed, dim=2))
            state = torch.relu(layer(state))

        output = self.classification_layer(state)
        return output


def BuildPolicyMLP(T, input_size, state_size, seed_size, query_size, number_of_primitives, output_size):
    initial_seed = torch.zeros((seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU(state_size, seed_size)
    seedToQueryFunction = ReluSeedToQueryFct(2, state_size, seed_size, query_size, Multi_Linear_layer)
    queryToAttentionFunction = QueryToAttentionFct(query_size, number_of_primitives)
    primitiveSets = MLP_Primitives(state_size, number_of_primitives, PolicyGeneratedLinearLayer)
    classification_layer = nn.Linear(state_size, output_size)

    return PolicyMLP(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, seedToQueryFct=seedToQueryFunction, queryToAttentionFct=queryToAttentionFunction,
                     classification_layer= classification_layer, primitives_set=primitiveSets)

def BuildResidualPolicyMLP(T, input_size, state_size, seed_size, query_size, number_of_primitives, output_size):
    initial_seed = torch.zeros((seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU(state_size, seed_size)
    seedToQueryFunction = ReluSeedToQueryFct(2, state_size, seed_size, query_size, Multi_Linear_layer)
    queryToAttentionFunction = QueryToAttentionFct(query_size, number_of_primitives)
    primitiveSets = MLP_Primitives(state_size, number_of_primitives, PolicyGeneratedResidualLinearLayer)
    classification_layer = nn.Linear(state_size, output_size)

    return PolicyMLP(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, seedToQueryFct=seedToQueryFunction, queryToAttentionFct=queryToAttentionFunction,
                     classification_layer= classification_layer, primitives_set=primitiveSets)

def BuildFullResidualPolicyMLP(T, input_size, state_size, seed_size, query_size, number_of_primitives, output_size):
    initial_seed = torch.zeros((seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU(state_size, seed_size)
    seedToQueryFunction = ReluSeedToQueryFct(2, state_size, seed_size, query_size, Multi_Linear_Residual_layer)
    queryToAttentionFunction = QueryToAttentionFct(query_size, number_of_primitives)
    primitiveSets = MLP_Primitives(state_size, number_of_primitives, PolicyGeneratedResidualLinearLayer)
    classification_layer = nn.Linear(state_size, output_size)

    return PolicyMLP(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, seedToQueryFct=seedToQueryFunction, queryToAttentionFct=queryToAttentionFunction,
                     classification_layer= classification_layer, primitives_set=primitiveSets)

def BuildFullResidualPolicyMLPWithPresetPrimitives(T, input_size, state_size, seed_size, output_size):
    initial_seed = torch.zeros((seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU(state_size, seed_size)
    seedToQueryFunction = ReluSeedToQueryFct(2, state_size, seed_size, state_size, Multi_Linear_Residual_layer)
    queryToAttentionFunction = QueryToAttentionFct(state_size, state_size)
    primitiveSets = MLP_PrimitivesPreSet(state_size, PolicyGeneratedResidualLinearLayer)
    classification_layer = nn.Linear(state_size, output_size)

    return PolicyMLP(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, seedToQueryFct=seedToQueryFunction, queryToAttentionFct=queryToAttentionFunction,
                     classification_layer= classification_layer, primitives_set=primitiveSets)

def BuildFullResidualPolicyMLPNoPrimitives(T, input_size, state_size, seed_size, query_size, number_of_primitives, output_size):
    initial_seed = torch.zeros((seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU(state_size, seed_size)
    seedToQueryFunction = ReluSeedToQueryFct(2, state_size, seed_size, query_size, Multi_Linear_Residual_layer)
    queryToAttentionFunction = QueryToAttentionFct(query_size, number_of_primitives)
    primitiveSets = MLP_NoPrimitives(PolicyGeneratedResidualLinearLayer)
    classification_layer = nn.Linear(state_size, output_size)

    return PolicyMLP(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, seedToQueryFct=seedToQueryFunction, queryToAttentionFct=queryToAttentionFunction,
                     classification_layer= classification_layer, primitives_set=primitiveSets)

def BuildFullResidualPolicyMLPNoAttention(T, input_size, state_size, seed_size, query_size, number_of_primitives, output_size):
    initial_seed = torch.zeros((seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU(state_size, seed_size)
    seedToQueryFunction = ReluSeedToQueryFct(2, state_size, seed_size, query_size, Multi_Linear_Residual_layer)
    queryToAttentionFunction = NoAttentionFct()
    primitiveSets = MLP_NoPrimitives(PolicyGeneratedResidualLinearLayer)
    classification_layer = nn.Linear(state_size, output_size)

    return PolicyMLP(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, seedToQueryFct=seedToQueryFunction, queryToAttentionFct=queryToAttentionFunction,
                     classification_layer= classification_layer, primitives_set=primitiveSets)

def BuildFullResidualPolicyMLPOnlyGru(T, input_size, state_size, seed_size, query_size, number_of_primitives, output_size):
    initial_seed = torch.zeros((state_size, seed_size))
    initial_layer = nn.Linear(input_size, state_size)
    stepwiseRNN = StepwiseGRU_layer_generator(state_size, seed_size)
    classification_layer = nn.Linear(state_size, output_size)


    torch.nn.init.kaiming_normal_(initial_seed)
    torch.nn.init.kaiming_normal_(initial_layer.weight)
    torch.nn.init.kaiming_normal_(classification_layer.weight)

    return PolicyMLPWithOnlyRNN(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, classification_layer= classification_layer,
                                layer_constructor=PolicyGeneratedResidualLinearLayer)








