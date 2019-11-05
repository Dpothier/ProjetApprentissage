import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.policy_cnn.submodules.stepwiseGRU import StepwiseGRU_layer_generator
from networks.policy_cnn.submodules.generator import ConvLayerGenerator
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

class PolicyCNN(nn.Module):
    # T is the number of recurrent computation steps
    # InitialLayer is a Linear layer of size batch x state x input
    # StepwiseRNN is an RNN computing one time step at a time have inputs of size batch x state and outputs of size batch x seed
    def __init__(self, T, initial_seed, initialLayer, stepwiseRNN, classification_layer, layer_generator):
        super().__init__()
        self.T = T
        self.initial_seed = Parameter(initial_seed, requires_grad=True)
        self.initialLayer = initialLayer
        self.stepwiseRNN = stepwiseRNN
        self.classification_layer = classification_layer
        self.layer_generator = layer_generator

    # X is size batch x input
    def __call__(self, X):
        batch_size = X.size()[0]
        output = torch.relu(self.initialLayer(X))
        state = F.adaptive_avg_pool2d(output, (1,1)).squeeze(3).squeeze(2)
        seed = self.initial_seed.expand(batch_size, -1, -1)
        for i in range(self.T):
            seed = self.stepwiseRNN(state, seed)
            layer = self.layer_generator(seed)
            output = torch.relu(layer(output))
            state = F.adaptive_avg_pool2d(output, (1, 1)).squeeze(3).squeeze(2)

        output = self.classification_layer(state)
        return output

def BuildPolicyCNN(T, in_channels, state_channels, seed_size, output_size):
    initial_seed = torch.zeros((state_channels, seed_size))
    initial_layer = nn.Conv2d(in_channels=in_channels, out_channels=state_channels, kernel_size=1)
    stepwiseRNN = StepwiseGRU_layer_generator(state_channels, seed_size)
    layer_generator = ConvLayerGenerator(kernel_size=3, state_channels=state_channels, seed_size=seed_size)
    classification_layer = nn.Linear(state_channels, output_size)


    torch.nn.init.kaiming_normal_(initial_seed)
    torch.nn.init.kaiming_normal_(initial_layer.weight)
    torch.nn.init.kaiming_normal_(classification_layer.weight)

    return PolicyCNN(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                     stepwiseRNN=stepwiseRNN, classification_layer= classification_layer,
                     layer_generator=layer_generator)








