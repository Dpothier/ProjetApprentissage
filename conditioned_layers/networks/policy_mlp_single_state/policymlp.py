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
# The output of each layer is the new observation of size obs_size. At t=0, the state != input and is blank vector
# Every layer should have obs_size as in_size and out_size. If obs_size != input_size, an input_layer is trained to produce a correctly-sized first hidden state
# The hidden state is the input of the state-update function, implemented by RNN. The state-update function outputs the new state of size batch_size x state_size
# The state is fed to the policy function. The policy output the next hidden layer tensor of size batch x obs_size x obs_size.
# The hidden layer is applied to the current hidden state, producing a new hidden state tensor of size batch x obs_size
# The new hidden state is the next observation of the state_update function.

class PolicyMLP_single_state(nn.Module):
    # T is the number of recurrent computation steps
    # InitialLayer is a Linear layer of size batch x state x input
    # StepwiseRNN is an RNN computing one time step at a time have inputs of size batch x state and outputs of size batch x seed
    def __init__(self, T, initial_seed, initialLayer, state_update, policy, classification_layer, layer_constructor):
        super().__init__()
        self.T = T
        self.initial_state = Parameter(initial_seed, requires_grad=True)
        self.initialLayer = initialLayer
        self.state_update = state_update
        self.policy = policy
        self.classification_layer = classification_layer
        self.layer_constructor = layer_constructor

    # X is size batch x input
    def __call__(self, X):
        batch_size = X.size()[0]
        X = X.view(-1, 784)
        hidden = torch.relu(self.initialLayer(X))
        state = self.initial_state.expand(batch_size, -1)
        for i in range(self.T):
            state = self.state_update(hidden, state)
            layer_weights = torch.tanh(self.policy(state))
            layer = self.layer_constructor(layer_weights)
            hidden = torch.relu(layer(hidden))

        output = self.classification_layer(state)
        return output


def Build_PolicyMLP_single_state(T, input_size, hidden_size, state_size, output_size):
    initial_seed = torch.zeros((state_size))
    initial_layer = nn.Linear(input_size, hidden_size)
    state_update = StepwiseGRU_layer_generator(hidden_size, state_size)
    policy = nn.Linear(state_size, hidden_size * hidden_size)
    classification_layer = nn.Linear(hidden_size, output_size)


    torch.nn.init.kaiming_normal_(initial_seed)
    torch.nn.init.kaiming_normal_(initial_layer.weight)
    torch.nn.init.kaiming_normal_(classification_layer.weight)

    return PolicyMLP_single_state(T=T, initial_seed=initial_seed, initialLayer=initial_layer,
                                  state_update=state_update, policy=policy, classification_layer= classification_layer,
                                  layer_constructor=PolicyGeneratedResidualLinearLayer)








