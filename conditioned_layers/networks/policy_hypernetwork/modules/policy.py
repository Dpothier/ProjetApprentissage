import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from networks.policy_hypernetwork.modules.stepwiseGRU import StepwiseGRU


class Policy(nn.Module):

    def __init__(self, channels=32, embedding_size=64, kernel_size = 3):
        super(Policy, self).__init__()
        self.channels = channels

        self.embedding_size = embedding_size

        self.kernel_size = kernel_size

        self.start_states = Parameter(torch.fmod(torch.randn(2,  self.embedding_size), 2), requires_grad=True)
        self.states = None

        self.state_update = StepwiseGRU(channels, embedding_size)

        self.layer_generator_weight = Parameter(torch.fmod(torch.randn((self.embedding_size, self.channels * self.kernel_size * self.kernel_size)), 2))
        self.layer_generator_bias = Parameter(torch.fmod(torch.randn((self.channels * self.kernel_size * self.kernel_size)), 2))

        self.kernel_embedding_generator_weight = Parameter(torch.fmod(torch.randn((self.embedding_size, self.channels * self.embedding_size)), 2))
        self.kernel_embedding_generator_bias = Parameter(torch.fmod(torch.randn((self.channels * self.embedding_size)), 2))

    def parse_kernel(self, z):
        batch_size = z.size()[0]
        h_in = torch.matmul(z, self.kernel_embedding_generator_weight) + self.kernel_embedding_generator_bias
        h_in = h_in.view(batch_size, self.channels, self.embedding_size)

        h_final = torch.matmul(h_in, self.layer_generator_weight) + self.layer_generator_bias
        kernel = h_final.view(batch_size, self.channels, self.channels, self.kernel_size, self.kernel_size)

        return kernel

    def forward(self, obs):
        self.states = self.state_update(obs, self.states)

        w1 = self.parse_kernel((self.states[:, 0, :]))
        w2 = self.parse_kernel((self.states[:, 1, :]))

        return w1, w2

    def init_states(self, batch_size):
        self.states = self.start_states.expand(batch_size, -1, -1)






