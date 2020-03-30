import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from networks.factorized_policy_hypernetwork.modules.state_update import StepwiseGRU, StateUpdateGRU, StateUpdateLSTM


class Policy(nn.Module):

    def __init__(self, channels=32, channels_factor_count=1, embedding_size=64, embedding_factor_count=1,
                 kernel_size=3):
        super(Policy, self).__init__()
        self.embedding_factor_count = embedding_factor_count
        self.channels_factor_count = channels_factor_count

        self.embedding_factors_size = embedding_size // embedding_factor_count  # if size is not divisible by count, size will be reduced to match
        self.embedding_size = self.embedding_factors_size * embedding_factor_count

        self.channels_factor_size = channels // channels_factor_count
        self.channels_size = self.channels_factor_size * channels_factor_count

        self.kernel_size = kernel_size

        self.start_states = Parameter(torch.fmod(
            torch.randn(2, self.channels_factor_count ** 2, self.embedding_factor_count, self.embedding_factors_size),
            2), requires_grad=True)
        self.states = None


        self.state_update = StateUpdateLSTM(self.channels_size, self.embedding_factors_size, self.channels_factor_count, self.embedding_factor_count)

        self.layer_generator_weight = Parameter(torch.fmod(
            torch.randn((self.embedding_size, self.channels_factor_size * self.kernel_size * self.kernel_size)), 2))
        self.layer_generator_bias = Parameter(
            torch.fmod(torch.randn((self.channels_factor_size * self.kernel_size * self.kernel_size)), 2))

        self.kernel_embedding_generator_weight = Parameter(
            torch.fmod(torch.randn((self.embedding_size, self.channels_factor_size * self.embedding_size)), 2))
        self.kernel_embedding_generator_bias = Parameter(
            torch.fmod(torch.randn((self.channels_factor_size * self.embedding_size)), 2))

    def parse_kernel(self, z):
        batch_size = z.size()[0]
        h_in = torch.matmul(z, self.kernel_embedding_generator_weight) + self.kernel_embedding_generator_bias
        h_in = h_in.view(batch_size, self.channels_factor_size, self.embedding_size)

        h_final = torch.matmul(h_in, self.layer_generator_weight) + self.layer_generator_bias
        kernel = h_final.view(batch_size, self.channels_factor_size, self.channels_factor_size, self.kernel_size, self.kernel_size)

        return kernel

    def compute_layer(self, z_list):
        ww = []
        for i in range(self.channels_factor_count):
            w = []
            for j in range(self.channels_factor_count):
                w.append(self.parse_kernel(z_list[:, i * self.channels_factor_count + j, :]))
            ww.append(torch.cat(w, dim=2))
        return torch.cat(ww, dim=1)

    def forward(self, obs):
        batch_size = obs.size()[0]

        self.states = self.state_update(obs, self.states)

        w1 = self.compute_layer(torch.reshape(self.states[:, 0, :, :, :],
                                              (batch_size, self.channels_factor_count ** 2, self.embedding_size)))

        w2 = self.compute_layer(torch.reshape(self.states[:, 1, :, :, :],
                                              (batch_size, self.channels_factor_count ** 2, self.embedding_size)))

        return w1, w2

    def init_states(self, batch_size):
        self.states = self.start_states.expand(batch_size, -1, -1, -1, -1)
        self.state_update.init_state(batch_size)
