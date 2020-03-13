import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Policy(nn.Module):

    def __init__(self, kernels_defined_per_embedding=16, channels_defined_per_embedding=16, start_kernel_embeddings=1, start_channel_embeddings=1, embedding_size=64, kernel_size = 3, z_dim = 64):
        super(Policy, self).__init__()
        self.kernels_defined_per_embedding = kernels_defined_per_embedding
        self.channels_defined_per_embedding = channels_defined_per_embedding

        self.embedding_size = embedding_size

        self.kernel_size = kernel_size

        self.first_convolution_embeddings = Parameter(torch.fmod(torch.randn(self.kernels_defined_per_embedding, self.channels_defined_per_embedding, self.embedding_size), 2), requires_grad=True)
        self.second_convolution_embeddings = Parameter(torch.fmod(torch.randn(self.kernels_defined_per_embedding, self.channels_defined_per_embedding, self.embedding_size), 2), requires_grad=True)

        self.layer_generator_bias = Parameter(torch.fmod(torch.randn((self.embedding_size, self.channels_defined_per_embedding * self.kernel_size * self.kernel_size)).cuda(), 2))
        self.layer_generator_bias = Parameter(torch.fmod(torch.randn((self.channels_defined_per_embedding * self.kernel_size * self.kernel_size)).cuda(), 2))

        self.kernel_embedding_generator_weight = Parameter(torch.fmod(torch.randn((self.embedding_size, self.kernels_defined_per_embedding * self.z_dim)).cuda(), 2))
        self.kernel_embedding_generator_bias = Parameter(torch.fmod(torch.randn((self.kernels_defined_per_embedding * self.embedding_size)).cuda(), 2))

    def parse_kernel(self, z):
        h_in = torch.matmul(z, self.kernel_embedding_generator_weight) + self.kernel_embedding_generator_bias
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.layer_generator_bias) + self.layer_generator_bias
        kernel = h_final.view(self.kernels_defined_per_embedding, self.channels_defined_per_embedding, self.kernel_size, self.kernel_size)

        return kernel

    def forward(self, obs, condensing_layer):
        z_list, self.h = self.state_update(obs, self.h)
        h = no_output_embs
        k = self.z_dim
        ww = []
        for i in range(h):
            w = []
            for j in range(k):
                w.append(self.parse_kernel(z_list[i * k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)





