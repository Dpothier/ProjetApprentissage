import torch
import torch.nn as nn
from torch.nn import Parameter

class LayerEmbedding(nn.Module):

    def __init__(self, z_num, z_dim):
        super(LayerEmbedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num
        self.z_dim = z_dim

        h, k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim).cuda(), 2), requires_grad=True))

    # def forward(self, hyper_net):
    #     ww = []
    #     h, k = self.z_num
    #     for i in range(h):
    #         w = []
    #         for j in range(k):
    #             w.append(hyper_net(self.z_list[i*k + j]))
    #         ww.append(torch.cat(w, dim=1))
    #     return torch.cat(ww, dim=0)
