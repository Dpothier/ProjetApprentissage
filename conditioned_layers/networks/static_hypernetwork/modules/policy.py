import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class Policy(nn.Module):

    def __init__(self, f_size = 3, layer_emb_size=64, out_size=16, in_size=16):
        super(Policy, self).__init__()
        self.layer_emb_size = layer_emb_size
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        self.w1 = Parameter(torch.fmod(torch.randn((self.layer_emb_size, self.out_size * self.f_size * self.f_size)).cuda(), 2))
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)).cuda(),2))

        self.w2 = Parameter(torch.fmod(torch.randn((self.layer_emb_size, self.in_size * self.layer_emb_size)).cuda(), 2))
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.layer_emb_size)).cuda(), 2))

    def parse_kernel(self, z):
        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.layer_emb_size)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel

    def forward(self, zs):
        z_list = zs.z_list
        h, k = zs.z_num
        ww = []
        for i in range(h):
            w = []
            for j in range(k):
                w.append(self.parse_kernel(z_list[i * k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)





