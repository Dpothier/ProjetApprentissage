import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from networks.static_hypernetwork.modules.layer_embeddings import LayerEmbedding
from networks.static_hypernetwork.modules.residual_block import ResidualBlock
from networks.static_hypernetwork.modules.policy import Policy


class PrimaryNetwork(nn.Module):

    def __init__(self, layer_emb_size=64, base_channel_count=16):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer_emb_size = layer_emb_size
        self.hope = Policy(z_dim=self.layer_emb_size)

        self.zs_size = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                        [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
                        [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size = [[base_channel_count, base_channel_count], [base_channel_count, base_channel_count],
                            [base_channel_count, base_channel_count], [base_channel_count, base_channel_count],
                            [base_channel_count, base_channel_count], [base_channel_count, base_channel_count],
                            [base_channel_count, 2 * base_channel_count], [2 * base_channel_count, 2 * base_channel_count],
                            [2 * base_channel_count, 2 * base_channel_count], [2 * base_channel_count, 2 * base_channel_count],
                            [2 * base_channel_count, 2 * base_channel_count], [2 * base_channel_count, 2 * base_channel_count],
                            [2 * base_channel_count, 4 * base_channel_count], [4 * base_channel_count, 4 * base_channel_count],
                            [4 * base_channel_count, 4 * base_channel_count], [4 * base_channel_count, 4 * base_channel_count],
                            [4 * base_channel_count, 4 * base_channel_count], [4 * base_channel_count, 4 * base_channel_count]]

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResidualBlock(self.filter_size[i][0], self.filter_size[i][1], downsample=down_sample))

        self.zs = nn.ModuleList()

        for i in range(36):
            self.zs.append(LayerEmbedding(self.zs_size[i], self.layer_emb_size))

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(64, 10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(18):
            # if i != 15 and i != 17:
            z_1 = self.zs[2*i]
            z_2 = self.zs[2*i + 1]
            w1 = self.hope(z_1)
            w2 = self.hope(z_2)

            # w1 = self.zs[2*i](self.hope)
            # w2 = self.zs[2*i+1](self.hope)
            x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1, 64))

        return x
