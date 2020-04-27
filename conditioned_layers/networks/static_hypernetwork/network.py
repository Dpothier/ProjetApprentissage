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
        self.layer_emb_size = layer_emb_size
        self.base_channel_count = base_channel_count

        self.conv1 = nn.Conv2d(3, base_channel_count, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channel_count)

        self.layer_emb_size = layer_emb_size
        self.Policy = Policy(layer_emb_size=self.layer_emb_size, out_size=base_channel_count, in_size=base_channel_count)

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
        self.final = nn.Linear(4 * base_channel_count, 10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        for i in range(18):
            # if i != 15 and i != 17:
            z_1 = self.zs[2*i]
            z_2 = self.zs[2*i + 1]
            w1 = self.Policy(z_1)
            w2 = self.Policy(z_2)

            x = self.res_net[i](x, w1, w2)

        x = self.global_avg(x)
        x = self.final(x.view(-1, 4 * self.base_channel_count))

        return x
