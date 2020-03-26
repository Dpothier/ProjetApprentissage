import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import time

from networks.factorized_policy_hypernetwork_fast.modules.layer_embeddings import LayerEmbedding
from networks.factorized_policy_hypernetwork_fast.modules.residual_block import ResidualBlock
from networks.factorized_policy_hypernetwork_fast.modules.policy import Policy


class PrimaryNetwork(nn.Module):

    def __init__(self, z_dim=64, filter_size=32, embedding_factor_count=1, channels_factor_count=1):
        super(PrimaryNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, filter_size, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filter_size)

        self.filter_size = filter_size

        self.z_dim = z_dim
        self.policy = Policy(channels=filter_size, embedding_size=z_dim,
                             embedding_factor_count=embedding_factor_count, channels_factor_count=channels_factor_count)

        self.res_net = nn.ModuleList()

        for i in range(18):
            down_sample = False
            if i > 5 and i % 6 == 0:
                down_sample = True
            self.res_net.append(ResidualBlock(self.filter_size, downsample=down_sample))

        self.zs = nn.ModuleList()

        for i in range(36):
            self.zs.append(LayerEmbedding([1, 1], self.z_dim))

        self.global_avg = nn.AvgPool2d(8)
        self.final = nn.Linear(self.filter_size, 10)

    def forward(self, x):
        batch_start = time.time()

        batch_size = x.size()[0]
        self.policy.init_states(batch_size)

        x = F.relu(self.bn1(self.conv1(x)))


        conv_start, conv_end = None, None
        policy_start, policy_end = None, None
        conv_time = 0
        policy_time = 0
        for i in range(18):
            obs = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(3).squeeze(2)
            policy_start = time.time()
            w1, w2 = self.policy(obs)
            policy_end = time.time()
            policy_time += policy_end - policy_start
            # if i != 15 and i != 17:
            # z_1 = self.zs[2*i]
            # z_2 = self.zs[2*i + 1]
            # w1 = self.hope(z_1)
            # w2 = self.hope(z_2)

            # w1 = self.zs[2*i](self.hope)
            # w2 = self.zs[2*i+1](self.hope)
            conv_start = time.time()
            x = self.res_net[i](x, w1, w2)
            conv_end = time.time()
            conv_time += conv_end - conv_start

        final_obs = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(3).squeeze(2)
        x = self.final(final_obs)

        batch_end = time.time()
        batch_time = batch_end - batch_start

        # print("Fast network")
        # print("\n Batch took {} ms".format(batch_time))
        # print("Conv took: {}, {}%".format(conv_time, conv_time/batch_time * 100))
        # print("Policy took: {}, {}%".format(policy_time, policy_time/batch_time * 100))


        return x
