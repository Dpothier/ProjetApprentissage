import torch.nn as nn
import torch.nn.functional as F
import torch
import time


class IdentityLayer(nn.Module):

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels=16, downsample = False):
        super(ResidualBlock,self).__init__()
        self.channels = channels

        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, conv1_w, conv2_w):
        residual = self.reslayer(x)

        out = torch.stack([
            F.conv2d(x[i, :, :, :].unsqueeze(0), conv1_w[i, :, :, :, :], stride=self.stride1, padding=1) for i, x_i in
            enumerate(torch.unbind(x, dim=0), 0)
        ], dim=0).squeeze(1)

        out = F.relu(self.bn1(out))

        out = torch.stack([
            F.conv2d(out[i, :, :, :].unsqueeze(0), conv2_w[i, :, :, :, :], padding=1) for i, x_i in
            enumerate(torch.unbind(x, dim=0), 0)
        ], dim=0).squeeze(1)

        out = self.bn2(out)
        
        out += residual

        out = F.relu(out)

        return out