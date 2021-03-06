import torch.nn as nn
import torch.nn.functional as F
import time


class IdentityLayer(nn.Module):

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_size=16, out_size=16, downsample = False):
        super(ResidualBlock,self).__init__()
        self.out_size = out_size
        self.in_size = in_size
        if downsample:
            self.stride1 = 2
            self.reslayer = nn.Conv2d(in_channels=self.in_size, out_channels=self.out_size, stride=2, kernel_size=1)
        else:
            self.stride1 = 1
            self.reslayer = IdentityLayer()

        self.bn1 = nn.BatchNorm2d(out_size)
        self.bn2 = nn.BatchNorm2d(out_size)

    def forward(self, x, conv1_w, conv2_w):
        residual = self.reslayer(x)

        out = F.conv2d(x, conv1_w, stride=self.stride1, padding=1)

        out = F.relu(self.bn1(out), inplace=True)

        out = F.conv2d(out, conv2_w, padding=1)

        out = self.bn2(out)


        out += residual

        out = F.relu(out)

        return out