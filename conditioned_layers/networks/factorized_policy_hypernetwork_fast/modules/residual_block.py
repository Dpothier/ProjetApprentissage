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
        batch_size, input_size, start_w_x, start_h_x = x.size()
        output_size1, w_k1, h_k1 = conv1_w.size()[1], conv1_w.size()[3], conv1_w.size()[4]
        output_size2, w_k2, h_k2 = conv2_w.size()[1], conv2_w.size()[3], conv2_w.size()[4]

        end_w_x = start_w_x // self.stride1
        end_h_x = start_h_x // self.stride1

        residual = self.reslayer(x)

        out = F.conv2d(x.view(1, batch_size * input_size, start_w_x, start_h_x),
                               conv1_w.view(batch_size * output_size1, input_size, w_k1, h_k1), stride=self.stride1, padding=1,
                               groups=batch_size).view((batch_size, output_size1, end_w_x, end_h_x))

        out = F.relu(self.bn1(out))

        out = F.conv2d(out.view(1, batch_size * input_size, end_w_x, end_h_x),
                               conv2_w.view(batch_size * output_size2, input_size, w_k2, h_k2), stride=1, padding=1,
                               groups=batch_size).view((batch_size, output_size2, end_w_x, end_h_x))

        out = self.bn2(out)

        out += residual

        out = F.relu(out)


        return out