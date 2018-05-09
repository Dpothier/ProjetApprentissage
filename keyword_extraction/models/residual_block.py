import torch.nn as nn
import torch.nn.functional as F
from models.spatial_dropout import SpatialDropout
from models.causal_conv import CausalConv1d

class ResidualCausalBlock(nn.Module):
    def __init__(self, in_depth, out_depth, dilation=1, p=0.0):
        super(ResidualCausalBlock, self).__init__()
        self.conv1 = CausalConv1d(in_depth, in_depth, kernel_size=3, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(in_depth)
        self.dropout1 = SpatialDropout(p)

        self.conv2 = CausalConv1d(in_depth, out_depth, kernel_size=3, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_depth)
        self.dropout2 = SpatialDropout(p)

        self.downsample = nn.Conv1d(in_depth, out_depth, 1) if in_depth != out_depth else None


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        residual = x

        out = self.dropout1(F.relu(self.bn1(self.conv1(x))))

        out = self.dropout2(F.relu(self.bn2(self.conv2(out))))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_depth, out_depth, dilation=1, p=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_depth, in_depth, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn1 = nn.BatchNorm1d(in_depth)
        self.dropout1 = SpatialDropout(p)

        self.conv2 = nn.Conv1d(in_depth, out_depth, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn2 = nn.BatchNorm1d(out_depth)
        self.dropout2 = SpatialDropout(p)

        self.downsample = nn.Conv1d(in_depth, out_depth, 1) if in_depth != out_depth else None


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        residual = x

        out = self.dropout1(F.relu(self.bn1(self.conv1(x))))

        out = self.dropout2(F.relu(self.bn2(self.conv2(out))))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out