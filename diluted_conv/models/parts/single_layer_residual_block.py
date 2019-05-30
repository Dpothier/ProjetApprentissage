import torch.nn as nn
import torch.nn.functional as F
from diluted_conv.models.parts.spatial_dropout import SpatialDropout
from diluted_conv.models.parts.causal_conv import CausalConv1d

class SingleLayerResidualBlock(nn.Module):
    def __init__(self, in_depth, out_depth, dilation=1, p=0.0):
        super(SingleLayerResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_depth, in_depth, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn1 = nn.BatchNorm1d(in_depth)
        self.dropout1 = SpatialDropout(p)

        self.downsample = nn.Conv1d(in_depth, out_depth, 1) if in_depth != out_depth else None


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        residual = x

        out = self.dropout1(F.relu(self.bn1(self.conv1(x))))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out