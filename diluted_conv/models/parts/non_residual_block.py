import torch.nn as nn
import torch.nn.functional as F
from diluted_conv.models.parts.spatial_dropout import SpatialDropout
from diluted_conv.models.parts.causal_conv import CausalConv1d


class NonResidualBlock(nn.Module):
    def __init__(self, in_depth, out_depth, dilation=1, p=0.0):
        super(NonResidualBlock, self).__init__()
        self.conv = nn.Conv1d(in_depth, in_depth, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn = nn.BatchNorm1d(in_depth)
        self.dropout = SpatialDropout(p)


    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)


    def forward(self, x):
        out = self.dropout(F.relu(self.bn(self.conv(x))))

        return out