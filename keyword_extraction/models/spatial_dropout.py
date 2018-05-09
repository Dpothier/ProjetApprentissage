import torch.nn as nn
from torch.autograd import Variable

class SpatialDropout(nn.Module):

    def __init__(self, p):
        super(SpatialDropout, self).__init__()
        self.drop_probability = p

    def forward(self, x):
        number_of_channel = x.shape[2]
        mask = Variable(torch.bernoulli(torch.ones(number_of_channel) * 1 - self.drop_probability).expand(x.shape))
        if x.is_cuda:
            mask = mask.cuda()

        if self.training:
            return mask * x
        else:
            return x * (1 - self.drop_probability)