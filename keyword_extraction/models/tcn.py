import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.residual_block import ResidualBlock
from models.residual_block import ResidualCausalBlock




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




class TCN(nn.Module):

    def __init__(self, embedding_vectors, p_first_layer=0.2, p_other_layers=0.5):
        super(TCN, self).__init__()
        vocabulary_size = embedding_vectors.shape[0]
        embedding_size = embedding_vectors.shape[1]
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.embeddings.weight.data.copy_(embedding_vectors)

        self.res1 = ResidualBlock(embedding_size, embedding_size, dilation=1, p=p_first_layer)
        self.res2 = ResidualBlock(embedding_size, embedding_size, dilation=2, p=p_other_layers)
        self.res3 = ResidualBlock(embedding_size, embedding_size, dilation=4, p=p_other_layers)
        self.res4 = ResidualBlock(embedding_size, embedding_size, dilation=8, p=p_other_layers)
        self.res5 = ResidualBlock(embedding_size, embedding_size, dilation=16, p=p_other_layers)
        self.res6 = ResidualBlock(embedding_size, embedding_size, dilation=32, p=p_other_layers)
        self.res7 = ResidualBlock(embedding_size, embedding_size, dilation=64, p=p_other_layers)
        self.process = ResidualBlock(embedding_size, 2, dilation=128, p=p_other_layers)
        self.material = ResidualBlock(embedding_size, 2, dilation=128, p=p_other_layers)
        self.task = ResidualBlock(embedding_size, 2, dilation=128, p=p_other_layers)
        pass

    def forward(self, x):
        x = self.embeddings(x).permute(1,2,0)

        out = F.relu(self.res1.forward(x))
        out = F.relu(self.res2.forward(out))
        out = F.relu(self.res3.forward(out))
        out = F.relu(self.res4.forward(out))
        out = F.relu(self.res5.forward(out))
        out = F.relu(self.res6.forward(out))
        out = F.relu(self.res7.forward(out))

        out_process = F.log_softmax(self.process.forward(out), dim=1)
        out_material = F.log_softmax(self.material.forward(out), dim=1)
        out_task = F.log_softmax(self.task.forward(out), dim=1)

        return out_process.permute(0,2,1), out_material.permute(0,2,1), out_task.permute(0,2,1)