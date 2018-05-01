import torch
import torch.nn as nn
import torch.nn.functional as F
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



class ResidualBlock(nn.Module):
    def __init__(self, in_depth, out_depth, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_depth, in_depth, kernel_size=3, dilation=dilation, padding=dilation)
        self.bn1 = nn.BatchNorm1d(in_depth)

        self.conv2 = nn.Conv1d(in_depth, out_depth, kernel_size=3, dilation=2*dilation, padding=2*dilation)
        self.bn2 = nn.BatchNorm1d(out_depth)

        self.downsample = nn.Conv1d(in_depth, out_depth, kernel_size=1)

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class TCN(nn.Module):

    def __init__(self, embedding_vectors):
        super(TCN, self).__init__()
        vocabulary_size = embedding_vectors.shape[0]
        embedding_size = embedding_vectors.shape[1]
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.embeddings.weight.data.copy_(embedding_vectors)

        self.res1 = ResidualBlock(embedding_size, embedding_size, dilation=1)
        self.res2 = ResidualBlock(embedding_size, embedding_size, dilation=4)
        self.res3 = ResidualBlock(embedding_size, embedding_size, dilation=16)
        self.res4 = ResidualBlock(embedding_size, embedding_size, dilation=64)
        self.process = ResidualBlock(embedding_size, 5, dilation=64)
        self.material = ResidualBlock(embedding_size, 5, dilation=64)
        self.task = ResidualBlock(embedding_size, 5, dilation=64)
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
        out_material = F.log_softmax(self.material.forward(x), dim=1)
        out_task = F.log_softmax(self.task.forward(x), dim=1)

        return out_process, out_material, out_task


class TCN_simple(nn.Module):

    def __init__(self, embedding_vectors):
        super(TCN_simple, self).__init__()
        vocabulary_size = embedding_vectors.shape[0]
        embedding_size = embedding_vectors.shape[1]
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.embeddings.weight.data.copy_(embedding_vectors)


        self.conv1 = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, dilation=1, padding=1)
        self.conv2 = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(embedding_size, embedding_size, kernel_size=3, dilation=4, padding=4)
        self.convProcess = nn.Conv1d(embedding_size, 5, kernel_size=3, dilation=1, padding=1)
        self.convMaterial = nn.Conv1d(embedding_size, 5, kernel_size=3, dilation=1, padding=1)
        self.convTask = nn.Conv1d(embedding_size, 5, kernel_size=3, dilation=1, padding=1)
        pass

    def forward(self, x):

        x = self.embeddings(x).permute(1,2,0)


        out = F.relu(self.conv1.forward(x))

        out = F.relu(self.conv2.forward(out))

        out = F.relu(self.conv3.forward(out))

        out_process = F.log_softmax(self.convProcess.forward(out), dim=1)
        out_material = F.log_softmax(self.convMaterial.forward(x), dim=1)
        out_task = F.log_softmax(self.convMaterial.forward(x), dim=1)

        return out_process, out_material, out_task