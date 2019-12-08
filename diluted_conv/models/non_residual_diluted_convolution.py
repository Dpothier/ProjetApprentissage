import torch.nn as nn
import torch.nn.functional as F
from diluted_conv.models.parts.non_residual_block import NonResidualBlock

class NonResidualDilutedConv(nn.Module):

    def __init__(self, embedding_vectors, p_first_layer=0.2, p_other_layers=0.5, dilution_function=lambda kernel, depth: 1, depth=8):
        super(NonResidualDilutedConv, self).__init__()
        vocabulary_size = embedding_vectors.shape[0]
        embedding_size = embedding_vectors.shape[1]
        self.embeddings = nn.Embedding(vocabulary_size, embedding_size)
        self.embeddings.weight.data.copy_(embedding_vectors)


        self.res_blocks = nn.ModuleList()
        for i in range(depth):
            p = p_first_layer if i == 0 else p_other_layers
            self.res_blocks.append(NonResidualBlock(embedding_size, embedding_size, dilation=dilution_function(3, i), p=p))

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.last_layer = nn.Linear(in_features=embedding_size, out_features=2)
        pass

    def forward(self, x):
        x = x.long()
        out = self.embeddings(x)
        out = out.permute(0,2,1)

        for res_block in self.res_blocks:
            out = F.relu(res_block.forward(, out)

        out = self.global_avg_pool(out).squeeze()
        out = self.last_layer(out)

        return out