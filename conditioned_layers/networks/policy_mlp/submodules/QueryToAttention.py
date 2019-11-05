import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class InjectedQueryToAttentionFct(nn.Module):
    # Keys are query_size x no_primitives
    def __init__(self, keys):
        super().__init__()
        self.keys = Parameter(keys)

    # Query is batch x output x query_size
    def __call__(self, query):
        batch_size = query.size()[0]
        output_size = query.size()[1]

        working_keys = self.keys.expand(batch_size, -1, -1) # output x query_size x no_primitives

        output = query.matmul(working_keys)
        output = torch.softmax(output, dim=2)
        return output

class QueryToAttentionFct(InjectedQueryToAttentionFct):
    def __init__(self, query_size, no_primitives):
        keys = torch.zeros((query_size, no_primitives))
        torch.nn.init.kaiming_normal_(keys)
        super().__init__(keys)


class NoAttentionFct(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, input):
        return torch.softmax(input, dim=2)