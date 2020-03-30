import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.parameter import Parameter

torch.cuda.set_device(0)
batch_size = 128
embedding_size = 64
channels_count = 32
embedding_factor_count = 2
channels_factor_count = 2

w_x = 32
h_x = 32

embedding_factors_size = embedding_size // embedding_factor_count  # if size is not divisible by count, size will be reduced to match
embedding_size = embedding_factors_size * embedding_factor_count

channels_factor_size = channels_count // channels_factor_count
channels = channels_factor_size * channels_factor_count

states = Parameter(torch.fmod(
            torch.randn(batch_size, 2, channels_factor_count ** 2, embedding_factor_count, embedding_factors_size),
            2), requires_grad=True).cuda()

base_image = torch.randn((batch_size, channels_count, w_x, h_x)).cuda()
obs = F.adaptive_avg_pool2d(base_image, (1, 1)).squeeze(3).squeeze(2)

number_of_states_to_update, channels_factor_count, embedding_factors_count  = states.size()[1], states.size()[2], states.size()[3]
obs = obs.unsqueeze(1).expand(-1, number_of_states_to_update, -1)
obs = obs.unsqueeze(2).expand(-1, -1, channels_factor_count, -1)
obs = obs.unsqueeze(3).expand(-1, -1, -1, embedding_factors_count, -1)
obs = obs.contiguous()

obs = obs.view(batch_size * number_of_states_to_update * channels_factor_count * embedding_factors_count, -1)
states = states.view(batch_size * number_of_states_to_update * channels_factor_count * embedding_factor_count, embedding_factors_size)

cell = nn.GRUCell(input_size=channels_count, hidden_size=embedding_factors_size, bias=True).cuda()

out = cell(obs, states)

out = out.view(batch_size, number_of_states_to_update, channels_factor_count, embedding_factor_count, embedding_factors_size)

print("done")


