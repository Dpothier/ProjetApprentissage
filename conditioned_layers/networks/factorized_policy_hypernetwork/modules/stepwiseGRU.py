import torch
import torch.nn as nn

# Implements a GRU, but computes a single step at a time. Parameter are injected
class Paramerized_StepwiseGRU(nn.Module):
    def __init__(self, update_mem, update_input, reset_mem, reset_input, candidate_mem, candidate_input):
        super().__init__()
        self.update_mem = update_mem
        self.update_input = update_input
        self.reset_mem = reset_mem
        self.reset_input = reset_input
        self.candidate_mem = candidate_mem
        self.candidate_input = candidate_input

        self.add_module("update_mem", update_mem)
        self.add_module("update_input", update_input)
        self.add_module("reset_mem", reset_mem)
        self.add_module("reset_input", reset_input)
        self.add_module("candidate_mem", candidate_mem)
        self.add_module("candidate_input", candidate_input)

    # x is batch x obs_size + pos_emb_size, h is batch x emb_kernels x emb_channels x emb_size
    # Output is batch x state_size x seed_size, it produces one seed for every unit of the layer for every instance
    def __call__(self, x, h):
        number_of_states_to_update, channels_factor_count, embedding_factors_count  = h.size()[1], h.size()[2], h.size()[3]
        x = x.unsqueeze(1).expand(-1, number_of_states_to_update, -1)
        x = x.unsqueeze(2).expand(-1, -1, channels_factor_count, -1)
        x = x.unsqueeze(3).expand(-1, -1, -1, embedding_factors_count, -1)

        z = torch.sigmoid(self.update_input(x) + self.update_mem(h))
        r = torch.sigmoid(self.reset_input(x) + self.reset_mem(h))
        candidate = torch.tanh(self.candidate_input(x) + r * self.candidate_mem(h))
        output_1 = z * h
        output_2 = (1 - z) * candidate
        output = output_1 + output_2
        return output


class StepwiseGRU(Paramerized_StepwiseGRU):
    def __init__(self, x_size, h_size):
        update_mem = nn.Linear(h_size, h_size)
        update_input = nn.Linear(x_size, h_size)
        reset_mem = nn.Linear(h_size, h_size)
        reset_input = nn.Linear(x_size, h_size)
        candidate_mem = nn.Linear(h_size, h_size)
        candidate_input = nn.Linear(x_size, h_size)

        torch.nn.init.kaiming_normal_(update_mem.weight)
        torch.nn.init.kaiming_normal_(update_input.weight)
        torch.nn.init.kaiming_normal_(reset_mem.weight)
        torch.nn.init.kaiming_normal_(reset_input.weight)
        torch.nn.init.kaiming_normal_(candidate_mem.weight)
        torch.nn.init.kaiming_normal_(candidate_input.weight)

        super().__init__(update_mem, update_input, reset_mem, reset_input, candidate_mem, candidate_input)