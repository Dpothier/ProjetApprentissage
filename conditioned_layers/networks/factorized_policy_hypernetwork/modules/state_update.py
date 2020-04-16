import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

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

class StateUpdateLSTM(nn.Module):
    def __init__(self, layer_count, channels_factor_size, embedding_factors_size, channels_factors_count, embedding_factors_count, bias=True):
        super().__init__()
        self.layer_count = layer_count
        self.channels_factors_size = channels_factor_size
        self.embedding_factors_size = embedding_factors_size
        self.channels_factors_count = channels_factors_count
        self.embedding_factors_count = embedding_factors_count
        self.states_to_update = 2

        cells = []
        hiddens = []
        cs = []

        for i in range(layer_count):
            hiddens.append(Parameter(torch.fmod(
            torch.randn(self.states_to_update * channels_factors_count ** 2 * embedding_factors_count, embedding_factors_size),
            2), requires_grad=True))
            cs.append(Parameter(torch.fmod(
                torch.randn(self.states_to_update * channels_factors_count ** 2 * embedding_factors_count,
                            embedding_factors_size),
                2), requires_grad=True))
            cells.append(nn.GRUCell(input_size=channels_factor_size, hidden_size=embedding_factors_size, bias=bias))

        self.base_hiddens = nn.ParameterList(hiddens)
        self.base_cs = nn.ParameterList(cs)
        self.cells = nn.ModuleList(cells)

        self.current_hiddens = None
        self.current_cs = None
        self.batch_size = None

    def init_state(self, batch_size):
        self.batch_size = batch_size

        self.current_hiddens = []
        for hidden in self.base_hiddens:
            self.current_hiddens.append(hidden.repeat((batch_size, 1)))

        for c in self.base_cs:
            self.current_cs.append(c.repeat((batch_size, 1)))

        pass

    def __call__(self, obs):
        obs = obs.repeat((self.states_to_update * self.channels_factors_count**2 * self.embedding_factors_count, 1))

        out = obs
        for i in range(self.layer_count):
            out, new_c = self.cells[i](out, self.current_hiddens[i], self.current_cs[i])
            self.current_hiddens[i] = out
            self.current_cs[i] = new_c


        out = out.view(self.batch_size, self.states_to_update, self.channels_factors_count**2, self.embedding_factors_count, self.embedding_factors_size)\
            .contiguous()
        return out


class StateUpdateGRU(nn.Module):

    def __init__(self, layer_count, channels_factor_size, embedding_factors_size, channels_factors_count, embedding_factors_count, bias=True):
        super().__init__()
        self.layer_count = layer_count
        self.channels_factors_size = channels_factor_size
        self.embedding_factors_size = embedding_factors_size
        self.channels_factors_count = channels_factors_count
        self.embedding_factors_count = embedding_factors_count
        self.states_to_update = 2

        cells = []
        hiddens = []

        for i in range(layer_count):
            hiddens.append(Parameter(torch.fmod(
            torch.randn(self.states_to_update * channels_factors_count ** 2 * embedding_factors_count, embedding_factors_size),
            2), requires_grad=True))
            cells.append(nn.GRUCell(input_size=channels_factor_size, hidden_size=embedding_factors_size, bias=bias))

        self.base_hiddens = nn.ParameterList(hiddens)
        self.cells = nn.ModuleList(cells)
        self.current_hiddens = None
        self.batch_size = None

    def init_state(self, batch_size):
        self.batch_size = batch_size

        self.current_hiddens = []
        for hidden in self.base_hiddens:
            self.current_hiddens.append(hidden.repeat((batch_size, 1)))

        pass

    def __call__(self, obs):
        obs = obs.repeat((self.states_to_update * self.channels_factors_count**2 * self.embedding_factors_count, 1))

        out = obs
        for i in range(self.layer_count):
            out = self.cells[i](out, self.current_hiddens[i])
            self.current_hiddens[i] = out


        out = out.view(self.batch_size, self.states_to_update, self.channels_factors_count**2, self.embedding_factors_count, self.embedding_factors_size)\
            .contiguous()
        return out


