''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conditioned_layers.training.multi_linear import MultiLinear
from conditioned_layers.networks.conditioned_transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_state, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_state = d_state

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)
        self.state_layer_norm = nn.LayerNorm(d_state)
        self.dropout = nn.Dropout(dropout)

        self.fc = GeneratedFullyConnectedLayer(d_state, n_head * d_v, d_model)
        self.fc_state = GeneratedFullyConnectedLayer(d_state, n_head * d_v, d_state)

        self.qs = GeneratedFullyConnectedLayer(d_state, d_model, n_head * d_k)
        self.ks = GeneratedFullyConnectedLayer(d_state, d_model, n_head * d_k)
        self.vs = GeneratedFullyConnectedLayer(d_state, d_model, n_head * d_v)

        self.seq_q = nn.Linear(d_state, n_head * d_k)
        nn.init.normal_(self.seq_q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

    def forward(self, q, k, v, state, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q


        q = self.qs(q, state).view(sz_b, len_q, n_head, d_k)
        k = self.ks(k, state).view(sz_b, len_k, n_head, d_k)
        v = self.vs(v, state).view(sz_b, len_v, n_head, d_v)

        state_queries = self.seq_q(state).unsqueeze(1).view(sz_b, 1, self.n_head, self.d_k)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        state_queries = state_queries.permute(2, 0, 1, 3).contiguous().view(-1, 1, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)
        state_update, _ = self.attention(state_queries, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        state_update = state_update.view(n_head, sz_b, 1, d_v)
        state_update = state_update.permute(1, 2, 0, 3).contiguous().view(sz_b, 1, -1)  # b x lq x (n*dv)

        output = self.fc(output, state)
        output = self.layer_norm(self.dropout(output) + residual)

        state_update = self.fc_state(state_update, state)
        state_update = self.state_layer_norm(self.dropout(state_update))

        return output, state_update, attn



class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, d_state, dropout=0.1):
        super().__init__()

        self.d_in = d_in
        self.d_hid = d_hid
        self.d_state = d_state

        self.w_1 = GeneratedFullyConnectedLayer(d_state, d_in, d_hid)
        self.w_2 = GeneratedFullyConnectedLayer(d_state, d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state):
        residual = x

        output = F.relu(self.w_1(x, state))

        output = self.w_2(output, state)

        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class StepwiseRNN(nn.Module):
    ''' An RNN-based module that computes a single time-step at a time for recurence over computation '''
    def __init__(self, state_size, input_size, output_size, dropout=0.1):
        super().__init__()
        self.register_buffer("initial_state", torch.zeros(state_size))
        self.state = self.initial_state
        self.w_state = nn.Linear(in_features=state_size+input_size, out_features=state_size)
        self.w_output = nn.Linear(in_features=state_size, out_features=output_size)

        self.layer_norm = nn.LayerNorm(output_size)
        self.dropout = nn.Dropout(dropout)


    def initialise_state(self, batch_size):
        self.state = self.initial_state.repeat(batch_size, 1)

    def forward(self, x):
        # update the hidden state
        self.state = torch.tanh(self.w_state(torch.cat([self.state, x], dim=1)))
        # compute the output vector
        out = self.w_output(self.state)
        out = self.layer_norm(self.dropout(out))

        return out

class GeneratedFullyConnectedLayer(nn.Module):
    ''' A fully connected layer where the weights are not learned directly, but are the result of a learnable function on a state of computation '''

    def __init__(self, d_state, d_in, d_out):
        super().__init__()
        self.d_state = d_state
        self.d_in = d_in
        self.d_out = d_out

        self.weight_generator = nn.Linear(d_state, d_in * d_out)
        self.bias_generator = nn.Linear(d_state, d_out)
        nn.init.xavier_normal_(self.weight_generator.weight)
        nn.init.xavier_normal_(self.bias_generator.weight)


    def forward(self, X, state):
        batch_size = X.shape[0]

        weights = self.weight_generator(state).view(batch_size, self.d_out, self.d_in)
        bias = self.bias_generator(state)
        return F.tanh(MultiLinear(X, weights, bias))

