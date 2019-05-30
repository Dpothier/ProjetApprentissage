''' Define the Layers '''
import torch.nn as nn
from conditioned_layers.networks.conditioned_transformer.base.SubLayers import MultiHeadAttention, PositionwiseFeedForward, GeneratedFullyConnectedLayer

__author__ = "Yu-Hsiang Huang"


class SimplerEncoderLayer(nn.Module):
    ''' Uses simpler sublayer to facilitate convergence '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, d_state, dropout=0.1):
        super(SimplerEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, d_state, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, d_state, dropout=dropout)
        self.state_ffn = GeneratedFullyConnectedLayer(d_state, d_state, d_state)


    def forward(self, enc_input, state, non_pad_mask=None, slf_attn_mask=None):
        enc_output, state_update, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, state, mask=slf_attn_mask)

        enc_output *= non_pad_mask

        enc_output= self.pos_ffn(enc_output, state)
        state_update = self.state_ffn(state_update, state_update.squeeze()).squeeze()
        enc_output *= non_pad_mask

        return enc_output, state_update, enc_slf_attn