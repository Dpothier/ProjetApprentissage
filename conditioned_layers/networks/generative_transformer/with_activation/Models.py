''' Define the Transformer model '''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networks.generative_transformer.Constants as Constants
from networks.generative_transformer.base.Layers import EncoderLayer
from networks.generative_transformer.base.SubLayers import StepwiseRNN

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, embeddings_vectors, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, d_state, dropout=0.1):
        super().__init__()

        self.d_state = d_state
        self.n_layers = n_layers

        n_position = len_max_seq + 1

        self.register_buffer("initial_state", torch.zeros(self.d_state))

        vocab_size = embeddings_vectors.shape[0]
        self.src_word_emb = nn.Embedding(vocab_size, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.generation_layer = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, d_state, dropout=dropout)

        self.state_layer = StepwiseRNN(d_state, d_state, d_state)
        self.layer_norm = nn.LayerNorm(d_state)

    def forward(self, src_seq, src_pos, return_attns=False):

        batch_size = src_seq.shape[0]

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq.long()) + self.position_enc(src_pos.long())

        state = self.initial_state.repeat(batch_size, 1)
        self.state_layer.initialise_state(batch_size)
        for i in range(self.n_layers):
            enc_output, state_update, enc_slf_attn = self.generation_layer(
                enc_output, state,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            state = self.state_layer(state_update)

            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            embeddings_vectors, len_max_seq,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, d_state=64, dropout=0.1):

        super().__init__()

        self.encoder = Encoder(
            embeddings_vectors=embeddings_vectors, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_state=d_state,
            dropout=dropout)

        self.classif_layer_1 = nn.Linear(d_model, d_inner)
        self.classif_layer_2 = nn.Linear(d_inner, 2)
        nn.init.xavier_normal_(self.classif_layer_1.weight)
        nn.init.xavier_normal_(self.classif_layer_2.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, X):
        src_seq, src_pos = X
        enc_output, *_ = self.encoder(src_seq, src_pos)

        seq_len = enc_output.shape[1]
        out = F.avg_pool1d(enc_output.transpose(1,2), kernel_size=seq_len).squeeze()

        out = F.relu(self.classif_layer_1(out))
        out = self.classif_layer_2(out)

        return out