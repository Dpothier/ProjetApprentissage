import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# A transformer layer that is conditioned on the computation state
# Uses BERT encoding, with the sequence and separator tokens
# Input: The previous layer representation of each element of the sequence
# Input: The current computation state
# Output: The new representation of each element of the sequence

class MetaTransformerLayer(nn.Module):

    def __init__(self, generator):
        self.generator = generator


    def forward(self, X, state):



class MetaMultiAttention(nn.Module):

    def __init__(self, n_head, d_model, d_state, dropout):
        super().__init__()
        assert d_model % n_head == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // n_head
        self.n_head = n_head

        self.linear_layers = nn.ModuleList([MetaLinear(d_model, d_model, d_state) for _ in range(3)])
        self.output_linear = MetaLinear(d_model, d_model, d_state)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, state, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, ((query, state), (key, state), (value, state)))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x, state)

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MetaPositionwiseFeedForward(nn.Module):




class MetaLinear(nn.Module):
    def __init__(self, input_size, output_size, state_size):


    def forward(self, input, state):



class TransformerLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)