import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_model(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self,embeddings_vectors, d_word_vec=512, d_model=512, d_inner=2048, dropout=0):

        super().__init__()

        vocab_size = embeddings_vectors.shape[0]
        self.src_word_emb = nn.Embedding(vocab_size, d_word_vec)

        self.encoder = nn.LSTM(input_size=d_word_vec, hidden_size=d_model, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.classif_layer_1 = nn.Linear(d_model, d_inner)
        self.classif_layer_2 = nn.Linear(d_inner, 2)
        nn.init.xavier_normal_(self.classif_layer_1.weight)
        nn.init.xavier_normal_(self.classif_layer_2.weight)

    def forward(self, X):
        src_seq, src_pos = X
        embedded_seq = self.src_word_emb(src_seq.long())

        out = self.encoder(embedded_seq)[0][:, -1, :]

        out = self.dropout(out)

        out = F.relu(self.classif_layer_1(out))
        out = self.classif_layer_2(out)

        return out