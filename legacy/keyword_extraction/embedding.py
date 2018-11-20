from torch import nn
import torch


def emb_layer(keyed_vectors, trainable=False):
    """Create an Embedding layer from the supplied gensim keyed_vectors."""
    emb_weights = torch.LongTensor(keyed_vectors.syn0)
    emb = nn.Embedding(*emb_weights.shape)
    emb.weight = nn.Parameter(emb_weights)
    emb.weight.requires_grad = trainable
    return emb