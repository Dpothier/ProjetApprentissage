from torchtext.vocab import Vectors
import torch
import os

def load(file_name):
    """
    Load the pre-trained embeddings from a file
    Then adds a vector and an entry in the word table for unknown tokens
    :param file_name: the embeddings file
    :return: the vocabulary and the word vectors
    """

    vectors = Vectors(name=file_name)

    embedding_size = vectors.vectors.shape[1]
    word_vectors = torch.cat((torch.zeros((1, embedding_size)), vectors.vectors), dim=0)

    word_table = {k: v+1 for k, v in vectors.stoi.items()}
    word_table['unk'] = 0

    return word_vectors, word_table