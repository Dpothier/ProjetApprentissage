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

class Vocab():

    def __init__(self, file_name=None):
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4

        self.stoi = None
        self.itos = None
        self.vectors = None

        if file_name != None:
            self.load_from_embeddings_table(file_name)

    def load_from_embeddings_table(self, file_name):
        vectors = Vectors(name=file_name)

        embedding_size = vectors.vectors.shape[1]
        self.vectors = torch.cat((torch.zeros((5, embedding_size)), vectors.vectors), dim=0)

        self.stoi= {k: v + 5 for k, v in vectors.stoi.items()}
        self.stoi['<pad>'] = self.pad_index
        self.stoi['<unk>'] = self.unk_index
        self.stoi['<eos>'] = self.eos_index
        self.stoi['<sos>'] = self.sos_index
        self.stoi['<mask>'] = self.mask_index

        self.itos = {k + 5: v for k,v in enumerate(vectors.itos)}
        self.itos[self.pad_index] = '<pad>'
        self.itos[self.unk_index] = '<unk>'
        self.itos[self.eos_index] = '<eos>'
        self.itos[self.sos_index] = '<sos>'
        self.itos[self.mask_index] = '<mask>'

    def __len__(self):
        return self.vectors.shape[0]

