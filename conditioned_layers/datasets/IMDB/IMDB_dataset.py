import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gensim
import torch
from torch.nn.utils.rnn import pad_sequence

class IMDBDataset(Dataset):
    """IMDB dataset"""

    def __init__(self, tsv_file, vocab_table, TEST_MODE = False):
        """
        Args:
            tsv_file (string): Path to the tsv file with data pairs.
            vocab_table (dict): Dictionary linking lemmas to index
        """
        self.datapairs_frame = pd.read_csv(tsv_file, sep="\t")

        review_texts = self.datapairs_frame.iloc[:, 1].values
        tokenized_reviews = [list(gensim.utils.tokenize(review)) for review in review_texts]

        self.indexed_reviews = [[vocab_table.get(str.lower(token), 0) for token in review] for review in tokenized_reviews]

        if TEST_MODE:
            self.datapairs_frame = self.datapairs_frame.iloc[:100, :]
            self.indexed_reviews = self.indexed_reviews[:100]


    def __len__(self):
        return len(self.indexed_reviews)

    def __getitem__(self, idx):
        X = torch.Tensor(self.indexed_reviews[idx])
        length = X.shape[0]
        X_pos = torch.Tensor(range(length)).half()

        X = torch.Tensor(self.indexed_reviews[idx]).half()
        Y = int(self.datapairs_frame.iloc[idx, 2])

        return (X, X_pos), Y

    def len_of_longest_sequence(self):
        return len(max(self.indexed_reviews, key=len))



def IMDBCollate(batch):
    """
           args:
               batch - list of (tensor, label)

           reutrn:
               xs - a tensor of all examples in 'batch' after padding
               ys - a LongTensor of all labels in batch
           """
    # find longest sequence
    batch_size = len(batch)
    if batch_size == 1:
        return batch[0][0], torch.LongTensor(batch[0][1]).unsqueeze(0)

    max_len = max(map(lambda x: x[0][0].shape[0], batch))
    # pad according to max_len
    batch = list(map(lambda x: ((pad_tensor(x[0][0], pad=max_len, dim=0), pad_tensor(x[0][1], pad=max_len, dim=0)), x[1]), batch))
    # stack all
    xs = torch.stack(list(map(lambda x: x[0][0], batch)), dim=0)
    xs_pos = torch.stack(list(map(lambda x: x[0][1], batch)), dim=0)
    ys = torch.LongTensor(list(map(lambda x: x[1], batch)))

    if batch_size == 1:
        xs = xs.unsqueeze(0)
        xs_pos = xs_pos.unsqueze(0)
        ys = ys.unsqueeze(0)

    return (xs, xs_pos), ys

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)