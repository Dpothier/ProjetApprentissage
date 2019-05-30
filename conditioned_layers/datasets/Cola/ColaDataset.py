import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gensim
import torch
from torch.nn.utils.rnn import pad_sequence

class ColaDataset(Dataset):
    """IMDB dataset"""

    def __init__(self, tsv_file, vocab_table, return_target=True, TEST_MODE=False):
        """
        Args:
            tsv_file (string): Path to the tsv file with data pairs.
            vocab_table (dict): Dictionary linking lemmas to index
        """

        self.return_target= return_target
        self.datapairs_frame = pd.read_csv(tsv_file, sep="\t")

        sentences = self.datapairs_frame.iloc[:, 1].values
        tokenized_reviews = [list(gensim.utils.tokenize(review)) for review in sentences]

        self.indexed_sentences = [[vocab_table.get(str.lower(token), 0) for token in review] for review in tokenized_reviews]

        if TEST_MODE:
            self.datapairs_frame = self.datapairs_frame.iloc[:100, :]
            self.indexed_sentences = self.indexed_sentences[:100]

        self.targets = self.datapairs_frame.iloc[:, 0].values


    def __len__(self):
        return len(self.indexed_sentences)

    def __getitem__(self, idx):
        X = torch.Tensor(self.indexed_sentences[idx])
        length = X.shape[0]
        X_pos = torch.Tensor(range(length))

        Y = self.targets[idx]

        if self.return_target:
            return X, X_pos, Y
        return X, X_pos

    def len_of_longest_sequence(self):
        return len(max(self.indexed_sentences, key=len))

    def get_collate_fn(self):
        return self.ColaCollate(self.return_target)

    class ColaCollate():
        def __init__(self, return_target):
            self.return_target = return_target


        def __call__(self, batch):
            max_len = max(map(lambda x: x[0][0].shape[0], batch))
            # pad according to max_len
            batch = list(map(
                lambda x: ((pad_tensor(x[0][0], pad=max_len, dim=0), pad_tensor(x[0][1], pad=max_len, dim=0)), x[1]),
                batch))
            # stack all
            xs = torch.stack(list(map(lambda x: x[0][0], batch)), dim=0)
            xs_pos = torch.stack(list(map(lambda x: x[0][1], batch)), dim=0)


            ys = torch.LongTensor(list(map(lambda x: x[1], batch)))

            if self.return_target:
                return (xs, xs_pos), ys

            return xs, xs_pos


# def ColaCollate(batch):
#     """
#            args:
#                batch - list of (tensor, label)
#
#            reutrn:
#                xs - a tensor of all examples in 'batch' after padding
#                ys - a LongTensor of all labels in batch
#            """
#     # find longest sequence
#     max_len = max(map(lambda x: x[0][0].shape[0], batch))
#     # pad according to max_len
#     batch = list(map(lambda x: ((pad_tensor(x[0][0], pad=max_len, dim=0), pad_tensor(x[0][1], pad=max_len, dim=0)), x[1]), batch))
#     # stack all
#     xs = torch.stack(list(map(lambda x: x[0][0], batch)), dim=0)
#     xs_pos = torch.stack(list(map(lambda x: x[0][1], batch)), dim=0)
#     ys = torch.LongTensor(list(map(lambda x: x[1], batch)))
#     return (xs, xs_pos), ys

# def ColaCollateNoTarget(batch):
#     """
#            args:
#                batch - list of (tensor, label)
#
#            reutrn:
#                xs - a tensor of all examples in 'batch' after padding
#                ys - a LongTensor of all labels in batch
#            """
#     # find longest sequence
#     max_len = max(map(lambda x: x[0].shape[0], batch))
#     # pad according to max_len
#     batch = list(map(lambda x: (pad_tensor(x[0], pad=max_len, dim=0), pad_tensor(x[1], pad=max_len, dim=0)), batch))
#     # stack all
#     xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
#     xs_pos = torch.stack(list(map(lambda x: x[1], batch)), dim=0)
#     return xs, xs_pos

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



