import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gensim
import torch
from torch.nn.utils.rnn import pad_sequence

class IMDBDataset(Dataset):
    """IMDB dataset"""

    def __init__(self, tsv_file, vocab_table):
        """
        Args:
            tsv_file (string): Path to the tsv file with data pairs.
            vocab_table (dict): Dictionary linking lemmas to index
        """
        self.datapairs_frame = pd.read_csv(tsv_file, sep="\t")

        review_texts = self.datapairs_frame.iloc[:, 1].values
        tokenized_reviews = [list(gensim.utils.tokenize(review)) for review in review_texts]

        self.indexed_reviews = [[vocab_table.get(str.lower(token), 0) for token in review] for review in tokenized_reviews]


    def __len__(self):
        return len(self.indexed_reviews)

    def __getitem__(self, idx):



        X = torch.Tensor(self.indexed_reviews[idx])
        Y = self.datapairs_frame.iloc[idx, 2]

        return X, Y

    def len_of_longest_review(self):
        return len(max(self.indexed_reviews, key=len))



