import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class SkillDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, tsv_file, vocab_file, TEST_MODE=False):
        """
        Args:
            tsv_file (string): Path to the tsv file with data pairs.
        """
        self.datapairs_frame = pd.read_csv(tsv_file, sep="\t")

        words = self.datapairs_frame.iloc[:, :2]
        words_substituted = words.applymap(lambda x: vocab_file.get(str.lower(x), 0))

        targets = self.datapairs_frame.iloc[:, 2]
        targets = targets.map(lambda x: int(x))

        self.datapairs_frame = pd.concat([words, words_substituted, targets], axis=1)

        if TEST_MODE:
            self.datapairs_frame = self.datapairs_frame.iloc[:100, :]



    def __len__(self):
        return len(self.datapairs_frame)

    def __getitem__(self, idx):

        hyponym = self.datapairs_frame.iloc[idx, 2]
        hypernym = self.datapairs_frame.iloc[idx, 3]
        targets = self.datapairs_frame.iloc[idx, 4]

        X = (hyponym, hypernym)
        Y = targets

        return X, Y

    def get_dataframe(self):
        return self.datapairs_frame



