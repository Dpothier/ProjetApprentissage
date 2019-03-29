import torchtext.data as data
import torchtext.datasets as datasets
from torchtext.vocab import GloVe


def load_IMDB():
    # set up fields
    TEXT = data.Field(lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False)

    # make splits for data
    train, test = datasets.IMDB.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=100))
    LABEL.build_vocab(train)

    return train, test
