import torch
import torch.nn as nn
from collections import Counter
from glove import Corpus, Glove

from torchtext import vocab
from torchtext import data
from torchtext.data import Example

from data_load.load import load_data

texts = data.Field(lower=True)
tags = data.Field(use_vocab=False, pad_token=3)
id = data.Field(sequential=False, use_vocab=False)

fields = [('id', id), ('texts', texts), ('process_tags', tags), ('material_tags', tags), ('task_tags', tags)]
fields_dict = {'id': ('id', id), 'texts': ('texts', texts), 'process_tags': ('process_tags',tags), 'material_tags': ('material_tags', tags), 'task_tags': ('task_tags', tags)}

loaded_train, train_indices = load_data('./data/train2', use_int_tags=True)
train_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_train]
train = data.Dataset(examples=train_examples, fields=fields)

texts_tokens = Counter([token for example in loaded_train for token in example['texts']])

model = Glove.load('./embeddings/semeval200.glove')

vectors = model.word_vectors
dictionary = model.dictionary
vocabulary = vocab.Vocab(texts_tokens)
vocabulary.set_vectors(stoi=dictionary, vectors=torch.tensor(vectors), dim=200)

# texts.build_vocab(train, vectors='glove.6B.300d')

texts.vocab = vocabulary


for key, value in texts_tokens.items():
    if key not in texts.vocab.freqs:
        print("Word do not have embedding:{}".format(key))

print("Number of tokens: {}".format(len([token for example in loaded_train for token in example['texts']])))
print("Size of whole vocabulary : {}".format(len(texts_tokens)))
print("Size of vocabulary with embeddings: {}".format(len(texts.vocab.vectors)))