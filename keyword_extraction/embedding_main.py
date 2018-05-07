import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn
import training
from collections import Counter

from torchtext import vocab
from torchtext import data
from torchtext.data import Example

from data_load.load import load_data
from models.tcn import TCN_simple
from models.tcn import TCN
import eval
import sys
from glove import Corpus, Glove



def get_tags_weight_ratio(examples):
    all_process = [tag for example in examples for tag in example['process_tags']]
    all_material = [tag for example in examples for tag in example['material_tags']]
    all_task = [tag for example in examples for tag in example['task_tags']]

    all_tags = all_process + all_material + all_task

    total_number_of_tags = len(all_tags)
    tags_count = Counter(all_tags)
    print("Number of tags: {}".format(len(tags_count)))
    tags_count_list = [tags_count[0], tags_count[1]]

    print("O counts of {}% of data".format(tags_count[3]/total_number_of_tags * 100))

    return torch.Tensor(tags_count_list)




texts = data.Field(lower=True)
tags = data.Field(use_vocab=False, pad_token=1)
id = data.Field(sequential=False, use_vocab=False)

fields = [('id', id), ('texts', texts), ('process_tags', tags), ('material_tags', tags), ('task_tags', tags)]
fields_dict = {'id': ('id', id), 'texts': ('texts', texts), 'process_tags': ('process_tags',tags), 'material_tags': ('material_tags', tags), 'task_tags': ('task_tags', tags)}
loaded_train, train_indices = load_data('./data/train2', use_int_tags=True, tag_scheme='IO')
loaded_val, val_indices = load_data('./data/dev', use_int_tags=True, tag_scheme='IO')
train_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_train]
val_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_val]


tags_weight = get_tags_weight_ratio(loaded_train)


texts_tokens = Counter([token for example in loaded_train for token in example['texts']])

model = Glove.load('./embeddings/whole_semeval200.glove')

vectors = model.word_vectors
dictionary = model.dictionary
vocabulary = vocab.Vocab(texts_tokens)
vocabulary.set_vectors(stoi=dictionary, vectors=torch.Tensor(vectors), dim=200)

train = data.Dataset(examples=train_examples, fields=fields)
val = data.Dataset(examples=val_examples, fields=fields)

texts.vocab = vocabulary


use_gpu = True if sys.argv[1] == 'gpu' else False

model = TCN(vocabulary.vectors)
if use_gpu:
    model = model.cuda()

dataset = ((train, train_indices), (val, val_indices))

# training_schedules = [(15,0.05),
#                       (30, 0.01),
#                       (100, 0.005)]

training_schedules = [(20,0.05),
                      (20, 0.01),
                      (20, 0.005),
                      (20, 0.001)]

history_process = training.train(model, dataset, history_file='./history/io_scheme.pdf', weight_decay=0.1, training_schedule=training_schedules, batch_size=32, use_gpu=use_gpu, class_weight=tags_weight, patience=100)

history_process.display()

print("Final scores on train set")
print(eval.calculateMeasures('./data/train2', './data/train_preds', 'rel'))

print("Final scores on test set")
print(eval.calculateMeasures('./data/dev', './data/val_preds', 'rel'))