import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn
import training
from collections import Counter

from torchtext import data
from torchtext.data import Example

from data_load.load import load_data
from models.tcn import TCN_simple
import eval


def get_tags_weight_ratio(examples):
    all_process = [tag for example in examples for tag in example['process_tags']]
    all_material = [tag for example in examples for tag in example['material_tags']]
    all_task = [tag for example in examples for tag in example['task_tags']]

    all_tags = all_process + all_material + all_task

    total_number_of_tags = len(all_tags)
    tags_count = Counter(all_tags)
    tags_count_list = [tags_count[0], tags_count[1], tags_count[2], tags_count[3], tags_count[4]]

    print("O counts of {}% of data".format(tags_count[3]/total_number_of_tags * 100))

    return torch.Tensor(tags_count_list)




texts = data.Field(lower=True)
tags = data.Field(use_vocab=False, pad_token=3)
id = data.Field(sequential=False, use_vocab=False)

fields = [('id', id), ('texts', texts), ('process_tags', tags), ('material_tags', tags), ('task_tags', tags)]
fields_dict = {'id': ('id', id), 'texts': ('texts', texts), 'process_tags': ('process_tags',tags), 'material_tags': ('material_tags', tags), 'task_tags': ('task_tags', tags)}
loaded_train, train_indices = load_data('./data/train2', use_int_tags=True)
loaded_val, val_indices = load_data('./data/dev', use_int_tags=True)
train_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_train]
val_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_val]


tags_weight = get_tags_weight_ratio(loaded_train)

train = data.Dataset(examples=train_examples, fields=fields)
val = data.Dataset(examples=val_examples, fields=fields)

texts.build_vocab(train, vectors='glove.6B.100d')

vocab = train.fields['texts'].vocab

model = TCN_simple(vocab.vectors)

dataset = ((train, train_indices), (val, val_indices))
history_process = training.train(model, dataset, n_epoch=10, batch_size=32, learning_rate=0.1, use_gpu=False, class_weight=tags_weight)

print("Final scores on train set")
print(eval.calculateMeasures('./data/train2', './data/train_preds', 'rel'))

print("Final scores on test set")
print(eval.calculateMeasures('./data/dev', './data/val_preds', 'rel'))