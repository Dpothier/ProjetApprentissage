import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn
import training

from torchtext import data
from torchtext.data import Example

from data_load.load import load_data
from models.tcn import TCN


texts = data.Field(lower=True)
tags = data.Field(use_vocab=False, pad_token=3)

fields = [('texts', texts), ('process_tags', tags), ('material_tags', tags), ('task_tags', tags)]
fields_dict = {'texts': ('texts', texts), 'process_tags': ('process_tags',tags), 'material_tags': ('material_tags', tags), 'task_tags': ('task_tags', tags)}
loaded_train = load_data('./data/train2', use_int_tags=True)
loaded_val = load_data('./data/dev', use_int_tags=True)
train_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_train]
val_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_val]


train = data.Dataset(examples=train_examples, fields=fields)
val = data.Dataset(examples=val_examples, fields=fields)

texts.build_vocab(train, vectors='glove.6B.100d')

vocab = train.fields['texts'].vocab

model = TCN(vocab.vectors)

#train_iter = data.Iterator(train, batch_size=32 , device=-1, repeat=False)
#val_iter = data.Iterator(val, batch_size=32 , device=-1, repeat=False)

history_process, history_material, history_task =\
    training.train(model, (train, val), n_epoch=3, batch_size=32, learning_rate=0.1, use_gpu=False)
