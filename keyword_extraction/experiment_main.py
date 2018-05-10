import torch
import training_hard as training
from collections import Counter

from torchtext import vocab
from torchtext import data
from torchtext.data import Example

from data_load.load import load_data
from models.tcn import TCN
import eval
import sys
from glove import Corpus, Glove
from training_module.trainer import Trainer
from training_module.trainer_soft_targets import TrainerSoftTarget



def get_tags_weight_ratio(examples):
    all_process = [tag for example in examples for tag in example['process_tags']]
    all_material = [tag for example in examples for tag in example['material_tags']]
    all_task = [tag for example in examples for tag in example['task_tags']]

    all_tags = all_process + all_material + all_task

    total_number_of_tags = len(all_tags)
    tags_count = Counter(all_tags)
    tags_count_list = [tags_count[0], tags_count[1]]

    print("1 counts of {}% of data".format(tags_count[1]/total_number_of_tags * 100))

    return torch.Tensor(tags_count_list)



orig_stdout = sys.stdout

texts = data.Field(lower=True)
tags = data.Field(use_vocab=False, pad_token=1)
id = data.Field(sequential=False, use_vocab=False)

fields = [('id', id), ('texts', texts), ('process_tags', tags), ('material_tags', tags), ('task_tags', tags)]
fields_dict = {'id': ('id', id), 'texts': ('texts', texts), 'process_tags': ('process_tags',tags), 'material_tags': ('material_tags', tags), 'task_tags': ('task_tags', tags)}
loaded_train, train_indices = load_data('./data/train2', use_int_tags=True, tag_scheme='IO')
loaded_val, val_indices = load_data('./data/dev', use_int_tags=True, tag_scheme='IO')
loaded_test, test_indices = load_data('./data/test', use_int_tags=True, tag_scheme='IO')
train_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_train]
val_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_val]
test_examples = [Example.fromdict(data_point, fields_dict) for data_point in loaded_test]


tags_weight = get_tags_weight_ratio(loaded_train)


texts_tokens = Counter([token for example in loaded_train for token in example['texts']])

model = Glove.load('./embeddings/whole_semeval200.glove')

vectors = model.word_vectors
dictionary = model.dictionary
vocabulary = vocab.Vocab(texts_tokens)
vocabulary.set_vectors(stoi=dictionary, vectors=torch.Tensor(vectors), dim=200)

train = data.Dataset(examples=train_examples, fields=fields)
val = data.Dataset(examples=val_examples, fields=fields)
test = data.Dataset(examples=test_examples, fields=fields)

texts.vocab = vocabulary


use_gpu = True if sys.argv[1] == 'gpu' else False



dataset = ((train, train_indices), (val, val_indices))

training_schedules = [(20,0.05),
                      (20, 0.01),
                      (20, 0.005),
                      (20, 0.001)]

weight_decay_values = [0.1,0.01,0.001]
dropout_values = [0.1,0.2,0.3]
batch_size = 32

results = []

absolute_min_val_loss = sys.maxsize
best_decay_value = 0
best_dropout_value = 0
absolute_best_model = None
for decay_value in weight_decay_values:
    for dropout_value in dropout_values:
        model = TCN(vocabulary.vectors, p_first_layer=dropout_value, p_other_layers=dropout_value)
        if use_gpu:
            model = model.cuda()

        trainer = TrainerSoftTarget(2, 0.8, use_gpu)

        f = open('./results/soft_target_{}_{}.txt'.format(decay_value, dropout_value), 'w')
        sys.stdout = f

        history, best_model = trainer.train(model, dataset, history_file='./history/removed_causal_conv.pdf', weight_decay=decay_value, training_schedule=training_schedules, batch_size=batch_size, use_gpu=use_gpu, class_weight=tags_weight, patience=100)

        sys.stdout = orig_stdout
        f.close()

        min_val_loss = min(history.history['val_loss'])
        if min_val_loss < absolute_min_val_loss:
            absolute_min_val_loss = min_val_loss
            best_decay_value = decay_value
            best_dropout_value = dropout_value
            absolute_best_model = best_model

        history.display()

torch.save(absolute_best_model.state_dict(), './model_dump/soft_classes_model')
#Evaluation on test set

f = open('./results/soft_target_final.txt'.format(decay_value, dropout_value), 'w')
sys.stdout = f

test_iter = data.Iterator(test, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
trainer = TrainerSoftTarget(2, 0.8, use_gpu)
trainer.validate(model, test_iter, test_indices, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/test_preds')

sys.stdout = orig_stdout
f.close()

print("Result for soft classes")
print("Best results obtained on decay: {}, dropout: {}".format(best_decay_value, best_dropout_value))
print("Final scores on test set")
print(eval.calculateMeasures('./data/test', './data/test_preds', 'rel'))

results = []

absolute_min_val_loss = sys.maxsize
best_decay_value = 0
best_dropout_value = 0
absolute_best_model = None
for decay_value in weight_decay_values:
    for dropout_value in dropout_values:
        model = TCN(vocabulary.vectors, p_first_layer=dropout_value, p_other_layers=dropout_value)
        if use_gpu:
            model = model.cuda()

        trainer = Trainer()

        f = open('./results/hard_target_{}_{}.txt'.format(decay_value, dropout_value), 'w')
        sys.stdout = f

        history, best_model = trainer.train(model, dataset, history_file='./history/removed_causal_conv.pdf', weight_decay=decay_value, training_schedule=training_schedules, batch_size=batch_size, use_gpu=use_gpu, class_weight=tags_weight, patience=100)

        sys.stdout = orig_stdout
        f.close()

        min_val_loss = min(history.history['val_loss'])
        if min_val_loss < absolute_min_val_loss:
            absolute_min_val_loss = min_val_loss
            best_decay_value = decay_value
            best_dropout_value = dropout_value
            absolute_best_model = best_model

        history.display()

torch.save(absolute_best_model.state_dict(), './model_dump/hard_classes_model')
#Evaluation on test set

f = open('./results/hard_target_final.txt'.format(decay_value, dropout_value), 'w')
sys.stdout = f

test_iter = data.Iterator(test, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
trainer = TrainerSoftTarget(2, 0.8, use_gpu)
trainer.validate(model, test_iter, test_indices, use_gpu=use_gpu, class_weight=tags_weight, ann_folder='./data/test_preds_hard')

sys.stdout = orig_stdout
f.close()

print("Result for hard classes")
print("Best results obtained on decay: {}, dropout: {}".format(best_decay_value, best_dropout_value))
print("Final scores on test set")
print(eval.calculateMeasures('./data/test', './data/test_preds_hards', 'rel'))