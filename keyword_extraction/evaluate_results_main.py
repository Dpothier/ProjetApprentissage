import torch
from collections import Counter

from torchtext import vocab
from torchtext import data
from torchtext.data import Example

from data_load.load import load_data
from models.tcn import TCN
from semeval import eval
import sys
from glove import Glove
from training_module.trainer import Trainer
from training_module.trainer_soft_targets import TrainerSoftTarget
from helper.class_weight import get_tags_weight_ratio



if __name__ == '__main__':

    use_gpu = True if sys.argv[1] == 'gpu' else False

    orig_stdout = sys.stdout

    texts = data.Field(lower=True)
    tags = data.Field(use_vocab=False, pad_token=1)
    id = data.Field(sequential=False, use_vocab=False)

    fields = [('id', id), ('texts', texts), ('process_tags', tags), ('material_tags', tags), ('task_tags', tags)]
    fields_dict = {'id': ('id', id), 'texts': ('texts', texts), 'process_tags': ('process_tags', tags),
                   'material_tags': ('material_tags', tags), 'task_tags': ('task_tags', tags)}
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

    model = TCN(vocabulary.vectors, p_first_layer=0.2, p_other_layers=0.2)
    model.load_state_dict(torch.load('./model_dump/soft_classes_model'))

    trainer = TrainerSoftTarget(2, 0.8, use_gpu)
    batch_size=32

    train_iter = data.Iterator(train, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
    val_iter = data.Iterator(val, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)
    test_iter = data.Iterator(test, batch_size=batch_size, device=-1 if use_gpu is False else None, repeat=False)

    trainer.validate(model, train_iter, train_indices, use_gpu=use_gpu, class_weight=tags_weight,
                     ann_folder='./data/train_preds')
    trainer.validate(model, val_iter, val_indices, use_gpu=use_gpu, class_weight=tags_weight,
                     ann_folder='./data/val_preds')
    trainer.validate(model, test_iter, test_indices, use_gpu=use_gpu, class_weight=tags_weight,
                     ann_folder='./data/test_preds')

    f = open('./results/soft_target_final_bis.txt', 'w')
    sys.stdout = f

    print("Result for soft classes on train set")
    print(eval.calculateMeasures('./data/train', './data/train_preds', 'rel'))
    print("Result for soft classes on val set")
    print(eval.calculateMeasures('./data/val', './data/val_preds', 'rel'))
    print("Result for soft classes on train set")
    print(eval.calculateMeasures('./data/test', './data/test_preds', 'rel'))

    sys.stdout = orig_stdout
    f.close()