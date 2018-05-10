import os
from data_load.tokenization import tokenize
import torch
from collections import Counter

from torchtext import vocab
from torchtext import data
from torchtext.data import Example

from glove import Glove
from helper.class_weight import get_tags_weight_ratio


def load_text(file):
    for l in file:
        text = l

    return text


def load_annotations(file):
    process_ann = []
    material_ann = []
    tasks_ann = []

    for l in file:
        annotation = l.split()
        if "T" in annotation[0]:
            coordinates = (int(annotation[2]), int(annotation[3]))
            if annotation[1] == "Process":
                process_ann.append(coordinates)
            if annotation[1] == "Material":
                material_ann.append(coordinates)
            if annotation[1] == "Task":
                tasks_ann.append(coordinates)

    return process_ann, material_ann, tasks_ann


def parse_BILOU_tags(spans, annotations, use_int_tag=True):
    B = 0 if use_int_tag else 'B'
    I = 1 if use_int_tag else 'I'
    L = 2 if use_int_tag else 'L'
    O = 3 if use_int_tag else 'O'
    U = 4 if use_int_tag else 'U'

    bilou_tags = []
    for i in range(len(spans)):
        bilou_tags.append(O)

    for annotation in annotations:
        start_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[0] <= span[1]]
        end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1] <= span[1]]
        if len(end_spans) == 0:
            end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1]-1 <= span[1]]
        start_span = start_spans[0]
        end_span = end_spans[0]
        if start_span == end_span:
            bilou_tags[start_span] = U
        else:
            bilou_tags[start_span] = B
            bilou_tags[end_span] = L
            for i in range(start_span+1, end_span):
                bilou_tags[i]= I
    return bilou_tags


def parse_IO_tags(spans, annotations, use_int_tag=True):
    I = 0 if use_int_tag else 'I'
    O = 1 if use_int_tag else 'O'

    bilou_tags = []
    for i in range(len(spans)):
        bilou_tags.append(O)

    for annotation in annotations:
        start_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[0] <= span[1]]
        end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1] <= span[1]]
        if len(end_spans) == 0:
            end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1]-1 <= span[1]]
        start_span = start_spans[0]
        end_span = end_spans[0]
        if start_span == end_span:
            bilou_tags[start_span] = I
        else:
            bilou_tags[start_span] = I
            bilou_tags[end_span] = I
            for i in range(start_span+1, end_span):
                bilou_tags[i]= I
    return bilou_tags


def load_data(folder, use_int_tags=True, tag_scheme='IO'):
    texts = []
    indices = {}

    tag_function = parse_IO_tags
    if tag_scheme == 'BILOU':
        tag_function= parse_BILOU_tags

    flist = os.listdir(folder)
    current_index = 0
    for f in flist:
        if not str(f).endswith(".txt"):
            continue
        f_text = open(os.path.join(folder, f), "r", encoding="utf8")
        f_ann = open(os.path.join(folder, f[0:-4]+".ann"), "r", encoding="utf8")

        text = load_text(f_text)

        process_ann, material_ann, tasks_ann = load_annotations(f_ann)
        tokens, spans = tokenize(text)
        texts.append((current_index,
                      tokens,
                      tag_function(spans, process_ann, use_int_tags),
                      tag_function(spans, material_ann, use_int_tags),
                      tag_function(spans, tasks_ann, use_int_tags)))

        indices[current_index] = (f[0:-4], tokens, spans)
        current_index += 1

    dict_texts = [{'id': text[0],
                   'texts': text[1],
                   'process_tags': text[2],
                   'material_tags': text[3],
                   'task_tags': text[4]} for text in texts]

    return dict_texts, indices


def prepare_dataset():
    texts = data.Field(lower=True)
    tags = data.Field(use_vocab=False, pad_token=1)
    id = data.Field(sequential=False, use_vocab=False)

    fields = [('id', id), ('texts', texts), ('process_tags', tags), ('material_tags', tags), ('task_tags', tags)]
    fields_dict = {'id': ('id', id), 'texts': ('texts', texts), 'process_tags': ('process_tags', tags),
                   'material_tags': ('material_tags', tags), 'task_tags': ('task_tags', tags)}
    loaded_train, train_extra = load_data('./data/train2', use_int_tags=True, tag_scheme='IO')
    loaded_val, val_extra = load_data('./data/dev', use_int_tags=True, tag_scheme='IO')
    loaded_test, test_extra = load_data('./data/test', use_int_tags=True, tag_scheme='IO')
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

    return (train, train_extra, val, val_extra, test, test_extra), \
            vocabulary, tags_weight

