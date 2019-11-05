import sklearn.metrics as metrics
from torch.utils.data import DataLoader
import torch
import numpy as np

def calculate_weight(data : DataLoader):
    targets = get_targets(data)

    if len(targets.shape) == 2:
        targets = np.argmax(targets, axis=1)

    number_of_classes = targets.unique().shape[0]

    occurences = torch.zeros(number_of_classes)
    for i in range(number_of_classes):
        occurences[i] = targets[targets == i].shape[0]

    max_occurence = occurences.max().repeat(number_of_classes)

    return max_occurence / occurences

def calculate_weight_mnist(data : DataLoader):
    targets = data.train_labels

    if len(targets.shape) == 2:
        targets = np.argmax(targets, axis=1)

    number_of_classes = targets.unique().shape[0]

    occurences = torch.zeros(number_of_classes)
    for i in range(number_of_classes):
        occurences[i] = targets[targets == i].shape[0]

    max_occurence = occurences.max().repeat(number_of_classes)

    return max_occurence / occurences

def get_targets(data: DataLoader):
    targets = None
    for X, y in data:
        if targets is None:
            targets = y
        else:
            targets = torch.cat((targets,y), dim=0)

    return targets

def flatten_and_discritize_preds(preds):
    preds_total = flatten_preds(preds)

    return discritize_preds(preds_total)


def flatten_preds(preds):
    preds_total = None
    for batch in preds:
        if preds_total is None:
            preds_total = batch
        else:
            preds_total = np.concatenate((preds_total, batch), axis=0)

    return preds_total

def discritize_preds(preds):
    preds_classes = np.argmax(preds, axis=1)

    return preds_classes


def produce_accuracy_precision_recall_f1(preds, true, average):
    accuracy = metrics.accuracy_score(preds, true)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(preds, true, average=average)

    return accuracy, precision, recall, f1

def filter_on_invocab_words(data, preds, true):
    filtered_ids_both = []
    filtered_preds_both = []
    filtered_true_both = []

    filtered_ids_hypo = []
    filtered_preds_hypo = []
    filtered_true_hypo = []

    filtered_ids_hyper = []
    filtered_preds_hyper = []
    filtered_true_hyper = []

    filtered_ids_none = []
    filtered_preds_none = []
    filtered_true_none = []

    for i in range(len(preds)):
        hyponym_index = data.iloc[i, 2]
        hypernym_index = data.iloc[i, 3]

        if hyponym_index != 0 and hypernym_index != 0:
            filtered_ids_both.append(i)
            filtered_preds_both.append(preds[i])
            filtered_true_both.append(true[i])

        if hyponym_index != 0 and hypernym_index == 0:
            filtered_ids_hypo.append(i)
            filtered_preds_hypo.append(preds[i])
            filtered_true_hypo.append(true[i])

        if hyponym_index == 0 and hypernym_index != 0:
            filtered_ids_hyper.append(i)
            filtered_preds_hyper.append(preds[i])
            filtered_true_hyper.append(true[i])

        if hyponym_index == 0 and hypernym_index == 0:
            filtered_ids_none.append(i)
            filtered_preds_none.append(preds[i])
            filtered_true_none.append(true[i])

    return (data.iloc[filtered_ids_both, :], filtered_preds_both, filtered_true_both), \
        (data.iloc[filtered_ids_hypo, :], filtered_preds_hypo, filtered_true_hypo), \
        (data.iloc[filtered_ids_hyper, :], filtered_preds_hyper, filtered_true_hyper), \
        (data.iloc[filtered_ids_none, :], filtered_preds_none, filtered_true_none)

def filter_on_multiple_words_skills(data, preds, true):
    filtered_ids_both = []
    filtered_preds_both = []
    filtered_true_both = []

    filtered_ids_hypo = []
    filtered_preds_hypo = []
    filtered_true_hypo = []

    filtered_ids_hyper = []
    filtered_preds_hyper = []
    filtered_true_hyper = []

    filtered_ids_none = []
    filtered_preds_none = []
    filtered_true_none = []

    for i in range (len(preds)):
        hyponym_words = data.iloc[i, 0].split(' ')
        hypernym_words = data.iloc[i, 1].split(' ')

        if len(hyponym_words) > 1 and len(hypernym_words) > 1:
            filtered_ids_both.append(i)
            filtered_preds_both.append(preds[i])
            filtered_true_both.append(true[i])

        if len(hyponym_words) > 1 and len(hypernym_words) == 1:
            filtered_ids_hypo.append(i)
            filtered_preds_hypo.append(preds[i])
            filtered_true_hypo.append(true[i])

        if len(hyponym_words) == 1 and len(hypernym_words) > 1:
            filtered_ids_hyper.append(i)
            filtered_preds_hyper.append(preds[i])
            filtered_true_hyper.append(true[i])

        if len(hyponym_words) == 1 and len(hypernym_words) == 1:
            filtered_ids_none.append(i)
            filtered_preds_none.append(preds[i])
            filtered_true_none.append(true[i])

    return (data.iloc[filtered_ids_both, :], filtered_preds_both, filtered_true_both), \
        (data.iloc[filtered_ids_hypo, :], filtered_preds_hypo, filtered_true_hypo), \
        (data.iloc[filtered_ids_hyper, :], filtered_preds_hyper, filtered_true_hyper), \
        (data.iloc[filtered_ids_none, :], filtered_preds_none, filtered_true_none)


