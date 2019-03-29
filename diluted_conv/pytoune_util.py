import sklearn.metrics as metrics
from torch.utils.data import DataLoader
import torch
import numpy as np

def calculate_weight(data : DataLoader):
    targets = get_targets(data)

    number_of_classes = targets.unique().shape[0]

    occurences = torch.zeros(number_of_classes)
    for i in range(number_of_classes):
        occurences[i] = targets[targets == i].shape[0]

    max_occurence = occurences.max()

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
    preds_total = None
    for batch in preds:
        if preds_total is None:
            preds_total = batch
        else:
            preds_total = np.concatenate((preds_total, batch), axis=0)

    preds_classes = np.argmax(preds_total, axis=1)

    return preds_classes


def produce_accuracy_precision_recall_f1(preds, true):
    accuracy = metrics.accuracy_score(preds, true)
    precision, recall, f1, support = metrics.precision_recall_fscore_support(preds, true, average='binary')

    return accuracy, precision, recall, f1



