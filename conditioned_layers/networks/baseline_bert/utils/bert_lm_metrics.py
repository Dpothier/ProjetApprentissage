from torch.utils.data import DataLoader
import torch

def sent_acc(y_pred, y_true, ignore_index=-100):
    y_pred_sentence, _ = y_pred
    y_true_sentence, _ = y_true
    y_pred_sentence = y_pred_sentence.argmax(1)

    weights = (y_true_sentence != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = ((y_pred_sentence == y_true_sentence).float() * weights).sum() / num_labels
    return acc_pred * 100

def words_acc(y_pred, y_true, ignore_index=-100):
    _, y_pred_words = y_pred
    _, y_true_words = y_true
    y_preds_words = y_pred_words.argmax(2)

    weights = (y_preds_words != ignore_index).float()
    num_labels = weights.sum()
    acc_pred = ((y_preds_words == y_true_words).float() * weights).sum() / num_labels
    return acc_pred * 100

def get_bert_sentence_targets(bert_data: DataLoader):
    targets = None
    for X, y in bert_data:
        sent_y = y[0]
        if targets is None:
            targets = sent_y
        else:
            targets = torch.cat((targets,sent_y), dim=0)

    return targets