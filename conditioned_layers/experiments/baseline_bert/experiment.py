import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import torch.nn as nn
import torch.optim as optim
from training.results import Results
from poutyne.framework import Model
from poutyne.framework.callbacks.clip_grad import ClipNorm
from poutyne.framework.callbacks.lr_scheduler import ExponentialLR
from poutyne.framework.callbacks.best_model_restore import BestModelRestore
from poutyne.framework.callbacks.earlystopping import EarlyStopping
from datasets.LM_pretrain.BertDataset import BERTDataset
from training.metrics_util import *
from networks.baseline_bert.bert import BERT
from networks.baseline_bert.language_model import BERTLM
from networks.baseline_bert.pretrain_loss import PretrainLoss
from networks.baseline_bert.utils.bert_lm_metrics import sent_acc, words_acc, get_bert_sentence_targets
from training.embeddings import load, Vocab
import sklearn
import os
import click


TEST_MODE = False
SEQ_LEN = 20

@click.command()
@click.option('-d', '--dataset', default="datasets/LM_pretrain/")
@click.option('-e', '--embeddings', default="embeddings/glove.6B.100d.txt")
@click.option('-g', '--gpu', default="gpu0")
def main(dataset, embeddings, gpu):
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    epochs = 1
    batch_size = 32

    if gpu == 'cpu':
        use_gpu = False
    elif gpu == 'gpu1':
        torch.cuda.set_device(1)
        use_gpu = True
    else:
        torch.cuda.set_device(0)
        use_gpu = True

    np.random.seed(133)

    output_folder_base = os.path.dirname(os.path.realpath(__file__)) + "/results/"

    learning_rates = [0.0001]
    weight_decays = [0.01]
    dropout_rates = [0.1]

    vocab = Vocab(embeddings)

    train = DataLoader(BERTDataset("{}train.txt".format(dataset), vocab, SEQ_LEN, test_mode=TEST_MODE), batch_size=batch_size)
    dev = DataLoader(BERTDataset("{}dev.txt".format(dataset), vocab, SEQ_LEN, test_mode=TEST_MODE), batch_size=batch_size)
    # test = DataLoader(ColaDataset("{}test.tsv".format(dataset_prefix), vocab_table, TEST_MODE=TEST_MODE), batch_size=batch_size)

    best_results = None
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            for dropout_rate in dropout_rates:
                output_folder = output_folder_base +"{}_{}_{}/".format(learning_rate, weight_decay, dropout_rate)
                results = Results(output_folder)

                save_hyperparameters(results, dataset, embeddings, learning_rate, epochs, batch_size, weight_decay, dropout_rate)

                # Create the classifier
                vocab_size = vocab.vectors.shape[0]
                bert = BERT( vocab_size= vocab_size, n_layers=6, dropout=dropout_rate, hidden=64, attn_heads=8)
                module = BERTLM(bert, vocab_size)

                print('Model defined')
                optimizer = optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
                gradient_clipping = ClipNorm(module.parameters(), 1)
                lr_scheduler = ExponentialLR(gamma=0.9)
                early_stopping = EarlyStopping(patience=2)
                best_model_restore = BestModelRestore()

                loss = PretrainLoss()


                model = Model(module, optimizer, loss, metrics=[sent_acc])

                if use_gpu:
                    model = model.cuda()

                model.fit_generator(train, dev, epochs=epochs, callbacks=[gradient_clipping, lr_scheduler, best_model_restore, early_stopping])

                test_loss, test_metrics, test_preds = model.evaluate_generator(dev, return_pred=True)
                train_loss, train_metrics, train_preds = model.evaluate_generator(train, return_pred=True)

                sentence_test_preds = [preds[0] for preds in test_preds]
                test_preds = flatten_and_discritize_preds(sentence_test_preds)
                test_true = get_bert_sentence_targets(dev)

                sentence_train_preds = [preds[0] for preds in train_preds]
                train_preds = flatten_and_discritize_preds(sentence_train_preds)
                train_true = get_bert_sentence_targets(train)

                write_results(results, "Results", test_preds, test_true, train_preds, train_true)
                results.save_model(model)

                train_accuracy, train_precision, train_recall, train_f1 = produce_accuracy_precision_recall_f1(
                    train_preds, train_true)
                test_accuracy, test_precision, test_recall, test_f1 = produce_accuracy_precision_recall_f1(test_preds,
                                                                                                           test_true)
                train_corr = sklearn.metrics.matthews_corrcoef(train_preds, train_true)
                dev_corr = sklearn.metrics.matthews_corrcoef(test_preds, test_true)

                current_results = {
                    "learning rate": learning_rate,
                    "weight decay": weight_decay,
                    "dropout rate": dropout_rate,
                    "train accuracy": train_accuracy,
                    "test accuracy": test_accuracy,
                    "precision": test_precision,
                    "recall": test_recall,
                    "f1": test_f1,
                    "train matthews corr": train_corr,
                    "dev matthews corr": dev_corr
                }

                if best_results is None:
                    best_results = current_results
                elif current_results["test accuracy"] > best_results["test accuracy"]:
                    best_results = current_results

            final_output = output_folder_base + "/summary/"
            final_results = Results(final_output)

            final_results.add_result_lines([
                "Stats on best model",
                "learning rate: {}".format(best_results["learning rate"]),
                "weight decay: {}".format(best_results["weight decay"]),
                "dropout rate: {}".format(best_results["dropout rate"]),
                "train accuracy: {}".format(best_results["train accuracy"]),
                "test accuracy: {}".format(best_results["test accuracy"]),
                "precision: {}".format(best_results["precision"]),
                "recall: {}".format(best_results["recall"]),
                "f1: {}".format(best_results["f1"]),
                "train matthews corr: {}".format(best_results["train matthews corr"]),
                "dev matthews corr: {}".format(best_results["dev matthews corr"])
            ])

def write_results(results, heading, test_preds, test_true, train_preds, train_true):
    accuracy, precision, recall, f1 = produce_accuracy_precision_recall_f1(test_preds, test_true)
    train_accuracy, _, _, _ = produce_accuracy_precision_recall_f1(train_preds, train_true)
    results.add_result_lines([
        heading,
        "Train Accuracy: {}".format(train_accuracy),
        "Test Accuracy: {}".format(accuracy),
        "Precision: {}".format(precision),
        "Recall: {}".format(recall),
        "f1: {}".format(f1)
        ])

def save_hyperparameters(results, dataset_prefix, embeddings_file, learning_rate, epochs, batch_size, decay_rate, dropout):
    results.add_result_lines([
        "Hyperparameters values",
        "Dataset: {}".format(dataset_prefix),
        "Embeddings: {}".format(embeddings_file),
        "Learning rate: {}".format(learning_rate),
        "Number of training epochs: {}".format(epochs),
        "Batch size: {}".format(batch_size),
        "Decay rate: {}".format(decay_rate),
        "Dropout rate: {}".format(dropout),
    ])




if __name__ == '__main__':
    main()