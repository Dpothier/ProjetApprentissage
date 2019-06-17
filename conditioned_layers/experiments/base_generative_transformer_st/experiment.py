import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import torch.nn as nn
import torch.optim as optim
from training.results import Results
from pytoune.framework import Model
from pytoune.framework.callbacks.clip_grad import ClipNorm
from pytoune.framework.callbacks.lr_scheduler import ExponentialLR
from pytoune.framework.callbacks.best_model_restore import BestModelRestore
from pytoune.framework.callbacks import EarlyStopping
from datasets.Cola.ColaDataset import ColaDataset
from training.metrics_util import *
from networks.generative_transformer.base.Models import Transformer
from training.embeddings import load
from training.loss import SoftCrossEntropyLoss, SoftenTargets
from training.dataloader import DataLoader
from training.random import set_random_seed
import os
import sklearn
import click


TEST_MODE = False
SEED = 133

@click.command()
@click.option('-d', '--dataset', default="datasets/Cola/clean/")
@click.option('-e', '--embeddings', default="embeddings/glove.6B.100d.txt")
@click.option('-g', '--gpu', default="gpu0")
def main(dataset, embeddings, gpu):
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    epochs = 50
    batch_size = 32

    if gpu == 'cpu':
        print("Setting computation on cpu")
        use_gpu = False
    elif gpu == 'gpu1':
        print("Setting computation on gpu1")
        torch.cuda.set_device(1)
        use_gpu = True
    else:
        print("Setting computation on gpu0")
        torch.cuda.set_device(0)
        use_gpu = True

    np.random.seed(133)
    torch.manual_seed(133)

    output_folder_base = os.path.dirname(os.path.realpath(__file__)) + "/results/"

    learning_rates = [0.00005, 0.00001]
    weight_decays = [0]
    dropout_rates = [0]

    word_vectors, vocab_table = load(embeddings)


    train = DataLoader(ColaDataset("{}train.tsv".format(dataset), vocab_table, TEST_MODE=TEST_MODE), batch_size=batch_size)
    dev = DataLoader(ColaDataset("{}dev.tsv".format(dataset), vocab_table, TEST_MODE=TEST_MODE), batch_size=batch_size)
    # test = DataLoader(ColaDataset("{}test.tsv".format(dataset_prefix), vocab_table, TEST_MODE=TEST_MODE), batch_size=batch_size)

    len_max_seq = max([train.dataset.len_of_longest_sequence(), dev.dataset.len_of_longest_sequence()])
    best_results = None
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            for dropout_rate in dropout_rates:
                output_folder = output_folder_base +"{}_{}_{}/".format(learning_rate, weight_decay, dropout_rate)
                results = Results(output_folder)

                save_hyperparameters(results, dataset, embeddings, learning_rate, epochs, batch_size, weight_decay, dropout_rate)
                set_random_seed(SEED)

                # Create the classifier
                module = Transformer(word_vectors, len_max_seq, d_word_vec=128, d_model=128, d_inner=1024, n_layers=6, n_head=8, d_k=32, d_v=32, d_state=64, dropout=0.1)

                classes_weight = calculate_weight(train)
                print("Weight of classes: {}".format(classes_weight))

                print('Model defined')
                optimizer = optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
                gradient_clipping = ClipNorm(module.parameters(), 1)
                lr_scheduler = ExponentialLR(gamma=0.9)
                early_stopping = EarlyStopping(patience=25)
                best_model_restore = BestModelRestore(monitor="val_acc", mode="max")

                loss = SoftCrossEntropyLoss(soft_target_fct=SoftenTargets(2, 0.8), weight=classes_weight)


                model = Model(module, optimizer, loss, metrics=['accuracy'])

                if use_gpu:
                    model = model.cuda()

                model.fit_generator(train, dev, epochs=epochs, callbacks=[gradient_clipping, lr_scheduler, best_model_restore])

                test_loss, test_metrics, test_preds = model.evaluate_generator(dev, return_pred=True)
                train_loss, train_metrics, train_preds = model.evaluate_generator(train, return_pred=True)


                test_preds = flatten_and_discritize_preds(test_preds)
                test_true = get_targets(dev)

                train_preds = flatten_and_discritize_preds(train_preds)
                train_true = get_targets(train)

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