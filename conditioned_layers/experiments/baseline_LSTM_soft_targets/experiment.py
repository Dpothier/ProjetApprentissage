import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import torch.nn as nn
import torch.optim as optim
from conditioned_layers.training.results import Results
from pytoune.framework import Model
from pytoune.framework.callbacks.clip_grad import ClipNorm
from pytoune.framework.callbacks.lr_scheduler import ExponentialLR
from pytoune.framework.callbacks.best_model_restore import BestModelRestore
from conditioned_layers.datasets.Cola.ColaDataset import ColaDataset
from conditioned_layers.training.metrics_util import *
from conditioned_layers.networks.baseline_lstm.lstm import LSTM_model
from conditioned_layers.training.embeddings import load
from conditioned_layers.training.loss import SoftCrossEntropyLoss, SoftenTargets
from conditioned_layers.training.dataloader import DataLoader
from conditioned_layers.training.random import set_random_seed
import os
import sklearn


TEST_MODE = False
SEED = 133


def main():
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    dataset_prefix = sys.argv[1]
    embeddings_file = sys.argv[2]
    gpu_usage = sys.argv[3]

    epochs = 50
    batch_size = 32

    if gpu_usage == 'cpu':
        use_gpu = False
    elif gpu_usage == 'gpu1':
        torch.cuda.set_device(1)
        use_gpu = True
    else:
        torch.cuda.set_device(0)
        use_gpu = True

    np.random.seed(133)
    torch.manual_seed(133)

    output_folder_base = os.path.dirname(os.path.realpath(__file__)) + "/results/"

    learning_rates = [0.0001, 0.00001]
    weight_decays = [0, 0.001, 0.005]
    dropout_rates = [0, 0.1, 0.2, 0.3]

    word_vectors, vocab_table = load(embeddings_file)


    train = DataLoader(ColaDataset("{}train.tsv".format(dataset_prefix), vocab_table,  TEST_MODE=TEST_MODE) , batch_size=batch_size)
    dev = DataLoader(ColaDataset("{}dev.tsv".format(dataset_prefix), vocab_table, TEST_MODE=TEST_MODE), batch_size=batch_size)
    # test = DataLoader(ColaDataset("{}test.tsv".format(dataset_prefix), vocab_table, TEST_MODE=TEST_MODE), batch_size=batch_size)

    len_max_seq = max([train.dataset.len_of_longest_sequence(), dev.dataset.len_of_longest_sequence()])
    best_model = None

    experiment_results = {}
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            for dropout_rate in dropout_rates:
                output_folder = output_folder_base +"{}_{}_{}/".format(learning_rate, weight_decay, dropout_rate)
                results = Results(output_folder)

                save_hyperparameters(results, dataset_prefix, embeddings_file, learning_rate, epochs, batch_size, weight_decay, dropout_rate)
                set_random_seed(SEED)

                # Create the classifier
                module = LSTM_model( word_vectors, d_word_vec=512, d_model=512, d_inner=2048, dropout=dropout_rate)

                classes_weight = calculate_weight(train)
                print("Weight of classes: {}".format(classes_weight))

                print('Model defined')
                optimizer = optim.Adam(module.parameters(), lr=learning_rate, weight_decay=weight_decay)
                gradient_clipping = ClipNorm(module.parameters(), 1)
                lr_scheduler = ExponentialLR(gamma=0.9)
                # early_stopping = EarlyStopping(patience=10)
                best_model_restore = BestModelRestore(monitor="val_acc")

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

                train_accuracy, train_precision, train_recall, train_f1 = produce_accuracy_precision_recall_f1(train_preds, train_true)
                test_accuracy, test_precision, test_recall, test_f1 = produce_accuracy_precision_recall_f1(test_preds, test_true)
                train_corr = sklearn.metrics.matthews_corrcoef(train_preds, train_true)
                dev_corr = sklearn.metrics.matthews_corrcoef(test_preds, test_true)

                experiment_results[(learning_rate, dropout_rate, weight_decay)] = {
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
                results.save_model(model)

        final_output = output_folder_base + "/summary/"
        final_results = Results(final_output)

        best_experiment_results = None
        for hyperparameters, results in experiment_results.items():
            if best_experiment_results is None:
                best_experiment_results = results
            elif results["test accuracy"] > best_experiment_results["test accuracy"]:
                best_experiment_results = results

        final_results.add_result_lines([
            "Stats on best model",
            "learning rate: {}".format(best_experiment_results["learning rate"]),
            "weight decay: {}".format(best_experiment_results["weight decay"]),
            "dropout rate: {}".format(best_experiment_results["dropout rate"]),
            "train accuracy: {}".format(best_experiment_results["train accuracy"]),
            "test accuracy: {}".format(best_experiment_results["test accuracy"]),
            "precision: {}".format(best_experiment_results["precision"]),
            "recall: {}".format(best_experiment_results["recall"]),
            "f1: {}".format(best_experiment_results["f1"]),
            "train matthews corr: {}".format(best_experiment_results["train matthews corr"]),
            "dev matthews corr: {}".format(best_experiment_results["dev matthews corr"])
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