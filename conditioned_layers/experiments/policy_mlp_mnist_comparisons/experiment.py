import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from training.results import Results
from poutyne.framework import Model
from poutyne.framework.callbacks.clip_grad import ClipNorm
from poutyne.framework.callbacks.lr_scheduler import ExponentialLR
from poutyne.framework.callbacks.best_model_restore import BestModelRestore
from poutyne.framework.callbacks import EarlyStopping
from poutyne.framework.callbacks import CSVLogger
from datasets.Cola.ColaDataset import ColaDataset
from training.metrics_util import *
import networks.policy_mlp.policymlp as policymlp
from training.embeddings import load
from training.loss import SoftCrossEntropyLoss, SoftenTargets
from training.random import set_random_seed
import os
import sklearn
import click


TEST_MODE = False
SEED = 133

@click.command()
@click.option('-g', '--gpu', default="gpu1")
def main(gpu):
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    epochs = 25
    batch_size = 32
    success_treshold = 0.95


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

    output_folder_base = os.path.dirname(os.path.realpath(__file__)) + "/results/"

    learning_rates = [0.001, 0.0001, 0.00001]
    Ts = [1,2,4,6]
    state_sizes = [8,16,32,64,128]
    seeds = [133, 256, 42, 79, 666, 123, 456, 789, 780, 682]

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_trainset = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=batch_size)
    mnist_devset = DataLoader(datasets.MNIST(root='./data', train=False, download=True, transform=transform), batch_size=batch_size)

    best_results = None
    all_average_results = []
    for learning_rate in learning_rates:
        for T in Ts:
            for state_size in state_sizes:
                output_folder = output_folder_base + "lr_{}_t_{}_state_size_{}/".format(learning_rate, T, state_size)
                results = Results(output_folder)
                save_hyperparameters(results, learning_rate, epochs, batch_size, T, state_size)
                seed_results = {}

                for seed in seeds:
                    set_random_seed(seed)

                    # Create the classifier
                    module = policymlp.BuildFullResidualPolicyMLPOnlyGru(T=T, input_size=784,state_size=state_size,query_size=state_size, number_of_primitives=state_size, seed_size=state_size, output_size=10)

                    classes_weight = calculate_weight(mnist_trainset)

                    optimizer = optim.Adam(module.parameters(), lr=learning_rate)
                    csvlogger = CSVLogger("{}/{}_log.csv".format(output_folder, seed))
                    best_model_restore = BestModelRestore(monitor="val_acc", mode="max")

                    loss = nn.CrossEntropyLoss(weight=classes_weight)


                    model = Model(module, optimizer, loss, metrics=['accuracy'])

                    if use_gpu:
                        model = model.cuda()

                    primitives_at_start = module.initial_seed.clone()
                    model.fit_generator(mnist_trainset, mnist_devset, epochs=epochs, callbacks=[best_model_restore, csvlogger])

                    primitives_at_end = module.initial_seed.clone()
                    print("Norm of the difference between primitives: {}".format((primitives_at_start - primitives_at_end).norm()))

                    test_loss, test_metrics, test_preds = model.evaluate_generator(mnist_devset, return_pred=True)
                    train_loss, train_metrics, train_preds = model.evaluate_generator(mnist_trainset, return_pred=True)


                    test_preds = flatten_and_discritize_preds(test_preds)
                    test_true = get_targets(mnist_devset)

                    train_preds = flatten_and_discritize_preds(train_preds)
                    train_true = get_targets(mnist_trainset)

                    write_results(results, "Results", test_preds, test_true, train_preds, train_true)
                    results.save_model(model)

                    train_accuracy, train_precision, train_recall, train_f1 = produce_accuracy_precision_recall_f1(
                        train_preds, train_true, "micro")
                    test_accuracy, test_precision, test_recall, test_f1 = produce_accuracy_precision_recall_f1(test_preds,
                                                                                                               test_true, "micro")
                    train_corr = sklearn.metrics.matthews_corrcoef(train_preds, train_true)
                    dev_corr = sklearn.metrics.matthews_corrcoef(test_preds, test_true)

                    seed_results[seed] = {
                        "train accuracy": train_accuracy,
                        "test accuracy": test_accuracy,
                        "precision": test_precision,
                        "recall": test_recall,
                        "f1": test_f1
                    }

                number_of_seeds = len(seeds)
                average_results = {
                    "learning rate": learning_rate,
                    "T": T,
                    "state size": state_size,
                    "train accuracy": sum([result["train accuracy"] for result in seed_results.values()]) / number_of_seeds,
                    "test accuracy": sum([result["test accuracy"] for result in seed_results.values()]) / number_of_seeds,
                    "precision": sum([result["precision"] for result in seed_results.values()]) / number_of_seeds,
                    "recall": sum([result["recall"] for result in seed_results.values()]) / number_of_seeds,
                    "f1": sum([result["f1"] for result in seed_results.values()]) / number_of_seeds,
                }
                all_average_results.append(average_results)
                if best_results is None:
                    best_results = average_results
                elif average_results["test accuracy"] > best_results["test accuracy"]:
                    best_results = average_results

    final_output = output_folder_base + "/summary/"
    final_results = Results(final_output)

    final_results.add_result_line("Following hyperparameters achieved success")
    for results in all_average_results:
        if results["test accuracy"] > success_treshold:
            final_results.add_result_line("Lr: {}, T: {}, State size: {}   Final test accuracy: {}".format(
                results["learning rate"], results["T"], results["state size"], results["test accuracy"]))



    final_results.add_result_lines([
        "Stats on best model",
        "learning rate: {}".format(best_results["learning rate"]),
        "T: {}".format(best_results["T"]),
        "state size: {}".format(best_results["state size"]),
        "train accuracy: {}".format(best_results["train accuracy"]),
        "test accuracy: {}".format(best_results["test accuracy"]),
        "precision: {}".format(best_results["precision"]),
        "recall: {}".format(best_results["recall"]),
        "f1: {}".format(best_results["f1"])
    ])

def write_results(results, heading, test_preds, test_true, train_preds, train_true):
    accuracy, precision, recall, f1 = produce_accuracy_precision_recall_f1(test_preds, test_true, "micro")
    train_accuracy, _, _, _ = produce_accuracy_precision_recall_f1(train_preds, train_true, "micro")
    results.add_result_lines([
        heading,
        "Train Accuracy: {}".format(train_accuracy),
        "Test Accuracy: {}".format(accuracy),
        "Precision: {}".format(precision),
        "Recall: {}".format(recall),
        "f1: {}".format(f1)
        ])

def save_hyperparameters(results, learning_rate, epochs, batch_size, T, state_size):
    results.add_result_lines([
        "Hyperparameters values",
        "Learning rate: {}".format(learning_rate),
        "Number of training epochs: {}".format(epochs),
        "Batch size: {}".format(batch_size),
        "T: {}".format(T),
        "state size {}".format(state_size),

    ])




if __name__ == '__main__':
    main()