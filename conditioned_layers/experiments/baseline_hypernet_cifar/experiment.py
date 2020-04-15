import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Subset
from training.results import Results
from poutyne.framework import Model

from poutyne.framework.callbacks.best_model_restore import BestModelRestore
from poutyne.framework.callbacks.lr_scheduler import MultiStepLR
from poutyne.framework.callbacks import CSVLogger
from training.metrics_util import *
from networks.static_hypernetwork.network import PrimaryNetwork
from training.random import set_random_seed
import os
import math
import click
import random


TEST_MODE = False
SEED = 133

@click.command()
@click.option('-g', '--gpu', default="gpu0")
@click.option('-f', '--fraction', default=0.5)
@click.option('-c', '--channels', default=16)
def main(gpu, fraction, channels):
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """


    batch_size = 128


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

    filter_size = channels **2

    learning_rates = [0.002, 0.001]
    weight_decays = [0.0005]
    layer_emb_sizes = [filter_size/4]
    base_channel_counts = [channels]
    seeds = [133, 42]
    epochs = 0

    lr_schedule = [75, 150, 200, 225, 250, 275]

    max_epoch = 1

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    cifar_trainset = datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform_train)
    fraction_indices = get_trainset_fraction_indices(cifar_trainset.targets, cifar_trainset.class_to_idx.values(), fraction)
    cifar_trainset = Subset(cifar_trainset, fraction_indices)

    cifar_devset = datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform_test)

    trainloader = DataLoader(cifar_trainset, batch_size=batch_size, num_workers=4)
    devloader = DataLoader(cifar_devset, batch_size=batch_size, num_workers=4)

    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            for layer_emb_size in layer_emb_sizes:
                for base_channel_count in base_channel_counts:
                    output_folder = output_folder_base + "lr_{}_wd_{}_layer_emb_{}_base_channels_{}_train_pct_{}/"\
                        .format(learning_rate, weight_decay, layer_emb_size, base_channel_count, fraction)
                    results = Results(output_folder)
                    save_hyperparameters(results, learning_rate, weight_decay, epochs, batch_size, layer_emb_size, base_channel_count, fraction)
                    seed_results = {}

                    for seed in seeds:
                        set_random_seed(seed)

                        # Create the classifier
                        module = PrimaryNetwork()

                        classes_weight = calculate_weight(trainloader)

                        optimizer = optim.Adam(module.parameters(), lr=learning_rate)
                        csvlogger = CSVLogger("{}/{}_log.csv".format(output_folder, seed))
                        best_model_restore = BestModelRestore(monitor="val_acc", mode="max")
                        lr_scheduler = MultiStepLR(milestones=lr_schedule, gamma=0.5)

                        loss = nn.CrossEntropyLoss(weight=classes_weight)


                        model = Model(module, optimizer, loss, metrics=['accuracy'])

                        if use_gpu:
                            model = model.cuda()

                        model.fit_generator(trainloader, devloader, epochs=max_epoch, callbacks=[best_model_restore, csvlogger, lr_scheduler])

                        test_loss, test_metrics, test_preds = model.evaluate_generator(devloader, return_pred=True)
                        train_loss, train_metrics, train_preds = model.evaluate_generator(trainloader, return_pred=True)


                        test_preds = flatten_and_discritize_preds(test_preds)
                        test_true = get_targets(devloader)

                        train_preds = flatten_and_discritize_preds(train_preds)
                        train_true = get_targets(trainloader)

                        train_accuracy, train_precision, train_recall, train_f1 = produce_accuracy_precision_recall_f1(
                            train_preds, train_true, "micro")
                        test_accuracy, test_precision, test_recall, test_f1 = produce_accuracy_precision_recall_f1(test_preds,
                                                                                                                   test_true, "micro")
                        seed_results[seed] = {
                            "train accuracy": train_accuracy.item(),
                            "test accuracy": test_accuracy.item(),
                            "precision": test_precision.item(),
                            "recall": test_recall.item(),
                            "f1": test_f1.item()
                        }
                        results.add_result(seed, seed_results[seed])

                    number_of_seeds = len(seeds)
                    average_results = {
                        "train accuracy": sum([result["train accuracy"] for result in seed_results.values()]) / number_of_seeds,
                        "test accuracy": sum([result["test accuracy"] for result in seed_results.values()]) / number_of_seeds,
                        "precision": sum([result["precision"] for result in seed_results.values()]) / number_of_seeds,
                        "recall": sum([result["recall"] for result in seed_results.values()]) / number_of_seeds,
                        "f1": sum([result["f1"] for result in seed_results.values()]) / number_of_seeds,
                    }
                    results.add_result("average", average_results)


def write_results(results, seed, test_preds, test_true, train_preds, train_true):
    accuracy, precision, recall, f1 = produce_accuracy_precision_recall_f1(test_preds, test_true, "micro")
    train_accuracy, _, _, _ = produce_accuracy_precision_recall_f1(train_preds, train_true, "micro")

    results.add_result(seed, {
        "Train Accuracy": train_accuracy.item(),
        "Test Accuracy": accuracy.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "f1": f1.item()
    })

def save_hyperparameters(results, learning_rate, weight_decay, epochs, batch_size, emb_layer_size, base_channel_count, training_pct):
    results.add_hyperparameters({
        "Learning rate": learning_rate,
        "Weight decay": weight_decay,
        "Number of training epochs": epochs,
        "Batch size": batch_size,
        "emb_layer_size": emb_layer_size,
        "base_channel_count": base_channel_count,
        "training_pct": training_pct
    })

def get_trainset_fraction_indices(targets, classes, fraction):
    indices = []
    for klass in classes:
        class_indices = [i for i, e in enumerate(targets) if e == klass]
        subset_size = math.floor(len(class_indices) * fraction)
        indices.extend(random.sample(class_indices, subset_size))

    random.shuffle(indices)
    return indices





if __name__ == '__main__':
    main()