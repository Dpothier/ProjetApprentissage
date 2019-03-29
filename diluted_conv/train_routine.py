import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from diluted_conv.results import Results
from pytoune.framework import Model
from pytoune.framework.callbacks.clip_grad import ClipNorm
from pytoune.framework.callbacks.lr_scheduler import ExponentialLR
from pytoune.framework.callbacks.earlystopping import EarlyStopping
from pytoune.framework.callbacks.best_model_restore import BestModelRestore
from torch.utils.data import DataLoader
from torchtext.datasets import SST, IMDB
from torchtext.vocab import GloVe
from diluted_conv.embeddings import load_embeddings
from diluted_conv.IMDB_dataset import IMDBDataset
from diluted_conv.models.diluted_convolution import TCN
from diluted_conv.pytoune_util import *
import os
from diluted_conv.pad_collate import PadCollate



EMBEDDINGS_DIM = 50
TEST_MODE = False


def main():
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    dataset_name = sys.argv[1]
    embeddings_file = sys.argv[2]
    output_folder = sys.argv[3]
    gpu_usage = sys.argv[4]

    lrs = [0.001, 0.0001, 0.00001]
    weight_decays = [0, 0.1, 0.2]
    epochs = 30
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

    for lr in lrs:
        for weight_decay in weight_decays:

            output_file = output_folder + "{}_{}.txt".format(lr, weight_decay)
            results = Results(output_folder)

            results.add_result_line("Using kernel sized dilation")
            save_hyperparameters(results, dataset_name, embeddings_file, lr, epochs, batch_size, weight_decay)

            word_vectors, vocab_table = load_embeddings(embeddings_file)

            train = DataLoader(IMDBDataset("{}train.csv".format(dataset_name), vocab_table), batch_size=batch_size, collate_fn=PadCollate(dim=0))
            test = DataLoader(IMDBDataset("{}test.csv".format(dataset_name), vocab_table), batch_size=batch_size, collate_fn=PadCollate(dim=0) )

            # Create the classifier
            module = TCN(embedding_vectors=word_vectors, dilution_function=lambda kernel, layer: (kernel-1) ** layer, depth=7)

            print('Model defined')
            optimizer = optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
            gradient_clipping = ClipNorm(module.parameters(), 1)
            lr_scheduler = ExponentialLR(gamma=0.9)
            early_stopping = EarlyStopping(patience=2)
            best_model_restore = BestModelRestore()

            loss = nn.CrossEntropyLoss()


            model = Model(module, optimizer, loss, metrics=['accuracy'])

            if use_gpu:
                model = model.cuda()



            total_time = os.times()
            model.fit_generator(train, test, epochs=epochs, callbacks=[gradient_clipping, lr_scheduler, early_stopping, best_model_restore])

            loss, metrics, preds = model.evaluate_generator(test, return_pred=True)

            preds = flatten_and_discritize_preds(preds)
            true = get_targets(test)

            write_result_for_filter(results, "Results", preds, true)

            # Save the best model to a file
            results.save_model(model)

def write_result_for_filter(results, heading, preds, true):
    accuracy, precision, recall, f1 = produce_accuracy_precision_recall_f1(preds, true)
    results.add_result_lines([
        heading,
        "Accuracy: {}".format(accuracy),
        "Precision: {}".format(precision),
        "Recall: {}".format(recall),
        "f1: {}".format(f1)
        ])

def save_hyperparameters(results, dataset_prefix, embeddings_file, learning_rate, epochs, batch_size, decay_rate):
    results.add_result_lines([
        "Hyperparameters values",
        "Dataset: {}".format(dataset_prefix),
        "Embeddings: {}".format(embeddings_file),
        "Learning rate: {}".format(learning_rate),
        "Number of training epochs: {}".format(epochs),
        "Batch size: {}".format(batch_size),
        "Decay rate: {}".format(decay_rate),
    ])



if __name__ == '__main__':
    print(os.path.abspath(os.curdir))
    main()