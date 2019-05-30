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
from diluted_conv.models.diluted_convolution import TCN, SingleLayerResidualTCN
from diluted_conv.models.non_residual_diluted_convolution import NonResidualDilutedConv
from diluted_conv.pytoune_util import *
import os
from diluted_conv.pad_collate import PadCollate
import time
from diluted_conv.callbacks import TimeCallback



EMBEDDINGS_DIM = 50
NUMBER_OF_EXPERIMENTS = 1
TEST_MODE = False
PATIENCE = 2
EPOCHS = 500
BATCH_SIZE = 32


def main():
    """
    Trains the LSTM-based integrated pattern-based and distributional method for hypernymy detection
    :return:
    """

    dataset_name = sys.argv[1]
    embeddings_file = sys.argv[2]
    base_output_folder = sys.argv[3]
    gpu_usage = sys.argv[4]

    lrs = [0.005, 0.001, 0.0005]
    weight_decays = [0, 0.01, 0.05, 0.1]

    if gpu_usage == 'cpu':
        use_gpu = False
    elif gpu_usage == 'gpu1':
        torch.cuda.set_device(1)
        use_gpu = True
    else:
        torch.cuda.set_device(0)
        use_gpu = True

    # np.random.seed(133)

    word_vectors, vocab_table = load_embeddings(embeddings_file)

    train = DataLoader(IMDBDataset("{}train.csv".format(dataset_name), vocab_table), batch_size=BATCH_SIZE,
                       collate_fn=PadCollate(dim=0))
    test = DataLoader(IMDBDataset("{}test.csv".format(dataset_name), vocab_table), batch_size=BATCH_SIZE,
                      collate_fn=PadCollate(dim=0))

    # one_layer_residual with base equal to kernel
    module = SingleLayerResidualTCN(embedding_vectors=word_vectors,
                                    dilution_function=lambda kernel, layer: (kernel) ** layer, depth=7)
    output_folder = base_output_folder + "single_layer_residual_base_equal_to_kernel/"
    experiment(train, test, module, word_vectors, lrs, weight_decays, dataset_name, embeddings_file, output_folder,
               use_gpu)

    # one_layer_residual with bigger base than kernel
    # module = SingleLayerResidualTCN(embedding_vectors=word_vectors,
    #                                 dilution_function=lambda kernel, layer: (kernel + 1) ** layer, depth=6)
    # output_folder = base_output_folder + "single_layer_residual_base_bigger_than_kernel"
    # experiment(train, test, module, word_vectors, lrs, weight_decays, dataset_name, embeddings_file, output_folder, use_gpu)
    #
    # # one_layer_residual with bigger base than kernel and depth 5 instead of 6
    # module = SingleLayerResidualTCN(embedding_vectors=word_vectors,
    #                                 dilution_function=lambda kernel, layer: (kernel + 1) ** layer, depth=5)
    # output_folder = base_output_folder + "single_layer_residual_base_bigger_than_kernel_depth_5"
    # experiment(train, test, module, word_vectors, lrs, weight_decays, dataset_name, embeddings_file, output_folder,
    #            use_gpu)

    # two_layer_residual_with_smaller_base_than_kernel
    # module = TCN(embedding_vectors=word_vectors,
    #                                 dilution_function=lambda kernel, layer: (kernel - 1) ** layer, depth=5)
    # output_folder = base_output_folder + "two_layer_residual_with_smaller_base_than_kernel"
    # experiment(train, test, module, word_vectors, lrs, weight_decays, dataset_name, embeddings_file, output_folder,
    #            use_gpu)
    #
    # #two_layer_residual_with_equal_base_and_kernel
    # module = TCN(embedding_vectors=word_vectors,
    #                                 dilution_function=lambda kernel, layer: (kernel) ** layer, depth=5)
    # output_folder = base_output_folder + "two_layer_residual_with_equal_base_and_kernel"
    # experiment(train, test, module, word_vectors, lrs, weight_decays, dataset_name, embeddings_file, output_folder,
    #            use_gpu)

    # two_layer_residual_with_bigger_base_than_kernel
    module = TCN(embedding_vectors=word_vectors,
                                    dilution_function=lambda kernel, layer: (kernel + 1) ** layer, depth=4)
    output_folder = base_output_folder + "two_layer_residual_with_bigger_base_than_kernel/"
    experiment(train, test, module, word_vectors, lrs, weight_decays, dataset_name, embeddings_file, output_folder,
               use_gpu)





def experiment(train, test, module, word_vectors, learning_rates, decay_rates, dataset_name, embeddings_file, output_folder, use_gpu):
    for lr in learning_rates:
        for weight_decay in decay_rates:

            output_file = output_folder + "{}_{}_".format(lr, weight_decay)
            results = Results(output_file)
            results.add_result_line("Using kernel sized dilation")
            save_hyperparameters(results, dataset_name, embeddings_file, lr, EPOCHS, BATCH_SIZE, weight_decay)

            for i in range(NUMBER_OF_EXPERIMENTS):
                results.add_result_line("Experiment: {}".format(i+1))
                # Create the classifier
                module = SingleLayerResidualTCN(embedding_vectors=word_vectors, dilution_function=lambda kernel, layer: (kernel) ** layer, depth=6)

                print('Model defined')
                optimizer = optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
                gradient_clipping = ClipNorm(module.parameters(), 1)
                lr_scheduler = ExponentialLR(gamma=0.9)
                early_stopping = EarlyStopping(patience=PATIENCE)
                best_model_restore = BestModelRestore()
                time_callback = TimeCallback()

                loss = nn.CrossEntropyLoss()


                model = Model(module, optimizer, loss, metrics=['accuracy'])

                if use_gpu:
                    model = model.cuda()



                model.fit_generator(train, test, epochs=EPOCHS, callbacks=[gradient_clipping, lr_scheduler, early_stopping, best_model_restore, time_callback])

                epoch_times = time_callback.epoch_times

                results.add_result_lines([
                    "Trained for {} epochs".format(len(epoch_times)),
                    "Average epoch time: {}".format(sum(epoch_times)/len(epoch_times))])


                loss, metrics, preds = model.evaluate_generator(test, return_pred=True)

                preds = flatten_and_discritize_preds(preds)
                true = get_targets(test)

                write_result_for_filter(results, "Results on test set", preds, true)

                loss, metrics, preds = model.evaluate_generator(train, return_pred=True)

                preds = flatten_and_discritize_preds(preds)
                true = get_targets(train)

                write_result_for_filter(results, "Results on train set", preds, true)

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