import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import os
from os import listdir
from os.path import isfile, isdir, join
import click
import re
from csv import DictReader, reader
from training.analyse_results.logs.log_files_processor import *
from training.analyse_results.logs.model_logs_processor import model_logs_processor
from training.analyse_results.logs.hyperparameters_extractor import *
from training.analyse_results.cheapest_successful_hyperparams_report import cheapest_successful_hyperparams_report
import training.analyse_results.architectural_metrics as model_types


base_folder = os.path.abspath("{}/grouped_results/".format(os.path.dirname(os.path.realpath(__file__))))
output_file = "{}/report.txt".format(os.path.dirname(os.path.realpath(__file__)))


models = [("Baseline CNN on mnist",
           model_logs_processor("{}/baseline_cnn_mnist".format(base_folder), max_accuracy_log_files_processor(), lr_t_state_size_extractor()),
           cheapest_successful_hyperparams_report(90, model_types.StandardCNN(in_channels=1, out_classes=10, width=28, length=28, kernel_size=3))),
          ( "Policy CNN on mnist",
            model_logs_processor("{}/policy_cnn_mnist".format(base_folder), max_accuracy_log_files_processor(), lr_t_state_size_seed_extractor()),
           cheapest_successful_hyperparams_report(90, model_types.PolicyCNN(in_channels=1, out_classes=10, width=28, length=28, kernel_size=3))),
          ("Baseline MLP on mnist",
            model_logs_processor("{}/baseline_mlp_mnist".format(base_folder), max_accuracy_log_files_processor(), lr_t_state_size_extractor()),
            cheapest_successful_hyperparams_report(90, model_types.StandardMLP(in_size=784, out_classes=10))),
          ("Policy MLP on mnist",
            model_logs_processor("{}/policy_mlp_mnist".format(base_folder), max_accuracy_log_files_processor(), lr_t_state_size_extractor()),
            cheapest_successful_hyperparams_report(90, model_types.PolicyMLP(in_size=784, out_classes=10)))
          ]

number_of_epochs = ["20"]


def main():
    with open(output_file, encoding="utf-8", mode="w+") as f:
        for name, log_processor, reporter in models:
            experiments_data = log_processor.process_experiments_on_model()

            for epoch in number_of_epochs:
                report = reporter.produce_report(experiments_data, epoch)
                f.write("Model: {}, training time: {}\n".format(name, epoch))
                f.write("Successful hyperparameters: \n")
                for hyperparams in report["success"]:
                    f.write("lr: {}, state_size: {}, seed_size: {}, t: {}\n".format(hyperparams["lr"], hyperparams["state_size"], hyperparams["seed_size"], hyperparams["t"]))
                f.write("Smallest successful hyperparams: {}\n".format(report["smallest_set"][0]))
                f.write("Smallest model has {} parameters\n".format(report["smallest_set"][1]))
                f.write("Fastest successful hyperparams: {}\n".format(report["fastest_set"][0]))
                f.write("Fastest model do {} operations per inferance\n".format(report["fastest_set"][1]))


if __name__ == '__main__':
    main()
