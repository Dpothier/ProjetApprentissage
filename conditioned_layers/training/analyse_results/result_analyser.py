import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import os
from os import listdir
from os.path import isfile, isdir, join
import click
import re
from csv import DictReader, reader
from training.analyse_results.logs.log_files_processor import max_accuracy_file_processor
from training.analyse_results.logs.model_logs_processor import experiment_logs_processor

class result_success_analyser():
    def __init__(self, model_directory, hyperparams_regex, logs_factory_method, success_threshold, model_size_method, model_operations_method):
        self.model_directory = model_directory
        self.hyperparams_regex = hyperparams_regex
        self.logs_factory_method = logs_factory_method
        self.success_threshold = success_threshold
        self.model_size_method = model_size_method
        self.model_operations_method = model_operations_method

        self.log_reader = experiment_logs_processor(max_accuracy_file_processor())
        self.aggregate = average_max_accuracy_logs_aggregator()





    def produce_success_report(self, training_epochs):
        successful_experiments = self.filter_successful_experiments(training_epochs)

        smaller_successful_experiment = self.find_smaller_model(successful_experiments)
        faster_successful_experiment = self.find_faster_model(successful_experiments)

        return {
            "successful_experiments" : successful_experiments,
            "smaller model size": smaller_successful_experiment[1],
            "smaller model": smaller_successful_experiment[0]["name"],
            "faster model operations": faster_successful_experiment[1],
            "faster model": faster_successful_experiment[0]["name"]
        }

    def filter_successful_experiments(self, training_epochs):
        successful_experiments = []
        for experiment in self.max_accuracies:
            max_accuracy_at_epoch = experiment[training_epochs]
            if max_accuracy_at_epoch > self.success_threshold:
                successful_experiments.append(experiment)

        return successful_experiments

    def find_smaller_model(self, experiments):
        smaller_model = None
        smaller_model_size = None

        for experiment in experiments:
            if smaller_model is None:
                smaller_model = experiment
                smaller_model_size = self.model_size_method(experiment)
            else:
                model_size = self.model_size_method(experiment)
                if model_size < smaller_model_size:
                    smaller_model_size = model_size
                    smaller_model = experiment

        return smaller_model, smaller_model_size

    def find_faster_model(self, experiments):
        faster_model = None
        faster_model_op_numbers = None
        for experiment in experiments:
            if faster_model is None:
                faster_model = experiment
                faster_model_op_numbers = self.model_operations_method(experiment)
            else:
                model_size = self.model_operations_method(experiment)
                if model_size < faster_model_op_numbers:
                    faster_model_op_numbers = model_size
                    faster_model = experiment

        return faster_model, faster_model_op_numbers


class ResultDataExtractor():

    def __init__(self, base_folder, hyperparams_extractor, log_reader, log_aggregator):
        self.base_folder = base_folder
        self.hyperparams_extractor = hyperparams_extractor
        self.log_reader = log_reader
        self.log_aggregator = log_aggregator

    def extract_data(self):
        experiments = os.listdir(self.base_folder)

        experiments_data = []
        for experiment in experiments:
            experiment_folder = "{}/{}".format(self.base_folder, experiment)
            experiment_data = {"hyperparameters": self.hyperparams_extractor(experiment_folder)}

            experiment_data.update(self.read_logs(experiment_folder))
            experiments_data.append(experiment_data)

        return experiments_data



    def read_logs(self, experiment_directory):
            logs = [f for f in listdir(experiment_directory) if f[-4:] == ".csv"]
            for log in logs:
                with open("{}/{}".format(experiment_directory, log), newline='') as csvfile:
                    experiment_logs = self.log_reader(csvfile)

            aggregated_logs = self.aggregate(experiment_logs)

            return aggregated_logs