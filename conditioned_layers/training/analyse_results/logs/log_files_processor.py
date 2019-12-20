from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import chain
from csv import DictReader
import collections, functools, operator

# Process the logs produced by each seed of an experiment.
# Each type of log extract different information
# Each log is processed sequentially, parsing each line with the overloaded parse_line method
# At the end of a file, the end_file method saves the extracted date for that file
# When all files are processed, aggregate_files produces the final metric for the experiment (could average the results across seed)
class log_files_processor():

    def process(self, file):
        with open(file, newline='') as file:
            reader = DictReader(file)
            for line in reader:
                self.parse_line(line)
            self.end_file()

        return self.aggregate_files()

    @abstractmethod
    def parse_line(self, line):
        pass

    @abstractmethod
    def end_file(self):
        pass

    # Should output a dictionary with a single key identifying the type of data extracted by the processor
    # The value of that key should be another dictionary with the actual keys and values extracted
    @abstractmethod
    def aggregate_files(self):
        pass

# If multiple metrics must be extracted from the logs, this class take charge of it seemlessly for the caller
class multi_log_files_processor(log_files_processor):

    def __init__(self, proccesors):
        self.processors = proccesors
        self.processed_files = []

    def process(self, line):
        for processor in self.processors:
            processor.process(line)

    def end_file(self):
        for processor in self.processors:
            processor.end_file()

    # The processors should have different names or they will overide one another
    def aggregate_files(self):
        outputs = {}
        for processor in self.processors:
            outputs.update(processor.output())

        return outputs



# Max accuracy is the highest accuracy obtained on the validation set up to a given epoch
# Expects the line to be fed in chronological order
class max_accuracy_log_files_processor(log_files_processor):

    def __init__(self):
        self.name = "max_val_acc"
        self.max_val_acc = 0
        self.max_values_acc_by_epoch = {}
        self.processed_files = []

    def process(self, line):
        val_acc = float(line["val_acc"])
        if val_acc > self.max_val_acc:
            self.max_val_acc = val_acc

        self.max_values_acc_by_epoch["{}".format(line["epoch"])] = self.max_val_acc

    def end_file(self):
        self.processed_files.append(self.max_values_acc_by_epoch)
        self.max_val_acc = 0
        self.max_values_acc_by_epoch = {}

    def aggregate_files(self):
        assert len(self.processed_files) > 0

        result = dict(functools.reduce(operator.add, map(collections.Counter, self.processed_files)))

        return {self.name: {key: result[key]/len(self.processed_files) for key in result.keys()}}




