import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import os
from os import listdir
from os.path import isfile, isdir, join
import click
import re
from csv import DictReader, reader


TEST_MODE = False
SEED = 133
number_of_epochs = [5,10,15,20]

@click.command()
def main():

    output_folder_base = os.path.dirname(os.path.realpath(__file__)) + "/results/"

    experiments_name = listdir(output_folder_base)
    experiments_results = []

    for experiment in experiments_name:
        logs = [f for f in listdir("{}/{}".format(output_folder_base, experiment)) if f[-4:] == ".csv"]

        seeds_results = []
        for log in logs:
            with open("{}/{}/{}".format(output_folder_base, experiment,log), newline='') as csvfile:
                reader = DictReader(csvfile)
                result_by_epoch = {}
                max_result = 0
                for index, row in enumerate(reader):
                    if float(row["val_acc"]) > max_result:
                        max_result = float(row["val_acc"])

                    if index + 1 in number_of_epochs:
                        result_by_epoch[index + 1] = max_result

                seeds_results.append(result_by_epoch)

        groups = re.match("lr_([\s\S]*?)_t_([\s\S]*?)_state_size_([\d]*)", experiment)
        experiment_result = {
            "lr":float(groups[1]),
            "t" : float(groups[2]),
            "state_size" : float(groups[3])
        }

        for epoch in number_of_epochs:
            experiment_result[epoch] = sum(f[epoch] for f in seeds_results) / len(seeds_results) if len(seeds_results) > 0 else 0

        experiments_results.append(experiment_result)


    for epochs in number_of_epochs:
        successes = []
        for experiment in experiments_results:
            if experiment[epochs] > 95:
                successes.append((experiment["t"], experiment["state_size"]))
        successes = list(set(successes))
        successes.sort(key=lambda item: (item[0], item[1]))
        print("List of hyperparams that obtain 95% in {} epochs".format(epochs))
        for success in successes:
            print("t: {}, state_size: {}".format(success[0], success[1]))


def count_parameters_in_mlp(t, state_size):
    

if __name__ == '__main__':
    main()