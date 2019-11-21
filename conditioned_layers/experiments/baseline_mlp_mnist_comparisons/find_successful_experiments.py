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
                result_by_epoch = []
                for row in reader:
                    result_by_epoch.append(row["val_acc"])

                if len(result_by_epoch) == 25:
                    seeds_results.append({
                                          "5" : float(result_by_epoch[4]),
                                          "10": float(result_by_epoch[9]),
                                          "15": float(result_by_epoch[14]),
                                          "20": float(result_by_epoch[19]),
                                          "25": float(result_by_epoch[24])})

        groups = re.match("lr_([\s\S]*?)_t_([\s\S]*?)_state_size_([\d]*)", experiment)
        lr = float(groups[1])
        t = float(groups[2])
        state_size = float(groups[3])

        avg_5_results  = sum(f["5"] for f in seeds_results) / len(seeds_results) if len(seeds_results) > 0 else 0
        avg_10_results = sum(f["10"] for f in seeds_results) / len(seeds_results) if len(seeds_results) > 0 else 0
        avg_15_results = sum(f["15"] for f in seeds_results) / len(seeds_results) if len(seeds_results) > 0 else 0
        avg_20_results = sum(f["20"] for f in seeds_results) / len(seeds_results) if len(seeds_results) > 0 else 0
        avg_25_results = sum(f["25"] for f in seeds_results) / len(seeds_results) if len(seeds_results) > 0 else 0

        experiments_results.append({
            "lr": lr,
            "t": t,
            "state_size": state_size,
            "avg_5_results": avg_5_results,
            "avg_10_results": avg_10_results,
            "avg_15_results": avg_15_results,
            "avg_20_results": avg_20_results,
            "avg_25_results": avg_25_results,
        })

    print("List of hyperparams that obtain 95% test results in 25 epochs")
    for experiment in experiments_results:
        if experiment["avg_25_results"] > 95:
            print("lr: {}, t: {}, state_size: {}".format(experiment["lr"], experiment["t"], experiment["state_size"]))

    print("List of hyperparams that obtain 95% test results in 20 epochs")
    for experiment in experiments_results:
        if experiment["avg_20_results"] > 95:
            print("lr: {}, t: {}, state_size: {}".format(experiment["lr"], experiment["t"], experiment["state_size"]))

    print("List of hyperparams that obtain 95% test results in 15 epochs")
    for experiment in experiments_results:
        if experiment["avg_15_results"] > 95:
            print("lr: {}, t: {}, state_size: {}".format(experiment["lr"], experiment["t"], experiment["state_size"]))

    print("List of hyperparams that obtain 95% test results in 10 epochs")
    for experiment in experiments_results:
        if experiment["avg_10_results"] > 95:
            print("lr: {}, t: {}, state_size: {}".format(experiment["lr"], experiment["t"], experiment["state_size"]))

    print("List of hyperparams that obtain 95% test results in 5 epochs")
    for experiment in experiments_results:
        if experiment["avg_5_results"] > 95:
            print("lr: {}, t: {}, state_size: {}, avg accuracy: {}".format(experiment["lr"], experiment["t"], experiment["state_size"], experiment["avg_5_results"]))




if __name__ == '__main__':
    main()