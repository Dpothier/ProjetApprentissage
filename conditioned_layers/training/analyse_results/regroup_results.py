import sys

sys.path.append('../common')
sys.path.append('/mnt/storage/dpothier/tmp/pytoune')

import os
from os import listdir
from os.path import isfile, isdir, join
import click
import shutil
import re
from csv import DictReader, reader

models = [
    {"output_path": "baseline_cnn_mnist",
     "input_paths": [
         "baseline_cnn_mnist_grid_T_4",
         "baseline_cnn_mnist_grid_T_8",
         "baseline_cnn_mnist_grid_T_12"
     ]},
    {"output_path": "policy_cnn_mnist",
     "input_paths": [
         "policy_cnn_mnist_grid_T_4",
         "policy_cnn_mnist_grid_T_8",
         "policy_cnn_mnist_grid_T_12"
     ]},
    {"output_path": "baseline_mlp_mnist",
     "input_paths": [
         "baseline_mlp_mnist_comparisons"
     ]},
    {"output_path": "policy_mlp_mnist",
     "input_paths": [
         "policy_mlp_mnist_comparisons"
     ]}
]

@click.command()
@click.option('-i', '--input')
def main(input):
    input_base_folder = input
    output_base_folders = os.path.abspath("{}/grouped_results/".format(os.path.dirname(os.path.realpath(__file__))))

    shutil.rmtree(output_base_folders)
    os.mkdir(output_base_folders)

    for model in models:
        for path in model["input_paths"]:
            input_path = "{}/{}/results".format(input_base_folder, path)
            directories = os.listdir(input_path)
            for directory in directories:
                if directory != "summary":
                    shutil.copytree("{}/{}".format(input_path, directory), "{}/{}/{}".format(output_base_folders, model["output_path"], directory))




if __name__ == '__main__':
    main()