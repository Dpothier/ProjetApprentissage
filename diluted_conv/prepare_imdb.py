import os
import pathlib
import random
import numpy as np
import pandas as pd

print(os.path.abspath(os.curdir))

def compile_dataset(input_directory, output_file):
    neg_directory = input_directory + "neg/"
    pos_directory = input_directory + "pos/"

    instances = []

    for filename in os.listdir(neg_directory):
        if filename.endswith(".txt"):
            f = open(neg_directory + filename)
            lines = f.read().replace('\t', ' ')
            if '\t' in lines:
                print("There are tabs in the review texts")
            instances.append({
                "review": lines,
                "sentiment": 0
            })

    for filename in os.listdir(pos_directory):
        if filename.endswith(".txt"):
            f = open(pos_directory + filename)
            lines = f.read().replace('\t', ' ')
            if '\t' in lines:
                print("There are tabs in the review texts")
            instances.append({
                "review": lines,
                "sentiment": 1
            })

    random.shuffle(instances)

    df = pd.DataFrame(instances)

    df.to_csv(output_file, sep="\t")


if __name__ == '__main__':
    compile_dataset("/home/dominique/git/ProjetApprentissage/diluted_conv/datasets/aclImdb/train/", "/home/dominique/git/ProjetApprentissage/diluted_conv/datasets/IMDB/train.csv")
    compile_dataset("/home/dominique/git/ProjetApprentissage/diluted_conv/datasets/aclImdb/test/", "/home/dominique/git/ProjetApprentissage/diluted_conv/datasets/IMDB/test.csv")