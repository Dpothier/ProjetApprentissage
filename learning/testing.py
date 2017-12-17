from sklearn.datasets import load_iris
from algorithms.SVM import SVM
from algorithms.MLP import MLP
from algorithms.ADABoost import ADABoost
from getData import get_carcomplaints_data
from sklearn.feature_extraction.text import CountVectorizer
import experiment.experiment_setup as setup
import csv

targets_bdrv = []
targets_carcomplaint = []
shared_targets = ['steering', 'brakes', 'electrical', 'engine', 'suspension']
data_points = []
with open('../data/bdrv_texts.csv', encoding="utf8") as f:
    rows = [{k: str(v) for k, v in row.items()}
            for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]
    for row in rows:
        if row['label'].lower() in shared_targets:
            targets_bdrv.append(row)


with open('../data/carcomplaints.csv', encoding="utf8") as f:
    rows = [{k: str(v) for k, v in row.items()}
         for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]
    for row in rows:
        if row['system'].lower() in shared_targets:
            targets_carcomplaint.append(row)

print(len(targets_bdrv))
print(len(targets_carcomplaint))
print(len(data_points))

