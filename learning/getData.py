import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

def get_bdrv_data():
    with open('../data/bdrv_texts.csv', encoding="utf8") as f:
        rows = [{k: str(v) for k, v in row.items()}
             for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

        data = []
        targets = []
        for row in rows:
            data.append(row['text'])
            targets.append(row['label'])

        encoder = LabelEncoder()
        numeric_targets = encoder.fit_transform(targets)

        return data, numeric_targets


def get_carcomplaints_data():
    with open('../data/carcomplaints.csv', encoding="utf8") as f:
        rows = [{k: str(v) for k, v in row.items()}
             for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

        data = []
        targets = []
        for row in rows:
            data.append(row['details'])
            targets.append(row['system'])

        encoder = LabelEncoder()
        numeric_targets = encoder.fit_transform(targets)

        return data, numeric_targets


def get_some_carcomplaints_data():
    data, numeric_targets = get_carcomplaints_data()
    data, numeric_targets = shuffle(data, numeric_targets)


    return data[: 50000], numeric_targets[: 50000]


def get_data_from_both_datasets():
    with open('../data/bdrv_texts.csv', encoding="utf8") as f:
        rows = [{k: str(v) for k, v in row.items()}
             for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

        bdrv_data = []
        bdrv_targets = []
        for row in rows:
            bdrv_data.append(row['text'])
            bdrv_targets.append(row['label'])