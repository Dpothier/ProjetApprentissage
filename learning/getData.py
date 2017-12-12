import csv
from sklearn.preprocessing import LabelEncoder

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
