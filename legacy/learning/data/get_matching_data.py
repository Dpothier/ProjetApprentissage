import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def get_data():
    with open('../data/tiny_carcomplaints.csv', encoding="utf8") as f:
        carcomplaints = [{k: str(v) for k, v in row.items()}
             for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

    with open('../data/bdrv.csv', encoding="utf8") as f:
        bdrv = [{k: str(v) for k, v in row.items()}
             for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

    with open('../data/carcomplaints_bdrv_link.csv', encoding="utf8") as f:
        matches = [{k: str(v) for k, v in row.items()}
             for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

    carcomplaints_ids = []
    bdrv_ids = []
    for match in matches:
        if not match['carcomplaints_id'] in carcomplaints_ids:
            carcomplaints_ids.append(match['carcomplaints_id'])

        if not match['bdrv_id'] in bdrv_ids:
            bdrv_ids.append(match['bdrv_id'])


    filtered_carcomplaints = []
    for row in carcomplaints:
        if row['id'] in carcomplaints_ids:
            filtered_carcomplaints.append(row)

    filtered_bdrv = []
    filtered_bdrv_ids = []
    for row in bdrv:
        if row['recall_id'] in bdrv_ids and row['recall_id'] not in filtered_bdrv_ids:
            filtered_bdrv.append(row)
            filtered_bdrv_ids.append(row['recall_id'])

    return filtered_carcomplaints, filtered_bdrv, matches

