import csv
import re


def row_is_good(row):
    if not row['system']:
        return False
    if row['details'] == 'for recall information older than 1980 contact 1-800-333-0510 or 613-993-9851':
        return False
    return True


with open('../data/bdrv.csv', encoding="utf8") as f:
    a = [{k: str(v) for k, v in row.items()}
         for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]


recall_ids = []

recall_texts = []

for row in a:
    if row["recall_id"] not in recall_ids and row_is_good(row):
        recall_ids.append(row["recall_id"])
        text = row["details"].replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').lower()
        recall_texts.append({
            'text': text,
            'label': row["system"]
        })

with open('../data/bdrv_texts.csv', encoding='utf8', mode="w") as f:
    fieldnames = ['label', 'text']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in recall_texts:
        writer.writerow(row)

print('new file contains {} data points'.format(len(recall_texts)))