from elasticsearch import Elasticsearch
import csv

with open('../data/bdrv.csv', encoding="utf8") as f:
    bdrv = [{k: str(v) for k, v in row.items()}
         for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

with open('../data/carcomplaints_bdrv_link.csv', encoding="utf8") as f:
    matches = [{k: str(v) for k, v in row.items()}
         for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

bdrv_ids = []
for match in matches:
    if not match['bdrv_id'] in bdrv_ids:
        bdrv_ids.append(match['bdrv_id'])

filtered_bdrv = []
filtered_bdrv_ids = []
for row in bdrv:
    if row['recall_id'] in bdrv_ids and row['recall_id'] not in filtered_bdrv_ids:
        filtered_bdrv.append(row)
        filtered_bdrv_ids.append(row['recall_id'])

bdrv_documents = []
next_id = 1
for line in filtered_bdrv:
    id = next_id
    recall_content = {
                    'recall_id': line['recall_id'],
                     'system': line['system'],
                     'make': line['make'],
                     'model': line['model'],
                     'year': line['year'],
                     'details': line['details']
                    }
    bdrv_documents.append((id, recall_content))
    next_id += 1



es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

if es.indices.exists("bdrv"):
    es.indices.delete("bdrv")

for recall_id, recall_content in bdrv_documents:
    es.index(index="small-bdrv", doc_type="recall", id=recall_id, body=recall_content)

es.search(index='small-bdrv')