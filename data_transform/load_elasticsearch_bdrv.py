from elasticsearch import Elasticsearch
import csv

with open('../data/bdrv.csv', encoding="utf8") as f:
    bdrv = [{k: str(v) for k, v in row.items()}
         for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

bdrv_documents = []
next_id = 1
for line in bdrv:
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
    es.index(index="bdrv", doc_type="recall", id=recall_id, body=recall_content)