from elasticsearch import Elasticsearch
import csv

with open('../data/tiny_carcomplaints.csv', encoding="utf8") as f:
    tiny_carcomplaints = [{k: str(v) for k, v in row.items()}
         for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]


es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

if es.indices.exists("carcomplaints"):
    es.indices.delete("carcomplains")

for complaint_content in tiny_carcomplaints:
    es.index(index="carcomplaints", doc_type="complaint", body=complaint_content)

    patate2000.