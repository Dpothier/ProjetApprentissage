from elasticsearch import Elasticsearch
import numpy as np

class ElasticScoreFeatureExtractor:

    def extract_feature(self, matches):
        es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

        scores = np.zeros((len(matches),1))
        for index, match in enumerate(matches):
            results = es.search(index='bdrv',
                      body={
                          "query": {
                                  "bool": {
                                          "must": {
                                                  "match": {
                                                      "details": match.carcomplaint['text']
                                                  }
                                              },
                                          "filter": {
                                              "match": {
                                                  "recall_id": match.bdrv['recall_id']
                                              }
                                          }
                                      }
                              }
                      })
            if len(results['hits']['hits']) != 0:
                scores[index, 0] = float(results['hits']['hits'][0]['_score'])
            else:
                scores[index, 0] = 0
        max_value = scores.max()
        print(max_value)
        return scores/max_value

