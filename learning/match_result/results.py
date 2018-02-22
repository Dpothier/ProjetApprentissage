from sklearn.externals import joblib
import os
from elasticsearch import Elasticsearch

def success_rate_on_n_most_likely(n, matching_data, probabilities):

    assert len(probabilities) == len(matching_data)

    probability_results = []
    for i in range(0, len(probabilities)):
        matched_row = matching_data[i]
        probability_result = (
            matched_row.bdrv['recall_id'],
            matched_row.carcomplaint['id'],
            int(matched_row.is_match),
            probabilities[i, 1]
        )
        probability_results.append(probability_result)

    carcomplaints_id_list = set((i[1] for i in probability_results))

    successes_count = 0
    matched = []
    not_matched = []
    for carcomplaints_id in carcomplaints_id_list:
        three_recall_with_highest_probability = sorted((i for i in probability_results if i[1] == carcomplaints_id),
                                                       key=lambda tup: tup[3], reverse=True)[0:n]
        if 1 in list((i[2] for i in three_recall_with_highest_probability)):
            successes_count += 1
            matched.append(carcomplaints_id)
        else:
            not_matched.append(carcomplaints_id)

    return successes_count / len(carcomplaints_id_list), matched, not_matched

def success_rate_on_n_most_likely_with_elasticsearch(n, carcomplaints, matches):
    es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    success = 0
    matched = []
    not_matched = []
    for carcomplaint in carcomplaints:
        response = es.search(index='small-bdrv',
                             body={"size": n, "query": {"match": {"details": carcomplaint['text']}}})
        hits = response['hits']['hits']
        returned_bdrv_ids = (hit['_source']['recall_id'] for hit in hits)
        bdrv_matches_for_complaint = (int(i['is_match']) for i in matches if
                                      i['carcomplaints_id'] == carcomplaint['id'] and i['bdrv_id'] in returned_bdrv_ids)
        if 1 in bdrv_matches_for_complaint:
            success += 1
            matched.append(carcomplaint['id'])
        else:
            not_matched.append((carcomplaint['id']))

    return success / len(carcomplaints), matched, not_matched


def get_n_most_likely_result_message(n, results, matches, carcomplaints, bdrv):
    rate = results[0]
    matched = results[1]
    not_matched = results[2]

    message = "Results for {} most likely\n".format(n)
    message += "Rate: {}\n\n".format(rate)
    for id in matched:
        match = [i for i in matches if i['carcomplaints_id'] == id][0]
        complaint = [i for i in carcomplaints if i['id'] == id][0]
        recall =  [i for i in bdrv if i['recall_id'] == match['bdrv_id']][0]

        message += "Matched complaint: {}\n".format(id)
        message += "\t carcomplaint text: {}\n".format(complaint['text'])
        message += "\t recall text:  {}\n\n".format(recall['details'])

    message += "\n"

    for id in not_matched:
        match = [i for i in matches if i['carcomplaints_id'] == id][0]
        complaint = [i for i in carcomplaints if i['id'] == id][0]
        recall = [i for i in bdrv if i['recall_id'] == match['bdrv_id']][0]

        message += "Not-Matched complaint: {}\n".format(id)
        message += "\t carcomplaint text: {}\n".format(complaint['text'])
        message += "\t recall text:  {}\n\n".format(recall['details'])

    return message
