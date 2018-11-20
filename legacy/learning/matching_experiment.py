from data.get_matching_data import get_data
from feature_extraction.class_probabilities_feature import ClassProbabilitiesExtractor
from feature_extraction.shared_vocab_feature import SharedVocabFeatureExtractor
from feature_extraction.longest_word_sequence_feature import LongestWordSequenceFeatureExtractor
from feature_extraction.Match import build_matches
from feature_extraction.class_probabilities_cosine_distance import ClassProbabilitiesCosineDistanceExtractor
from feature_extraction.elastic_score_feature import ElasticScoreFeatureExtractor
from dictionary import TerminologicalDictionary
import numpy as np
from experiment.experiment_configuration import *
from sklearn.externals import joblib
import os
from match_result.results import success_rate_on_n_most_likely
from match_result.results import success_rate_on_n_most_likely_with_elasticsearch
from match_result.results import get_n_most_likely_result_message
from feature_extraction.extraction_setup import start_extraction

dictionary = TerminologicalDictionary()

carcomplaints, bdrv, matches = get_data()

class_probabilities_extractor = ClassProbabilitiesCosineDistanceExtractor('../results/bdrv/svm/vectorizer_by count_stemming_all.pkl',
                                        '../results/bdrv/svm/model_by count_stemming_all.pkl')

shared_vocab_extractor = SharedVocabFeatureExtractor(1, 2)
shared_vocab_dict_extractor = SharedVocabFeatureExtractor(1, 2, dictionary)
longest_word_sequence_extractor = LongestWordSequenceFeatureExtractor()
elastic_score_feature_extractor = ElasticScoreFeatureExtractor()

matched_rows = build_matches(bdrv, carcomplaints, matches)

extracted_data = start_extraction(matched_rows)\
                    .use_class_probabilities('../results/bdrv/svm/vectorizer_by count_stemming_all.pkl',
                                             '../results/bdrv/svm/model_by count_stemming_all.pkl')\
                    .use_elastic_score()\
                    .use_stemmed_shared_vocabulary(1,2)\
                    .use_longest_word_sequence()\
                    .extract_features()

targets = np.array([int(row.is_match) for row in matched_rows])

hyperparameters = {
    "c" : [10,100,1000],
    "sigma": [1, 2, 4, 8, 16, 32, 64]
}

use_SVM(True)\
    .train_on_k_fold(10)\
    .use_hyperparameters_grid_search(hyperparameters)\
    .use_internal_validation_set(0.2)\
    .use_mean_accuracies_metrics()\
    .use_mean_training_time_metric().output_to_file("../results/matching/count_postprocessing_stemming")\
    .use_mean_confusion_matrix_metric()\
    .use_model_dumper("../results/matching/model")\
    .execute_with_data('Matching prediction accuracy', extracted_data, targets)


classifier = joblib.load(os.path.abspath("../results/matching/model_0.pkl"))

probabilities = classifier.predict_proba(extracted_data)

suffix = "test"

with open("../results/matching/n_most_likely/1_trained_{}.txt".format(suffix), mode="w+", encoding="utf8") as file:
    file.writelines("{}".format(get_n_most_likely_result_message(1, success_rate_on_n_most_likely(1, matched_rows, probabilities), matches, carcomplaints, bdrv)))

with open("../results/matching/n_most_likely/3_trained_{}.txt".format(suffix), mode="w+", encoding="utf8") as file:
    file.writelines("{}".format(get_n_most_likely_result_message(3, success_rate_on_n_most_likely(3, matched_rows, probabilities), matches,
                                         carcomplaints, bdrv)))

with open("../results/matching/n_most_likely/5_trained_{}.txt".format(suffix), mode="w+", encoding="utf8") as file:
    file.writelines("{}".format(get_n_most_likely_result_message(5, success_rate_on_n_most_likely(5, matched_rows, probabilities), matches,
                                         carcomplaints, bdrv)))

# with open("../results/matching/n_most_likely/1_elastic_{}.txt".format(suffix), mode="w+", encoding="utf8") as file:
#     file.writelines("{}".format(get_n_most_likely_result_message(1, success_rate_on_n_most_likely_with_elasticsearch(1, carcomplaints, matches),
#                                          matches, carcomplaints, bdrv)))
#
# with open("../results/matching/n_most_likely/3_elastic_{}.txt".format(suffix), mode="w+", encoding="utf8") as file:
#     file.writelines("{}".format(get_n_most_likely_result_message(3, success_rate_on_n_most_likely_with_elasticsearch(3, carcomplaints, matches),
#                                          matches, carcomplaints, bdrv)))
#
# with open("../results/matching/n_most_likely/5_elastic_{}.txt".format(suffix), mode="w+", encoding="utf8") as file:
#     file.writelines("{}".format(get_n_most_likely_result_message(5, success_rate_on_n_most_likely_with_elasticsearch(5, carcomplaints, matches),
#                                         matches, carcomplaints, bdrv)))








