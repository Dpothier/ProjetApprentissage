from data.get_matching_data import get_data
from feature_extraction.class_probabilities_feature import ClassProbabilitiesExtractor
from feature_extraction.Match import build_matches
from dictionary import TerminologicalDictionary
import numpy as np
from experiment.experiment_configuration import *
from sklearn.externals import joblib
import os

dictionary = TerminologicalDictionary()

carcomplaints, bdrv, matches = get_data()

extractor = ClassProbabilitiesExtractor('../results/bdrv/svm/vectorizer_by count_stemming_all.pkl',
                                        '../results/bdrv/svm/model_by count_stemming_all.pkl')

matched_rows = build_matches(bdrv, carcomplaints, matches)

extracted_data = extractor.extract_feature(matched_rows)
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
    .use_mean_confusion_matrix_metric("../results/matching/model")\
    .use_model_dumper()\
    .execute_with_data('Matching prediction accuracy', extracted_data, targets)


classifier = joblib.load(os.path.abspath("../results/matching/model_0.pkl"))

probabilities = classifier.predict_proba(extracted_data)
