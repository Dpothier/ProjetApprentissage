from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import KFold
from algorithms.helper import run_classifier
from sklearn.model_selection import train_test_split


class ADABoost:
    def __call__(self, train_data, test_data, train_target, test_target):
            return run_classifier(AdaBoostClassifier(),
                                    data_experiment[train_indices],
                                    data_experiment[test_indices],
                                    target_experiment[train_indices],
                                    target_experiment[test_indices])