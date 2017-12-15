from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import KFold
from algorithms.helper import run_classifier
from sklearn.model_selection import train_test_split


class ADABoost:
    def __init__(self, sets_constructor):
        self.sets_constructor = sets_constructor

    def produce_results(self, train_data, test_data, train_target, test_target):
        return run_classifier(AdaBoostClassifier(), train_data, test_data, train_target, test_target)

    def run_experiment_with_hyperparameters(self, train_data, validation_data, train_target, validation_target, hyperparameters):
        return run_classifier(AdaBoostClassifier(), train_data, validation_data, train_target, validation_target)

    def optimize_hyperparameters(self, data, targets):
        return self.sets_constructor(data, targets, self.produce_results), ()
