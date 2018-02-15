import numpy as np
from algorithms.helper import run_classifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self, sets_constructor, cs, sigmas):
        self.cs = cs
        self.sigmas = sigmas
        self.sets_constructor = sets_constructor
        self.current_c = 0
        self.current_sigma = 0
        self.current_model = None
        self.best_hyperparameters = (0, 0) #C, then sigma
        self.best_model = None


    def produce_results(self, train_data, test_data, train_target, test_target):
        self.current_model = SVC(C=self.current_c, gamma=(1 / (2 * self.current_sigma ** 2)), probability=True)
        results = run_classifier(self.current_model,
                              train_data, test_data, train_target, test_target)
        results['model'] = self.current_model
        return results

    def run_experiment_with_hyperparameters(self, train_data, validation_data, train_target, validation_target,hyperparameters):
        self.current_model = SVC(C=hyperparameters[0], gamma=(1 / (2 * hyperparameters[1] ** 2)), probability=True)
        results = run_classifier(self.current_model, train_data, validation_data, train_target, validation_target)
        results['model'] = self.current_model
        return results

    def optimize_hyperparameters(self, data, targets):

        min_sigma = self.get_min_sigma(data)

        best_result = (0, 0, 0)
        best_result_hyperparameters = (0,0)
        for i in range(0, len(self.cs)):
            for j in range(0, len(self.sigmas)):
                self.current_c = self.cs[i]
                self.current_sigma = min_sigma * self.sigmas[j]
                result = self.sets_constructor(data, targets, self.produce_results)
                if result[0] > best_result[0]:
                    best_result = result
                    best_result_hyperparameters = (self.current_c, self.current_sigma)
                    best_result_model = self.current_model

        return best_result, best_result_hyperparameters, best_result_model

    def get_min_sigma(self, data):
        distance_matrix = pairwise_distances(data, metric='cityblock')
        return distance_matrix[distance_matrix != 0].min()