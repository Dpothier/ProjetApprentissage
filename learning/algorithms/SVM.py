import numpy as np
from algorithms.helper import run_classifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split


class SVM:
    def __init__(self, sets_constructor):
        self.sets_constructor = sets_constructor
        self.current_c = 0
        self.current_sigma = 0
        self.best_hyperparameters = (0, 0) #C, then sigma


    def produce_results(self, train_data, test_data, train_target, test_target):
        return run_classifier(SVC(C=self.current_c, gamma=(1 / (2 * self.current_sigma ** 2))),
                              train_data, test_data, train_target, test_target)

    def run_experiment_with_hyperparameters(self, train_data, validation_data, train_target, validation_target,hyperparameters):
        return run_classifier(SVC(C=hyperparameters[0], gamma=(1 / (2 * hyperparameters[1] ** 2))),
                              train_data, validation_data, train_target, validation_target)

    def optimize_hyperparameters(self, data, targets):
        cs = [1e-08, 1e-07, 1e-06, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        min_sigma = self.get_min_sigma(data)
        sigmas = [min_sigma, 2 * min_sigma, 4 * min_sigma, 8 * min_sigma, 16 * min_sigma, 32 * min_sigma,
                  64 * min_sigma]

        best_result = (0, 0, 0)
        best_result_hyperparameters = (0,0)
        for i in range(0, len(cs)):
            for j in range(0, len(sigmas)):
                self.current_c = cs[i]
                self.current_sigma = sigmas[j]
                result = self.sets_constructor(data, targets, self.produce_results)
                if result[0] > best_result[0]:
                    best_result = result
                    best_result_hyperparameters = (self.current_c, self.current_sigma)

        #best_result = run_classifier(SVC(C=best_c, gamma=(1 / (2 * best_sigma ** 2))),
        #                             data_experiment,
        #                             data_validation,
        #                             target_experiment,
        #                             target_validation)

        return best_result, best_result_hyperparameters

    def get_min_sigma(self, data):
        distance_matrix = pairwise_distances(data, metric='cityblock')
        return distance_matrix[distance_matrix != 0].min()