import numpy as np
from sklearn import cluster
from sklearn.metrics import silhouette_score
from algorithms.helper import run_clustering


class KMeans:
    def __init__(self, sets_constructor, max_k):
        self.sets_constructor = sets_constructor
        self.max_k = max_k
        self.current_k = 2
        self.best_hyperparameter = 2

    def produce_results(self, train_data, test_data, train_target, test_target):
        algo = cluster.KMeans(n_clusters=self.current_k)
        return run_clustering(algo, train_data, test_data)

    def run_experiment_with_hyperparameters(self, train_data, validation_data, train_target, validation_target, hyperparameters):
        algo = cluster.KMeans(n_clusters=hyperparameters)
        return run_clustering(algo, train_data, validation_data)

    def optimize_hyperparameters(self, data, targets):
        best_result = (0, 0, 0)
        while self.current_k < self.max_k:
            result = self.sets_constructor(data, targets, self.produce_results)
            if result[0] > best_result[0]:
                best_result = result
                self.best_hyperparameter = self.current_k
            self.current_k += 2

        return best_result, self.best_hyperparameter