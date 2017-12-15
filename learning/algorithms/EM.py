from sklearn.mixture import GaussianMixture
from algorithms.helper import run_clustering


class EM_Gaussian:
    def __init__(self, sets_constructor, max_k):
        self.sets_constructor = sets_constructor
        self.max_k = max_k
        self.current_k = 2
        self.best_hyperparameter = 0

    def produce_results(self, train_data, test_data, train_target, test_target):
        algo = GaussianMixture(n_components=self.current_k)
        try:
            return run_clustering(algo, train_data.toarray(), test_data.toarray())
        except AttributeError:
            return run_clustering(algo, train_data, test_data)

    def run_experiment_with_hyperparameters(self, train_data, validation_data, train_target, validation_target, hyperparameters):
        return run_clustering(GaussianMixture(n_components=hyperparameters),
                              train_data.toarray(), validation_data.toarray())

    def optimize_hyperparameters(self, data, targets):
        best_result = (0, 0, 0)
        while self.current_k < self.max_k:
            result = self.sets_constructor(data, targets, self.produce_results)
            if result[0] > best_result[0]:
                best_result = result
                self.best_hyperparameter = self.current_k
            self.current_k += 2

        return best_result, self.best_hyperparameter
