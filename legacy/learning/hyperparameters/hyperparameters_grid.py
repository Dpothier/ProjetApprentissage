import numpy as np

class HyperparameterGrid():

    def __init__(self, hyperparameter_grid_dict):
        self.hyperparameters_keys = list(hyperparameter_grid_dict.keys())
        self.hyperparameters_lists = list(hyperparameter_grid_dict.values())
        self.number_of_hyperparameter = len(self.hyperparameters_keys)
        self.indices = np.zeros(self.number_of_hyperparameter)
        self.number_of_permutations = 1
        for hyperparameter in self.hyperparameters_lists:
            self.number_of_permutations *= len(hyperparameter)

    def iterate_hyperparameters(self):
        for i in range(0, self.number_of_permutations):
            next_hyperparameter = {}
            val = i
            for j in range(0, len(self.hyperparameters_keys)):
                number_of_values_for_hyperparameter = len(self.hyperparameters_lists[j])
                next_hyperparameter[self.hyperparameters_keys[j]] = self.hyperparameters_lists[j][val % number_of_values_for_hyperparameter]
                val //= number_of_values_for_hyperparameter
            yield next_hyperparameter
