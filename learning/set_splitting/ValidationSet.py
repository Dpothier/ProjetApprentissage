from sklearn.model_selection import train_test_split

class ValidationSet:

    def __init__(self, learning_algorithm, validation_size):
        self.learning_algorithm = learning_algorithm
        self.validation_size = validation_size

    def run_experiment(self, data, targets):
        data_train, data_validation, targets_train, targets_validation = train_test_split(data, targets, test_size=self.validation_size)

        results, hyperparameters, model, confusion_matrix = self.learning_algorithm.optimize_hyperparameters(data_train, targets_train)

        best_results, model = self.learning_algorithm.run_experiment_with_hyperparameters\
            (data_train, data_validation, targets_train, targets_validation, hyperparameters)

        return best_results, hyperparameters, model


class ValidationFromOtherDataset:
    def __init__(self, learning_algorithm, data_validation, targets_validation):
        self.learning_algorithm = learning_algorithm
        self.data_validation = data_validation
        self.targets_validation = targets_validation

    def run_experiment(self, data, targets):

        results, hyperparameters, model = self.learning_algorithm.optimize_hyperparameters(data, targets)

        best_results, model = self.learning_algorithm.run_experiment_with_hyperparameters \
            (data, self.data_validation, targets, self.targets_validation, hyperparameters)

        return best_results, hyperparameters, model


class NoValidationSet:
    def __init__(self, learning_algorithm):
        self.learning_algorithm = learning_algorithm

    def run_experiment(self, data, targets):
        return self.learning_algorithm.optimize_hyperparameters(data, targets)
