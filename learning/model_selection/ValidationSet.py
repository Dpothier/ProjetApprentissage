from sklearn.model_selection import train_test_split

class ValidationSet:

    def __init__(self, learning_algorithm, validation_size):
        self.learning_algorithm = learning_algorithm
        self.validation_size = validation_size

    def produce_results_validation(self, data, targets):
        data_train, data_validation, targets_train, targets_validation = train_test_split(data, targets, test_size=self.validation_size)

        results, hyperparameters = self.learning_algorithm.optimize_hyperparameters(data_train, targets_train)

        best_results = self.learning_algorithm.produce_results_validation(data_train, data_validation, targets_train, targets_validation)

        return best_results, hyperparameters