class HyperparametersGridOptimizer:

    def __init__(self, trainer, hyperparameters_grid, accuracy_result_producer):
        self.trainer = trainer
        self.hyperparameters_grid = hyperparameters_grid
        self.accuracy_result_producer = accuracy_result_producer

    def __call__(self):
        best_accuracy = -1
        best_hyperparameters = None
        for hyperparameter_set in self.hyperparameters_grid.iterate_hyperparameters():
            result = self.trainer(hyperparameter_set)
            accuracy = self.accuracy_result_producer(result)
            if accuracy['mean_test_accuracy'] > best_accuracy:
                best_accuracy = accuracy['mean_test_accuracy']
                best_hyperparameters = hyperparameter_set

        return best_hyperparameters