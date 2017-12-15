import numpy as np
from sklearn.model_selection import KFold
from algorithms.helper import run_classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class MLP:
    def __init__(self, sets_constructor):
        self.sets_constructor = sets_constructor
        self.current_layer = 0
        self.current_neuron = 0
        self.best_hyperparameters = (0, 0) # Number of layers, then number of neurons per layer


    def produce_results(self, train_data, test_data, train_target, test_target):
        return run_classifier(MLPClassifier(hidden_layer_sizes=(self.current_neuron,) * self.current_layer),
                              train_data, test_data, train_target, test_target)

    def produce_results_validation(self, train_data, validation_data, train_target, validation_target):
        return run_classifier(MLPClassifier(hidden_layer_sizes=(self.best_hyperparameters[1],) * self.best_hyperparameters[0]),
                              train_data, validation_data, train_target, validation_target)

    def optimize_hyperparameters(self, data, targets):
        layers = [1, 2, 3, 4, 5, 6, 7, 8]
        neurons_per_layer = [50, 75, 100, 125, 150, 175, 200]

        best_result = (0,0,0)
        for i in range(0, len(layers)):
            for j in range(0, len(neurons_per_layer)):
                self.current_layer = layers[i]
                self.current_neuron = neurons_per_layer[j]
                result = self.sets_constructor(data, targets, self.produce_results)
                if result[0] > best_result[0]:
                    best_result = result
                    self.best_hyperparameters = (self.current_layer, self.current_neuron)

        #best_result = run_classifier(SVC(C=best_c, gamma=(1 / (2 * best_sigma ** 2))),
        #                             data_experiment,
        #                             data_validation,
        #                             target_experiment,
        #                             target_validation)

        return best_result + self.best_hyperparameters

