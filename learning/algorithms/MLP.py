import numpy as np
from sklearn.model_selection import KFold
from algorithms.helper import run_classifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


class MLP:
    def __call__(self, data, target):
        data_experiment, data_validation, target_experiment, target_validation = train_test_split(data, target, test_size=0.2)
        layers = [1, 2, 3, 4, 5, 6, 7, 8]
        neurons_per_layer = [50, 75, 100, 125, 150, 175, 200]

        results = np.zeros((len(layers), len(neurons_per_layer), 3))

        for i in range(0, len(layers)):
            for j in range(0, len(neurons_per_layer)):
                number_of_layers = layers[i]
                number_of_neurons = neurons_per_layer[j]
                kfold = KFold(n_splits=10)
                training_accuracies = []
                test_accuracies = []
                execution_times = []
                for train_indices, test_indices in kfold.split(data_experiment, target_experiment):
                    result = run_classifier(MLPClassifier(hidden_layer_sizes=(number_of_neurons,) * number_of_layers),
                                            data_experiment[train_indices],
                                            data_experiment[test_indices],
                                            target_experiment[train_indices],
                                            target_experiment[test_indices])

                    training_accuracies.append(result[0])
                    test_accuracies.append(result[1])
                    execution_times.append(result[2])

                results[i, 0] = sum(training_accuracies) / len(training_accuracies)
                results[i, 1] = sum(test_accuracies) / len(test_accuracies)
                results[i, 2] = sum(execution_times) / len(execution_times)


        best = np.unravel_index(np.argmax(results[:, :, 1]), results[:, :, 1].shape)
        best_number_of_layer = layers[best[0]]
        best_number_of_neurons = neurons_per_layer[best[1]]

        best_result = run_classifier(MLPClassifier(hidden_layer_sizes=(best_number_of_neurons,) * best_number_of_layer),
                                     data_experiment,
                                     data_validation,
                                     target_experiment,
                                     target_validation)

        return best_result + (best_number_of_layer, best_number_of_neurons)