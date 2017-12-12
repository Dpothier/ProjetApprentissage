import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from question2_classifier import run_classifier
from latex_table import output_to_table
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neural_network import MLPClassifier

def find_best_hyperparameters_for_mlp(X_train, X_validation, y_train, y_validation):

    layers = [1, 2, 3, 4, 5, 6 , 7, 8]
    neurons_per_layer = [ 50, 75, 100, 125, 150, 175, 200]

    results = np.zeros((len(layers), len(neurons_per_layer), 3))

    for i in range(0, len(layers)):
        for j in range(0, len(neurons_per_layer)):
            number_of_layers = layers[i]
            number_of_neurons = neurons_per_layer[j]
            kfold = KFold()
            training_accuracies = []
            test_accuracies = []
            execution_times = []
            for train_indices, test_indices in kfold.split(X_train, y_train):
                result = run_classifier(MLPClassifier(hidden_layer_sizes=(number_of_neurons,)*number_of_layers),
                                        X_train[train_indices],
                                        X_train[test_indices],
                                        y_train[train_indices],
                                        y_train[test_indices])

                training_accuracies.append(result[0])
                test_accuracies.append(result[1])
                execution_times.append(result[2])

            results[i, 0] = sum(training_accuracies) / len(training_accuracies)
            results[i, 1] = sum(test_accuracies) / len(test_accuracies)
            results[i, 2] = sum(execution_times) / len(execution_times)

    #output_to_table("knn", results, "Number of neighbours",
    #                ["Training Accuracy", "Testing Accuracy", "Execution time"],
    #                ks,
    #                "Knn pour différents nombres de voisins",
    #                "tab:q2_knn")

    best = np.unravel_index(np.argmax(results[:,:, 1]), results[:, :, 1].shape)
    best_number_of_layer = layers[best[0]]
    best_number_of_neurons = neurons_per_layer[best[1]]

    best_result = run_classifier(MLPClassifier(hidden_layer_sizes=(best_number_of_neurons,)*best_number_of_layer),
                                 X_train,
                                 X_validation,
                                 y_train,
                                 y_validation)

    return best_result + (best_number_of_layer, best_number_of_neurons)


def get_min_sigma(data):
    distance_matrix = pairwise_distances(data, metric='cityblock')
    return distance_matrix[distance_matrix != 0].min()
