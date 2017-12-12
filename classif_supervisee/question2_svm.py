import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from question2_classifier import run_classifier
from latex_table import output_to_table
from sklearn.metrics.pairwise import pairwise_distances

def find_best_hyperparameters_for_svm(X_train, X_validation, y_train, y_validation):

    cs = [1e-08, 1e-07, 1e-06, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    min_sigma = get_min_sigma(np.append(X_train, X_validation, axis=0))
    sigmas = [min_sigma, 2*min_sigma, 4*min_sigma, 8*min_sigma, 16*min_sigma, 32*min_sigma, 64*min_sigma]

    results = np.zeros((len(cs), len(sigmas),3))

    print(range(0, len(cs)))
    for i in range(0, len(cs)):
        for j in range(0, len(sigmas)):
            c = cs[i]
            sigma = sigmas[j]
            kfold = KFold()
            training_accuracies = []
            test_accuracies = []
            execution_times = []
            for train_indices, test_indices in kfold.split(X_train, y_train):
                result = run_classifier(SVC(C=c, gamma=(1/(2 * sigma**2))),
                                        X_train[train_indices],
                                        X_train[test_indices],
                                        y_train[train_indices],
                                        y_train[test_indices])

                training_accuracies.append(result[0])
                test_accuracies.append(result[1])
                execution_times.append(result[2])
            results[i, j, 0] = sum(training_accuracies) / len(training_accuracies)
            results[i, j, 1] = sum(test_accuracies) / len(test_accuracies)
            results[i, j, 2] = sum(execution_times) / len(execution_times)

        #output_to_table("knn", results, "Number of neighbours",
        #                ["Training Accuracy", "Testing Accuracy", "Execution time"],
        #                ks,
        #                "Knn pour différents nombres de voisins",
        #                "tab:q2_knn")

    best = np.unravel_index(np.argmax(results[:,:, 1]), results[:, :, 1].shape)
    best_c = cs[best[0]]
    best_sigma = sigmas[best[1]]

    best_result = run_classifier(SVC(C=best_c, gamma=(1/(2 * best_sigma**2))),
                                 X_train,
                                 X_validation,
                                 y_train,
                                 y_validation)

    return best_result + (best_c, best_sigma)


def get_min_sigma(data):
    distance_matrix = pairwise_distances(data, metric='cityblock')
    return distance_matrix[distance_matrix != 0].min()
