import numpy as np
from algorithms.helper import run_classifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split


class SVM:
    def __call__(self, data, target):
        data_experiment, data_validation, target_experiment, target_validation = train_test_split(data, target,
                                                                                                  test_size=0.2)
        kfold = KFold(n_splits=10, shuffle=True)

        cs = [1e-08, 1e-07, 1e-06, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
        min_sigma = self.get_min_sigma(data)
        sigmas = [min_sigma, 2 * min_sigma, 4 * min_sigma, 8 * min_sigma, 16 * min_sigma, 32 * min_sigma,
                  64 * min_sigma]
        results = np.zeros((len(cs), len(sigmas), 3))

        for train_index, test_index in kfold.split(data_experiment, target_experiment):
            train_data = data[train_index]
            train_target = target[train_index]

            test_data = data[test_index]
            test_target = target[test_index]
            for i in range(0, len(cs)):
                for j in range(0, len(sigmas)):
                    c = cs[i]
                    sigma = sigmas[j]
                    training_accuracies = []
                    test_accuracies = []
                    execution_times = []

                    result = run_classifier(SVC(C=c, gamma=(1 / (2 * sigma ** 2))),
                                            train_data,
                                            test_data,
                                            train_target,
                                            test_target)

                    training_accuracies.append(result[0])
                    test_accuracies.append(result[1])
                    execution_times.append(result[2])
                    results[i, j, 0] = sum(training_accuracies) / len(training_accuracies)
                    results[i, j, 1] = sum(test_accuracies) / len(test_accuracies)
                    results[i, j, 2] = sum(execution_times) / len(execution_times)

        best = np.unravel_index(np.argmax(results[:, :, 1]), results[:, :, 1].shape)
        best_c = cs[best[0]]
        best_sigma = sigmas[best[1]]

        best_result = run_classifier(SVC(C=best_c, gamma=(1 / (2 * best_sigma ** 2))),
                                     data_experiment,
                                     data_validation,
                                     target_experiment,
                                     target_validation)

        return best_result + (best_c, best_sigma)

    def get_min_sigma(self, data):
        distance_matrix = pairwise_distances(data, metric='cityblock')
        return distance_matrix[distance_matrix != 0].min()