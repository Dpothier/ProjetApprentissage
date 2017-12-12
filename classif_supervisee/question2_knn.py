import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from question2_classifier import run_classifier
from latex_table import output_to_table

def find_best_hyperparameters_for_knn(X_train, X_validation, y_train, y_validation):

    ks = [1, 3, 5, 7, 9, 11, 23]

    results = np.zeros((len(ks), 3))

    for i in range(0, len(ks)):
        k = ks[i]
        kfold = KFold()
        training_accuracies = []
        test_accuracies = []
        execution_times = []
        for train_indices, test_indices in kfold.split(X_train, y_train):
            result = run_classifier(KNeighborsClassifier(n_neighbors=k),
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

    best = np.argmax(results[:, 1])
    best_k = ks[best]

    best_result = run_classifier(KNeighborsClassifier(n_neighbors=best_k),
                                 X_train,
                                 X_validation,
                                 y_train,
                                 y_validation)

    return best_result + (best_k, "")


