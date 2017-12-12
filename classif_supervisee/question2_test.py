import numpy as np
from question2_data import get_data
from question2_knn import find_best_hyperparameters_for_knn
from question2_svm import find_best_hyperparameters_for_svm
from question2_mlp import find_best_hyperparameters_for_mlp
from latex_table import output_to_table


X_train, X_validation, y_train, y_validation = get_data()

results_knn = find_best_hyperparameters_for_knn(X_train, X_validation, y_train, y_validation)
results_svm = find_best_hyperparameters_for_svm(X_train, X_validation, y_train, y_validation)
results_mlp = find_best_hyperparameters_for_mlp(X_train, X_validation, y_train, y_validation)
print(results_knn)
print(results_svm)
print(results_mlp)
results = np.array([[results_knn[0], results_knn[1], results_knn[2], results_knn[3], results_knn[4]],
                    [results_svm[0], results_svm[1], results_svm[2], results_svm[3], results_svm[4]],
                    [results_mlp[0], results_mlp[1], results_mlp[2], results_mlp[3], results_mlp[4]]])


output_to_table("question2_results",results, "Algorithme",
                ["Taux de classement entraînement", "Taux de classement entraînement","Temps d'exécution",
                 "Premier hyperparamètre", "Deuxième hyperparamètre" ],
                ["Knn", "SVM", "MLP"],
                "Résultats d'optimisations d'hyperparamètres pour trois méthodes par discriminant",
                "tab:q2_results")
