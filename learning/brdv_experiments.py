import experiment.experiment_setup as setup
import vectorization.VectorizerSet as sets
from data.getData import get_bdrv_data
from dictionary import TerminologicalDictionary
from experiment.experiment_configuration import *

data, targets = get_bdrv_data()

dictionary = TerminologicalDictionary()

# vectorizer_set = sets.count_postprocessing_single(dictionary)
#
# experiment = use_ADA_boost()\
#                 .train_on_k_fold(10)\
#                 .use_mean_accuracies_metrics()\
#                 .use_mean_training_time_metric()\
#                 .output_to_file("../results/bdrv/adaboost/count_postprocessing_single")
#
# for vectorizer in vectorizer_set:
#     vectorized_data = vectorizer[1].fit_transform(data)
#     experiment.execute_with_data(vectorizer[0], vectorized_data, targets)
#
#
# vectorizer_set = sets.count_postprocessing_stemming(dictionary)
#
# experiment = use_ADA_boost()\
#                 .train_on_k_fold(10)\
#                 .use_mean_accuracies_metrics()\
#                 .use_mean_training_time_metric()\
#                 .output_to_file("../results/bdrv/adaboost/count_postprocessing_stemming")
#
# for vectorizer in vectorizer_set:
#     vectorized_data = vectorizer[1].fit_transform(data)
#     experiment.execute_with_data(vectorizer[0], vectorized_data, targets)
#
# vectorizer_set = sets.count_postprocessing_lemma(dictionary)
#
# experiment = use_ADA_boost()\
#                 .train_on_k_fold(10)\
#                 .use_mean_accuracies_metrics()\
#                 .use_mean_training_time_metric()\
#                 .output_to_file("../results/bdrv/adaboost/count_postprocessing_lemma")
#
# for vectorizer in vectorizer_set:
#     vectorized_data = vectorizer[1].fit_transform(data)
#     experiment.execute_with_data(vectorizer[0], vectorized_data, targets)

# setup.With_kfold(10)\
#     .use_EM(20)\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
#     .output_to_file("../results/bdrv/em/count_postprocessing_single")\
#     .output_to_console()\
#     .go(data, targets)
#
# setup.With_kfold(10)\
#     .use_EM(20)\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
#     .output_to_file("../results/bdrv/em/count_postprocessing_stemming")\
#     .output_to_console()\
#     .go(data, targets)
#
# setup.With_kfold(10)\
#     .use_EM(20)\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
#     .output_to_file("../results/bdrv/em/count_postprocessing_lemma")\
#     .output_to_console()\
#     .go(data, targets)
#
# setup.With_kfold(10)\
#     .use_MLP()\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
#     .output_to_file("../results/bdrv/mlp/count_postprocessing_single")\
#     .output_to_console()\
#     .go(data, targets)
#
# setup.With_kfold(10)\
#     .use_MLP()\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
#     .output_to_file("../results/bdrv/mlp/count_postprocessing_stemming")\
#     .output_to_console()\
#     .go(data, targets)
#
# setup.With_kfold(10)\
#     .use_MLP()\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
#     .output_to_file("../results/bdrv/mlp/count_postprocessing_lemma")\
#     .output_to_console()\
#     .go(data, targets)


MLP_hyperparameters = {
    "depth": [1, 2, 4, 6, 8],
    "width": [100]
}
vectorizer_set = sets.count_postprocessing_single(dictionary)

experiment = use_MLP()\
                .train_on_k_fold(10)\
                .use_hyperparameters_grid_search(MLP_hyperparameters)\
                .use_internal_validation_set(0.2)\
                .use_mean_accuracies_metrics()\
                .use_mean_training_time_metric()\
                .output_to_file("../results/bdrv/mlp/count_postprocessing_single")

for vectorizer in vectorizer_set:
    vectorized_data = vectorizer[1].fit_transform(data)
    experiment.execute_with_data(vectorizer[0], vectorized_data, targets)


# vectorizer_set = sets.count_postprocessing_stemming(dictionary)
#
# experiment = use_MLP()\
#                 .train_on_k_fold(10)\
#                 .use_hyperparameters_grid_search(MLP_hyperparameters)\
#                 .use_internal_validation_set(0.2)\
#                 .use_mean_accuracies_metrics()\
#                 .use_mean_training_time_metric()\
#                 .output_to_file("../results/bdrv/mlp/count_postprocessing_stemming")
#
# for vectorizer in vectorizer_set:
#     vectorized_data = vectorizer[1].fit_transform(data)
#     experiment.execute_with_data(vectorizer[0], vectorized_data, targets)
#
# vectorizer_set = sets.count_postprocessing_lemma(dictionary)
#
# experiment = use_MLP()\
#                 .train_on_k_fold(10)\
#                 .use_hyperparameters_grid_search(MLP_hyperparameters)\
#                 .use_internal_validation_set(0.2)\
#                 .use_mean_accuracies_metrics()\
#                 .use_mean_training_time_metric()\
#                 .output_to_file("../results/bdrv/mlp/count_postprocessing_lemma")
#
# for vectorizer in vectorizer_set:
#     vectorized_data = vectorizer[1].fit_transform(data)
#     experiment.execute_with_data(vectorizer[0], vectorized_data, targets)

# setup.With_kfold(10)\
#     .use_nb()\
#     .use_test_set_results()\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
#     .output_to_file("../results/bdrv/nb/count_postprocessing_single")\
#     .output_to_console()\
#     .go(data, targets)
#
# setup.With_kfold(10)\
#     .use_nb()\
#     .use_test_set_results()\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
#     .output_to_file("../results/bdrv/nb/count_postprocessing_stemming")\
#     .output_to_console()\
#     .go(data, targets)
#
# setup.With_kfold(10)\
#     .use_nb()\
#     .use_test_set_results()\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
#     .output_to_file("../results/bdrv/nb/count_postprocessing_lemma")\
#     .output_to_console()\
#     .go(data, targets)


#cs = [1e-08, 1e-07, 1e-06, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
cs = [10,100,1000]
sigmas = [1, 2, 4, 8, 16, 32, 64]

# setup.With_kfold(10)\
#     .use_SVM()\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
#     .output_to_file("../results/bdrv/svm/count_postprocessing_single")\
#     .output_to_console()\
#     .go(data, targets)

# setup.With_kfold(10)\
#     .use_SVM(cs, sigmas)\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
#     .output_to_file("../results/bdrv/svm/count_postprocessing_stemming")\
#     .output_to_console()\
#     .output_model_to_file("../results/bdrv/svm/model")\
#     .output_vectorizer_to_file("../results/bdrv/svm/vectorizer")\
#     .go(data, targets)

# setup.With_kfold(10)\
#     .use_SVM()\
#     .use_validation_set(0.2)\
#     .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
#     .output_to_file("../results/bdrv/svm/count_postprocessing_lemma")\
#     .output_to_console()\
#     .go(data, targets)