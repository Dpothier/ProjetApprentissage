from getData import get_bdrv_data
from getData import get_some_carcomplaints_data
import experiment.experiment_setup as setup
import vectorization.VectorizerSet as sets
from dictionary import TerminologicalDictionary


data, targets = get_bdrv_data()
data_test, targets_test = get_some_carcomplaints_data()

dictionary = TerminologicalDictionary()

setup.With_kfold(10)\
    .use_MLP()\
    .use_external_validation_set(data_test, targets_test)\
    .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
    .output_to_file("../results/generalisation/mlp/count_postprocessing_single")\
    .output_to_console()\
    .go(data, targets)

setup.With_kfold(10)\
    .use_MLP()\
    .use_external_validation_set(data_test, targets_test)\
    .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
    .output_to_file("../results/generalisation/mlp/count_postprocessing_stemming")\
    .output_to_console()\
    .go(data, targets)

setup.With_kfold(10)\
    .use_MLP()\
    .use_external_validation_set(data_test, targets_test)\
    .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
    .output_to_file("../results/generalisation/mlp/count_postprocessing_lemma")\
    .output_to_console()\
    .go(data, targets)

