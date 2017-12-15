from getData import get_bdrv_data
from getData import get_some_carcomplaints_data
import experiment.experiment_setup as setup
import vectorization.VectorizerSet as sets
from dictionary import TerminologicalDictionary


data, targets = get_bdrv_data()
data_test, targets_test = get_some_carcomplaints_data()

dictionary = TerminologicalDictionary()

setup.With_preselected_test_set(data_test, targets_test)\
    .use_nb()\
    .use_test_set_results()\
    .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
    .output_to_file("../results/bdrv/nb/count_postprocessing_single")\
    .output_to_console()\
    .go(data, targets)

setup.With_preselected_test_set(data_test, targets_test)\
    .use_nb()\
    .use_test_set_results()\
    .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
    .output_to_file("../results/bdrv/nb/count_postprocessing_stemming")\
    .output_to_console()\
    .go(data, targets)

setup.With_preselected_test_set(data_test, targets_test)\
    .use_nb()\
    .use_test_set_results()\
    .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
    .output_to_file("../results/bdrv/nb/count_postprocessing_lemma")\
    .output_to_console()\
    .go(data, targets)

