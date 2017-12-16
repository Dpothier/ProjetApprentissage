from getData import get_carcomplaints_data
import experiment.experiment_setup as setup
import vectorization.VectorizerSet as sets
from dictionary import TerminologicalDictionary


data, targets = get_carcomplaints_data()

dictionary = TerminologicalDictionary()

setup.With_single_split(0.2)\
    .use_MLP()\
    .use_validation_set(0.2)\
    .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
    .output_to_file("../results/carcomplaints/mlp/count_postprocessing_single")\
    .output_to_console()\
    .go(data, targets)

setup.With_single_split(0.2)\
    .use_MLP()\
    .use_validation_set(0.2)\
    .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
    .output_to_file("../results/carcomplaints/mlp/count_postprocessing_stemming")\
    .output_to_console()\
    .go(data, targets)

setup.With_single_split(0.2)\
    .use_MLP()\
    .use_validation_set(0.2)\
    .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
    .output_to_file("../results/carcomplaints/mlp/count_postprocessing_lemma")\
    .output_to_console()\
    .go(data, targets)
