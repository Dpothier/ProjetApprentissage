from getData import get_carcomplaints_data
import experiment.experiment_setup as setup
import vectorization.VectorizerSet as sets
from dictionary import TerminologicalDictionary

print("starting experiment")
data, targets = get_carcomplaints_data()

dictionary = TerminologicalDictionary()

print("WIth single preprocessing")
setup.With_single_split(0.3)\
    .use_nb()\
    .use_test_set_results()\
    .test_on_multiple_pretreatment(sets.count_postprocessing_single(dictionary))\
    .output_to_file("../results/carcomplaints/nb/count_postprocessing_single")\
    .output_to_console()\
    .go(data, targets)

print("Combination with stemming")
setup.With_single_split(0.3)\
    .use_nb()\
    .use_test_set_results()\
    .test_on_multiple_pretreatment(sets.count_postprocessing_stemming(dictionary))\
    .output_to_file("../results/carcomplaints/nb/count_postprocessing_stemming")\
    .output_to_console()\
    .go(data, targets)

print("Combination with lemmatization")
setup.With_single_split(0.3)\
    .use_nb()\
    .use_test_set_results()\
    .test_on_multiple_pretreatment(sets.count_postprocessing_lemma(dictionary))\
    .output_to_file("../results/carcomplaints/nb/count_postprocessing_lemma")\
    .output_to_console()\
    .go(data, targets)

