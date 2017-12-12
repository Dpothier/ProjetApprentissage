from Experiment import ExperimentSet
from classify import classify_with_NB
from getData import get_bdrv_data
from vectorization.VectorizerSet import ngram_count_tf_idf
import vectorization.VectorizerSet as vectorSets
from dictionary import TerminologicalDictionary
from classification.classify_kmeans import Clustering_kmeans

texts, labels = get_bdrv_data()

dictionary = TerminologicalDictionary()

experiment_set = ExperimentSet(Clustering_kmeans(50),
                               vectorSets.dictionary_amplification(dictionary))

with open("../results/dict_amplification_comparison.txt", mode="w", encoding="utf8") as f:
    for result in experiment_set.get_experiment_results(texts, labels):
        f.writelines('Mean accuracy for {}: {} \n'.format(result[0], result[1]))
        print('Mean accuracy for {}: {}'.format(result[0], result[1]))



