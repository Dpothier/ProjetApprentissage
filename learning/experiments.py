import vectorization.VectorizerSet as vectorSets
from Experiment import ExperimentSet
from classification.classify import classify_with_NB
from classification.classify import classify_with_NB_with_no_folds
from dictionary import TerminologicalDictionary
from getData import get_bdrv_data
from getData import get_carcomplaints_data
from getData import get_some_carcomplaints_data
import nltk
nltk.download('averaged_perceptron_tagger')
from vectorization.VectorizerSet import ngram_count_tf_idf
from dictionary import TerminologicalDictionary
from classification.classify_kmeans import Clustering_kmeans

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

texts_brdv, labels_brdv = get_bdrv_data()
texts_carcomplaints, labels_carcomplaints = get_some_carcomplaints_data()

dictionary = TerminologicalDictionary()


experiment_set = ExperimentSet(Clustering_kmeans(50),
                               vectorSets.count_postprocessing(dictionary))

#experiment_set = ExperimentSet(classify_with_NB,
#                               vectorSets.closed_vocab(dictionary))


with open("../results/closed_vocab.txt", mode="w", encoding="utf8") as f:
    for result in experiment_set.get_experiment_results(texts_brdv, labels_brdv):
        f.writelines('Mean accuracy for {}: {} \n'.format(result[0], result[1]))
        print('Mean accuracy for {}: {}'.format(result[0], result[1]))



