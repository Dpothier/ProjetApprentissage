import vectorization.VectorizerSet as vectorSets
from Experiment import ExperimentSet
from classification.classify import classify_with_NB
from classification.classify import classify_with_NB_with_no_folds
from dictionary import TerminologicalDictionary
from getData import get_bdrv_data
from getData import get_carcomplaints_data
from getData import get_some_carcomplaints_data
import nltk
from vectorization.VectorizerSet import ngram_count_tf_idf
import vectorization.VectorizerSet as vectorSets
from dictionary import TerminologicalDictionary
from classification.classify_kmeans import Clustering_kmeans

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

texts_brdv, labels_brdv = get_bdrv_data()
texts_carcomplaints, labels_carcomplaints = get_some_carcomplaints_data()

dictionary = TerminologicalDictionary()

experiment_set = ExperimentSet(classify_with_NB,
                               vectorSets.lemmatization(dictionary))

with open("../results/lemmatization_brdv.txt", mode="w", encoding="utf8") as f:
    for result in experiment_set.get_experiment_results(texts_brdv, labels_brdv):
        f.writelines('Mean accuracy for {}: {} \n'.format(result[0], result[1]))
        print('Mean accuracy for {}: {}'.format(result[0], result[1]))



