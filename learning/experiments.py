import vectorization.VectorizerSet as vectorSets
from Experiment import ExperimentSet
from Experiment import MetaExperimentSet
from classification.classify import classify_with_NB
from classification.classify import classify_with_NB_with_no_folds
from dictionary import TerminologicalDictionary
from getData import get_bdrv_data
from getData import get_carcomplaints_data
from getData import get_some_carcomplaints_data
import nltk
from dictionary import TerminologicalDictionary
from classification.classify_kmeans import Clustering_kmeans
from classification.classify_bernoulli_EM import Clustering_Gaussian_Mixture
from classification import SVM

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

texts_brdv, labels_brdv = get_bdrv_data()
texts_carcomplaints, labels_carcomplaints = get_some_carcomplaints_data()

dictionary = TerminologicalDictionary()

#experiment_set = MetaExperimentSet(Clustering_kmeans(50), vectorSets.metaset_std(dictionary), "../results/Kmeans/")

#experiment_set.execute_experiments(texts_brdv, labels_brdv)

experiment_set = ExperimentSet(SVM.SVM(),
                              vectorSets.ngram_count_tf_idf(dictionary))


with open("../results/svm_ngram_count_tf_idf.txt", mode="w", encoding="utf8") as f:
    for result in experiment_set.get_experiment_results(texts_brdv, labels_brdv):
        f.writelines('Mean accuracy for {}: {} \n'.format(result[0], result[1]))
        print('Mean accuracy for {}: {}'.format(result[0], result[1]))



