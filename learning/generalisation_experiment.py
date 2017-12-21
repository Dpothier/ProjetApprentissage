from getData import get_data_from_both_datasets
import experiment.experiment_setup as setup
import vectorization.VectorizerSet as sets
from dictionary import TerminologicalDictionary
import vectorization.VectorizerBuilder as builder


data_train, targets_train, data_validation, targets_validation = get_data_from_both_datasets()

dictionary = TerminologicalDictionary()

vectorizer_SVM = builder.Use_count(1).and_stemming().and_stop_words().as_vectorizer()
vectorizer_MLP = builder.Use_count(1).and_stemming().and_stop_words().as_vectorizer()

train_count = len(data_train)

all_data = data_train.copy()
all_data.extend(data_validation[:train_count])

print("Starting vectorization SVM")
all_vectors_SVM = vectorizer_SVM.fit_transform(all_data)
train_vectors_SVM = all_vectors_SVM[:train_count]
validation_vectors_SVM = all_vectors_SVM[train_count:]

print("Vectorization done")

setup.With_kfold(10)\
    .use_SVM()\
    .use_external_validation_set(validation_vectors_SVM, targets_validation[:train_count])\
    .use_raw_data()\
    .output_to_file("../results/generalisation/svm/count_postprocessing_single")\
    .output_to_console()\
    .go(train_vectors_SVM, targets_train)

print("Starting vectorization SVM")
all_vectors_MLP = vectorizer_MLP.fit_transform(all_data)
train_vectors_MLP = all_vectors_SVM[:train_count]
validation_vectors_MLP = all_vectors_SVM[train_count:]

print("Vectorization done")

setup.With_kfold(10)\
    .use_MLP()\
    .use_external_validation_set(validation_vectors_MLP, targets_validation[:train_count])\
    .use_raw_data()\
    .output_to_file("../results/generalisation/MLP/count_postprocessing_single")\
    .output_to_console()\
    .go(train_vectors_MLP, targets_train)