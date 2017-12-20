from getData import get_data_from_both_datasets
import experiment.experiment_setup as setup
import vectorization.VectorizerSet as sets
from dictionary import TerminologicalDictionary
import vectorization.VectorizerBuilder as builder


data_train, targets_train, data_validation, targets_validation = get_data_from_both_datasets()

dictionary = TerminologicalDictionary()

vectorizer = builder.Use_count(1).and_stemming().and_pos_filter().as_vectorizer()

train_count = len(data_train)

all_data = data_train.copy()
all_data.extend(data_validation[:train_count])

print("Starting vectorization")
all_vectors = vectorizer.fit_transform(all_data)
train_vectors = all_vectors[:train_count]
validation_vectors = all_vectors[train_count:]

print("Vectorization done")

setup.With_kfold(10)\
    .use_SVM()\
    .use_external_validation_set(validation_vectors, targets_validation[:train_count])\
    .use_raw_data()\
    .output_to_file("../results/generalisation/svm/count_postprocessing_single")\
    .output_to_console()\
    .go(train_vectors, targets_train)