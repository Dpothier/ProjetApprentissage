class Experiment:
    vectorize = None
    classify =  None

    def __init__(self, vectorize, classify):
        self.vectorize = vectorize
        self.classify = classify

    def get_experiment_result(self, data, targets):
        vectors = self.vectorize(data)
        return self.classify(vectors, targets)


class ExperimentSet:
    def __init__(self, algorithm, vectorizer_set):
        self.algorithm = algorithm
        self.vectorizer_set = vectorizer_set

    def get_experiment_results(self, data, targets):
        for vectorizer in self.vectorizer_set:
            vector = vectorizer[1].fit_transform(data)
            accuracy = self.algorithm.run_experiment(vector, targets)
            yield((vectorizer[0], accuracy))

class MetaExperimentSet:
    def __init__(self, classifying_method, metaset, results_path):
        self.classifying_method = classifying_method
        self.metaset = metaset
        self.results_path = results_path

    def execute_experiments(self, data, targets):
        for experiment_set in self.metaset:
            experiment = ExperimentSet(self.classifying_method,
                                           experiment_set[1])

            with open("{}/{}.txt".format(self.results_path, experiment_set[0]), mode="w", encoding="utf8") as f:
                for result in experiment.get_experiment_results(data, targets):
                    f.writelines('Mean accuracy for {}: {} \n'.format(result[0], result[1]))
                    print('Mean accuracy for {}: {}'.format(result[0], result[1]))