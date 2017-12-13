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
    classifier_method = None
    vectorizer_set = None

    def __init__(self, classifier_factory, vectorizer_set):
        self.classifier_method = classifier_factory
        self.vectorizer_set = vectorizer_set

    def get_experiment_results(self, data, targets):
        for vectorizer in self.vectorizer_set:
            vector = vectorizer[1].fit_transform(data)
            accuracy = self.classifier_method(vector, targets)
            yield((vectorizer[0], accuracy))
