from sklearn.naive_bayes import MultinomialNB
from algorithms.helper import run_classifier

class NB:
    def __call__(self, train_data, test_data, train_target, test_target):
        classifier = MultinomialNB()
        return run_classifier(classifier, train_data, test_data, train_target, test_target)