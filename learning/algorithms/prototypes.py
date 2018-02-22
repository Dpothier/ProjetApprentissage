from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier


class SvmPrototype:
    def __init__(self, use_probability):
        self.expected_hyperparameters = 2
        self.use_probability = use_probability

    def __call__(self, hyperparameters):
        return SVC(C=hyperparameters["c"], gamma=(1 / (2 * hyperparameters["sigma"] ** 2)), probability=self.use_probability, class_weight='balanced')


class NbPrototype:
    def __init__(self):
        self.expected_hyperparameters = 0

    def __call__(self, hyperparameters):
        return MultinomialNB()


class MlpPrototype:
    def __init__(self):
        self.expected_hyperparameters = 2

    def __call__(self, hyperparameters):
        return MLPClassifier(hidden_layer_sizes=(hyperparameters['width'],) * hyperparameters['depth'])


class AdaBoostPrototype:
    def __init__(self):
        self.expected_hyperparameters = 0

    def __call__(self, hyperparameters):
       return AdaBoostClassifier()
