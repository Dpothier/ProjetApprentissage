from algorithms.prototypes import *
from hyperparameters.hyperparameters_grid import HyperparameterGrid
from experiment.experiment_factory import *


def use_SVM(use_probability):
    return ExperimentConfiguration(SvmPrototype(use_probability))


def use_MLP():
    return ExperimentConfiguration(MlpPrototype())


def use_NB():
    return ExperimentConfiguration(NbPrototype())


def use_ADA_boost():
    return ExperimentConfiguration(AdaBoostPrototype())


class ExperimentConfiguration:
    X = None
    y = None
    number_of_fold = None
    single_fold_size = None
    defined_hyperparameters = None
    hyperparameters_grid = None
    validation_set_size = None
    validation_X = None
    validation_y = None
    metrics_producers = []
    output_channel = [ConsoleOutput()]
    experiment_name = ""

    def __init__(self, algorithm_prototype):
        self.algorithm_prototype = algorithm_prototype

    def train_on_k_fold(self, k):
        assert self.number_of_fold is None, 'Number of fold is already set'

        assert self.single_fold_size is None, 'Training is already configured to use single fold'

        self.number_of_fold = k

        return self

    def train_on_single_fold(self, test_size):
        assert self.number_of_fold is None, 'Training is already configured to use k fold'

        assert self.single_fold_size is None, 'Size of test set size on single fold already set'

        self.single_fold_size = test_size

        return self

    def use_fixed_hyperparameters(self, hyperparameters_dict):
        """The keys are the hyperparameters names"""
        assert self.hyperparameters_grid is None, 'Hyperparameters are already configured to use grid search'

        assert len(hyperparameters_dict) == self.algorithm_prototype.expected_hyperparameters,\
            'Hyperparameters_list is not of the size expected by the learning algorithm'

        self.defined_hyperparameters = hyperparameters_dict
        return self

    def use_hyperparameters_grid_search(self, hyperparameters_grid):
        """ Each item in the hyperparameters_grid dict is a list containing the possible values for that hyperparameter.
            All hyperparameters values will be tried in a grid fashion"""

        self.hyperparameters_grid = HyperparameterGrid(hyperparameters_grid)
        return self

    def use_internal_validation_set(self, validation_set_size):
        assert self.hyperparameters_grid is not None or self.defined_hyperparameters is not None,\
            "Validation is not necessary without hyperparameters"
        assert self.validation_set_size is None, "Validation set size is already set"
        assert self.validation_X is None and self.validation_y is None, "Can't use internal and external validation set at once"

        self.validation_set_size = validation_set_size
        return self

    def use_external_validation_set(self, validation_X, validation_y):
        assert self.hyperparameters_grid is not None or self.defined_hyperparameters is not None, \
            "Validation is not necessary without hyperparameters"
        assert self.validation_X is None and self.validation_y is None, 'Validation set is already set'
        assert self.validation_set_size is None, "Can't use internal and external validation set at once"

        self.validation_X = validation_X,
        self.validation_y = validation_y
        return self

    def use_mean_accuracies_metrics(self):
        self.metrics_producers.append(MeanAccuracyMetricProducer())
        return self

    def use_mean_training_time_metric(self):
        self.metrics_producers.append(MeanTrainingTimeMetricProducer())
        return self

    def use_mean_confusion_matrix_metric(self):
        self.metrics_producers.append(MeanConfusionMatrixMetricProducer())
        return self

    def use_model_dumper(self, filename):
        self.metrics_producers.append(ModelDumper(filename))
        return self

    def output_to_file(self, filename):
        self.output_channel.append(FileOutput(filename))
        return self

    def execute_with_data(self, experiment_name, X, y):
        assert X.shape[0] == y.shape[0], 'X and y must have the same number of instances'
        self.experiment_name = experiment_name
        self.X = X
        self.y = y

        ExperimentFactory().build_experiment(self).execute()