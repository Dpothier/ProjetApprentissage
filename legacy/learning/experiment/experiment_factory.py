from metrics.metrics_producers import *
from output.outputs import *
from trainers.trainers import *
from hyperparameters.hyperparameter_grid_optimizer import HyperparametersGridOptimizer

class ExperimentFactory:

    def build_experiment(self, config):
        result_producer = self._prepare_result_producer(config)
        hyperparameters_optimizer = self._prepare_hyperparameters_optimizer(config)
        output = OutputCentral(config.experiment_name, config.metrics_producers, config.output_channel)

        return Experiment(hyperparameters_optimizer, result_producer, output)

    def _prepare_result_producer(self, config):
        if config.hyperparameters_grid is not None:
            self._prepare_validation_set(config)
            return PreFoldedTrainer(config.algorithm_prototype, config.X, config.validation_X, config.y, config.validation_y)
        elif config.single_fold_size is not None:
            return SingleFoldTrainer(config.algorithm_prototype, config.single_fold_size, config.X, config.y)
        elif config.number_of_fold is not None:
            return KFoldTrainer(config.algorithm_prototype, config.number_of_fold, config.X, config.y)


    def _prepare_validation_set(self, config):
        if config.validation_set_size is not None:
            config.X, config.validation_X, config.y, config.validation_y = \
                train_test_split(config.X, config.y, test_size=config.validation_set_size)

    def _prepare_hyperparameters_optimizer(self, config):
        if config.defined_hyperparameters is None and config.hyperparameters_grid is None:
            return lambda: {}

        if config.defined_hyperparameters is not None:
            return lambda: config.defined_hyperparameters

        if config.hyperparameters_grid is not None:
            trainer = self._get_hyperparameter_trainer(config)
            return HyperparametersGridOptimizer(trainer, config.hyperparameters_grid, MeanAccuracyMetricProducer())

    def _get_hyperparameter_trainer(self, config):
        assert config.single_fold_size is not None or config.number_of_fold is not None

        if config.single_fold_size is not None:
            return SingleFoldTrainer(config.algorithm_prototype, config.single_fold_size, config.X, config.y)
        if config.number_of_fold is not None:
            return KFoldTrainer(config.algorithm_prototype, config.number_of_fold, config.X, config.y)



class Experiment:

    def __init__(self, optimize_hyperparameters, produce_results, output_results):
        self.optimize_hyperparameters = optimize_hyperparameters
        self.produce_results = produce_results
        self.output_results = output_results

    def execute(self):
        hyperparameters = self.optimize_hyperparameters()
        results = self.produce_results(hyperparameters)
        self.output_results(results)