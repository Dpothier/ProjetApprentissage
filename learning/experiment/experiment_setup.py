from set_splitting.KFold import KFold
from set_splitting.single_split import SingleSplit
from algorithms.NB import NB
from algorithms.ADABoost import ADABoost
from algorithms.KMeans import KMeans
from algorithms.EM import EM_Gaussian
from algorithms.SVM import SVM
from algorithms.MLP import MLP
from set_splitting.ValidationSet import ValidationSet
from set_splitting.ValidationSet import NoValidationSet
from experiment.Experiment import ExperimentSet
from vectorization.null_preatreatment import NullPreatreatment
from output.to_result_file import ResultFileOutput
from output.to_console import ConsoleOutput
from output.output_central import OutputCentral

def With_kfold(k):
    return AlgorithmSelector(KFold(k))

def With_single_split(test_size):
    return AlgorithmSelector(SingleSplit(test_size))

class AlgorithmSelector:

    def __init__(self, set_splitter):
        self.set_splitter = set_splitter

    def use_nb(self):
        return ValidationSelector(NB(self.set_splitter))

    def use_ADABoost(self):
        return ValidationSelector(ADABoost(self.set_splitter))

    def use_KMeans(self, max_k):
        return ValidationSelector(KMeans(self.set_splitter, max_k))

    def use_EM(self, max_k):
        return ValidationSelector(EM_Gaussian(self.set_splitter, max_k))

    def use_SVM(self):
        return ValidationSelector(SVM(self.set_splitter))

    def use_MLP(self):
        return ValidationSelector(MLP(self.set_splitter))


class ValidationSelector:

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def use_validation_set(self, size):
        return PretreatmentSelector(ValidationSet(self.algorithm, size))

    def use_test_set_results(self):
        return PretreatmentSelector(NoValidationSet(self.algorithm))


class PretreatmentSelector:
    def __init__(self, validation_policy):
        self.validation_policy = validation_policy

    def use_raw_data(self):
        return OutputSelector(ExperimentSet(self.validation_policy, [("No pretreatments", NullPreatreatment())]))

    def test_on_multiple_pretreatment(self, pretreatments):
        return OutputSelector(ExperimentSet(self.validation_policy, pretreatments))

class OutputSelector:
    def __init__(self, experiment):
        self.experiment = experiment
        self.outputs = []
        self.experiment_with_output = None

    def output_to_file(self, filepath):
        self.outputs.append(ResultFileOutput(filepath))
        return self


    def output_to_console(self):
        self.outputs.append(ConsoleOutput())
        return self

    def go(self, data, targets):
        output_central = OutputCentral(self.outputs, self.experiment)
        output_central.go(data, targets)
