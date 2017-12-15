
class OutputCentral:
    def __init__(self, outputs, experiment):
        self.outputs = outputs
        self.experiment = experiment

    def go(self, data, targets):
        for result in self.experiment.get_experiment_results(data, targets):
            for output in self.outputs:
                output(result)