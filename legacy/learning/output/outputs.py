from datetime import datetime

class ConsoleOutput:

    def __call__(self,experiment_name, metrics):
        for k, v in metrics.items():
            print('{}: {}'.format(k, v))

class FileOutput:
    def __init__(self, filename):
        self.filename = filename

    def __call__(self,experiment_name, metrics):

        output_file = open("{}.txt".format(self.filename), mode="a+", encoding="utf8")
        output_file.writelines('Results for experiment {} at {}\n'.format(experiment_name, str(datetime.now())))
        for k, v in metrics.items():
            output_file.writelines('{}: {}\n'.format(k, v))

        output_file.close()


class OutputCentral:
    metrics = {}

    def __init__(self, experiment_name, metrics_producers, output_channels):
        self.metrics_producers = metrics_producers
        self.output_channels = output_channels
        self.experiment_name = experiment_name

    def __call__(self, results):
        for metrics_producer in self.metrics_producers:
            self.metrics = {**self.metrics, **metrics_producer(results)}

        for output_channel in self.output_channels:
            output_channel(self.experiment_name, self.metrics)
