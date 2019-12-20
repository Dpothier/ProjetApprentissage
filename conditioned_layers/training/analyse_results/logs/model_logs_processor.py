from csv import DictReader
from os import listdir


class model_logs_processor():

    def __init__(self, folder, logs_processor, hyperparameters_extractor):
        self.model_folder = folder
        self.logs_processor = logs_processor
        self.hyperparameters_extractor = hyperparameters_extractor

    def process_experiments_on_model(self):
        experiments_folders = [f for f in listdir(self.model_folder)]

        experiments_stats = []
        for experiment in experiments_folders:
            experiments_stats.append(self.process_experiment("{}/{}".format(self.model_folder, experiment)))

        return experiments_stats


    def process_experiment(self, experiment_name):
        experiment_data = {}
        experiment_data["hyperparameters"] = self.hyperparameters_extractor.extract_from(experiment_name)

        logs = [f for f in listdir("{}".format(experiment_name)) if f[-4:] == ".csv"]
        for log in logs:
            self.process_file("{}/{}".format(experiment_name, log))

        experiment_data.update(self.logs_processor.aggregate_files())
        return experiment_data


    def process_file(self, file_name):
        with open(file_name, newline='') as file:
            reader = DictReader(file)
            for line in reader:
                self.logs_processor.process(line)

        self.logs_processor.end_file()

