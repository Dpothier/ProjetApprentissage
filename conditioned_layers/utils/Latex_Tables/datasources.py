import abc
import yaml
import collections
import numpy as np

class Datasource(abc.ABC):
    @abc.abstractmethod
    def load(self, rows):
        pass

class ExperimentResultDatasource(Datasource):
    def __init__(self, result_folder: str):
        self.result_folder = result_folder

    def load(self, experiment_list: list) -> dict:
        output = collections.OrderedDict()
        for experiment in experiment_list:
            with open("{}/{}/results.yaml".format(self.result_folder, experiment), mode="r", encoding="utf-8") as f:
                content = yaml.load(f)
                output[experiment] = content["results"]["average"]

        return output

class ResourceUsageDatasource(Datasource):
    def __init__(self, source_file: str):
        self.source_file = source_file

    def load(self, experiment_list: list) -> dict:
        with open(self.source_file, mode="r", encoding="utf-8") as f:
            content = yaml.load(f)
            output = collections.OrderedDict()
            for experiment in experiment_list:
                output[experiment] = {
                    "memory": int(content[experiment]["memory"]["total"]),
                    "computation": int(content[experiment]["computation"]["total"])
                }

            return output


class CompositeDataSource(Datasource):
    def __init__(self, datasources: list):
        self.datasources = datasources

    def load(self, experiment_list: list) -> dict:
        temp_data = []
        output_data = {}
        for source in self.datasources:
            temp_data.append(source.load(experiment_list))

        for experiment in experiment_list:
            combined_data = {}
            for data in temp_data:
                combined_data.update(data[experiment])
            output_data[experiment] = combined_data

        return output_data


class DataLoader:
    def __init__(self, source: Datasource):
        self.source = source

    def load_data(self, row_names: list, column_names: list) -> np.array:
        raw_data = self.source.load(row_names)
        rows = []
        for row_name in row_names:
            row = []
            for column in column_names:
                row.append(raw_data[row_name][column])
            rows.append(row)

        return np.array(rows, dtype=object)