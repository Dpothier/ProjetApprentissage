import numpy as np
import yaml
import copy


def filter_value_of_interests(data: dict, values_of_interests: list) -> dict:
    return {key : value for (key,value) in data.items() if key in values_of_interests}


def to_numpy_array(data: dict, values_of_interest: dict) -> np.array:
    source_list = list(values_of_interest.values())
    for key, value in data.items():
        list.extend(list(value.values()))

    return np.array(source_list)

def create_numpy_array(ressources_file, results_directory, experiment_list, values_of_interests):
    use_names_experiment = list(experiment_list.keys())

    ressources = load_resources(ressources_file, use_names_experiment)
    results = load_results(results_directory, use_names_experiment)
    combined = combine_resources_and_results(ressources, results)

    return to_numpy_array(combined, values_of_interests)



