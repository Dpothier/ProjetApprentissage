

class cheapest_successful_hyperparams_report():

    def __init__(self, success_threshold, model_type):
        self.success_threshold = success_threshold
        self.model_type = model_type

    def produce_report(self, data, training_time):
        successful_hyperparams = self.evaluate_success(data, training_time)

        fastest_set = self.find_fastest(successful_hyperparams)
        smallest_set = self.find_smallest(successful_hyperparams)

        return {"success": successful_hyperparams,
                "fastest_set": fastest_set,
                "smallest_set": smallest_set}


    def evaluate_success(self, data, training_time):
        successful_hyperparams = []

        for experiment in data:
            hyperparameters = experiment["hyperparameters"]

            average_max_accuracy = experiment["max_val_acc"][training_time]

            if average_max_accuracy >= self.success_threshold:
                successful_hyperparams.append(hyperparameters)

        return successful_hyperparams

    def find_fastest(self, hyperparameters_sets):
        fastest_set = None
        fastest_ops = None
        for set in hyperparameters_sets:
            ops = self.model_type.count_operations(**set)
            if not fastest_set:
                fastest_set = set
                fastest_ops = ops
            else:
                if ops < fastest_ops:
                    fastest_set = set
                    fastest_ops = ops

        return (fastest_set, fastest_ops)

    def find_smallest(self, hyperparameters_sets):
        smallest_set = None
        smallest_size = None
        for set in hyperparameters_sets:
            size = self.model_type.count_parameters(**set)
            if not smallest_set:
                smallest_set = set
                smallest_size = size
            else:
                if size < smallest_size:
                    smallest_set = set
                    smallest_size = size

        return (smallest_set, smallest_size)

