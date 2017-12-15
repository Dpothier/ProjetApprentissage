

class PreselectedTestSet:
    def __init__(self, data_test, targets_test):
        self.data_test = data_test
        self.targets_test = targets_test

    def __call__(self, data, targets, produce_results):
        return produce_results(data, self.data_test, targets, self.targets_test)