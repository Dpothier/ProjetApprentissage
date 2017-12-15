from sklearn.model_selection import train_test_split


class SingleSplit:
    def __init__(self, test_size):
        self.test_size = test_size

    def __call__(self, data, targets, produce_results):
        data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=self.test_size)
        return produce_results(data_train, data_test, targets_train, targets_test)