
class DistinctSplits:
    def __init__(self, k):
        self.k = k

    def __call__(self, data, targets, classifier):
        length_of_dataset = len(data)
        size_of_split = length_of_dataset // self.k
        remains = length_of_dataset % self.k
        start_index = 0
        remains_used = 0
        results = []
        while start_index < length_of_dataset:
            end_index = start_index + size_of_split - 1
            if remains_used < remains:
                end_index += 1
                remains_used += 1

        data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=self.test_size)
        return classifier.produce_results(data_train, data_test, targets_train, targets_train)