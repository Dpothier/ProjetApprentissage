from sklearn.model_selection import KFold
import numpy as np

class KFold:

    def __init__(self, k):
        self.k = k

    def __call__(self, data, targets, classifier):
        kfold = KFold(n_splits=self.k)

        results = []
        for train_index, test_index in kfold.split(data):
            train_data = data[train_index]
            train_target = targets[train_index]

            test_data = data[test_index]
            test_target = targets[test_index]

            result = classifier.produce_results(train_data, test_data, train_target, test_target)

            results.append(result)

        return tuple(np.mean(results, axis=0))
