from sklearn.model_selection import KFold

class KFold:

    def __init__(self, learningAlgorithm, k):
        self.learningAlgorithm = learningAlgorithm
        self.k = k

    def __call__(self, data, targets):
        kfold = KFold(n_splits=self.k)

        training_accuracy = []
        test_accuracy = []
        execution_time = []
        for train_index, test_index in kfold.split(data):
            train_data = data[train_index]
            train_target = targets[train_index]

            test_data = data[test_index]
            test_target = targets[test_index]

            result = self.learningAlgorithm(train_data, test_data, train_target, test_target)

            training_accuracy.append(result[0])
            test_accuracy.append(result[1])
            execution_time.append(result[2])

        return sum(training_accuracy)/len(training_accuracy), \
            sum(test_accuracy)/len(test_accuracy),\
            sum(execution_time)/len(execution_time)
