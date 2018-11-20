from algorithms.helper import current_milli_time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def train(model, X_train, X_test, y_train, y_test):

    start_time = current_milli_time()

    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    execution_time = current_milli_time() - start_time

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "execution_time": execution_time,
        "model": model
    }

class PreFoldedTrainer():

    def __init__(self, algorithm_prototype, X_train, X_test, y_train, y_test):
        self.algorithm_prototype = algorithm_prototype
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, hyperparameters):
        model = self.algorithm_prototype(hyperparameters)
        return [train(model, self.X_train, self.X_test, self.y_train, self.y_test)]


class SingleFoldTrainer():

    def __init__(self,algorithm_prototype, test_size, X, y):
        self.algorithm_prototype = algorithm_prototype
        self.test_size = test_size
        self.X = X
        self.y = y


    def __call__(self, hyperparameters):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size)
        model = self.algorithm_prototype(hyperparameters)
        return [train(model, X_train, X_test, y_train, y_test)]


class KFoldTrainer():
    def __init__(self,algorithm_prototype, k, X, y):
        self.algorithm_prototype = algorithm_prototype;
        self.k = k
        self.X = X
        self.y = y

    def __call__(self, hyperparameters):
        kfold = KFold(n_splits=self.k)

        results = []
        for train_index, test_index in kfold.split(self.X):
            X_train = self.X[train_index]
            y_train = self.y[train_index]
            X_test = self.X[test_index]
            y_test = self.y[test_index]
            model = self.algorithm_prototype(hyperparameters)
            results.append(train(model, X_train, X_test, y_train, y_test))

        return results