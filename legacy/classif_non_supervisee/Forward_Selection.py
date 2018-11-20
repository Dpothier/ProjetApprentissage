import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ForwardSelection:
    classifier = None
    d = 0
    k = 0
    selectedFeatures = []

    def __init__(self, classifier, k):
        self.classifier = classifier
        self.k = k

    def fit(self, X, y):
        self.d = X.shape[1]

        selected_features = []
        while len(selected_features) < self.k:
            selected_features = self.select_new_feature(X, y, selected_features)

        self.selected_features = selected_features

    def select_new_feature(self, X, y, selected_features):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        max_accuracy = 0
        feature_for_max_accuracy = 0
        for i in range(0, self.d):
            if i not in selected_features:
                tentative_selected_features = selected_features + [i]
                classifier = self.classifier()
                classifier.fit(X_train[:, tentative_selected_features], y_train)
                pred = classifier.predict(X_test[:, tentative_selected_features])
                accuracy = accuracy_score(y_test, pred)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    feature_for_max_accuracy = i
        return selected_features + [feature_for_max_accuracy]

    def get_support(self, indices=False):
        if indices:
            return self.selected_features
        else:
            mask = np.zeros(self.d, dtype=bool)
            mask[self.selected_features] = True
            return mask

    def transform(self, X):
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

