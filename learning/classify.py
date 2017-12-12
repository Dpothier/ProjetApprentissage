from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def classify_with_NB(data_vector, targets):
    kfold = KFold(n_splits=10, shuffle=True)

    accuracy_scores = []
    for train_index, test_index in kfold.split(data_vector):
        train_data = data_vector[train_index]
        train_target = targets[train_index]

        test_data = data_vector[test_index]
        test_target = targets[test_index]

        classifier = MultinomialNB()
        classifier.fit(train_data, train_target)
        predictions = classifier.predict(test_data)

        accuracy_scores.append(accuracy_score(test_target, predictions))

    return sum(accuracy_scores)/len(accuracy_scores)


