from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time




def classify_with_NB_with_no_folds(data_vector, targets):
    X_train, X_test, y_train, y_test = train_test_split(data_vector, targets, test_size=0.2)

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return accuracy_score(y_test, predictions)


current_milli_time = lambda: int(round(time.time() * 1000))

def run_classifier(classifier, X_train, X_test, y_train, y_test):
    start_time = current_milli_time()

    classifier.fit(X_train, y_train)
    pred_train = classifier.predict(X_train)
    pred_test = classifier.predict(X_test)

    accuracy_training = accuracy_score(y_train, pred_train)
    accuracy_test = accuracy_score(y_test, pred_test)
    execution_time = current_milli_time() - start_time

    return accuracy_training, accuracy_test, execution_time

