import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from Forward_Selection import ForwardSelection


def classify_with_feature_selection(data, target, seed, fonction):
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.5,
                                                                        random_state=seed)
    feature_selector = SelectKBest(fonction, k = 10)
    selected_train_data = feature_selector.fit_transform(train_data, train_target)
    selected_test_data = feature_selector.transform(test_data)

    accuracy = classify(selected_train_data, selected_test_data, train_target, test_target, seed)
    return accuracy, feature_selector.get_support(True)


def classify_with_no_feature_selection(data, target, seed):
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.5, random_state=seed)
    return classify(train_data, test_data,train_target, test_target, seed)


def classify_with_RFE(data, target, seed):
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.5, random_state=seed)

    feature_selector = RFE(LinearSVC(random_state=seed), 10, step=1)
    selected_train_data = feature_selector.fit_transform(train_data, train_target)
    selected_test_data = feature_selector.transform(test_data)

    accuracy = classify(selected_train_data, selected_test_data, train_target, test_target, seed)

    return accuracy, feature_selector.get_support(True)


def classify_with_forward_selection(data,target,seed):
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.5, random_state=seed)

    feature_selector = ForwardSelection(LinearSVC, 10)
    selected_train_data = feature_selector.fit_transform(train_data, train_target)
    selected_test_data = feature_selector.transform(test_data)

    accuracy = classify(selected_train_data, selected_test_data, train_target, test_target, seed)

    return accuracy, feature_selector.get_support(True)


def classify(train_data, test_data, train_target, test_target, seed):
    classifier = LinearSVC(random_state=seed)
    classifier.fit(train_data, train_target)
    pred = classifier.predict(test_data)

    accuracy = accuracy_score(test_target, pred)
    return accuracy


if __name__ == '__main__':
    data = pd.read_csv("C:\\Users\\Dominique\\PycharmProjects\\ML_homework2\\csdmc-spam-binary\\data").as_matrix()
    target = pd.read_csv("C:\\Users\\Dominique\\PycharmProjects\\ML_homework2\\csdmc-spam-binary\\target").as_matrix()[:, 0]
    features = pd.read_csv("C:\\Users\\Dominique\\PycharmProjects\\ML_homework2\\csdmc-spam-binary\\features").as_matrix()

    random_seed = 0
    chi2_accuracy, chi2_features_indices = classify_with_feature_selection(data, target, random_seed, chi2)
    mutual_info_accuracy, mutual_info_features_indices = classify_with_feature_selection(data, target, random_seed, mutual_info_classif)
    RFE_accuracy, RFE_features_indices = classify_with_RFE(data, target, random_seed)
    Forward_accuracy, Forward_features_indices = classify_with_forward_selection(data, target, random_seed)
    no_selection_accuracy = classify_with_no_feature_selection(data, target, random_seed)

    print("Chi2 accuracy: {}".format(chi2_accuracy))
    print("Chi2 selected features:")
    print(features[chi2_features_indices])

    print("Mutual info accuracy: {}".format(mutual_info_accuracy))
    print("Mutual info selected features:")
    print(features[mutual_info_features_indices])

    print("RFE accuracy: {}".format(RFE_accuracy))
    print("RFE selected features:")
    print(features[RFE_features_indices])

    print("Forward selection accuracy: {}".format(Forward_accuracy))
    print("Forward selection selected features:")
    print(features[Forward_features_indices])

    print("No selection accuracy: {}".format(no_selection_accuracy))
