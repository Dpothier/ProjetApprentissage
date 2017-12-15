from sklearn.datasets import load_iris
from algorithms.SVM import SVM
from algorithms.MLP import MLP
from algorithms.ADABoost import ADABoost
from getData import get_bdrv_data
from sklearn.feature_extraction.text import CountVectorizer

data, targets = get_bdrv_data()

iris = load_iris()
svm = SVM()
mlp = MLP()
adaBoost = ADABoost()
vectorizer = CountVectorizer()
vector = vectorizer.fit_transform(data)
print("data vectorized")
result = adaBoost(vector, targets)
print(result)
