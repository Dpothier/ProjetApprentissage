from sklearn.datasets import load_iris
from classification.SVM import SVM
from getData import get_bdrv_data
from sklearn.feature_extraction.text import CountVectorizer

data, targets = get_bdrv_data()

iris = load_iris()
svm = SVM()
vectorizer = CountVectorizer()
vector = vectorizer.fit_transform(data)
result = svm(vector, targets)
print(result)