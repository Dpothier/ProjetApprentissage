import numpy as np
import sklearn.datasets as datasets
from sklearn.decomposition import PCA
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_number_of_dimensions_for_percentage_of_variance(percentage, data):
    pca = PCA()
    pca.fit(data)
    ratios = np.sort(pca.explained_variance_ratio_)[::-1]
    sum_of_ratio = 0
    for i in range(0, len(ratios)):
        sum_of_ratio += ratios[i]
        if sum_of_ratio >= percentage:
            return i+1

def get_most_influential_dimension_on_principal_variant(data):
    pca = PCA()
    pca.fit(data)
    ratios = pca.explained_variance_ratio_
    components = pca.components_
    main_component_squared = np.power(components[np.argmax(ratios)],2)
    number_of_significant_dim = get_number_of_significant_dimension(main_component_squared)

    return np.argpartition(main_component_squared, -number_of_significant_dim)[-number_of_significant_dim:]

#Since we know the squared component vector sums to 1, we know a sum of 0.6
#will be comparably the same amongst any number of dimensions
def get_number_of_significant_dimension(component):
    sorted_main_component_squared = np.sort(component)[::-1]
    number_of_dim_before_60_percent_of_length = 0
    percent_of_length = 0
    while percent_of_length < 0.2:
        percent_of_length += sorted_main_component_squared[number_of_dim_before_60_percent_of_length]
        number_of_dim_before_60_percent_of_length += 1
    return number_of_dim_before_60_percent_of_length

def draw_data_using_main_components(data, target, datasetname):
    target_type = np.unique(target)
    pca = PCA()
    trans_data = pca.fit_transform(data)
    savedFile = PdfPages("PCA_{}.pdf".format(datasetname))

    for type in target_type:
        data_of_class = trans_data[np.where(target == type)]
        pyplot.scatter(data_of_class[:, 0],
                       data_of_class[:, 1], cmap=pyplot.cm.jet)


    pyplot.xlabel("First main component")
    pyplot.ylabel("Second main component")

    savedFile.savefig()
    savedFile.close()
    pyplot.show()


def draw_data_using_lda(data, target, datasetname):
    target_type = np.unique(target)

    lda = LinearDiscriminantAnalysis()
    trans_data = lda.fit_transform(data, target)

    savedFile = PdfPages("LDA_{}.pdf".format(datasetname))

    for type in target_type:
        data_of_class = trans_data[np.where(target == type)]
        pyplot.scatter(data_of_class[:, 0],
                       data_of_class[:, 1], cmap=pyplot.cm.jet)

    pyplot.xlabel("First dimension of projected hyperplane")
    pyplot.ylabel("Second dimension of projected hyperplane")

    savedFile.savefig()
    savedFile.close()
    pyplot.show()

def classify_using_PCA(data, target):
    number_of_classes = len(np.unique(target))
    pca = PCA(n_components=number_of_classes-1) #Limit the number of components to have the same as LDA
    trans_data = pca.fit_transform(data)

    return get_accuracy_with_nearest_centroid(trans_data, target)

def classify_using_LDA(data, target):
    pca = LinearDiscriminantAnalysis()
    trans_data = pca.fit_transform(data, target)
    return get_accuracy_with_nearest_centroid(trans_data, target)


def get_accuracy_with_nearest_centroid(data, target):
    classifier = NearestCentroid()
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.4)
    classifier.fit(train_data, train_target)
    predictions = classifier.predict(test_data)
    return accuracy_score(test_target, predictions)

iris = datasets.load_iris()
digits = datasets.load_digits()
olivetti = datasets.fetch_olivetti_faces()

iris_number_of_dim = get_number_of_dimensions_for_percentage_of_variance(0.6, iris.data)
iris_most_influential_dim = get_most_influential_dimension_on_principal_variant(iris.data)
iris_PCA_accuracy = classify_using_PCA(iris.data, iris.target)
iris_LDA_accuracy = classify_using_LDA(iris.data, iris.target)
print("Iris needs {} dimension to reach 60% of variance and the most influential dimensions are {}"
      .format(iris_number_of_dim, np.sort(iris_most_influential_dim)))
print("Iris has {} accuracy with PCA and {} accuracy with LDA"
      .format(iris_PCA_accuracy, iris_LDA_accuracy))
draw_data_using_main_components(iris.data, iris.target, "iris")
draw_data_using_lda(iris.data, iris.target, "iris")


digits_number_of_dim = get_number_of_dimensions_for_percentage_of_variance(0.6, digits.data)
digits_most_influential_dim = get_most_influential_dimension_on_principal_variant(digits.data)
digits_PCA_accuracy = classify_using_PCA(digits.data, digits.target)
digits_LDA_accuracy = classify_using_LDA(digits.data, digits.target)
print("Digits needs {} dimension to reach 60% of variance and the most dimensions are is {}"
      .format(digits_number_of_dim, np.sort(digits_most_influential_dim)))
print("Digits has {} accuracy with PCA and {} accuracy with LDA"
      .format(digits_PCA_accuracy, digits_LDA_accuracy))
draw_data_using_main_components(digits.data, digits.target, "digits")
draw_data_using_lda(digits.data, digits.target, "digits")

olivetti_number_of_dim = get_number_of_dimensions_for_percentage_of_variance(0.6, olivetti.data)
olivetti_most_influential_dim = get_most_influential_dimension_on_principal_variant(olivetti.data)
olivetti_PCA_accuracy = classify_using_PCA(olivetti.data, olivetti.target)
olivetti_LDA_accuracy = classify_using_LDA(olivetti.data, olivetti.target)
print("Olivetti needs {} dimension to reach 60% of variance and the most dimensions are is {}"
      .format(olivetti_number_of_dim, np.sort(olivetti_most_influential_dim)))
print("Olivetti has {} accuracy with PCA and {} accuracy with LDA"
      .format(olivetti_PCA_accuracy, olivetti_LDA_accuracy))
draw_data_using_main_components(olivetti.data, olivetti.target, "olivetti")
draw_data_using_lda(olivetti.data, olivetti.target, "olivetti")

