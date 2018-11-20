import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import v_measure_score
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from question1_algo import Bernouilli_EM


def classify(n, pred, target):
    print("classification for {} clusters".format(n))
    y = np.zeros(len(target))
    for i in range(0, n):
        cluster_targets = target[np.where(pred == i)]
        if len(np.where(cluster_targets == 1)[0]) >= len(np.where(cluster_targets == 0)[0]):
            y[np.where(pred == i)[0]] = 1
            #print("cluster {} is legitimate with {} members".format(i, len(cluster_targets)))

    print("y for {} clusters: {}: {}".format(n, len(np.where(y == 0)[0]), len(np.where(y == 1)[0])))
    print(" ")
    return y


def test_for_n_clusters(n, data, target):
    algo = Bernouilli_EM(n_clusters=n, random_state=0)
    algo.fit(data)
    predictions = algo.predict(data)
    classification = classify(n, predictions, target)

    rand = adjusted_rand_score(classification, target)
    mutual_info = adjusted_mutual_info_score(classification, target)
    v_measure = v_measure_score(classification, target)

    return n, rand, mutual_info, v_measure


if __name__ == '__main__':
    data = pd.read_csv("C:\\Users\\Dominique\\PycharmProjects\\ML_homework2\\csdmc-spam-binary\\data").as_matrix()[:, 0:100] #Only the first 100 variables
    target = pd.read_csv("C:\\Users\\Dominique\\PycharmProjects\\ML_homework2\\csdmc-spam-binary\\target").as_matrix()[:, 0]
    features = pd.read_csv("C:\\Users\\Dominique\\PycharmProjects\\ML_homework2\\csdmc-spam-binary\\features").as_matrix()

    results = []
    for n in range(2, 51, 2):
        results.append(test_for_n_clusters(n, data, target))

    results = np.array(results)

    savedFile = PdfPages("./Clustering_score_measures_EM.pdf")
    colors = ["red", "blue", "green"]

    rand = pyplot.plot(results[:, 0], results[:, 1], color=colors[0], label="Rand")
    mutual = pyplot.plot(results[:, 0], results[:, 2], color=colors[1], label="Mutual Information")
    V = pyplot.plot(results[:, 0], results[:, 3], color=colors[2], label="V")

    pyplot.legend()

    pyplot.xlabel("Number of clusters")
    pyplot.ylabel("Value of various measures")

    savedFile.savefig()
    savedFile.close()
    pyplot.show()
