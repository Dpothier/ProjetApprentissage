import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeans:


    def __init__(self, k):
        self.k = k

    def __call__(self, data_vector, target):
        results = []
        for n in range(2, self.k+1, 2):
            algo = KMeans(n_clusters=n)
            clusters_label = algo.fit_predict(data_vector)
            silhouette = silhouette_score(data_vector, clusters_label)
            results.append((n, silhouette))
        return results
