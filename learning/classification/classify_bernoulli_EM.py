import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


class Clustering_bernoulli_EM:
    def __init__(self, k):
        self.k = k

    def __call__(self, data_vector, target):
        results = []
        # for n in range(2, self.k+1, 2):
        # algo = Bernouilli_EM(n_clusters=n, random_state=0)
        # algo.fit(data_vector)
        # predictions = algo.predict(data_vector)
        # silhouette = silhouette_score(data_vector, predictions)
        # results.append([n, silhouette])
        return results


class Clustering_Gaussian_Mixture:
    def __init__(self, k):
        self.k = k

    def __call__(self, data_vector, target):
        results = []
        for n in range(2, self.k+1, 2):
            algo = GaussianMixture(n_components=n)
            data_array = data_vector.toarray()
            algo.fit(data_array)
            predictions = algo.predict(data_array)
            silhouette = silhouette_score(data_array, predictions)
            print([n, silhouette])
            results.append([n, silhouette])
        return results