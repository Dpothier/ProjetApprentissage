import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# def test_for_n_clusters(n, data_vector, target):
#     algo = KMeans(n_clusters=n, random_state=0)
#     algo.fit(data_vector)
#     predictions = algo.predict(data_vector)
    
#     silhouette_score = silhouette_score(data_vector, predictions)

#     return n, silhouette_score


class Clustering_kmeans:

 	def __init__(self, k):
 		self.k = k
 		
 	def __call__(self, data_vector, target):
 		results = []
 		for n in range(2, self.k+1, 2):
 			algo = KMeans(n_clusters=n, random_state=0)
			algo.fit(data_vector)
			predictions = algo.predict(data_vector)
			silhouette_score = silhouette_score(data_vector, predictions)
			results.append(n, silhouette_score)
 		return results