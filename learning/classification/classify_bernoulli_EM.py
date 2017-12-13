import numpy as np
from sklearn.metrics import silhouette_score
from question1_algo import Bernouilli_EM


class Clustering_bernoulli_EM:

	def __init__(self, k):
		self.k = k
 		
	def __call__(self, data_vector, target):
		results = []
		for n in range(2, self.k+1, 2):
			algo = Bernouilli_EM(n_clusters=n, random_state=0)
			algo.fit(data_vector)
			predictions = algo.predict(data_vector)
			silhouette = silhouette_score(data_vector, predictions)
			results.append([n, silhouette])
		return results

