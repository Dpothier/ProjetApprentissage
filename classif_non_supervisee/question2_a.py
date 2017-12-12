import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv("./csdmc-spam-binary/data").as_matrix()
target = pd.read_csv("./csdmc-spam-binary/target").as_matrix()
features = pd.read_csv("./csdmc-spam-binary/features").as_matrix()

algo = KMeans(n_clusters=5)
algo.fit(data)
predictions = algo.predict(data)

for i in range(0,5):
    cluster_members = data[np.where(predictions == i)[0], :]
    members_word_count = np.sum(cluster_members, axis=0)
    print("For cluster {}, among the {} documents, the 5 most common words are:".format(i, cluster_members.shape[0]))
    print(features[np.argpartition(members_word_count, -5)[-5:]])

word_count = np.sum(data, axis=0)
print("Overall, the 5 most common words are: ")
print(features[np.argpartition(word_count, -5)[-5:]])