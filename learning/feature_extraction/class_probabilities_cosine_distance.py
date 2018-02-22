from sklearn.externals import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ClassProbabilitiesCosineDistanceExtractor:

    def __init__(self, vectorizer_filename, model_filename):
        self.vectorizer = joblib.load(os.path.abspath(vectorizer_filename))
        self.model = joblib.load(os.path.abspath(model_filename))


    def extract_feature(self, matches):
        """Receive a list of Matches, extract class probabilities for bdrv and carcomplaints and return joint result"""
        bdrv_text = []
        carcomplaints_text = []
        for match in matches:
            bdrv_text.append(match.bdrv['details'])
            carcomplaints_text.append(match.carcomplaint['text'])

        bdrv_probabilities = self.extract_probabilities(bdrv_text)
        carcomplaints_probabilities = self.extract_probabilities(carcomplaints_text)
        return np.diag(cosine_similarity(bdrv_probabilities, carcomplaints_probabilities))[..., np.newaxis]



    def extract_probabilities(self, data):
        vectorized_data = self.vectorizer.transform(data)

        probabilities = self.model.predict_proba(vectorized_data)

        return probabilities

