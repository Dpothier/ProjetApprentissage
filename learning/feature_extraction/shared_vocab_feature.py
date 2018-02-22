from sklearn.externals import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from vectorization.VectorizerBuilder import Use_binary_count

class SharedVocabFeatureExtractor:

    def __init__(self, lower_n, upper_n, dictionary = None, use_stemming=True):
        self.lower_n = lower_n
        self.upper_n = upper_n
        self.dictionary = dictionary
        self.use_stemming = use_stemming


    def extract_feature(self, matches):
        """Receive a list of Matches, vectorizes all texts, then computes the vocabulary overlap for each complaint/recall pair"""
        carcomplaints_ids = []
        bdrv_ids = []
        texts = []
        for match in matches:
            if match.carcomplaint['id'] not in carcomplaints_ids:
                carcomplaints_ids.append(match.carcomplaint['id'])
                texts.append(match.carcomplaint['text'])
            if match.bdrv['recall_id'] not in bdrv_ids:
                bdrv_ids.append(match.bdrv['recall_id'])
                texts.append(match.bdrv['details'])

        n_gram_shared = np.zeros((len(matches), len(range(self.lower_n, self.upper_n))))
        for n_index, n in enumerate(range(self.lower_n, self.upper_n)):
            if self.use_stemming:
                vectorizer = Use_binary_count(n).and_stemming().as_vectorizer()
            else:
                vectorizer = Use_binary_count(n).as_vectorizer()
            vectorizer.fit(texts)


            for match_index, match in enumerate(matches):
                carcomplaint_vector = vectorizer.transform([match.carcomplaint['text']])
                recall_vector = vectorizer.transform([match.bdrv['details']])
                if self.dictionary is not None:
                    vocabulary = vectorizer.vocabulary_
                    for term, index in vocabulary.items():
                        if term not in self.dictionary:
                            carcomplaint_vector[:, index] = 0
                            recall_vector[:, index] = 0
                n_gram_shared[match_index, n_index] = np.sum(carcomplaint_vector[carcomplaint_vector == recall_vector])

        return n_gram_shared

