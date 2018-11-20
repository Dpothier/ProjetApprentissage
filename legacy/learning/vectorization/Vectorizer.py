import numpy as np

class Vectorizer:
    vectorizer = None

    def __init__(self, vectorizer, post_vectorizers):
        self.vectorizer = vectorizer
        self.post_vectorizers = post_vectorizers
        self.post_vectorizers.sort(key=lambda x: x[0])

    def fit_transform(self, X):
        vector = self.vectorizer.fit_transform(X)
        return self.apply_post_vectorizers(vector)

    def transform(self, X):
        vector = self.vectorizer.transform(X)
        return self.apply_post_vectorizers(vector)

    def fit(self, X):
        self.vectorizer.fit(X)


    def apply_post_vectorizers(self, vector):
        vector = vector.tolil()

        for post_vectorizer in self.post_vectorizers:
            vector = post_vectorizer[1](vector, self.vectorizer)

        vector = vector.tocsr()
        return vector

class DictAmpPostVectorizer:

    def __init__(self, dictionary, amp_factor):
        self.dictionary = dictionary
        self.amp_factor = amp_factor

    def __call__(self, vector, vectorizer):
        amplified_vector = vector.copy()
        vocabulary = vectorizer.vocabulary_
        for term, index in vocabulary.items():
            if term in self.dictionary:
                amplified_vector[:, index] = amplified_vector[:, index] * self.amp_factor

        return amplified_vector