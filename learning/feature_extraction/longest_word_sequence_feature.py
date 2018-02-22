from sklearn.externals import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize

class LongestWordSequenceFeatureExtractor:

    def extract_feature(self, matches):
        """Receive a list of Matches, compare each texts to find the longest token sequence. The length of that sequence is the feature"""

        longest_word_sequence = np.zeros((len(matches), 1))
        for index, match in enumerate(matches):
            lts = self.longest_token_sequence(word_tokenize(match.bdrv['details']), word_tokenize(match.carcomplaint['text']))
            longest_word_sequence[index][0] = len(lts)

        return longest_word_sequence


    def longest_token_sequence(self, a, b):
        lengths = [[0 for j in range(len(b)+1)] for i in range(len(a)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                if x == y:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
        # read the substring out from the matrix
        result = []
        x, y = len(a), len(b)
        while x != 0 and y != 0:
            if lengths[x][y] == lengths[x-1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else:
                assert a[x-1] == b[y-1]
                result = [a[x-1]] + result
                x -= 1
                y -= 1
        return result