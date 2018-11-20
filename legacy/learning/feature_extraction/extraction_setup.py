from feature_extraction.class_probabilities_feature import ClassProbabilitiesExtractor
from feature_extraction.class_probabilities_cosine_distance import ClassProbabilitiesCosineDistanceExtractor
from feature_extraction.elastic_score_feature import ElasticScoreFeatureExtractor
from feature_extraction.longest_word_sequence_feature import LongestWordSequenceFeatureExtractor
from feature_extraction.shared_vocab_feature import SharedVocabFeatureExtractor
import numpy as np

def start_extraction(matches):
    return FeatureExtractionConfigurator(matches)


class FeatureExtractionConfigurator:

    def __init__(self, matches):
        self.matches = matches
        self.feature_extractors = []

    def use_class_probabilities(self, vectorizer_filename, probabilities_model_filename):
        self.feature_extractors.append(ClassProbabilitiesExtractor(vectorizer_filename, probabilities_model_filename))
        return self

    def use_class_probabilities_cosine_distance(self,vectorizer_filename, probabilities_model_filename):
        self.feature_extractors.append(ClassProbabilitiesCosineDistanceExtractor(vectorizer_filename, probabilities_model_filename))
        return self

    def use_elastic_score(self):
        self.feature_extractors.append(ElasticScoreFeatureExtractor())
        return self

    def use_longest_word_sequence(self):
        self.feature_extractors.append(LongestWordSequenceFeatureExtractor())
        return self

    def use_shared_vocabulary(self, lower_n, upper_n):
        self.feature_extractors.append(SharedVocabFeatureExtractor(lower_n, upper_n, use_stemming=False))
        return self

    def use_stemmed_shared_vocabulary(self, lower_n, upper_n):
        self.feature_extractors.append(SharedVocabFeatureExtractor(lower_n, upper_n, use_stemming=True))
        return self

    def use_shared_named_entities(self, lower_n, upper_n, shared_entities):
        self.feature_extractors.append(SharedVocabFeatureExtractor(lower_n, upper_n, dictionary=shared_entities, use_stemming=False))
        return self

    def extract_features(self):
        concatenated_features = None
        for extractor in self.feature_extractors:
            if concatenated_features is None:
                concatenated_features = extractor.extract_feature(self.matches)
            else:
                concatenated_features = np.concatenate((concatenated_features, extractor.extract_feature(self.matches)), axis=1)

        return concatenated_features
