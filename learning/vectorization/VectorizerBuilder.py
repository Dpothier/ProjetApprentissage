from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from vectorization import tokenizers as tokens
from vectorization.Vectorizer import Vectorizer
from vectorization import Vectorizer as vectorizer



def Use_count(ngram):
    return VectorizerConfigurator(CountVectorizer(ngram_range=(1, ngram)))

def Use_tf(ngram):
    return VectorizerConfigurator(TfidfVectorizer(use_idf=False, ngram_range=(1, ngram)))


def Use_tfidf(ngram):
    return VectorizerConfigurator(TfidfVectorizer(use_idf=True, ngram_range=(1, ngram)))


class VectorizerConfigurator:

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.post_vectorizers = []
        self.post_tokenizers = []

    def and_stemming(self):
        self.post_tokenizers.append((0, tokens.StemmerPostTokenizer()))
        return self

    def and_pos_filter(self):
        self.post_tokenizers.append((3, tokens.POSFilterPostTokenizer()))
        return self

    def and_lemmatization(self):
        self.post_tokenizers.append((2, tokens.LemmatizerPostTokenizer()))
        return self

    def and_lemmatization_with_pos(self):
        self.post_tokenizers.append((2, tokens.LemmatizerWithPosPostTokenizer()))
        return self

    def and_dict_amp(self, dictionary, factor):
        self.post_vectorizers.append((1, vectorizer.DictAmpPostVectorizer(dictionary, factor)))
        return self

    def and_stop_words(self):
        self.post_tokenizers.append((4, tokens.StopWordPostTokenizer()))
        return self

    def and_closed_vocab(self, dictionary):
        self.post_tokenizers.append((5, tokens.ClosedVocabularyTokenizer(dictionary)))
        return self

    def as_vectorizer(self):
        tokenizer = tokens.Tokenizer(tokens.NLTKTokenizer(), self.post_tokenizers)
        self.vectorizer.tokenizer = tokenizer
        vectorizer_with_post_vectorizers = Vectorizer(self.vectorizer, self.post_vectorizers)
        return vectorizer_with_post_vectorizers