import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords


class Tokenizer:

    def __init__(self, tokenizer, post_tokenizer):
        post_tokenizer.sort(key=lambda x: x[0])
        self.tokenizer = tokenizer
        self.post_tokenizer = post_tokenizer

    def __call__(self, doc):
        tokens = self.tokenizer(doc)

        for postprocessor in self.post_tokenizer:
            tokens = postprocessor[1](tokens)

        return tokens


class NLTKTokenizer:
    def __call__(self, doc):
        return word_tokenize(doc)

class StemmerPostTokenizer:
    porter_stemmer = None
    def __init__(self):
        self.porter_stemmer = PorterStemmer()

    def __call__(self, tokens):
        return [self.porter_stemmer.stem(t) for t in tokens]

class POSFilterPostTokenizer:
    def __call__(self, tokens):
        tagged_tokens = nltk.pos_tag(tokens)
        filtered_tag_tokens = [token[0] for token in tagged_tokens if token[1] in ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR",
                                                                                   "JJS", "RB", "RBR", "RBS", "VB", "VBD",
                                                                                   "VBG", "VBN", "VBP", "VBZ"]]
        return filtered_tag_tokens

class RareWordsPostTokenizer:
    rare_words = []
    def __init__(self, rare_words=[]):
        self.rare_words = rare_words

    def __call__(self, tokens):
        return [token for token in tokens if token not in self.rare_words]

class WordNetLemmatizerPostTokenizer:
    lemmatizer = None

    def __init__(self):
        self.lemmatizer =  WordNetLemmatizer()

    def __call__(self, tokens):
        return [self.lemmatizer.lemmatize(t) for t in tokens]

class DictionaryAmplificationPostTokenizer:
    dictionary = None
    amplification_factor = 3

    def __init__(self, dictionary, amplification_factor):
        self.dictionary = dictionary
        self.amplification_factor = amplification_factor


    def __call__(self, tokens):
        amplified_tokens = []
        for token in tokens:
            if token in self.dictionary.unigram_dic:
                amplified_tokens.extend([token] * self.amplification_factor)
            else:
                amplified_tokens.append(token)
        return amplified_tokens

class StopWordPostTokenizer:

    def __call__(self, tokens):
        stopWords = set(stopwords.words('english'))
        wordsFiltered = []

        for t in tokens:
            if t not in stopWords:
                wordsFiltered.append(t)

        return wordsFiltered