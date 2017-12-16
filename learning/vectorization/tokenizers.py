import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from vectorization.helper import get_wordnet_pos


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
        print("starting stemming")
        return [self.porter_stemmer.stem(t) for t in tokens]

class POSFilterPostTokenizer:
    def __call__(self, tokens):
        print("starting POS filtering")
        tagged_tokens = nltk.pos_tag(tokens)
        filtered_tag_tokens = [token[0] for token in tagged_tokens if token[1] in ["NN", "NNS", "NNP", "NNPS", "JJ", "JJR",
                                                                                   "JJS", "RB", "RBR", "RBS", "VB", "VBD",
                                                                                   "VBG", "VBN", "VBP", "VBZ"]]
        return filtered_tag_tokens

class RareWordsPostTokenizer:
    def __init__(self, rare_words=[]):
        self.rare_words = rare_words

    def __call__(self, tokens):
        print("starting rare words filtering")
        return [token for token in tokens if token not in self.rare_words]

class LemmatizerPostTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, tokens):
        print("starting lemmatization")
        return [self.lemmatizer.lemmatize(t) for t in tokens]

class LemmatizerWithPosPostTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, tokens):
        print("starting lemmatization with POS tagging")
        tagged_tokens = nltk.pos_tag(tokens)
        return [self.lemmatizer.lemmatize(t[0], get_wordnet_pos(t[1])) for t in tagged_tokens]

class DictionaryAmplificationPostTokenizer:
    dictionary = None
    amplification_factor = 3

    def __init__(self, dictionary, amplification_factor):
        self.dictionary = dictionary
        self.amplification_factor = amplification_factor


    def __call__(self, tokens):
        print("starting dictionary amplification")
        amplified_tokens = []
        for token in tokens:
            if token in self.dictionary.unigram_dic:
                amplified_tokens.extend([token] * self.amplification_factor)
            else:
                amplified_tokens.append(token)
        return amplified_tokens

class StopWordPostTokenizer:

    def __call__(self, tokens):
        print("starting stop words")
        stopWords = set(stopwords.words('english'))
        wordsFiltered = []

        for t in tokens:
            if t not in stopWords:
                wordsFiltered.append(t)

        return wordsFiltered

class ClosedVocabularyTokenizer:

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, tokens):
        print("starting closed dictionary filter")
        return [token for token in tokens if token in self.dictionary.unigram_dic]


