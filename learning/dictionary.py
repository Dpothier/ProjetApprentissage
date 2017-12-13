import csv
from nltk.corpus import stopwords

def load_dictionary():
    with open('../data/engdictionary.csv', encoding="utf8") as f:
        rows = [{k: str(v) for k, v in row.items()}
             for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

        terms = []
        for row in rows:
            terms.append(row['term'])

        return terms


def parse_dictionary_unigram(terms):
    return parse_dictionary(terms, 1)

def parse_dictionary_bigram(terms):
    return parse_dictionary(terms, 2)

def parse_dictionary(terms, ngram):
    stopWords = set(stopwords.words('english'))
    parsed_terms = []
    for term in terms:
        words = term.split()
        non_stop_words = [word for word in words if word not in stopWords]
        for i in range(0, len(non_stop_words ) - ngram + 1):
            term_to_add = non_stop_words[i:i + ngram]
            if term_to_add not in parsed_terms:
                parsed_terms.append(' '.join(non_stop_words[i:i + ngram]))

    return parsed_terms

class TerminologicalDictionary:
    def __init__(self):
        terms = load_dictionary()
        self.unigram_dic = parse_dictionary_unigram(terms)
        self.bigram_dic = parse_dictionary_bigram(terms)


    def __contains__(self, item):
        if item in self.unigram_dic:
            return True
        if item in self.bigram_dic:
            return True
        return False
