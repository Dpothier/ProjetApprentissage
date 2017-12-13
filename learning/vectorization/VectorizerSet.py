from vectorization.VectorizerBuilder import Use_count
from vectorization.VectorizerBuilder import Use_tf
from vectorization.VectorizerBuilder import Use_tfidf


def ngram_count_tf_idf(dictionary):
    return [("by count", Use_count(1).as_vectorizer()),
            ("by count, bigram", Use_count(2).as_vectorizer()),
            ("by count, trigram", Use_count(3).as_vectorizer()),
            ("by tf", Use_tf(1).as_vectorizer()),
            ("by tf, bigram", Use_tf(2).as_vectorizer()),
            ("by tf, trigram", Use_tf(3).as_vectorizer()),
            ("by tfidf", Use_tfidf(1).as_vectorizer()),
            ("by tfidf, bigram", Use_tfidf(2).as_vectorizer()),
            ("by tfidf, trigram", Use_tfidf(3).as_vectorizer()),
            ]

def count_postprocessing(dictionary):
    return [("by count", Use_count(1).as_vectorizer()),
            ("by count, stemming", Use_count(1).and_stemming().as_vectorizer()),
            ("by count, pos", Use_count(1).and_pos_filter().as_vectorizer()),
            ("by count, stemming, pos", Use_count(1).and_pos_filter().and_stemming().as_vectorizer()),
            ("by count, lemma", Use_count(1).and_lemmatization().as_vectorizer()),
            ("by count, lemma, pos", Use_count(1).and_lemmatization().and_pos_filter().as_vectorizer()),
            ("by count, amp 3", Use_count(1).and_dictionnary_amplification(dictionary, 3).as_vectorizer()),
            ("by count, amp, lemma", Use_count(1).and_dictionnary_amplification(dictionary, 3).and_lemmatization().as_vectorizer()),
            ("By count, amp, lemma, pos", Use_count(1).and_dictionnary_amplification(dictionary, 3).and_lemmatization().as_vectorizer())]

def dictionary_amplification(dictionary):
    return [("by count, amp 2", Use_count(1).and_dictionnary_amplification(dictionary, 2).as_vectorizer()),
            ("by count, amp 3", Use_count(1).and_dictionnary_amplification(dictionary, 3).as_vectorizer()),
            ("by count, amp 4", Use_count(1).and_dictionnary_amplification(dictionary, 4).as_vectorizer()),
            ("by count, amp 10", Use_count(1).and_dictionnary_amplification(dictionary, 10).as_vectorizer()),
            ("by count, amp 15", Use_count(1).and_dictionnary_amplification(dictionary, 15).as_vectorizer()),
            ("by tf, amp 2", Use_tf(1).and_dictionnary_amplification(dictionary, 2).as_vectorizer()),
            ("by tf, amp 3", Use_tf(1).and_dictionnary_amplification(dictionary, 3).as_vectorizer()),
            ("by tf, amp 4", Use_tf(1).and_dictionnary_amplification(dictionary, 4).as_vectorizer()),
            ("by tf, amp 10", Use_tf(1).and_dictionnary_amplification(dictionary, 10).as_vectorizer()),
            ("by tf, amp 15", Use_tf(1).and_dictionnary_amplification(dictionary, 15).as_vectorizer()),
            ("by tf, amp 25", Use_tf(1).and_dictionnary_amplification(dictionary, 25).as_vectorizer()),
            ("by tf, amp 40", Use_tf(1).and_dictionnary_amplification(dictionary, 40).as_vectorizer()),
            ("by tfidf, amp 2", Use_tfidf(1).and_dictionnary_amplification(dictionary, 2).as_vectorizer()),
            ("by tfidf, amp 3", Use_tfidf(1).and_dictionnary_amplification(dictionary, 3).as_vectorizer()),
            ("by tfidf, amp 4", Use_tfidf(1).and_dictionnary_amplification(dictionary, 4).as_vectorizer()),
            ("by tfidf, amp 10", Use_tfidf(1).and_dictionnary_amplification(dictionary, 10).as_vectorizer()),
            ("by tfidf, amp 15", Use_tfidf(1).and_dictionnary_amplification(dictionary, 15).as_vectorizer()),
            ("by tfidf, amp 25", Use_tfidf(1).and_dictionnary_amplification(dictionary, 25).as_vectorizer()),
            ("by tfidf, amp 40", Use_tfidf(1).and_dictionnary_amplification(dictionary, 40).as_vectorizer()),]

def lemmatization(dictionary):
    return [("by count, lemma", Use_count(1).and_lemmatization().as_vectorizer()),
            ("by count, lemma, stop", Use_count(1).and_lemmatization().and_stop_words().as_vectorizer()),
            ("By count, lemma, pos", Use_count(1).and_pos_filter().and_lemmatization().as_vectorizer()),
            ("By count, lemma, pos, stop", Use_count(1).and_pos_filter().and_stop_words().and_lemmatization().as_vectorizer())
            ]

def closed_vocab(dictionary):
    return [('by count, closed vocab', Use_count(1).and_closed_vocab(dictionary).as_vectorizer()),
            ('by tf, closed vocab', Use_tf(1).and_closed_vocab(dictionary).as_vectorizer()),
            ('by tfidf, closed vocab', Use_tfidf(1).and_closed_vocab(dictionary).as_vectorizer())]