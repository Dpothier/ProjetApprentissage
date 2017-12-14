from vectorization.VectorizerBuilder import Use_count
from vectorization.VectorizerBuilder import Use_tf
from vectorization.VectorizerBuilder import Use_tfidf

def metaset_std(dictionary):
    return [("count_postprocessing_single", count_postprocessing_single(dictionary)),
            ("count_postprocessing_stemming", count_postprocessing_stemming(dictionary)),
            ("count_postprocessing_lemma", count_postprocessing_lemma(dictionary)),
            ("tf_postprocessing_single", tf_postprocessing_single(dictionary)),
            ("tf_postprocessing_stemming", tf_postprocessing_stemming(dictionary)),
            ("tf_postprocessing_lemma", tf_postprocessing_lemma(dictionary)),
            ("tfidf_postprocessing_single", tfidf_postprocessing_single(dictionary)),
            ("tfidf_postprocessing_stemming", tfidf_postprocessing_stemming(dictionary)),
            ("tfidf_postprocessing_lemma", tfidf_postprocessing_lemma(dictionary)),]


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

def count_postprocessing_single(dictionary):
    return [("by count", Use_count(1).as_vectorizer()),
            ("by count, stemming", Use_count(1).and_stemming().as_vectorizer()),
            ("by count, pos", Use_count(1).and_pos_filter().as_vectorizer()),
            ("by count, stop", Use_count(1).and_stop_words().as_vectorizer()),
            ("by count, lemma", Use_count(1).and_lemmatization().as_vectorizer()),
            ("by count, amp 4", Use_count(1).and_dict_amp(dictionary, 4).as_vectorizer()),]

def count_postprocessing_stemming(dictionary):
    return [("by count, stemming, pos", Use_count(1).and_stemming().and_pos_filter().as_vectorizer()),
            ("by count, stemming, stop", Use_count(1).and_stemming().and_stop_words().as_vectorizer()),
            ("by count, stemming, amp 4", Use_count(1).and_stemming().and_dict_amp(dictionary, 4).as_vectorizer()),
            ("by count, stemming, all", Use_count(1).and_stemming().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 4).as_vectorizer()),]

def count_postprocessing_lemma(dictionary):
    return [("by count, lemma, pos", Use_count(1).and_lemmatization().and_pos_filter().as_vectorizer()),
            ("by count, lemma, stop", Use_count(1).and_lemmatization().and_stop_words().as_vectorizer()),
            ("by count, lemma, amp 4", Use_count(1).and_lemmatization().and_dict_amp(dictionary, 4).as_vectorizer()),
            ("by count, lemma, all", Use_count(1).and_lemmatization().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 4).as_vectorizer()),]

def tf_postprocessing_single(dictionary):
    return [("by tf", Use_tf(1).as_vectorizer()),
            ("by tf, stemming", Use_tf(1).and_stemming().as_vectorizer()),
            ("by tf, pos", Use_tf(1).and_pos_filter().as_vectorizer()),
            ("by tf, stop", Use_tf(1).and_stop_words().as_vectorizer()),
            ("by tf, lemma", Use_tf(1).and_lemmatization().as_vectorizer()),
            ("by tf, amp 40", Use_tf(1).and_dict_amp(dictionary, 40).as_vectorizer()),]

def tf_postprocessing_stemming(dictionary):
    return [("by tf, stemming, pos", Use_tf(1).and_stemming().and_pos_filter().as_vectorizer()),
            ("by tf, stemming, stop", Use_tf(1).and_stemming().and_stop_words().as_vectorizer()),
            ("by tf, stemming, amp 40", Use_tf(1).and_stemming().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("by tf, stemming, all", Use_tf(1).and_stemming().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def tf_postprocessing_lemma(dictionary):
    return [("by tf, lemma, pos", Use_tf(1).and_lemmatization().and_pos_filter().as_vectorizer()),
            ("by tf, lemma, stop", Use_tf(1).and_lemmatization().and_stop_words().as_vectorizer()),
            ("by tf, lemma, amp 40", Use_tf(1).and_lemmatization().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("by tf, lemma, all", Use_tf(1).and_lemmatization().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def tfidf_postprocessing_single(dictionary):
    return [("by tfidf", Use_tfidf(1).as_vectorizer()),
            ("by tfidf, stemming", Use_tfidf(1).and_stemming().as_vectorizer()),
            ("by tfidf, pos", Use_tfidf(1).and_pos_filter().as_vectorizer()),
            ("by tfidf, stop", Use_tfidf(1).and_stop_words().as_vectorizer()),
            ("by tfidf, lemma", Use_tfidf(1).and_lemmatization().as_vectorizer()),
            ("by tfidf, amp 40", Use_tfidf(1).and_dict_amp(dictionary, 40).as_vectorizer()),]

def tfidf_postprocessing_stemming(dictionary):
    return [("by tfidf, stemming, pos", Use_tfidf(1).and_stemming().and_pos_filter().as_vectorizer()),
            ("by tfidf, stemming, stop", Use_tfidf(1).and_stemming().and_stop_words().as_vectorizer()),
            ("by tfidf, stemming, amp 40", Use_tfidf(1).and_stemming().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("by tfidf, stemming, all", Use_tfidf(1).and_stemming().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def tfidf_postprocessing_lemma(dictionary):
    return [("by tfidf, lemma, pos", Use_tfidf(1).and_lemmatization().and_pos_filter().as_vectorizer()),
            ("by tfidf, lemma, stop", Use_tfidf(1).and_lemmatization().and_stop_words().as_vectorizer()),
            ("by tfidf, lemma, amp 40", Use_tfidf(1).and_lemmatization().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("by tfidf, lemma, all", Use_tfidf(1).and_lemmatization().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def dictionary_amplification(dictionary):
    return [("by count, amp 2", Use_count(1).and_dict_amp(dictionary, 2).as_vectorizer()),
            ("by count, amp 3", Use_count(1).and_dict_amp(dictionary, 3).as_vectorizer()),
            ("by count, amp 4", Use_count(1).and_dict_amp(dictionary, 4).as_vectorizer()),
            ("by count, amp 10", Use_count(1).and_dict_amp(dictionary, 10).as_vectorizer()),
            ("by count, amp 15", Use_count(1).and_dict_amp(dictionary, 15).as_vectorizer()),
            ("by tf, amp 2", Use_tf(1).and_dict_amp(dictionary, 2).as_vectorizer()),
            ("by tf, amp 3", Use_tf(1).and_dict_amp(dictionary, 3).as_vectorizer()),
            ("by tf, amp 4", Use_tf(1).and_dict_amp(dictionary, 4).as_vectorizer()),
            ("by tf, amp 10", Use_tf(1).and_dict_amp(dictionary, 10).as_vectorizer()),
            ("by tf, amp 15", Use_tf(1).and_dict_amp(dictionary, 15).as_vectorizer()),
            ("by tf, amp 25", Use_tf(1).and_dict_amp(dictionary, 25).as_vectorizer()),
            ("by tf, amp 40", Use_tf(1).and_dict_amp(dictionary, 40).as_vectorizer()),
            ("by tfidf, amp 2", Use_tfidf(1).and_dict_amp(dictionary, 2).as_vectorizer()),
            ("by tfidf, amp 3", Use_tfidf(1).and_dict_amp(dictionary, 3).as_vectorizer()),
            ("by tfidf, amp 4", Use_tfidf(1).and_dict_amp(dictionary, 4).as_vectorizer()),
            ("by tfidf, amp 10", Use_tfidf(1).and_dict_amp(dictionary, 10).as_vectorizer()),
            ("by tfidf, amp 15", Use_tfidf(1).and_dict_amp(dictionary, 15).as_vectorizer()),
            ("by tfidf, amp 25", Use_tfidf(1).and_dict_amp(dictionary, 25).as_vectorizer()),
            ("by tfidf, amp 40", Use_tfidf(1).and_dict_amp(dictionary, 40).as_vectorizer()), ]

def closed_vocab(dictionary):
    return [('by count, closed vocab', Use_count(1).and_closed_vocab(dictionary).as_vectorizer()),
            ('by tf, closed vocab', Use_tf(1).and_closed_vocab(dictionary).as_vectorizer()),
            ('by tfidf, closed vocab', Use_tfidf(1).and_closed_vocab(dictionary).as_vectorizer())]