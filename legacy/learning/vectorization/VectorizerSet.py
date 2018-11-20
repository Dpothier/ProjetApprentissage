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
    return [("count", Use_count(1).as_vectorizer()),
            ("count_bigram", Use_count(2).as_vectorizer()),
            ("count_trigram", Use_count(3).as_vectorizer()),
            ("tf", Use_tf(1).as_vectorizer()),
            ("tf_bigram", Use_tf(2).as_vectorizer()),
            ("tf_trigram", Use_tf(3).as_vectorizer()),
            ("tfidf", Use_tfidf(1).as_vectorizer()),
            ("tfidf_bigram", Use_tfidf(2).as_vectorizer()),
            ("tfidf_trigram", Use_tfidf(3).as_vectorizer()),
            ]

def count_postprocessing_single(dictionary):
    return [("count", Use_count(1).as_vectorizer()),
            ("count_stemming", Use_count(1).and_stemming().as_vectorizer()),
            ("count_pos", Use_count(1).and_pos_filter().as_vectorizer()),
            ("count_stop", Use_count(1).and_stop_words().as_vectorizer()),
            ("count_lemma", Use_count(1).and_lemmatization().as_vectorizer()),
            #("count_amp 4", Use_count(1).and_dict_amp(dictionary, 4).as_vectorizer()),
     ]

def count_postprocessing_stemming(dictionary):
    return [("count_stemming_pos", Use_count(1).and_stemming().and_pos_filter().as_vectorizer()),
            ("by count_stemming_stop", Use_count(1).and_stemming().and_stop_words().as_vectorizer()),
            #("by count_stemming_amp 4", Use_count(1).and_stemming().and_dict_amp(dictionary, 4).as_vectorizer()),
            ("by count_stemming_all", Use_count(1).and_stemming().and_stop_words().and_pos_filter().as_vectorizer()),
            ]

def count_postprocessing_lemma(dictionary):
    return [("by count_lemma_pos", Use_count(1).and_lemmatization().and_pos_filter().as_vectorizer()),
            ("by count_lemma_stop", Use_count(1).and_lemmatization().and_stop_words().as_vectorizer()),
            #("by count_lemma_amp 4", Use_count(1).and_lemmatization().and_dict_amp(dictionary, 4).as_vectorizer()),
            ("by count_lemma_all", Use_count(1).and_lemmatization().and_stop_words().and_pos_filter().as_vectorizer()),]

def tf_postprocessing_single(dictionary):
    return [("tf", Use_tf(1).as_vectorizer()),
            ("tf_stemming", Use_tf(1).and_stemming().as_vectorizer()),
            ("tf_pos", Use_tf(1).and_pos_filter().as_vectorizer()),
            ("tf_stop", Use_tf(1).and_stop_words().as_vectorizer()),
            ("tf_lemma", Use_tf(1).and_lemmatization().as_vectorizer()),
            ("tf_amp40", Use_tf(1).and_dict_amp(dictionary, 40).as_vectorizer()),]

def tf_postprocessing_stemming(dictionary):
    return [("tf_stemming_pos", Use_tf(1).and_stemming().and_pos_filter().as_vectorizer()),
            ("tf_stemming_stop", Use_tf(1).and_stemming().and_stop_words().as_vectorizer()),
            ("tf_stemming_amp 40", Use_tf(1).and_stemming().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("tf_stemming_all", Use_tf(1).and_stemming().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def tf_postprocessing_lemma(dictionary):
    return [("tf_lemma_pos", Use_tf(1).and_lemmatization().and_pos_filter().as_vectorizer()),
            ("tf_lemma_stop", Use_tf(1).and_lemmatization().and_stop_words().as_vectorizer()),
            ("tf_lemma_amp40", Use_tf(1).and_lemmatization().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("tf_lemma_all", Use_tf(1).and_lemmatization().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def tfidf_postprocessing_single(dictionary):
    return [("tfidf", Use_tfidf(1).as_vectorizer()),
            ("tfidf_stemming", Use_tfidf(1).and_stemming().as_vectorizer()),
            ("tfidf_pos", Use_tfidf(1).and_pos_filter().as_vectorizer()),
            ("tfidf_stop", Use_tfidf(1).and_stop_words().as_vectorizer()),
            ("tfidf_lemma", Use_tfidf(1).and_lemmatization().as_vectorizer()),
            ("tfidf_amp40", Use_tfidf(1).and_dict_amp(dictionary, 40).as_vectorizer()),]

def tfidf_postprocessing_stemming(dictionary):
    return [("tfidf_stemming_pos", Use_tfidf(1).and_stemming().and_pos_filter().as_vectorizer()),
            ("tfidf_stemming_stop", Use_tfidf(1).and_stemming().and_stop_words().as_vectorizer()),
            ("tfidf_stemming_amp40", Use_tfidf(1).and_stemming().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("tfidf_stemming_all", Use_tfidf(1).and_stemming().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def tfidf_postprocessing_lemma(dictionary):
    return [("tfidf_lemma_pos", Use_tfidf(1).and_lemmatization().and_pos_filter().as_vectorizer()),
            ("tfidf_lemma_stop", Use_tfidf(1).and_lemmatization().and_stop_words().as_vectorizer()),
            ("tfidf_lemma_amp40", Use_tfidf(1).and_lemmatization().and_dict_amp(dictionary, 40).as_vectorizer()),
            ("tfidf_lemma_all", Use_tfidf(1).and_lemmatization().and_stop_words().and_pos_filter().
                and_dict_amp(dictionary, 40).as_vectorizer()),]

def dictionary_amplification(dictionary):
    return [("count_amp2", Use_count(1).and_dict_amp(dictionary, 2).as_vectorizer()),
            ("count_amp3", Use_count(1).and_dict_amp(dictionary, 3).as_vectorizer()),
            ("count_amp4", Use_count(1).and_dict_amp(dictionary, 4).as_vectorizer()),
            ("count_amp10", Use_count(1).and_dict_amp(dictionary, 10).as_vectorizer()),
            ("count_amp15", Use_count(1).and_dict_amp(dictionary, 15).as_vectorizer()),
            ("tf_amp2", Use_tf(1).and_dict_amp(dictionary, 2).as_vectorizer()),
            ("tf_amp3", Use_tf(1).and_dict_amp(dictionary, 3).as_vectorizer()),
            ("tf_amp4", Use_tf(1).and_dict_amp(dictionary, 4).as_vectorizer()),
            ("tf_amp10", Use_tf(1).and_dict_amp(dictionary, 10).as_vectorizer()),
            ("tf_amp15", Use_tf(1).and_dict_amp(dictionary, 15).as_vectorizer()),
            ("tf_amp25", Use_tf(1).and_dict_amp(dictionary, 25).as_vectorizer()),
            ("tf_amp40", Use_tf(1).and_dict_amp(dictionary, 40).as_vectorizer()),
            ("tfidf_amp2", Use_tfidf(1).and_dict_amp(dictionary, 2).as_vectorizer()),
            ("tfidf_amp3", Use_tfidf(1).and_dict_amp(dictionary, 3).as_vectorizer()),
            ("tfidf_amp4", Use_tfidf(1).and_dict_amp(dictionary, 4).as_vectorizer()),
            ("tfidf_amp10", Use_tfidf(1).and_dict_amp(dictionary, 10).as_vectorizer()),
            ("tfidf_amp15", Use_tfidf(1).and_dict_amp(dictionary, 15).as_vectorizer()),
            ("tfidf_amp25", Use_tfidf(1).and_dict_amp(dictionary, 25).as_vectorizer()),
            ("tfidf_amp40", Use_tfidf(1).and_dict_amp(dictionary, 40).as_vectorizer()), ]

def closed_vocab(dictionary):
    return [('count_closed_vocab', Use_count(1).and_closed_vocab(dictionary).as_vectorizer()),
            ('tf_closed_vocab', Use_tf(1).and_closed_vocab(dictionary).as_vectorizer()),
            ('tfidf_closed_vocab', Use_tfidf(1).and_closed_vocab(dictionary).as_vectorizer())]