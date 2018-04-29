import nltk
from nltk.tokenize import TreebankWordTokenizer


def tokenize(text):
    sentence_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    tokenizer = TreebankWordTokenizer()
    sentences_list = sentence_tokenizer.tokenize(text)
    sentences_span = sentence_tokenizer.span_tokenize(text)
    total_tokens = []
    total_spans = []
    for i in range(len(sentences_list)):
        word_tokens = tokenizer.tokenize(sentences_list[i])
        word_spans = tokenizer.span_tokenize(sentences_list[i])
        word_spans = [(span[0] + sentences_span[i][0], span[1] + sentences_span[i][0]) for span in word_spans]
        total_tokens += word_tokens
        total_spans += word_spans

    return total_tokens, total_spans