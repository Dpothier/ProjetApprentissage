from keyword_extraction.data_load.corpus import load_corpus, NERCorpusParser, word_tokenize


def main():
    parser = NERCorpusParser(word_tokenize)
    train_documents = load_corpus('./data/scienceie/scienceie2017_train/train2')
    train_corpus = [e for d in train_documents for e in parser.parse(d)]
    dev_documents = load_corpus('./data/scienceie/scienceie2017_dev/dev')
    dev_corpus = [e for d in dev_documents for e in parser.parse(d)]
    test_documents = load_corpus('./data/scienceie/semeval_articles_test')
    test_corpus = [e for d in test_documents for e in parser.parse(d)]


if __name__ == '__main__':
    main()
