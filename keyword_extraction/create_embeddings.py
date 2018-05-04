import os
from data_load.load import load_data
from glove import Corpus, Glove
from data_load.tokenization import tokenize_no_span




loaded_train, train_indices = load_data('./data/train2', use_int_tags=True)
folder = './data/wikipedia_filtered'
flist = os.listdir(folder)
texts = []
for f in flist:
    with open(os.path.join(folder, f), "r", encoding="utf8") as f_text:
        for line in f_text:
            texts.append(tokenize_no_span(line))



texts.extend([example['texts'] for example in loaded_train])

corpus = Corpus()

corpus.fit(texts, window=10)

glove = Glove(no_components=200, learning_rate=0.05)

glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)

glove.add_dictionary(corpus.dictionary)

glove.save('./embeddings/wiki-semeval200.glove')

model = Glove.load('./embeddings/wiki-semeval200.glove')


