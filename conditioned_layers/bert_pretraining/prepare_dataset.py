import click
import sys
import csv
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os
import random
import math

# Input a csv file containing one document per line and output:
# Input a folder containing one book per file
# Output a file with all sentences of all documents, one sentence per line for use with wordpiece
# Output a corpus, split in train and dev set, with two consecutive sentences seperated by a tab on each line, to pretrain a BERT model

@click.command()
@click.option('-w', '--wikipedia', default="bert_pretraining/documents_utf8_filtered_20pageviews.csv")
@click.option('-b', '--books', default="bert_pretraining/books/")
@click.option('-s', '--sentences', default="bert_pretraining/wordpiece_sentences.txt")
@click.option('-c', '--corpus', default="bert_pretraining/corpus/")
def main(wikipedia, books, sentences, corpus):
    documents = get_documents(wikipedia, books)
    produce_sentence_list(documents, sentences)
    train_documents, dev_documents = split_train_dev(documents)
    produce_corpus(train_documents, corpus + "train.txt")
    produce_corpus(dev_documents, corpus + "dev.txt")

def get_documents(wikipedia_file, book_folder):
    documents = []
    with open(wikipedia_file, encoding="utf-8", mode="r") as w:
        reader = csv.reader(w)
        for line in tqdm(reader):
            documents.append(line[1])

    files = os.listdir(book_folder)
    for file in tqdm(files):
        with open(book_folder + file, encoding="utf-8", mode="r") as f:
            lines = f.readlines()
            whole_text = " ".join(lines)
            documents.append(whole_text)

    return documents

def produce_sentence_list(documents, output_file):
    with open(output_file, encoding="utf-8", mode="w+") as o:
        for document in tqdm(documents):
            sentences = nltk.sent_tokenize(document)
            for sentence in sentences:
                o.write(sentence + "\n")

def produce_corpus(documents, output_file):
    with open(output_file, encoding="utf-8", mode="w+") as c:
        for document in tqdm(documents):
            sentences = sent_tokenize(document)
            tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
            for index in tqdm(range(len(tokenized_sentences))):
                # if there is a next sentence in the document
                if index < len(sentences) - 1:
                    line = " ".join(tokenized_sentences[index]) + "\t" + " ".join(tokenized_sentences[index + 1])
                    c.write(line + "\n")


def split_train_dev(documents):
    random.shuffle(documents)
    number_of_documents = len(documents)
    split_point = math.ceil(number_of_documents * 0.8)
    return documents[split_point:], documents[:split_point]

if __name__ == "__main__":
    nltk.download('punkt')
    csv.field_size_limit(sys.maxsize)
    main()