import click
import sys
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import os
import csv

# Takes a folder containing a document in each file and output a file where every sentence is on its own line

@click.command()
@click.option('-i', '--input', default="bert_pretraining/books/")
@click.option('-o', '--output', default="bert_pretraining/books_sentences.txt")
@click.option('-c', '--corpus', default="bert_pretraining/wikipedia_corpus.tsv")
def main(input, output, corpus):
    with open(output, encoding="utf-8", mode="w+") as o:
        with open(corpus, encoding="utf-8", mode="w+") as c:
            writer = csv.writer(c, delimiter="\t")
            files = os.listdir(input)
            for file in tqdm(files):
                with open(input + file, encoding="utf-8", mode="r") as f:
                    lines = f.readlines()
                    whole_text = " ".join(lines)
                    sentences = sent_tokenize(whole_text)
                    for index in range(len(sentences)):
                    # if there is a next sentence in the document
                        if index < len(sentences) - 1:
                            writer.writerow([sentences[index], sentences[index + 1]])
                        o.write(sentences[index] + "\n")


if __name__ == "__main__":
    nltk.download('punkt')
    main()