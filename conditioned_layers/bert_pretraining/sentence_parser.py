import click
import sys
import csv
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Takes a csv file containing one document per line and output:
# a txt file with one sentence per line for use with WordPiece,
# a tsv file with sentences paired with the following one for each documents for training BERT
@click.command()
@click.option('-f', '--file', default="bert_pretraining/documents_utf8_filtered_20pageviews.csv")
@click.option('-o', '--output', default="bert_pretraining/wikipedia_sentences.txt")
@click.option('-c', '--corpus', default="bert_pretraining/wikipedia_corpus.tsv")
def main(file, output, corpus):
    with open(file, encoding="utf-8", mode="r") as f:
        with open(output, encoding="utf-8", mode="w+") as o:
            with open(corpus, encoding="utf-8", mode="w+") as c:
                writer = csv.writer(c, delimiter="\t")
                reader = csv.reader(f)
                for line in tqdm(reader):
                    sentences = sent_tokenize(line[1])
                    for index in range(len(sentences)):
                        # if there is a next sentence in the document
                        if index < len(sentences) - 1:
                            writer.writerow([sentences[index], sentences[index+1]])
                        o.write(sentences[index] + "\n")



if __name__ == "__main__":
    nltk.download('punkt')
    csv.field_size_limit(sys.maxsize)
    main()