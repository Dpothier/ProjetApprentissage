import nltk
import csv
import click
from tqdm import tqdm

# Takes two partial corpora and fuse them together.
# Also tokenize every sentence and format the sequences so they can be used to train BERT

@click.command()
@click.option('-f', '--first', default="bert_pretraining/wikipedia_corpus.tsv")
@click.option('-s', '--second', default="bert_pretraining/books_corpus.tsv")
@click.option('-o', '--output', default="bert_pretraining/.txt")
def main(first, second, output):
    with open(output, encoding="utf-8", mode="w+") as o:
        with open(first, encoding="utf-8", mode="r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                formated_line = format_line(line)
                o.write(formated_line)
        with open(second, encoding="utf-8", mode="r") as s:
            lines = s.readlines()
            for line in tqdm(lines):
                formated_line = format_line(line)
                o.write(formated_line)

def format_line(line):
    first_sentence, second_sentence = line[0], line[1]
    first_sent_tokens = nltk.word_tokenize(first_sentence)
    second_sent_tokens = nltk.word_tokenize(second_sentence)

    return " ".join(first_sent_tokens) + "\t" + " ".join(second_sent_tokens) + "\n"


if __name__ == "__main__":
    main()