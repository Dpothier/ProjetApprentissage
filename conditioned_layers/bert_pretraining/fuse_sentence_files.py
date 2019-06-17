import click
from tqdm import tqdm

# Takes a folder containing a document in each file and output a file where every sentence is on its own line

@click.command()
@click.option('-f', '--first', default="bert_pretraining/wikipedia_sentences.csv")
@click.option('-s', '--second', default="bert_pretraining/books_sentences.txt")
@click.option('-o', '--output', default="bert_pretraining/all_sentences.txt")
def main(first, second, output):
    with open(output, encoding="utf-8", mode="w+") as o:
        with open(first, encoding="utf-8", mode="r") as f:
            lines = f.readlines()
            for line in tqdm(lines):
                o.write(line + "\n")
        with open(second, encoding="utf-8", mode="r") as s:
            lines = s.readlines()
            for line in tqdm(lines):
                o.write(line + "\n")


if __name__ == "__main__":
    main()