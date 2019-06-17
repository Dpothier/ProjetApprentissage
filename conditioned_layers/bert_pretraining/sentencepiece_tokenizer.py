import click
from tqdm import tqdm
import sentencepiece as spm

# Takes a folder containing a document in each file and output a file where every sentence is on its own line

@click.command()
@click.option('-i', '--input', default="bert_pretraining/all_sentences.txt")
@click.option('-o', '--output', default="bert_pretraining/bert_embeddings")
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
spm.SentencePieceTrainer_Train()
if __name__ == "__main__":
    main()