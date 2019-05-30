import pandas as pd


train = pd.read_csv("raw/train.tsv", sep="\t")

train = train.drop([train.columns[0], train.columns[2]], axis=1)

train.to_csv("clean/train.tsv", sep="\t", index=False)

val = pd.read_csv("raw/dev.tsv", sep="\t")

val = val.drop([val.columns[0], val.columns[2]], axis= 1)

val.to_csv("clean/dev.tsv", sep="\t", index=False)

test = pd.read_csv("raw/test.tsv", sep="\t")

val = val.drop([0], axis=0)

test.to_csv("clean/test.tsv", sep="\t", index=False)