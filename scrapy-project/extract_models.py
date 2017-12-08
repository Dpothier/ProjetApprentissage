import csv
from collections import Counter

with open('bdrv.csv', encoding="utf8") as f:
    a = [{k: str(v) for k, v in row.items()}
         for row in csv.DictReader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)]

b = []
for row in a:
    year = int(row["year"])
    if year > 2012:
        b.append("{} {} {}".format(row["make"], row["model"], row["year"]))


c = Counter(b)
d = c.most_common(40)
e = [x[0] for x in d]
print(e)

with open('models.txt', encoding="utf8", mode="w") as f:
    for model in e:
        f.write("{} \n".format(model))

