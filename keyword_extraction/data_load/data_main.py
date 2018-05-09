from nltk.tokenize import TreebankWordTokenizer
import eval
import os
import nltk
from operator import itemgetter
from data_load.tokenization import tokenize
from data_load.load import load_data
from data_load.save import save_annotation_file
from data_load.corpus import load_corpus, NERCorpusParser, word_tokenize

def extract_annotations(annotation_file):
    process_ann = []
    material_ann = []
    tasks_ann = []

    with open(annotation_file, encoding="utf8") as f:
        for l in f:
            annotation = l.split()
            if "T" in annotation[0]:
                coordinates = (int(annotation[2]), int(annotation[3]))
                if annotation[1] == "Process":
                    process_ann.append(coordinates)
                if annotation[1] == "Material":
                    material_ann.append(coordinates)
                if annotation[1] == "Task":
                    tasks_ann.append(coordinates)

    return process_ann, material_ann, tasks_ann

def single_span_matches_whole_annotation(annotations, span):
    return len([annotation for annotation in annotations if annotation[0] == span[0] and annotation[1] == span[1]]) != 0

def span_begin_matches_annotation_begin(annotations, span):
    return len([annotation for annotation in annotations if annotation[0] == span[0] and annotation[1] != span[1]]) != 0

def extract_BILOU(spans, annotations):
    bilou_tags = []
    for i in range(len(spans)):
        bilou_tags.append("O")

    for annotation in annotations:
        start_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[0] <= span[1]]
        end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1] <= span[1]]
        if len(end_spans) == 0:
            end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1]-1 <= span[1]]
        start_span = start_spans[0]
        end_span = end_spans[0]
        if start_span == end_span:
            bilou_tags[start_span] = "U"
        else:
            bilou_tags[start_span] = "B"
            bilou_tags[end_span] = "L"
            for i in range(start_span+1, end_span):
                bilou_tags[i]="I"

    longest_I_streak = 0
    current_I_streak = 0
    for tag in bilou_tags:
        if tag == "I":
            current_I_streak += 1
        else:
            if current_I_streak > longest_I_streak:
                longest_I_streak = current_I_streak
            current_I_streak = 0

    if bilou_tags[-1] == "I":
        print("Abnormally long I streak")
        print(bilou_tags)

    return bilou_tags

def bilou_to_annotation_file(filename, tokens, spans, bilou_list):
    with open(filename, encoding='utf8', mode="w") as f:
        entity_index = 1
        for entity_type, bilou in bilou_list.items():
            current_entity = []
            for i in range(len(bilou)):
                if bilou[i] == "U":
                    f.write("T{}\t{} {} {}\t{}\n".format(entity_index, entity_type, spans[i][0], spans[i][1], tokens[i]))
                    entity_index +=1

                if bilou[i] == "B":
                    current_entity.append(i)

                if bilou[i] == "I":
                    current_entity.append(i)

                if bilou[i] == "L":
                    current_entity.append(i)
                    start_index = spans[current_entity[0]][0]
                    last_index = spans[current_entity[-1]][1]
                    words = ' '.join([tokens[x] for x in current_entity])
                    f.write("T{}\t{} {} {}\t{}\n".format(entity_index, entity_type, start_index, last_index, words))
                    current_entity = []
                    entity_index += 1


def print_alignments(tokens, spans, bilou_list):
    line = ""
    just = max(len(x) for x in tokens)
    for i in range(len(tokens)):
        line += str(tokens[i]).ljust(just)

    print(line)
    line = ""
    for i in range(len(spans)):
        line += str(spans[i][0]).ljust(just)
    print(line)
    for bilou in bilou_list:
        line = ""
        for i in range(len(bilou)):
            line += str(bilou[i].ljust(just))
        print(line)

folder_gold = '../data/train2'
folder_pred = '../data/train_pred'


texts, extras = load_data('../data/train2', use_int_tags=True, tag_scheme='IO')

number_of_overlaps = 0
number_of_punctuation = 0
for i in range(len(texts)):
    id = texts[i]['id']
    file_name, tokens, spans = extras[id]
    process_list = texts[i]['process_tags']
    material_list = texts[i]['material_tags']
    task_list = texts[i]['task_tags']
    save_annotation_file('{}/{}.ann'.format(folder_pred, file_name), tokens, spans,
                         {"Process": process_list, "Material": material_list, "Task": task_list}, tag_scheme='IO')
    mins = [span[0] for index, span in enumerate(spans)]
    overlaps = [index for index, span in enumerate(spans) if span[1] in mins]
    for overlap in overlaps:
        if tokens[overlap + 1] == "." or tokens[overlap + 1] == ",":
            number_of_overlaps += 1
            print("{}: Overlaping tokens: {}, {}".format(number_of_overlaps, tokens[overlap], tokens[overlap + 1]))

    for token in tokens:
        if token == "," or token == ".":
            number_of_punctuation += 1


parser = NERCorpusParser(word_tokenize)
train_documents = load_corpus('../data/train2')
train_corpus = [e for d in train_documents for e in parser.parse(d)]
dev_documents = load_corpus('../data/dev')
dev_corpus = [e for d in dev_documents for e in parser.parse(d)]
test_documents = load_corpus('../data/test')
test_corpus = [e for d in test_documents for e in parser.parse(d)]


print(eval.calculateMeasures('../data/train2', '../data/train_pred', 'rel'))
print("Number of overlap:{}".format(number_of_overlaps))
print("Number of punctuation: {}".format(number_of_punctuation))
