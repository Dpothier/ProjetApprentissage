from nltk.tokenize import TreebankWordTokenizer
import eval
import os
import nltk
from operator import itemgetter


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



def tokenize(text, total_of_sentences):
    sentence_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    tokenizer = TreebankWordTokenizer()
    sentences_list = sentence_tokenizer.tokenize(text)
    sentences_span = sentence_tokenizer.span_tokenize(text)
    total_tokens = []
    total_spans = []
    for i in range(len(sentences_list)):
        word_tokens = tokenizer.tokenize(sentences_list[i])
        word_spans = tokenizer.span_tokenize(sentences_list[i])
        word_spans = [(span[0] + sentences_span[i][0], span[1] + sentences_span[i][0]) for span in word_spans]
        total_tokens += word_tokens
        total_spans += word_spans


    return total_tokens, total_spans, total_of_sentences + len(sentences_span)

folder_gold = '../data/train2'
folder_pred = '../data/train_pred'
flist_gold = os.listdir(folder_gold)
number_of_overlaps = 0
number_of_punctuation = 0
total_of_sentences = 0
for f in flist_gold:
    if not str(f).endswith(".txt"):
        continue
    f_ann = str(f)[0:-4] + ".ann"
    f_text = open(os.path.join(folder_gold, f), "r", encoding="utf8")

    for l in f_text:
        text = l

    tokens_list, tokens_spans, total_of_sentences = tokenize(text, total_of_sentences)

    mins = [span[0] for index, span in enumerate(tokens_spans)]
    overlaps = [index for index, span in enumerate(tokens_spans) if span[1] in mins]
    for overlap in overlaps:
        if tokens_list[overlap + 1] == "." or tokens_list[overlap + 1] == ",":
            number_of_overlaps += 1
            print("{}: Overlaping tokens: {}, {}".format(number_of_overlaps, tokens_list[overlap], tokens_list[overlap + 1]))

    for token in tokens_list:
        if token == "," or token == ".":
            number_of_punctuation += 1

    number_of_tokens = len(tokens_list)
    length_of_text = len(text)

    try:
        proccess_ann, material_ann, task_ann = extract_annotations(os.path.join(folder_gold, f_ann))
    except ValueError as e:
        print("Unable to process: {}".format(os.path.join(folder_gold, f_ann)))
        print(e)

    process_bilou = extract_BILOU(tokens_spans, proccess_ann)
    material_bilou = extract_BILOU(tokens_spans, material_ann)
    task_bilou = extract_BILOU(tokens_spans, task_ann)

    if str(f) == "S2212671612001291.txt": #548 565
        print_alignments(tokens_list,tokens_spans, [process_bilou, material_bilou, task_bilou])

    bilou_to_annotation_file(os.path.join(folder_pred, f_ann),tokens_list, tokens_spans, {"Process": process_bilou, "Material": material_bilou, "Task": task_bilou})

print(eval.calculateMeasures('../data/train2', '../data/train_pred', 'rel'))
print("Number of overlap:{}".format(number_of_overlaps))
print("Number of punctuation: {}".format(number_of_punctuation))
print("Total of sentences: {}".format(total_of_sentences))
