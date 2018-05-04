import os
from data_load.tokenization import tokenize

def load_text(file):
    for l in file:
        text = l

    return text


def load_annotations(file):
    process_ann = []
    material_ann = []
    tasks_ann = []

    for l in file:
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


def parse_BILOU_tags(spans, annotations, use_int_tag=True):
    B = 0 if use_int_tag else 'B'
    I = 1 if use_int_tag else 'I'
    L = 2 if use_int_tag else 'L'
    O = 3 if use_int_tag else 'O'
    U = 4 if use_int_tag else 'U'

    bilou_tags = []
    for i in range(len(spans)):
        bilou_tags.append(O)

    for annotation in annotations:
        start_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[0] <= span[1]]
        end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1] <= span[1]]
        if len(end_spans) == 0:
            end_spans = [index for index, span in enumerate(spans) if span[0] <= annotation[1]-1 <= span[1]]
        start_span = start_spans[0]
        end_span = end_spans[0]
        if start_span == end_span:
            bilou_tags[start_span] = U
        else:
            bilou_tags[start_span] = B
            bilou_tags[end_span] = L
            for i in range(start_span+1, end_span):
                bilou_tags[i]= I
    return bilou_tags


def load_data(folder, use_int_tags=True):
    texts = []
    indices = {}

    flist = os.listdir(folder)
    current_index = 0
    for f in flist:
        if not str(f).endswith(".txt"):
            continue
        f_text = open(os.path.join(folder, f), "r", encoding="utf8")
        f_ann = open(os.path.join(folder, f[0:-4]+".ann"), "r", encoding="utf8")

        text = load_text(f_text)

        process_ann, material_ann, tasks_ann = load_annotations(f_ann)
        tokens, spans = tokenize(text)
        texts.append((current_index,
                      tokens,
                      parse_BILOU_tags(spans, process_ann, use_int_tags),
                      parse_BILOU_tags(spans, material_ann, use_int_tags),
                      parse_BILOU_tags(spans, tasks_ann, use_int_tags)))

        indices[current_index] = (f[0:-4], tokens, spans)
        current_index += 1

    dict_texts = [{'id': text[0],
                   'texts': text[1],
                   'process_tags': text[2],
                   'material_tags': text[3],
                   'task_tags': text[4]} for text in texts]

    return dict_texts, indices
