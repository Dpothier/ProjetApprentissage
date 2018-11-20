

def save_annotation_file(filename, tokens, spans, bilou_list, tag_scheme='IO'):
    if tag_scheme == 'BILOU':
        save_annotation_file_bilou(filename, tokens, spans, bilou_list)

    if tag_scheme == 'IO':
        save_annotation_file_IO(filename, tokens, spans, bilou_list)


def save_annotation_file_bilou(filename, tokens, spans, bilou_list):
    B = 0
    I = 1
    L = 2
    O = 3
    U = 4

    with open(filename, encoding='utf8', mode="w") as f:
        entity_index = 1
        for entity_type, bilou in bilou_list.items():
            current_entity = []
            for i in range(len(tokens)):
                if bilou[i] == U:
                    f.write("T{}\t{} {} {}\t{}\n".format(entity_index, entity_type, spans[i][0], spans[i][1], tokens[i]))
                    entity_index +=1

                if bilou[i] == B:
                    current_entity.append(i)

                if bilou[i] == I:
                    current_entity.append(i)

                if bilou[i] == L:
                    current_entity.append(i)
                    start_index = spans[current_entity[0]][0]
                    last_index = spans[current_entity[-1]][1]
                    words = ' '.join([tokens[x] for x in current_entity])
                    f.write("T{}\t{} {} {}\t{}\n".format(entity_index, entity_type, start_index, last_index, words))
                    current_entity = []
                    entity_index += 1

def save_annotation_file_IO(filename, tokens, spans, bilou_list):
    I = 0
    O = 1

    with open(filename, encoding='utf8', mode="w") as f:
        entity_index = 1
        for entity_type, bilou in bilou_list.items():
            current_entity = []
            for i in range(len(tokens)):
                if bilou[i] == I:
                    current_entity.append(i)

                if bilou[i] == O:
                    if len(current_entity) > 0:
                        start_index = spans[current_entity[0]][0]
                        last_index = spans[current_entity[-1]][1]
                        words = ' '.join([tokens[x] for x in current_entity])
                        f.write("T{}\t{} {} {}\t{}\n".format(entity_index, entity_type, start_index, last_index, words))
                        current_entity = []
                        entity_index += 1