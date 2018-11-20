import itertools
from abc import abstractmethod
from os import listdir
from os.path import isfile, join
from random import shuffle
from typing import List, Tuple

from nltk import re
from nltk.tokenize import TreebankWordTokenizer, PunktSentenceTokenizer
from nltk.data import load
from torch.utils.data import DataLoader

# from utils.ontology_parsing import parse_ontology

tokenizer = TreebankWordTokenizer()
word_tokenize = tokenizer.tokenize

sent_tokenizer = load('tokenizers/punkt/english.pickle')
# from nltk.tokenize import sent_tokenize as sent_tokenizer

NUM_REGEX = re.compile("\d")


class ExampleX(object):
    pass


class ExampleY(object):
    pass


class Example:
    def __init__(self, x: ExampleX, y: ExampleY):
        self.x = x
        self.y = y


class NERExampleX(ExampleX):
    def __init__(self, tokens):
        self.tokens = tokens


class NERExampleY(ExampleY):
    def __init__(self, labels):
        self.labels = labels


class NERExample(Example):
    def __init__(self, x: NERExampleX, y: NERExampleY):
        super().__init__(x, y)


class RelationExampleX(ExampleX):
    def __init__(self, tokens, entities, e1_type, e2_type, e1_offsets, e2_offsets, e1_name, e2_name, e1_id=None,
                 e2_id=None):
        self.e2_id = e2_id
        self.e1_id = e1_id
        self.e2_name = e2_name
        self.e1_name = e1_name
        self.e2_offsets = e2_offsets
        self.e1_offsets = e1_offsets
        self.e1_type = e1_type
        self.e2_type = e2_type
        self.ontology_mapping = '-'.join(sorted([self.e1_type, self.e2_type]))
        self.tokens = tokens
        self.entities = entities


class RelationExampleY(ExampleY):
    def __init__(self, relation):
        self.relation = relation


class RelationExample(Example):
    def __init__(self, x: RelationExampleX, y: RelationExampleY):
        super().__init__(x, y)


class Sentence(object):
    def __init__(self, txt, entities, relations, relative_offsets):
        self.relative_offsets = relative_offsets
        self.txt = txt
        self.entities = self.__remap_entities_relatively(entities)
        self.relations = relations

    def __remap_entities_relatively(self, entities):
        new_entities = list()
        for entity in entities:
            entity[1][1] -= self.relative_offsets[0]
            entity[1][2] -= self.relative_offsets[0]
            new_entities.append(entity)
        return new_entities


class Document(object):
    def __init__(self, txt, ann):
        self.txt = txt
        self.ann = ann
        self.entities, self.relations = self.__parse_annotations()
        self.sentences = self.__parse_sentences()

    def __parse_entities(self):
        """
        Parses entities annotation to have a form of
        [ID, [TYPE, BEGIN, END], TEXT]
        List[Tuple]
        """
        annotations = [ann.split('\t') for ann in self.ann if ann[0] == 'T']  # Keep only entities
        annotations = [(ann[0], ann[1].split(' '), ann[2][:-1]) for ann in annotations]
        annotations = [(ann[0], [ann[1][0], int(ann[1][1]), int(ann[1][2])], ann[2]) for ann in annotations]
        annotations.sort(key=lambda x: x[1][1])
        return annotations

    def __parse_relations(self):
        """
        Parses relation annotation to have a form of
        [ID, TYPE, ARG1, ARG2]
        List[Tuple]
        """
        annotations = [ann.split('\t') for ann in self.ann if ann[0] == 'R']  # Keep only relations
        annotations = [[ann[0]] + ann[1].split(' ') for ann in annotations]
        annotations = [(ann[0], ann[1], ann[2].split(':')[1], ann[3].split(':')[1]) for ann in annotations]
        return annotations

    def __parse_annotations(self):
        entities = self.__parse_entities()
        relations = self.__parse_relations()
        return entities, relations

    def __find_annotations_in_range(self, begin, end):
        entities = list()
        entities_ids = set()
        relations = list()
        for entity in self.entities:
            if entity[1][1] >= begin and entity[1][2] <= end:
                entities.append(entity)
                entities_ids.add(entity[0])
        for relation in self.relations:
            if relation[2] in entities_ids or relation[3] in entities_ids:
                relations.append(relation)
        return entities, relations

    def __parse_sentences(self):
        sentences = list()
        for offsets in sent_tokenizer.span_tokenize(self.txt):
            begin, end = offsets
            entities, relations = self.__find_annotations_in_range(begin, end)
            sentence = Sentence(
                self.txt[begin:end],
                entities,
                relations,
                [begin, end]
            )
            sentences.append(sentence)
        return sentences


class ManyExamplesParser(object):
    @abstractmethod
    def parse(self, document: Document) -> List[Example]:
        pass


class NERCorpusParser(ManyExamplesParser):
    """
    Returns two list of tokens,
    First list is the xs
    Second is the ys
    """

    def __init__(self, tokenize_func):
        self.tokenize_func = tokenize_func

    def parse(self, document: Document) -> List[NERExample]:
        examples = list()
        for sentence in document.sentences:
            tokens = list()
            labels = list()
            last_index = 0
            for ann in sentence.entities:
                # Parse whats before
                new_tokens = word_tokenize(sentence.txt[last_index:ann[1][1]])
                tokens += new_tokens
                labels += ['O' for _ in range(len(new_tokens))]

                # Parse annotation
                ann_tokens = word_tokenize(sentence.txt[ann[1][1]:ann[1][2]])
                tokens += ann_tokens
                labels += ['B-' + ann[1][0]]
                for _ in ann_tokens[1:]:
                    labels += ['I-' + ann[1][0]]
                last_index = ann[1][2]

            ann_tokens = word_tokenize(sentence.txt[last_index:])
            tokens += ann_tokens
            for _ in ann_tokens:
                labels += ['O']
            assert len(tokens) == len(labels)  # sanity check
            tokens = [flatten_numbers(token) for token in tokens]  # Replace numbers by zeros
            examples.append(NERExample(
                NERExampleX(tokens),
                NERExampleY(labels)
            ))
        return examples


class RelationCorpusParser(ManyExamplesParser):
    NO_RELATION = 'NoRelation'

    def __init__(self, tokenize_func):
        self.tokenize_func = tokenize_func
        self.ontology = parse_ontology()

    def __remove_weird_chars(self, s: str):
        return s

    def __parse_entities(self, annotations):
        annotations = [ann.split('\t') for ann in annotations if ann[0] == 'T']  # Keep only entities
        annotations = [(ann[0], ann[1].split(' '), ann[2][:-1]) for ann in annotations]
        annotations = [(ann[0], [ann[1][0], int(ann[1][1]), int(ann[1][2])], self.__remove_weird_chars(ann[2])) for ann
                       in
                       annotations]
        annotations.sort(key=lambda x: x[0][1])
        return annotations

    def __parse_relations(self, annotations):
        annotations = [ann.split('\t') for ann in annotations if ann[0] == 'R']  # Keep only entities
        annotations = [[ann[0]] + ann[1].split(' ') for ann in annotations]
        annotations = [(ann[0], ann[1], ann[2].split(':')[1], ann[3].split(':')[1]) for ann in annotations]
        return annotations

    def __parse_annotations(self, annotations):
        entities = self.__parse_entities(annotations)
        relations = self.__parse_relations(annotations)
        return entities, relations

    def get_relation(self, relations, e1, e2):
        for r in relations:
            if e1 in r and e2 in r:
                return r[1]
        return self.NO_RELATION

    def __tokenize(self, txt, e1, e2):
        e1_start_idx = e1[1][1]
        e1_end_idx = e1[1][2]
        e2_start_idx = e2[1][1]
        e2_end_idx = e2[1][2]
        bef = word_tokenize(txt[:e1_start_idx])[-3:]
        e1_tokens = word_tokenize(txt[e1_start_idx:e1_end_idx])
        bet = word_tokenize(txt[e1_end_idx:e2_start_idx])
        e2_tokens = word_tokenize(txt[e2_start_idx:e2_end_idx])
        aft = word_tokenize(txt[e2_end_idx:])[:3]

        tokens = bef + e1_tokens + bet + e2_tokens + aft
        entities = list()
        for _ in bef:
            entities.append('O')

        # entities.append(e1[1][0])
        entities += ['B-' + e1[1][0]]
        for _ in e1_tokens[1:]:
            entities += ['I-' + e1[1][0]]

        for _ in bet:
            entities.append('O')

        # entities.append(e2[1][0])
        entities += ['B-' + e2[1][0]]
        for _ in e2_tokens[1:]:
            entities += ['I-' + e2[1][0]]

        for _ in aft:
            entities.append('O')

        e1_offsets = list()
        e1_start_token = len(bef)
        e1_end_token = len(bef) + len(e1_tokens) - 1
        for i, _ in enumerate(tokens):
            if i < e1_start_token:
                e1_offsets.append(i - e1_start_token)
            elif i > e1_end_token:
                e1_offsets.append(i - e1_end_token)
            else:
                e1_offsets.append(0)

        e2_offsets = list()
        e2_start_token = len(bef) + len(e1_tokens) + len(bet)
        e2_end_token = len(bef) + len(e1_tokens) + len(bet) + len(e2_tokens) -2
        for i, _ in enumerate(tokens):
            if i < e2_start_token:
                e2_offsets.append(i - e2_start_token)
            elif i > e2_end_token:
                e2_offsets.append(i - e2_end_token - 1)
            else:
                e2_offsets.append(0)

        assert len(tokens) == len(entities) == len(e1_offsets) == len(e2_offsets)  # sanity check
        e1_offsets = [str(e) for e in e1_offsets]
        e2_offsets = [str(e) for e in e2_offsets]
        tokens = [flatten_numbers(token) for token in tokens]  # Replace numbers by zeros
        return tokens, entities, e1_offsets, e2_offsets

    def parse(self, document: Document):
        for sentence in document.sentences:
            txt = sentence.txt
            candidates = list(itertools.product(sentence.entities, sentence.entities))

            distinct = set()
            filtered_candidates = list()
            for c in candidates:
                nums = '-'.join(sorted([c[0][0], c[1][0]]))
                if nums not in distinct:
                    filtered_candidates.append(c)
                    distinct.add(nums)

            for c in filtered_candidates:
                e1 = c[0]
                e2 = c[1]
                combination_of_entities = '-'.join(sorted([e1[1][0], e2[1][0]]))
                if e1[0] == e2[0] or combination_of_entities not in self.ontology:
                    continue

                if int(e1[1][1]) > int(e2[1][1]):
                    tmp = e2
                    e2 = e1
                    e1 = tmp

                rel = self.get_relation(sentence.relations, e1[0], e2[0])
                tokens, entities, e1_offsets, e2_offsets = self.__tokenize(txt, e1, e2)
                yield RelationExample(
                    RelationExampleX(
                        tokens,
                        entities,
                        e1[1][0],
                        e2[1][0],
                        e1_offsets,
                        e2_offsets,
                        e1[2],
                        e2[2],
                        e1[0],
                        e2[0]
                    ),
                    RelationExampleY(rel)
                )


class AnnFileLoader(object):
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path

    def load_text_file(self, f):
        with open(join(self.corpus_path, f + '.txt'), encoding='utf-8') as fhandle:
            lines = [l[:-1] for l in fhandle.readlines()]
            txt = ' '.join(lines)
        fhandle.close()
        return txt

    def load_ann_file(self, f):
        with open(join(self.corpus_path, f + '.ann'), encoding='utf-8') as fhandle:
            annotations = fhandle.readlines()
        fhandle.close()
        return annotations

    def load_document(self, f) -> Document:
        try:
            return Document(
                self.load_text_file(f),
                self.load_ann_file(f)
            )
        except Exception:
            print(f)


class Corpus:
    def __init__(self, examples, name, f=lambda x: x):
        self.examples = examples
        self.name = name
        self.transform = f

    def __iter__(self):
        for e in self.examples:
            yield self.transform(e)

    def __getitem__(self, item):
        return self.transform(self.examples[item])

    def __len__(self):
        return len(self.examples)


class CorpusLoader(DataLoader):
    def __init__(self, corpus: Corpus, **kwargs):
        super(CorpusLoader, self).__init__(corpus, **kwargs)


def load_corpus(corpus_path) -> List[Document]:
    fileloader = AnnFileLoader(corpus_path)
    onlyfiles = list({f.split('.')[0] for f in listdir(corpus_path) if isfile(join(corpus_path, f))})
    return [fileloader.load_document(f) for f in sorted(onlyfiles)]


def partition_examples(examples: List[Example]):
    shuffle(examples)
    t = examples[:int(len(examples) * 0.8)]
    test_filenames = examples[int(len(examples) * 0.8):]
    train_filenames = t[:int(len(t) * 0.8)]
    dev_filenames = t[int(len(t) * 0.8):]
    return train_filenames, dev_filenames, test_filenames


def partition_corpus(parser, corpus_path, filter=lambda x: x) -> Tuple[List[Example], List[Example], List[Example]]:
    documents = load_corpus(corpus_path)
    t = documents[:int(len(documents) * 0.8)]
    test_documents = documents[int(len(documents) * 0.8):]
    train_documents = t[:int(len(t) * 0.8)]
    dev_documents = t[int(len(t) * 0.8):]
    train_examples = [e for d in train_documents for e in parser.parse(d) if filter(e)]
    dev_examples = [e for d in dev_documents for e in parser.parse(d) if filter(e)]
    test_documents = [e for d in test_documents for e in parser.parse(d) if filter(e)]
    return train_examples, dev_examples, test_documents


def partition_corpus_for_ner(corpus_path):
    parser = NERCorpusParser(word_tokenize)
    return partition_corpus(parser, corpus_path)


def partition_corpus_for_rel(corpus_path, filter=lambda x: x):
    parser = RelationCorpusParser(word_tokenize)
    return partition_corpus(parser, corpus_path, filter)

def load_conll_corpus(corpus_path):
    splits_names = ['train', 'valid', 'test']
    splits = list()
    parser = NERCorpusParser(word_tokenize)
    for corpus_name in splits_names:
        corpus = join(corpus_path, corpus_name)
        documents = load_corpus(corpus)
        parsed_documents = [e for d in documents for e in parser.parse(d)]
        splits.append(parsed_documents)
    return splits



def replace_int_by_zero(char):
    try:
        int(char)
        return '0'
    except:
        return char


def flatten_numbers(token):
    return ''.join([replace_int_by_zero(c) if NUM_REGEX.match(c) else c for c in token])
