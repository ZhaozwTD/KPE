import configparser
import json

import networkx as nx
import nltk
import spacy
from tqdm import tqdm

config = configparser.ConfigParser()
config.read("config.cfg")
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could", "get", "in", "up", "may", "wanter"]

useful_relations = ['AtLocation', 'LocatedNear', 'CapableOf', 'Causes',
                    'CausesDesire', 'MotivatedByGoal', 'CreatedBy', 'Desires', 'HasProperty',
                    'HasSubevent', 'DefinedAs', 'MadeOf',
                    'PartOf', 'HasA', 'UsedFor']
blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])

concept2id = None
relation2id = None
id2relation = None
id2concept = None


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def is_invalid(word):
    return len(word) <= 2 or len(word.split('_')) > 4


def extract_english_cpnet():
    relation_and_entity = []

    with open(config["paths"]["conceptnet"], 'r', encoding="utf8") as f:
        for line in tqdm(f.readlines(), desc='extract and filter triples'):
            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                rel = toks[1].split("/")[-1]
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()
                weight = json.loads(toks[4])["weight"]

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if weight < 1.0:
                    continue
                if rel not in useful_relations or is_invalid(head) or is_invalid(tail):
                    continue

                relation_and_entity.append("\t".join([rel, head, tail, str(weight)]))

    with open(config["paths"]["conceptnet_preprocess"], "w", encoding="utf8") as f:
        f.write("\n".join(relation_and_entity))


def get_all_concept_relation():
    relations = set()
    concepts = set()
    print('get all concepts and relations ......')
    with open(config["paths"]['conceptnet_preprocess'], 'r', encoding="utf8") as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            rel, c1, c2 = line[0], line[1], line[2]
            if rel not in useful_relations:
                continue
            relations.add(rel)
            concepts.add(c1)
            concepts.add(c2)
    print('Done')

    print('write concepts and relations to corresponding file ......')
    with open(config["paths"]["concept_vocab"], "w", encoding="utf8") as c:
        for i in tqdm(concepts):
            c.write(i + '\n')

    with open(config["paths"]["relation_vocab"], "w", encoding="utf8") as r:
        for i in tqdm(relations):
            r.write(i + '\n')
    print('Done')


def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("relation2id done")


def save_graph():
    global concept2id, relation2id, id2relation, id2concept, blacklist
    load_resources()
    graph = nx.MultiDiGraph()
    print('construct conceptnet graph')

    with open(config["paths"]["conceptnet_preprocess"], "r", encoding="utf8") as f:
        lines = f.readlines()

        def not_save(cpt):
            if cpt in blacklist:
                return True
            for t in cpt.split("_"):
                if t in nltk_stopwords:
                    return True
            return False

        for line in tqdm(lines, desc="saving to graph"):
            ls = line.strip().split('\t')

            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])

            if not_save(ls[1]) or not_save(ls[2]):
                continue

            if subj == obj:  # delete loops
                continue

            graph.add_edge(subj, obj, rel=rel, weight=weight)

    nx.write_gpickle(graph, config["paths"]["conceptnet_graph"])


# extract_english_cpnet()
get_all_concept_relation()
save_graph()
