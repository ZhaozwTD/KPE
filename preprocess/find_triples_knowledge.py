import configparser
import json
import logging
import os
import sys
from collections import Counter
from string import Template

import networkx as nx
import spacy
import torch
from elasticsearch import Elasticsearch
from simcse import SimCSE
from tqdm import tqdm

config = configparser.ConfigParser()
config.read("./preprocess/config.cfg")
es = Elasticsearch(hosts="http://120.92.13.161:9200")
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])
simcse_model = SimCSE("../sup-simcse-roberta-large")
logging.getLogger("elasticsearch").setLevel(logging.WARNING)

with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
    cpnet_vocab = [l.strip() for l in list(f.readlines())]

cpnet_vocab = set([c.replace("_", " ") for c in cpnet_vocab])
blacklist = set(
    ["from", "as", "more", "either", "in", "and", "on", "an", "when", "too", "to", "i", "do", "can", "be", "that", "or",
     "the", "a", "of", "for", "is", "was", "will", "The", "-PRON-", "actually", "likely", "possibly", "want",
     "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to", "is a",
     "one", "something", "sometimes", "everybody", "somebody", "could", "could_be", "mine", "us", "em",
     "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "A", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst",
     "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af",
     "affected", "affecting", "after", "afterwards", "ag", "again", "against", "ah", "ain", "aj", "al", "all", "allow",
     "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst",
     "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone",
     "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appreciate", "approximately", "ar", "are",
     "aren", "arent", "arise", "around", "as", "aside", "ask", "asking", "at", "au", "auth", "av", "available", "aw",
     "away", "awfully", "ax", "ay", "az", "b", "B", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "been",
     "before", "beforehand", "beginnings", "behind", "below", "beside", "besides", "best", "between", "beyond", "bi",
     "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but",
     "bx", "by", "c", "C", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "cc", "cd", "ce", "certain",
     "certainly", "cf", "cg", "ch", "ci", "cit", "cj", "cl", "clearly", "cm", "cn", "co", "com", "come", "comes", "con",
     "concerning", "consequently", "consider", "considering", "could", "couldn", "couldnt", "course", "cp", "cq", "cr",
     "cry", "cs", "ct", "cu", "cv", "cx", "cy", "cz", "d", "D", "d2", "da", "date", "dc", "dd", "de", "definitely",
     "describe", "described", "despite", "detail", "df", "di", "did", "didn", "dj", "dk", "dl", "do", "does", "doesn",
     "doing", "don", "done", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "E",
     "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "eg", "ei", "eight", "eighty", "either", "ej", "el",
     "eleven", "else", "elsewhere", "em", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es",
     "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone",
     "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "F", "f2", "fa", "far", "fc", "few",
     "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "five", "fix", "fj", "fl", "fn", "fo", "followed",
     "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front",
     "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "G", "ga", "gave", "ge", "get", "gets", "getting",
     "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr",
     "greetings", "gs", "gy", "h", "H", "h2", "h3", "had", "hadn", "happens", "hardly", "has", "hasn", "hasnt", "have",
     "haven", "having", "he", "hed", "hello", "help", "hence", "here", "hereafter", "hereby", "herein", "heres",
     "hereupon", "hes", "hh", "hi", "hid", "hither", "hj", "ho", "hopefully", "how", "howbeit", "however", "hr", "hs",
     "http", "hu", "hundred", "hy", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "ie", "if",
     "ig", "ignored", "ih", "ii", "ij", "il", "im", "immediately", "in", "inasmuch", "inc", "indeed", "index",
     "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "inward",
     "io", "ip", "iq", "ir", "isn", "it", "itd", "its", "iv", "ix", "iy", "iz", "j", "J", "jj", "jr", "js", "jt",
     "ju", "just", "k", "K", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "ko", "l", "L", "l2", "la", "largely",
     "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets",
     "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr",
     "ls", "lt", "ltd", "m", "M", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me",
     "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mill", "million", "mine", "miss", "ml", "mn", "mo",
     "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "my",
     "n", "N", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "neither",
     "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none",
     "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "novel", "now", "nowhere", "nr", "ns", "nt",
     "ny", "o", "O", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi",
     "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq",
     "or", "ord", "os", "ot", "otherwise", "ou", "ought", "our", "out", "outside", "over", "overall", "ow", "owing",
     "own", "ox", "oz", "p", "P", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular",
     "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed",
     "please", "plus", "pm", "pn", "po", "poorly", "pp", "pq", "pr", "predominantly", "presumably", "previously",
     "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "Q", "qj", "qu",
     "que", "quickly", "quite", "qv", "r", "R", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really",
     "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively",
     "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm",
     "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "S", "s2", "sa", "said", "saw", "say", "saying",
     "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "seem", "seemed", "seeming", "seems", "seen",
     "sent", "seven", "several", "sf", "shall", "shan", "shed", "shes", "show", "showed", "shown", "showns", "shows",
     "si", "side", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somehow",
     "somethan", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified",
     "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially",
     "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "sz", "t", "T", "t1", "t2", "t3", "take",
     "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx",
     "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter",
     "thereby", "thered", "therefore", "therein", "thereof", "therere", "theres", "thereto", "thereupon", "these",
     "they", "theyd", "theyre", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou",
     "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip",
     "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried",
     "tries", "truly", "try", "trying", "ts", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "U", "u201d",
     "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto",
     "uo", "up", "upon", "ups", "ur", "us", "used", "useful", "usefully", "usefulness", "using", "usually", "ut", "v",
     "V", "va", "various", "vd", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt",
     "vu", "w", "W", "wa", "was", "wasn", "wasnt", "way", "we", "wed", "welcome", "well", "well-b", "went", "were",
     "weren", "werent", "what", "whatever", "whats", "when", "whence", "whenever", "where", "whereafter", "whereas",
     "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who",
     "whod", "whoever", "whole", "whom", "whomever", "whos", "whose", "why", "wi", "widely", "with", "within",
     "without", "wo", "won", "wonder", "wont", "would", "wouldn", "wouldnt", "www", "x", "X", "x1", "x2", "x3", "xf",
     "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "Y", "y2", "yes", "yet", "yj", "yl", "you",
     "youd", "your", "youre", "yours", "yr", "ys", "yt", "z", "Z", "zero", "zi", "zz"])


def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)
    print("concept2id and relation2id done")


def extract_concept(sent):
    global cpnet_vocab
    sent = sent.lower()
    doc = nlp(sent)
    res = set()

    # word length == 1
    for t in doc:
        if t.lemma_ in cpnet_vocab and t.lemma_ not in blacklist:
            if t.pos_ == "NOUN" or t.pos_ == "VERB" or t.pos_ == "PROPN":
                res.add(t.lemma_)

    for word_length in [2, 3]:
        for i in range(len(doc) - word_length + 1):
            t = doc[i:i + word_length]
            if t.lemma_ in cpnet_vocab and all([i not in blacklist for i in str(t).split()]):
                res.add(t.lemma_)

    return list(res)


def judge_concept(concept):
    if concept in cpnet_vocab:
        return [concept]
    else:
        return [i for i in concept.split() if i in cpnet_vocab and i not in blacklist]


def filter_triples(sent, triples, max_triples=50):
    N = len(triples)
    if N == 0:
        return triples
    else:
        rel_count = Counter([t[1] for t in triples])
        all_weights = []

        for t in triples:
            if len(id2concept[t[0]].split('_')) > 1:
                all_weights.append((t, 300))
            elif id2concept[t[2]] in sent.split():
                all_weights.append((t, 300))
            else:
                all_weights.append((t, t[3] * N / rel_count[t[1]]))

        all_weights = sorted(all_weights, key=lambda x: -x[1])

        head_entity = {}
        res = []
        for ind, (t, s) in enumerate(all_weights):
            if head_entity.get(t[0], -1) == -1:
                head_entity[t[0]] = 0
            else:
                head_entity[t[0]] += 1
            if head_entity[t[0]] < 5:
                res.append(t)

        final_triples = [(id2concept[t[0]], t[1], id2concept[t[2]]) for t in res]

        return final_triples[:max_triples]


def find_triples(sent, topic_prompt, is_self_extract=True):
    if is_self_extract:
        concepts = extract_concept(sent)
    else:
        concepts = []
    if topic_prompt != None:
        top_concept = judge_concept(topic_prompt)
        concepts.extend(top_concept)
    triples = []
    concepts = set(concepts)
    if len(concepts) != 0:
        concepts = [concept2id[i.replace(' ', '_')] for i in list(concepts) if i not in blacklist]
        for src in concepts:
            tmp = []
            try:
                tgt_muster = cpnet[src]
                for tgt in tgt_muster.keys():
                    rel_list = cpnet[src][tgt]
                    for key in rel_list:
                        tmp.append((src, rel_list[key]['rel'], tgt, rel_list[key]['weight']))
                triples.extend(tmp)
            except:
                continue

    return filter_triples(sent, triples, max_triples=int(config["parameters"]["max_triples"]))


def cal_similarity(str1, str2):
    with torch.no_grad():
        similarities = simcse_model.similarity(str1, str2)

    return similarities


def find_knowledge(sent, cands, triples, max_triples_knowledge, max_query_knowledge):
    commonsense_knowledge = []
    for ind, triple in enumerate(triples):
        h, r, t = ' '.join(triple[0].split('_')), id2relation[triple[1]], ' '.join(triple[2].split('_'))
        temp = Template(relation_template[r])
        query = temp.substitute(head=h, tail=t)

        body = {"query": {"match": {"context_cut": query}}}
        es_res = es.search(index="generics_kb_2022080201", body=body, filter_path=['hits.hits'])
        current_triple_knowledge = [(sample['_source']['context'], (h, r, t)) for sample in es_res['hits']['hits']
                                    if h in sample['_source']['context'] and t in sample['_source']['context']]
        commonsense_knowledge.extend(current_triple_knowledge)

    triple_knowledge = []
    for i in commonsense_knowledge:
        sim = cal_similarity(sent, i[0])
        triple_knowledge.append((i, sim))

    triple_knowledge = sorted(triple_knowledge, key=lambda x: -x[1])[:int(max_triples_knowledge)]
    triple_knowledge = [{'t_knowledge': k[0][0], 'triple': k[0][1]} for k in triple_knowledge]

    query_knowledge = []

    body = {"query": {"match": {"context_cut": sent}}}
    es_res = es.search(index="generics_kb_2022080201", body=body, filter_path=['hits.hits'])
    query_knowledge.extend(
        [(sample['_source']['context'], cal_similarity(sample['_source']['context'], sent))
         for sample in es_res['hits']['hits']])

    query_knowledge = sorted(query_knowledge, key=lambda x: -x[1])[:int(max_query_knowledge)]
    query_knowledge = [k[0] for k in query_knowledge]

    return triple_knowledge, query_knowledge


def process(input_path, output_path, type, dataset_name):
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

        for ind, data in enumerate(tqdm(data_list, desc='extract knowledge for {} dataset'.format(type))):
            query = data['query']
            if type == 'test':
                answer = []
            else:
                answer = data['answer']
            if dataset_name == 'csqa':
                cands = data['cands']
            else:
                cands = []

            topic_prompt = data.get('topic_prompt', None)
            triples = find_triples(query, topic_prompt)
            triple_knowledge, query_knowledge = find_knowledge(query, cands, triples,
                                                               config["parameters"]["max_triples_knowledge"],
                                                               config["parameters"]["max_query_knowledge"])

            examples.append(
                {'query': query, 'cands': cands, 'answer': answer, 'triple_knowledge': triple_knowledge,
                 'query_knowledge': query_knowledge})

    with open(output_path, 'w') as f:
        f.write(json.dumps(examples, indent=4))


dataset_name = sys.argv[1]
load_resources()
cpnet = nx.read_gpickle(config["paths"]["conceptnet_graph"])
DATA_PATH = config["paths"][dataset_name + "_dir"]
genericsKG_path = config["paths"]["genericsKG_path"]
TYPES = ['dev', 'test', 'train']

relation_template = {
    "MotivatedByGoal": "$head is motivated by $tail",
    "MadeOf": "$head can be made of $tail",
    "CapableOf": "$head is capable of $tail",
    "PartOf": "$head is part of $tail",
    "UsedFor": "$head can be used for $tail",
    "Desires": "$head desires $tail",
    "CreatedBy": "$head is created by $tail",
    "HasProperty": "$head has property $tail",
    "DefinedAs": "$head is defined as $tail",
    "Causes": "$head can cause $tail",
    "HasSubevent": "$head has a subevent of $tail",
    "LocatedNear": "$head is located near $tail",
    "CausesDesire": "$head causes desire for $tail",
    "HasA": "$head has a $tail",
    "AtLocation": "$head can be found at $tail",
}

genericsKG, genericsKG_scores = [], []
with open(genericsKG_path, 'r', encoding='utf-8') as f:
    for index, line in enumerate(f.readlines()[1:]):
        line = line.strip().split('\t')
        commonsense_sent, score = line[-2].lower(), float(line[-1])
        genericsKG.append(commonsense_sent.split(" "))
        genericsKG_scores.append(score)

for data_type in TYPES:
    in_path = os.path.join(DATA_PATH, '.'.join([data_type, dataset_name, 'json']))
    out_path = os.path.join(DATA_PATH, '.'.join([data_type, dataset_name, 'knowledge', 'json']))
    process(in_path, out_path, data_type, dataset_name)

print('Finish!')
