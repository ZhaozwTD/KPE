import configparser
import json
import random
import re

import spacy
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import T5Tokenizer, RobertaTokenizer

from .util import cache

config = configparser.ConfigParser()
config.read("./preprocess/config.cfg")
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat'])

mask_token = {
    "t5": "<extra_id_0>",
    "roberta": "<mask>"
}
pad_token = {
    "t5": "<pad>",
    "roberta": "<pad>"
}


def load_resources():
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

    return concept2id, relation2id, id2relation, id2concept


def get_query_knowledge(args, knowledge: list, tokenizer, mask_token, pad_token):
    query_knowledge = []
    vocab = tokenizer.get_vocab()

    for k in knowledge:
        token = tokenizer(k, max_length=args.max_len, padding='max_length', truncation=True)
        input_ids, attn_mask = token.input_ids, token.attention_mask

        length = sum(attn_mask)
        mask_label = [vocab[pad_token]] * args.max_len
        for ind in range(args.max_len):
            if ind < length:
                if random.random() < 0.15:
                    if random.random() < 0.8:
                        mask_label[ind] = input_ids[ind]
                        input_ids[ind] = vocab[mask_token]
                    elif random.random() < 0.9:
                        mask_label[ind] = input_ids[ind]
                        input_ids[ind] = random.choice(list(range(len(vocab))))
                    else:
                        mask_label[ind] = input_ids[ind]

                else:
                    continue

        query_knowledge.append((input_ids, attn_mask, mask_label))

    return query_knowledge


def replace_masked_tokens(args, sent: str, triple, tokenizer, mask_token, pad_token, relation2id):
    head, rel, tail = triple
    sent = re.sub(r'[^\w\s]', '', sent)

    for i in range(len(sent.split())):
        length = len(head.split())
        if head in ' '.join(sent.split()[i:i + length]):
            head = ' '.join(sent.split()[i:i + length])
            break

    for i in range(len(sent.split())):
        length = len(tail.split())
        if tail in ' '.join(sent.split()[i:i + length]):
            tail = ' '.join(sent.split()[i:i + length])
            break

    vocab = tokenizer.get_vocab()

    token = tokenizer(' ' + sent, max_length=args.max_len, padding='max_length', truncation=True)
    sent_token, att_mask = token.input_ids, token.attention_mask

    if args.model_type == 'roberta':
        head_token, tail_token = tokenizer.tokenize(' ' + head), tokenizer.tokenize(' ' + tail)
    else:
        head_token, tail_token = tokenizer.tokenize(head), tokenizer.tokenize(tail)

    head_token, tail_token = tokenizer.convert_tokens_to_ids(head_token), tokenizer.convert_tokens_to_ids(tail_token)

    replace_token = tail_token
    mask_replace_token = [vocab[mask_token]] * len(replace_token)

    mask_sent = sent_token.copy()
    for i in range(len(mask_sent) - len(replace_token) + 1):
        if mask_sent[i:i + len(replace_token)] == replace_token:
            mask_sent[i:i + len(replace_token)] = mask_replace_token
            break

    mask_sent_label = [vocab[pad_token] if mask_sent[ind] != vocab[mask_token] else sent_token[ind]
                       for ind in range(len(mask_sent))]

    # relation-level task
    for i in range(len(sent_token)):
        if sent_token[i:i + len(head_token)] == head_token:
            head_index = i
            break
    for i in range(len(sent_token)):
        if sent_token[i:i + len(tail_token)] == tail_token:
            tail_index = i
            break

    head_tail_index_label = [[head_index, head_index + len(head_token)], [tail_index, tail_index + len(tail_token)]]

    return mask_sent, att_mask, mask_sent_label, head_tail_index_label, relation2id[rel]


def get_triple_knowledge(args, knowledge: list, tokenizer, mask_token, pad_token, relation2id):
    triple_knowledge = []
    sign = len(knowledge)
    if sign == 0:
        triple_knowledge.append(([], [], [], [[-1, -1], [-1, -1]], -1))

    else:
        for k in knowledge:
            mask_sent, att_mask, mask_sent_label, head_tail_index, rel_id = replace_masked_tokens(args,
                                                                                                  k['t_knowledge'],
                                                                                                  k['triple'],
                                                                                                  tokenizer,
                                                                                                  mask_token, pad_token,
                                                                                                  relation2id)
            triple_knowledge.append((mask_sent, att_mask, mask_sent_label, head_tail_index, rel_id))

    return triple_knowledge, sign


def get_knowledge(args, query_knowledge, triple_knowledge, sign):
    knowledge_id, att_mask, label, head_tail_index, rel_id = [], [], [], [], []
    if sign == 0:
        for k in random.sample(query_knowledge, args.num_knowledge):
            knowledge_id.append(k[0])
            att_mask.append(k[1])
            label.append(k[2])
            head_tail_index.append([[-1, -1], [-1, -1]])
            rel_id.append(-1)

    else:
        all_knowledge = query_knowledge + triple_knowledge
        for k in random.sample(all_knowledge, args.num_knowledge):
            knowledge_id.append(k[0])
            att_mask.append(k[1])
            label.append(k[2])
            if len(k) > 3:
                head_tail_index.append(k[3])
                rel_id.append(k[4])
            else:
                head_tail_index.append([[-1, -1], [-1, -1]])
                rel_id.append(-1)

    return knowledge_id, att_mask, label, head_tail_index, rel_id


class CSQA2Dataset(Dataset):
    def __init__(self, args, data_path, type, relation2id):
        self.args = args
        self.answer_list = args.csqa2_answer_list
        self.data_type = type  # 'train', 'dev', 'test'
        if args.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(args.t5_model_type)
        elif args.model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model_type)

        self.mask_token = mask_token[args.model_type]
        self.pad_token = pad_token[args.model_type]
        self.relation2id = relation2id

        cache_dir = './cache'
        postfix = f'cache_csqa2_{args.model_type}_{type}'
        all_data = self.load_data(data_path, file_dir=cache_dir, postfix=postfix)

        self.token_ids = all_data['token_ids']
        self.attention_mask = all_data['attention_mask']

        self.knowledge_id = all_data['knowledge_id']
        self.knowledge_att_mask = all_data['knowledge_attn_mask']
        self.knowledge_label = all_data['knowledge_label']
        self.signs = all_data['sign']

        self.head_tail_index = all_data['head_tail_index']
        self.rel_id = all_data['rel_id']

        self.labels = all_data['labels']
        self.cands = all_data['cands']

    @cache
    def load_data(self, data_path, file_dir='./cache', postfix=f'cache'):
        all_token_ids, all_attention_mask, = [], []
        all_knowledge_id, all_knowledge_att_mask, all_knowledge_label, all_signs = [], [], [], []
        all_head_tail_index, all_rel_id, all_labels, all_cands = [], [], [], []
        with open(data_path, 'r') as f:
            for line in tqdm(json.load(f), desc="load dataset"):
                token_id, att_mask = self.get_token_mask(line['query'], self.answer_list)
                all_token_ids.append(token_id)
                all_attention_mask.append(att_mask)

                query_knowledge = get_query_knowledge(self.args, line['query_knowledge'], self.tokenizer,
                                                      self.mask_token, self.pad_token)

                triple_knowledge, sign = get_triple_knowledge(self.args,
                                                              line['triple_knowledge'],
                                                              self.tokenizer,
                                                              self.mask_token, self.pad_token, self.relation2id)

                knowledge_id, knowledge_attn_mask, knowledge_label, head_tail_index, rel_id = \
                    get_knowledge(self.args, query_knowledge, triple_knowledge, sign)

                all_knowledge_id.append(torch.tensor(knowledge_id))
                all_knowledge_att_mask.append(torch.tensor(knowledge_attn_mask))
                all_knowledge_label.append(torch.tensor(knowledge_label))
                all_head_tail_index.append(torch.tensor(head_tail_index))
                all_rel_id.append(torch.tensor(rel_id))
                all_signs.append(torch.tensor(sign))

                all_labels.append(line['answer'])
                all_cands.append('#'.join(self.answer_list))

        data = {'token_ids': all_token_ids, 'attention_mask': all_attention_mask,
                'knowledge_id': all_knowledge_id, 'knowledge_attn_mask': all_knowledge_att_mask,
                'knowledge_label': all_knowledge_label,
                'head_tail_index': all_head_tail_index, 'rel_id': all_rel_id,
                'sign': all_signs, 'labels': all_labels, 'cands': all_cands}

        return data

    def get_token_mask(self, sent, answer_list):
        token_ids = []
        attention_masks = []
        for answer in answer_list:
            input_id, attention_mask = self.generate_template(sent, answer)
            token_ids.append(input_id)
            attention_masks.append(attention_mask)

        return torch.stack(token_ids), torch.stack(attention_masks)

    def generate_template(self, sent, answer):
        sent = f'Question: {sent} The answer is {answer}'
        if self.args.model_type == 'roberta':
            token = self.tokenizer(' ' + sent, max_length=self.args.max_len, padding='max_length', truncation=True,
                                   return_tensors='pt')

        else:
            token = self.tokenizer(sent, max_length=self.args.max_len, padding='max_length', truncation=True,
                                   return_tensors='pt')

        return (token.input_ids, token.attention_mask)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        token_id = self.token_ids[idx]
        attention_mask = self.attention_mask[idx]
        knowledge_id = self.knowledge_id[idx]
        knowledge_att_mask = self.knowledge_att_mask[idx]
        knowledge_label = self.knowledge_label[idx]
        head_tail_index = self.head_tail_index[idx]
        rel_id = self.rel_id[idx]
        sign = self.signs[idx]

        label = self.labels[idx]
        cands = self.cands[idx]

        return (
            token_id, attention_mask,
            knowledge_id, knowledge_att_mask, knowledge_label,
            head_tail_index, rel_id, sign,
            label, cands
        )


class OpenbookQADataset(Dataset):
    def __init__(self, args, data_path, type, relation2id):
        self.args = args
        self.data_type = type  # 'train', 'dev', 'test'
        if args.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(args.t5_model_type)
        elif args.model_type == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model_type)

        self.mask_token = mask_token[args.model_type]
        self.pad_token = pad_token[args.model_type]
        self.relation2id = relation2id

        cache_dir = './cache'
        postfix = f'cache_openbookqa_{args.model_type}_{type}'
        all_data = self.load_data(data_path, file_dir=cache_dir, postfix=postfix)

        self.token_ids = all_data['token_ids']
        self.attention_mask = all_data['attention_mask']

        self.knowledge_id = all_data['knowledge_id']
        self.knowledge_att_mask = all_data['knowledge_attn_mask']
        self.knowledge_label = all_data['knowledge_label']
        self.signs = all_data['sign']

        self.head_tail_index = all_data['head_tail_index']
        self.rel_id = all_data['rel_id']

        self.labels = all_data['labels']
        self.cands = all_data['cands']

    @cache
    def load_data(self, data_path, file_dir='./cache', postfix=f'cache'):
        all_token_ids, all_attention_mask, = [], []
        all_knowledge_id, all_knowledge_att_mask, all_knowledge_label, all_signs = [], [], [], []
        all_head_tail_index, all_rel_id, all_labels, all_cands = [], [], [], []
        with open(data_path, 'r') as f:
            for line in tqdm(json.load(f), desc="load dataset"):
                token_id, att_mask = self.get_token_mask(line['query'], line['cands'])
                all_token_ids.append(token_id)
                all_attention_mask.append(att_mask)

                query_knowledge = get_query_knowledge(self.args, line['query_knowledge'], self.tokenizer,
                                                      self.mask_token, self.pad_token)

                triple_knowledge, sign = get_triple_knowledge(self.args,
                                                              line['triple_knowledge'],
                                                              self.tokenizer,
                                                              self.mask_token, self.pad_token, self.relation2id)

                knowledge_id, knowledge_attn_mask, knowledge_label, head_tail_index, rel_id = \
                    get_knowledge(self.args, query_knowledge, triple_knowledge, sign)

                all_knowledge_id.append(torch.tensor(knowledge_id))
                all_knowledge_att_mask.append(torch.tensor(knowledge_attn_mask))
                all_knowledge_label.append(torch.tensor(knowledge_label))
                all_head_tail_index.append(torch.tensor(head_tail_index))
                all_rel_id.append(torch.tensor(rel_id))
                all_signs.append(torch.tensor(sign))

                all_labels.append(line['answer'])
                all_cands.append('#'.join(line['cands']))

        data = {'token_ids': all_token_ids, 'attention_mask': all_attention_mask,
                'knowledge_id': all_knowledge_id, 'knowledge_attn_mask': all_knowledge_att_mask,
                'knowledge_label': all_knowledge_label,
                'head_tail_index': all_head_tail_index, 'rel_id': all_rel_id,
                'sign': all_signs, 'labels': all_labels, 'cands': all_cands}

        return data

    def get_token_mask(self, sent, answer_list):
        token_ids = []
        attention_masks = []
        for answer in answer_list:
            input_id, attention_mask = self.generate_template(sent, answer)
            token_ids.append(input_id)
            attention_masks.append(attention_mask)

        return torch.stack(token_ids), torch.stack(attention_masks)

    def generate_template(self, sent, answer):
        sent = f'{sent} {answer}'
        if self.args.model_type == 'roberta':
            token = self.tokenizer(' ' + sent, max_length=self.args.max_len, padding='max_length', truncation=True,
                                   return_tensors='pt')

        else:
            token = self.tokenizer(sent, max_length=self.args.max_len, padding='max_length', truncation=True,
                                   return_tensors='pt')

        return (token.input_ids, token.attention_mask)

    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, idx):
        token_id = self.token_ids[idx]
        attention_mask = self.attention_mask[idx]
        knowledge_id = self.knowledge_id[idx]
        knowledge_att_mask = self.knowledge_att_mask[idx]
        knowledge_label = self.knowledge_label[idx]
        head_tail_index = self.head_tail_index[idx]
        rel_id = self.rel_id[idx]
        sign = self.signs[idx]

        label = self.labels[idx]
        cands = self.cands[idx]

        return (
            token_id, attention_mask,
            knowledge_id, knowledge_att_mask, knowledge_label,
            head_tail_index, rel_id, sign,
            label, cands
        )
