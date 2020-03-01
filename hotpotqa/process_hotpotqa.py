import re
import json
from tqdm import tqdm, trange
from cogqa_utils import find_start_end_after_tokenized, find_start_end_before_tokenized
from transformers import AutoModel, AutoTokenizer
from itertools import chain
import os
import pickle
import logging
from buffer import Buffer
from utils import DEFAULT_MODEL_NAME

def process(DATA_PATH, HOTPOTQA_PATH, DEFAULT_MODEL_NAME, suffix='train'):
    cnt = 0
    with open(HOTPOTQA_PATH, 'r') as fin:
        dataset = json.load(fin)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    batches = []
    for data in tqdm(dataset):
        try:
            flag_ans = False
            question = [tokenizer.cls_token] + tokenizer.tokenize('yes no ' + data['question']) + [tokenizer.sep_token]
            q, q_property = [question], [[('relevance', 1), ('blk_type', 0)]]
            if suffix != 'test':
                if data['answer'] in ['yes', 'no']:
                    pos = 1 + (data['answer'] == 'no')
                    q_property[0].extend([('start', pos), ('end', pos)])
                    flag_ans = True
            else:
                q_property[0].append(('_id', data['_id']))

            d, properties = [], []
            for entity, sentences in data['context']:
                bgn_idx = len(d)
                for sen_idx, sen in enumerate(sentences):
                    if len(sen) == 0: 
                        continue
                    d.append(tokenizer.tokenize(sen))
                    properties.append([])
                    if suffix == 'test':
                        properties[-1].append(('origin', (entity, sen_idx)))
                if suffix == 'test':
                    continue
                for sup_entity, sup_idx in data['supporting_facts']:
                    if sup_entity == entity:
                        properties[bgn_idx + sup_idx].append(('relevance', 1))
                        ret = find_start_end_after_tokenized(tokenizer, d[bgn_idx + sup_idx], [data['answer']])
                        if ret is not None:
                            start, end = ret[0]
                            properties[bgn_idx + sup_idx].extend([('start', start), ('end', end)])
                            flag_ans = True
        except Exception as e:
            logging.error((data['_id'], e))
        else:
            if flag_ans or suffix == 'test':
                # batches.append((q, q_property, d, properties))
                qbuf, cnt = Buffer.split_document_into_blocks(q, tokenizer, properties=q_property, cnt=cnt)
                dbuf, cnt = Buffer.split_document_into_blocks(d, tokenizer, properties=properties, cnt=cnt)
                batches.append((qbuf, dbuf))
            else: 
                logging.warning((data['_id'], data['question']))

    with open(os.path.join(DATA_PATH, 'hotpotqa_{}_{}.pkl'.format(suffix, DEFAULT_MODEL_NAME)), 'wb') as fout:
        pickle.dump(batches, fout)


if __name__ == "__main__":
    HOTPOTQA_PATH_test = '/home/mingding/cognew/hotpot_dev_distractor_v1.json'
    HOTPOTQA_PATH_train = '/home/mingding/cognew/hotpot_train_v1.1.json'

    root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    DATA_PATH = os.path.join(root_dir, 'data')
    os.makedirs(DATA_PATH, exist_ok=True)
    process(DATA_PATH, HOTPOTQA_PATH_test, DEFAULT_MODEL_NAME, 'test')
    process(DATA_PATH, HOTPOTQA_PATH_train, DEFAULT_MODEL_NAME, 'train')
