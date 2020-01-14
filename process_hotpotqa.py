# %%
import re
import json
from tqdm import tqdm, trange
from utils import DEFAULT_MODEL_NAME
from cogqa_utils import find_start_end_after_tokenized, find_start_end_before_tokenized
from transformers import AutoModel, AutoTokenizer
from itertools import chain
import os
import pickle
import logging
# %%
def process(DATA_PATH, HOTPOTQA_PATH, suffix='train'):
    with open(HOTPOTQA_PATH, 'r') as fin:
        dataset = json.load(fin)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    batches = []
    for data in tqdm(dataset):
        try:
            flag_ans = False
            question = [tokenizer.cls_token] + tokenizer.tokenize(data['question']) + ['yes', 'no', tokenizer.sep_token]
            q, q_property = [question], [[('relevance', 1), ('blk_type', 0)]]
            if suffix != 'test':
                if data['answer'] in ['yes', 'no']:
                    pos = len(question) - 2 - (data['answer'] == 'yes')
                    q_property[0].extend([('start', pos), ('end', pos)])
                    flag_ans = True
            else:
                q_property[0].append(('_id', data['_id']))

            d, properties = [], []
            for entity, sentences in data['context']:
                bgn_idx = len(d)
                for sen_idx, sen in enumerate(sentences):
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
                batches.append((q, q_property, d, properties))
            else: 
                logging.warning((data['_id'], data['question']))

    with open(os.path.join(DATA_PATH, 'hotpotqa_{}_{}.pkl'.format(suffix, DEFAULT_MODEL_NAME)), 'wb') as fout:
        pickle.dump(batches, fout)

HOTPOTQA_PATH = '/home/mingding/cognew/hotpot_dev_distractor_v1.json'
DATA_PATH = './'
process(DATA_PATH, HOTPOTQA_PATH, 'test')

# %%
