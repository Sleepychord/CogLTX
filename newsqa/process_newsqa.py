# %%
import re
import json
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer
from itertools import chain
import os
import sys
import pickle
import logging
import pdb
from bisect import bisect_left

root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)
from buffer import Buffer, Block
from utils import DEFAULT_MODEL_NAME, convert_caps
from hotpotqa.cogqa_utils import find_start_end_after_tokenized
# %%
data_dir = '/home/mingding/newsqa/maluuba/newsqa/'
output_path = './data'
with open(os.path.join(data_dir, 'combined-newsqa-data-v1.json'), 'r') as fin:
    dataset = json.load(fin)

# %%
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
train_batches, test_json, cnt = [], {}, 0
for data in tqdm(dataset['data'][:500]):
    sentences = data['text'].split('\n\n\n\n')
    sen_offsets = []
    for i, sen in enumerate(sentences):
        sen_offsets.append(len(sen) + 4 + (sen_offsets[-1] if i > 0 else -1))
    tokenized_sentences = [tokenizer.tokenize(sen) for sen in sentences]
    article_buf = None

    for i, raw_q in enumerate(data['questions']):
        try:
            flag_ans = False
            qid = f'{data["storyId"]}_{i}'
            if 'badQuestion' in raw_q['consensus']:
                continue
            question = [tokenizer.cls_token] + tokenizer.tokenize('None ' + raw_q['q'])
            q, q_property = [question], [[('relevance', 3), ('blk_type', 0), ('_id', qid)]]

            d_properties = [[] for sen in sentences]
            if 'noAnswer' in raw_q['consensus']:
                q_property[0].extend([('start', 1, 1), ('end', 1, 1)])
                flag_ans = True
            elif 's' in raw_q['consensus']:
                # find relevance
                start_sen_idx, end_sen_idx = bisect_left(sen_offsets, raw_q['consensus']['s']), bisect_left(sen_offsets, raw_q['consensus']['e'] - 1)
                if start_sen_idx != end_sen_idx:
                    logging.warning(f"s and e are not in the same sentence. {qid} {data['text'][raw_q['consensus']['s']:raw_q['consensus']['e']]}")
                else:
                    d_properties[start_sen_idx].append(('relevance', 3))
                    flag_ans = True
                    s, e = raw_q['consensus']['s'], raw_q['consensus']['e']
                    s, e = find_start_end_after_tokenized(tokenizer, tokenized_sentences[start_sen_idx], [data['text'][s:e]])[0]
                    d_properties[start_sen_idx].extend([('start', s, 1), ('end', e, 1)])
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise KeyboardInterrupt
            raise e
            logging.error((qid, e))
        else:
            # pdb.set_trace()
            if flag_ans:
                qbuf, cnt = Buffer.split_document_into_blocks(q, tokenizer, properties=q_property, cnt=cnt)
                dbuf, cnt = Buffer.split_document_into_blocks(tokenized_sentences, tokenizer, properties=d_properties, cnt=cnt)
                if article_buf is None:
                    article_buf = dbuf
                else: # carefully build new dbuf with shared ids
                    for i, blk in enumerate(dbuf):
                        blk.ids = article_buf[i].ids
                if data['type'] != 'test':
                    train_batches.append((qbuf, dbuf))
                else:
                    test_json[qid] = 'None' if 'noAnswer' in raw_q['consensus'] else (data['text'][raw_q['consensus']['s']:raw_q['consensus']['e']]) 
            else: 
                logging.warning((qid, raw_q['q']))

# %%
DATA_PATH = os.path.join(root_dir, 'data')
with open(os.path.join(DATA_PATH, 'toynewsqa_{}_{}.pkl'.format('train', DEFAULT_MODEL_NAME)), 'wb') as fout:
    pickle.dump(train_batches, fout)
# with open(os.path.join(DATA_PATH, 'newsqa_test.json'), 'w') as fout:
#     json.dump(test_json, fout)
