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
import string


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
invalid_chrs = set(string.punctuation + string.whitespace)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
train_batches, test_json, test_batches, cnt = [], {}, [], 0
for data in tqdm(dataset['data']):
    # sentences = re.split(r'\n+', data['text'])
    last_newline, sentences, sen_offsets = -1, [], []
    for i, ch in enumerate(data['text']+'\n'):
        if ch == '\n':
            if last_newline + 1 < i:
                sentences.append(data['text'][last_newline + 1: i])
                sen_offsets.append(i)
            last_newline = i
    tokenized_sentences = [tokenizer.tokenize(sen) for sen in sentences]
    article_buf = None

    for i, raw_q in enumerate(data['questions']):
        try:
            flag_ans = False
            qid = f'{data["storyId"]}_{i}'
            if 'isQuestionBad' in raw_q and raw_q['isQuestionBad'] > 0 or 'badQuestion' in raw_q['consensus']:
                continue
            question = [tokenizer.cls_token] + tokenizer.tokenize('None ' + raw_q['q'])
            q, q_property = [question], [[('relevance', 3), ('blk_type', 0), ('_id', qid)]]

            d_properties = [[] for sen in sentences]
            if 'noAnswer' in raw_q['consensus']:
                q_property[0].extend([('start', 1, 1), ('end', 1, 1)])
                flag_ans = True
            elif 's' in raw_q['consensus']:
                s, e = raw_q['consensus']['s'], raw_q['consensus']['e']
                while e > s and data['text'][e - 1] in invalid_chrs:
                    e -= 1
                if s >= e:
                    logging.warning(f"the span is invalid. {qid} {data['text'][s:e]}")
                    continue
                # find relevance
                start_sen_idx, end_sen_idx = bisect_left(sen_offsets, s), bisect_left(sen_offsets, e)
                if start_sen_idx != end_sen_idx:
                    logging.warning(f"s and e are not in the same sentence. {qid} {data['text'][s:e]}")
                    continue
                else:
                    d_properties[start_sen_idx].append(('relevance', 3))
                    ret = find_start_end_after_tokenized(tokenizer, tokenized_sentences[start_sen_idx], [data['text'][s:e]])
                    if ret is not None:
                        ss, ee = ret[0]
                        d_properties[start_sen_idx].extend([('start', ss, 1), ('end', ee, 1)])
                        flag_ans = True
                    else:
                        logging.warning(f"remapping fails. {qid} {data['text'][s:e]}")
        except Exception as err:
            if isinstance(err, KeyboardInterrupt):
                raise KeyboardInterrupt
            raise err
            logging.error((qid, err))
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
                    test_batches.append((qbuf, dbuf))
                    test_json[qid] = 'None' if 'noAnswer' in raw_q['consensus'] else (data['text'][s:e]) 
            else: 
                logging.warning((qid, raw_q['q']))

# %%
DATA_PATH = os.path.join(root_dir, 'data')
with open(os.path.join(DATA_PATH, 'newsqa_{}_{}.pkl'.format('train', DEFAULT_MODEL_NAME)), 'wb') as fout:
    pickle.dump(train_batches, fout)
with open(os.path.join(DATA_PATH, 'newsqa_{}_{}.pkl'.format('test', DEFAULT_MODEL_NAME)), 'wb') as fout:
    pickle.dump(test_batches, fout)
with open(os.path.join(DATA_PATH, 'newsqa_test.json'), 'w') as fout:
    json.dump(test_json, fout)
