# %%
import gensim.downloader as api
import re
import numpy as np
from tqdm import tqdm
import logging
from gensim.summarization import bm25
def remove_special_split(blk):
    return re.sub(r'</s>|<pad>|<s>|\W', ' ', str(blk)).lower().split()
def _init_relevance_glove(qbuf, dbuf, word_vectors, threshold=0.5):
    dvecs = [
        np.stack([word_vectors[w] for w in remove_special_split(blk)]) for blk in dbuf
    ] # num_doc * sen_len * hidden_size
    qvec = np.stack([word_vectors[w] for w in remove_special_split(qbuf)])
     # num_query * query_len * hidden_size
    best = (-1, -100000)
    for i, dvec in enumerate(dvecs):
        r = np.matmul(qvec, dvec.T).mean()
        if r > best[1]:
            best = (i, r)
        if r > threshold:
            dbuf[i].relevance = max(dbuf[i].relevance, 1)
    if best[0] >= 0:
        dbuf[best[0]].relevance = max(dbuf[best[0]].relevance, 1)
def _init_relevance_bm25(qbuf, dbuf, threshold=0.15):
    docs = [remove_special_split(blk) for blk in dbuf]
    model = bm25.BM25(docs)
    scores = model.get_scores(remove_special_split(qbuf))
    max_score = max(scores)
    # print(scores)
    # print([b.relevance for b in dbuf])
    if max_score > 0:
        for i, blk in enumerate(dbuf):
            if 1 - scores[i] / max_score < threshold:
                blk.relevance = max(blk.relevance, 1)
        return True
    return False

def init_relevance(a, method='bm25'):
    print('Initialize relevance...')
    if method == 'glove':
        word_vectors = api.load("glove-wiki-gigaword-100")
        for qbuf, dbuf in tqdm(a):
            _init_relevance_glove(qbuf, dbuf, word_vectors)
    elif method == 'bm25':
        total = 0
        for qbuf, dbuf in tqdm(a):
            total += _init_relevance_bm25(qbuf, dbuf)
        print(f'Initialized {total} question-document pairs!')
    else:
        raise NotImplementedError
    
# %%
if __name__ == "__main__":
    from data_helper import *
    a = SimpleListDataset('./data/newsqa_train_roberta-large.pkl')[:100]
    init_relevance(a)

# %%
