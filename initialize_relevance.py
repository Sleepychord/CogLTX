# %%
import gensim.downloader as api
import re
import numpy as np
from tqdm import tqdm
import logging
from gensim.summarization import bm25
def remove_special_split(blk):
    return re.sub(r'</s>|<pad>|<s>|\W', ' ', str(blk)).lower().split()
def _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transforms=[], threshold=0.15):
    for transform_func in conditional_transforms:
        qbuf, dbuf = transform_func(qbuf, dbuf)
    dvecs = []
    for blk in dbuf:
        doc = [word_vectors[w] for w in remove_special_split(blk) if w in word_vectors]
        if len(doc) > 0:
            dvecs.append(np.stack(doc))
        else:
            dvecs.append(np.zeros((1, 100)))
    # num_doc * sen_len * hidden_size 
    qvec = np.stack([word_vectors[w] for w in remove_special_split(qbuf) if w in word_vectors])
     # num_query * query_len * hidden_size
    scores = [np.matmul(qvec, dvec.T).mean() for dvec in dvecs]
    max_score_abs = max(scores) - min(scores) + 1e-6
    # print(scores)
    # print([b.relevance for b in dbuf])
    for i, blk in enumerate(dbuf):
        if 1 - scores[i] / max_score_abs < threshold:
            blk.relevance = max(blk.relevance, 1)
    return True

def _init_relevance_bm25(qbuf, dbuf, conditional_transforms=[], threshold=0.15):
    for transform_func in conditional_transforms:
        qbuf, dbuf = transform_func(qbuf, dbuf)
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

def init_relevance(a, method='bm25', conditional_transforms=[]):
    print('Initialize relevance...')
    total = 0
    if method == 'glove':
        word_vectors = api.load("glove-wiki-gigaword-100")
        for qbuf, dbuf in tqdm(a):
            total += _init_relevance_glove(qbuf, dbuf, word_vectors, conditional_transforms)
    elif method == 'bm25':
        for qbuf, dbuf in tqdm(a):
            total += _init_relevance_bm25(qbuf, dbuf, conditional_transforms)
    else:
        pass
    print(f'Initialized {total} question-document pairs!')

# %%
if __name__ == "__main__":
    from data_helper import *
    a = SimpleListDataset('./data/newsqa_train_roberta-large.pkl')[:100]
    init_relevance(a)

# %%
