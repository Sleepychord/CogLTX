# %%
from gensim.summarization import bm25

def init_bm25(qbuf, dbuf):
    q = qbuf