# %%
import os
from collections import defaultdict

path = '/home/mingding/hindex'
edges = defaultdict(list)
h, node_topic = defaultdict(int), defaultdict(list)
topic_dict = {}
with open(os.path.join(path, 'AMiner-Author.txt'), 'r') as fin:
    # with open(os.path.join(path, 'aminer-hindex.nodelabel'), 'w') as fout:
        n = 1
        topic_cnt = 0
        for line in fin:
            if line.startswith('#index'):
                idx = int(line.split()[-1])
                if idx != n:
                    raise ValueError('Missing index {}, {}'.format(n, idx))
            if line.startswith('#hi'):
                hindex = int(line.split()[-1])
                h[n] = hindex
                # fout.write('{} {}\n'.format(t, hindex))
                n += 1
            if line.startswith('#t'):
                topics = line[3:].lower().strip().split(';')
                for topic in topics:
                    if topic not in topic_dict:
                        topic_dict[topic] = topic_cnt
                        topic_cnt += 1
                    node_topic[n].append(topic)
        n -= 1
with open(os.path.join(path, 'AMiner-Coauthor.txt'), 'r') as fin:
    # with open(os.path.join(path, 'aminer-hindex.edgelist'), 'w') as fout:
        for line in fin:
            x, y = line[1:].split()[:2]
            x, y = int(x), int(y)
            # fout.write('{} {}\n'.format(x, y))
            edges[x].append(y)
            edges[y].append(x)
# %%
import gensim.downloader as api
word_vectors = api.load("glove-wiki-gigaword-100")
# %%
import random
import numpy as np
def bfs(starts, num):
    global n, edges
    vis, q = set(starts), starts
    w = 0
    def push(x):
        q.append(x)
        vis.add(x)
    while len(q) < num:
        if w == len(q):
            seed = random.randint(1, n)
            while seed in vis:
                seed = random.randint(1, n)
            push(seed)
            continue
        x = q[w]
        w += 1
        for y in edges[x]:
            if not y in vis:
                push(y)
    return q[:num]

def export(nodes, identifier=None):
    if identifier is None:
        identifier = len(nodes)
    with open(os.path.join(path, 'aminer_hindex_{}.nodelabel'.format(identifier)), 'w') as fout:
        for x in nodes:
            fout.write('{} {}\n'.format(x, h[x]))
    with open(os.path.join(path, 'aminer_hindex_{}.edgelist'.format(identifier)), 'w') as fout:
        node_set = set(nodes)
        for x in nodes:
            for y in edges[x]:
                if y in node_set and y > x:
                    fout.write('{} {}\n'.format(x, y))
    with open(os.path.join(path, 'aminer_hindex_{}.nodefeatures'.format(identifier)), 'w') as fout:
        for x in nodes:
            tmp = np.zeros(100)
            tmp_cnt = 0
            for topic in node_topic[x]:
                for part in topic.strip().split():
                    try:
                        tmp += word_vectors[part]
                        tmp_cnt += 1
                    except KeyError as e:
                        print(e)
            if tmp_cnt > 0:
                tmp /= tmp_cnt
            tmp_str = np.array2string(tmp, formatter={'float_kind':'{0:.3f}'.format}, max_line_width=100000)
            fout.write('{} {}\n'.format(x, tmp_str[1:-1]))

export(bfs([], 5000), 'rand1_5000')

# %%
top_scholars = sorted([(len(edges[k]), k) for k in edges.keys()], reverse=True)
export(bfs([top_scholars[0][1]], 5000), 'top1_5000')

selected = random.sample(top_scholars[:200], 10)
export(bfs([k for v, k in selected], 5000), 'rand20intop200_5000')





# %%
