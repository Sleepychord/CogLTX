import torch
import torch.nn.functional as F
from utils import CAPACITY

import random

def train_encoders(qenc, kenc, pnbuf, qbuf, device, temp=1.):
    # enc is from {id,att_mask,type_id} to {_dump, (hidden_size) tensor}

    # Key block encodings
    pbuf = pnbuf.filtered(lambda blk, idx: hasattr(blk, 'relevance'))
    psize, nsize = len(pbuf), len(pnbuf) - len(pbuf)
    kids, katt_masks, ktype_ids = pnbuf.export_as_batch() # each (key_batch_size, block_size)
    keys = F.normalize(kenc(kids, katt_masks, ktype_ids)[1], dim=1) # keys (key_batch_size, hidden_size)
    labels = torch.zeros(len(pnbuf), device=device)
    labels[:psize] = 1.

    # Query encodings
    qbuf = qbuf.merge(pbuf)
    qids, qatt_masks, qtype_ids = qbuf.export(device=device) # (capacity)
    qids = qids.view(1, -1).expand(psize, -1)
    qatt_masks = torch.zeros(psize, len(qatt_masks), device=device)
    for i, t in enumerate(qbuf.block_ends()): 
        qatt_masks[i, :t] = 1
    qtype_ids = qtype_ids.view(1, -1).expand(psize, -1)
    queries = F.normalize(qenc(qids, qatt_masks, qtype_ids)[1], dim=1) # queries (query_batch_size, hidden_size)

    # TODO hybrid bank

    # retrieval loss
    products = queries.matmul(keys.transpose(0, 1)) # (query_batch_size, key_batch_size)
    loss = -torch.sum(labels * F.log_softmax(products / temp, dim=1)) / psize
    return loss 

def train_introspector(introspector, bufs, device):
    # introspector is from {ids,att_masks,type_ids} to {(seq_len, 1) 0/1 tensor}

    max_len = max([buf.calc_size() for buf in bufs])
    inputs = torch.zeros(4, len(bufs), max_len, dtype=torch.long, device=device)
    for i, buf in enumerate(bufs):
        buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        buf.export_relevance(out=inputs[3, i])
    loss = introspector(*inputs)[0].mean()
    return loss

def train_QA_reasoner(reasoner, bufs, device):
    # reasoner is from {ids,att_masks,type_ids} to {(seq_len, 2) tensor}

    max_len = max([buf.calc_size() for buf in bufs])
    inputs = torch.zeros(5, len(bufs), max_len, dtype=torch.long, device=device)
    for i, buf in enumerate(bufs):
        buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        buf.export_start_end(out=(inputs[3, i], inputs[4, i]))
    loss = reasoner(*inputs)[0].mean()
    return loss

def infer_QA_reason(reasoner, buf, device, top_k=3):
    inputs = torch.zeros(3, 1, buf.calc_size(), dtype=torch.long, device=device)
    buf.export(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))
    start_logits, end_logits = reasoner(*inputs)[:2]
    top_start_logits, top_start_indices = torch.topk(start_logits.squeeze(0), k=top_k)
    top_end_logits, top_end_indices = torch.topk(end_logits.squeeze(0), k=top_k)
    
    ret = []
    for start_pos in top_start_indices:
        for end_pos in top_end_indices:
            if end_pos - start_pos < 0:
                adds = -10000
            elif end_pos - start_pos > 8:
                adds = -10
            else:
                adds = 0
            ret.append((adds + start_logits[start_pos] + end_logits[end_pos], start_pos, end_pos))
    ret.sort(reverse=True)

    # backtrace
    start_pos, end_pos = ret[0][1], ret[0][2] + 1
    
    return inputs[0, 0, start_pos:end_pos]

def infer_replay(kenc, qenc, introspector, dbuf, qbuf, times=[2,2,1,1,1], device):
    '''
        times: increased number of blocks each replay.
    '''
    kids, katt_masks, ktype_ids = dbuf.export_as_batch() # each (key_batch_size, block_size)
    keys = F.normalize(kenc(kids, katt_masks, ktype_ids)[1], dim=1) # keys (key_batch_size, hidden_size)

    inputs = torch.zeros(3, 1, CAPACITY, dtype=torch.long, device=device)
    pos_set = [] # the poses of B blks in qbuf
    for inc in range(times):
        num_to_keep = len(qbuf) + inc
        qbuf.export(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))
        query = F.normalize(qenc(*inputs)[1], dim=-1)
        # batch_size of query is 1, so that the matmul becomes
        products = (keys * query.squeeze_(dim=0)).sum(dim=1)

        # fill the buffer up
        indices = products.argsort(descending=True)
        qbuf_size = qbuf.calc_size()
        for idx in indices:
            if qbuf_size + len(dbuf[idx]) > CAPACITY:
                break
            if dbuf[idx].pos in pos_set:
                continue
            qbuf_size += len(dbuf[idx])
            qbuf.insert(dbuf[idx])
        
        # keep only num_to_keep blks
        qbuf.export(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))
        relevance_token = torch.sigmoid(introspector(*inputs)[0].view(-1))

        ends = qbuf.block_ends()
        relevance_blk = torch.ones(len(ends), device='cpu')
        for i in range(1, len(ends)): # the 0-th blk is the query
            relevance_blk[i] = relevance_token[ends[i-1]:ends[i]].mean().cpu()
        keeped_indices = relevance_blk.argsort(descending=True)
        if len(keeped_indices) > num_to_keep:
            keeped_indices = keeped_indices[:num_to_keep]
        else:
            break
        qbuf = qbuf.filtered(lambda blk, idx: idx in keeped_indices)
        pos_set = []
        for blk in qbuf:
            if blk.blk_type == 1:
                pos_set.append(blk.pos)
    return qbuf

def construct_introspect_batch(dbuf, qbuf, size):
    return qbuf.marry(dbuf, size)

def construct_reasoning_batch(dbuf, qbuf, size):
    result_buf = dbuf.filtered(lambda blk, idx: hasattr(blk, 'start') or hasattr(blk, 'end'))
    other_buf = dbuf.filtered(lambda blk, idx: not hasattr(blk, 'start') and not hasattr(blk, 'end'))
    qbuf = qbuf.merge(result_buf)
    return qbuf.marry(other_buf, size, min_positive_sample=0)

def infer_supporting_facts(introspector, qbuf, top_k=5, threshold=0.5, device):
    ids, type_ids, att_masks = qbuf.export(device=device)
    relevance_token = introspector(ids.unsqueeze(0), type_ids.unsqueeze(0), att_masks.unsqueeze(0))[0].view(-1).sigmoid_()
    ends = qbuf.block_ends()
    relevance_blk = torch.ones(len(ends), device='cpu')
    for i in range(1, len(ends)):
        relevance_blk[i] = relevance_token[ends[i-1]:ends[i]].mean().cpu()
    relevances, indices = relevance_blk.sort(descending=True)
    ret = set()
    for i, idx in enumerate(indices[:top_k]):
        if relevances[i] > threshold and hasattr(qbuf[idx], 'origin'):
            ret.add(qbuf[idx].origin)
    return list(ret)