import torch
import torch.nn.functional as F

from utils import CAPACITY

def mem_replay(kenc, qenc, introspector, dbuf, qbuf, times=[2,2,1,1,1], device):
    '''
        times: increased number of blocks each replay.
    '''
    # TODO Here we assume the key embeddings can be generated in an one-time feed, FIX ME.
    kids, katt_masks, ktype_ids = dbuf.export_as_batch(device=device) # each (key_batch_size, block_size)
    keys = F.normalize(kenc(kids, katt_masks, ktype_ids)[1], dim=1) # keys (key_batch_size, hidden_size)

    inputs = torch.zeros(3, 1, CAPACITY, dtype=torch.long, device=device)
    B_set = [] # the poses of B blks in qbuf
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
            if dbuf[idx] in B_set:
                continue
            qbuf_size += len(dbuf[idx])
            qbuf.insert(dbuf[idx])
        
        # if introspector is not ready
        if introspector is None:
            return qbuf

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
        B_set = []
        for blk in qbuf:
            if blk.blk_type == 1:
                B_set.append(blk)
    return qbuf

