import torch
import torch.nn.functional as F

from utils import CAPACITY
from buffer import Buffer

def _score_blocks(qbuf, relevance_token):
    ends = qbuf.block_ends()
    relevance_blk = torch.ones(len(ends), device='cpu')
    for i in range(len(ends)): 
        if qbuf[i].blk_type > 0: # query
            relevance_blk[i] = (relevance_token[ends[i-1]:ends[i]]).mean()
    return relevance_blk

def positional_smoothing(buf, relevance_blk, factor_forward=0.1, factor_backward=0.3):
    ret = torch.zeros_like(relevance_blk)
    for i, blk in enumerate(buf):
        rest = 1.   
        if i > 0 and buf[i-1].pos == blk.pos - 1:
            rest -= factor_forward
            ret[i] += relevance_blk[i-1] * factor_forward
        if i < len(buf) - 1 and buf[i+1].pos == blk.pos + 1:
            rest -= factor_backward
            ret[i] += relevance_blk[i+1] * factor_backward
        ret[i] += relevance_blk[i] * rest
        ret[i] = max(ret[i], relevance_blk[i])
    return ret

def mem_replay(introspector, qbuf, dbuf, device, times='3,5', batch_size_inference=16):
    '''
        times: increased number of blocks each replay.
    '''
    times = [int(x) for x in times.split(',')]
    inputs = torch.zeros(3, batch_size_inference, CAPACITY, dtype=torch.long, device=device)
    B_set = [] # the poses of B blks in qbuf
    for k, inc in enumerate(times):
        num_to_keep = len(qbuf) + inc
        # stage one: continuous
        estimations = torch.zeros(len(dbuf), device='cpu')
        bufs, t = qbuf.fill(dbuf), 0
        for i in range((len(bufs) - 1) // batch_size_inference + 1):
            l, r = batch_size_inference * i, min(len(bufs), batch_size_inference * (i + 1))
            for j, buf in enumerate(bufs[l:r]):
                buf.export(out=(inputs[0, j], inputs[1, j], inputs[2, j]))
            logits = introspector(*inputs[:,:r-l]).sigmoid_()
            for j, buf in enumerate(bufs[l:r]):
                estimation = _score_blocks(buf, logits[j])[len(qbuf):]
                estimations[t: t + len(estimation)] = estimation
                t += len(estimation)
        assert t == len(dbuf)

        # estimations = positional_smoothing(dbuf, estimations)
        # fill the buffer up
        indices = estimations.argsort(descending=True)
        qbuf_size = qbuf.calc_size()
        for idx in indices:
            if qbuf_size + len(dbuf[idx]) > CAPACITY:
                break
            if dbuf[idx] in B_set:
                continue
            qbuf_size += len(dbuf[idx])
            qbuf.insert(dbuf[idx])

        # keep only num_to_keep blks
        qbuf.export(out=(inputs[0, 0], inputs[1, 0], inputs[2, 0]))
        relevance_token = torch.sigmoid(introspector(*inputs[:, :1]).view(-1))
        relevance_blk = _score_blocks(qbuf, relevance_token)
        keeped_indices = relevance_blk.argsort(descending=True)
        if len(keeped_indices) > num_to_keep and k < len(times) - 1:
            keeped_indices = keeped_indices[:num_to_keep]
        else:   
            return qbuf, relevance_blk
        # manually filtering
        filtered_qbuf, filtered_relevance_blk = Buffer(), []
        for i, blk in enumerate(qbuf):
            if i in keeped_indices:
                filtered_qbuf.blocks.append(blk)
                filtered_relevance_blk.append(relevance_blk[i])
        qbuf = filtered_qbuf
        # record the blocks already in the qbuf
        B_set = [blk for blk in qbuf if blk.blk_type == 1]
    return filtered_qbuf, torch.tensor(filtered_relevance_blk)

