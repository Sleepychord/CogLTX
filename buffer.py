import torch
from copy import copy
from transformers import AutoTokenizer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME
import random
from bisect import bisect_left
from itertools import chain
class Block:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    def __init__(self, ids, pos, blk_type=1, **kwargs):
        self.ids = ids
        self.pos = pos
        self.blk_type = blk_type # 0 sentence A, 1 sentence B
        self.relevance = 0
        self.estimation = 0
        self.__dict__.update(kwargs)
    def __lt__(self, rhs):
        return self.blk_type < rhs.blk_type or (self.blk_type == rhs.blk_type and self.pos < rhs.pos)
    def __ne__(self, rhs):
        return self.pos != rhs.pos or self.blk_type != rhs.blk_type
    def __len__(self):
        return len(self.ids)
    def __str__(self):
        return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))

class Buffer:
    @staticmethod
    def split_document_into_blocks(d, tokenizer, cnt=0, hard=True, properties=None):
        '''
            d: [['word', '##piece'], ...] # a document of tokenized sentences 
            properties: [
                            [
                                (name: str, value: any), # len(2) tuple, sentence level property
                                (name: str, position: int, value: any) # len(3) tuple, token level property
                            ],
                            []... # len(d) lists
                        ]
        '''
        ret = Buffer()
        updiv = lambda a,b: (a - 1) // b + 1
        if hard:
            for sid, tsen in enumerate(d):
                psen = properties[sid] if properties is not None else []
                num = updiv(len(tsen), BLOCK_SIZE) # cls
                bsize = updiv(len(tsen), num)
                for i in range(num):
                    st, en = i * bsize, min((i + 1) * bsize, len(tsen))
                    cnt += 1
                    tmp = tsen[st: en] + [tokenizer.sep_token]
                    # inject properties into blks
                    tmp_kwargs = {}
                    for p in psen:
                        if len(p) == 2:
                            tmp_kwargs[p[0]] = p[1]
                        elif len(p) == 3:
                            if st <= p[1] < en:
                                tmp_kwargs[p[0]] = (p[1] - st, p[2])
                        else:
                            raise ValueError('Invalid property {}'.format(p))
                    ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs))
        else:
            # d is only a list of tokens, not split. 
            # properties are also a list of tuples.
            end_tokens = {'\n':0, '.':1, '?':1, '!':1, ',':2}
            for k, v in list(end_tokens.items()):
                end_tokens['Ġ' + k] = v
            sen_cost, break_cost = 4, 8
            poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
            poses.insert(0, (-1, 0))
            if poses[-1][0] < len(d) - 1:
                poses.append((len(d) - 1, 0))
            x = 0
            while x < len(poses) - 1:
                if poses[x + 1][0] - poses[x][0] > BLOCK_SIZE:
                    poses.insert(x + 1, (poses[x][0] + BLOCK_SIZE, break_cost))
                x += 1
            # simple dynamic programming
            best = [(0, 0)]
            for i, (p, cost) in enumerate(poses):
                if i == 0:
                    continue    
                best.append((-1, 100000))
                for j in range(i-1, -1, -1):
                    if p - poses[j][0] > BLOCK_SIZE:
                        break
                    value = best[j][1] + cost + sen_cost
                    if value < best[i][1]:
                        best[i] = (j, value)
                assert best[i][0] >= 0
            intervals, x = [], len(poses) - 1
            while x > 0:
                l = poses[best[x][0]][0 ]
                intervals.append((l + 1, poses[x][0] + 1))
                x = best[x][0]
            if properties is None:
                properties = []
            for st, en in reversed(intervals):
                # copy from hard version
                cnt += 1
                tmp = d[st: en] + [tokenizer.sep_token]
                # inject properties into blks
                tmp_kwargs = {}
                for p in properties:
                    if len(p) == 2:
                        tmp_kwargs[p[0]] = p[1]
                    elif len(p) == 3:
                        if st <= p[1] < en:
                            tmp_kwargs[p[0]] = (p[1] - st, p[2])
                    else:
                        raise ValueError('Invalid property {}'.format(p))
                ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt, **tmp_kwargs))

        return ret, cnt

    def __init__(self):
        self.blocks = []

    def __add__(self, buf):
        ret = Buffer()
        ret.blocks = self.blocks + buf.blocks
        return ret

    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, key):
        return self.blocks[key]

    def __str__(self):
        return ''.join([str(b)+'\n' for b in self.blocks])
        
    def clone(self):
        ret = Buffer()
        ret.blocks = self.blocks.copy()
        return ret

    def calc_size(self):
        return sum([len(b) for b in self.blocks])

    def block_ends(self):
        t, ret = 0, []
        for b in self.blocks:
            t += len(b)
            ret.append(t)
        return ret

    def insert(self, b, reverse=True):
        if not reverse:
            for index in range(len(self.blocks) + 1):
                if index >= len(self.blocks) or b < self.blocks[index]:
                    self.blocks.insert(index, b)
                    break
        else:
            for index in range(len(self.blocks), -1, -1):
                if index == 0 or self.blocks[index - 1] < b:
                    self.blocks.insert(index, b)
                    break

    def merge(self, buf):
        ret = Buffer()
        t1, t2 = 0, 0
        while t1 < len(self.blocks) or t2 < len(buf):
            if t1 < len(self.blocks) and (t2 >= len(buf) or self.blocks[t1] < buf.blocks[t2]):
                ret.blocks.append(self.blocks[t1])
                t1 += 1
            else:
                ret.blocks.append(buf.blocks[t2])
                t2 += 1
        return ret
    
    def filtered(self, fltr: 'function blk, index->bool', need_residue=False):
        ret, ret2 = Buffer(), Buffer()
        for i, blk in enumerate(self.blocks):
            if fltr(blk, i):
                ret.blocks.append(blk)
            else:
                ret2.blocks.append(blk)
        if need_residue:
            return ret, ret2
        else:
            return ret
            
    def random_sample(self, size):
        assert size <= len(self.blocks)
        index = sorted(random.sample(range(len(self.blocks)), size))
        ret = Buffer()
        ret.blocks = [self.blocks[i] for i in index]
        return ret
    # def fill_(self, buf, is_prior=None):
    #     indices = list(range(len(buf)))
    #     random.shuffle(indices)
    #     # First fill the blks with priority
    #     if is_prior is not None:
    #         t = 0
    #         for i, idx in enumerate(indices):
    #             if is_prior(buf[idx]):
    #                 indices[t], indices[i] = indices[i], indices[t]
    #                 t += 1
    #     tmp_size = self.calc_size()
    #     for idx in indices:
    #         if tmp_size + len(buf[idx]) > CAPACITY:
    #             break
    #         else:
    #             tmp_size += len(buf[idx])
    #             self.insert(buf[idx])
    #     return self
    # def marry(self, buf, size):
    #     return [self.clone().fill_(buf) for i in range(size)]
    
    def sort_(self):
        self.blocks.sort()
        return self

    def fill(self, buf):
        ret, tmp_buf, tmp_size = [], self.clone(), self.calc_size()
        for blk in buf:
            if tmp_size + len(blk) > CAPACITY:
                ret.append(tmp_buf)
                tmp_buf, tmp_size = self.clone(), self.calc_size()
            tmp_buf.blocks.append(blk)
            tmp_size += len(blk)
        ret.append(tmp_buf)
        return ret

    def export(self, device=None, length=None, out=None):
        if out is None:
            if length is None:
                total_length = self.calc_size()
                if total_length > CAPACITY:
                    raise ValueError('export inputs larger than capacity')
            else:
                total_length = length * len(self.blocks)
            ids, att_masks, type_ids = torch.zeros(3, total_length, dtype=torch.long, device=device)
        else: # must be zeros and big enough
            ids, att_masks, type_ids = out
            att_masks.zero_()
        t = 0
        for b in self.blocks:
            ids[t:t + len(b)] = torch.tensor(b.ids, dtype=torch.long, device=device) # id
            # if b.blk_type == 1:
            #     type_ids[t:w] = 1 # sentence B
            att_masks[t:t + len(b)] = 1 # attention_mask
            t += len(b) if length is None else length
        return ids, att_masks, type_ids

    def export_as_batch(self, device, length=BLOCK_SIZE+1, add_cls=False):
        ids, att_masks, type_ids = self.export(device, length, add_cls=add_cls)
        return ids.view(-1, length), att_masks.view(-1, length), type_ids.view(-1, length)

    def export_relevance(self, device, length=None, dtype=torch.long, out=None):
        if out is None:
            total_length = self.calc_size() if length is None else length * len(self.blocks)
            relevance = torch.zeros(total_length, dtype=dtype, device=device)
        else:
            relevance = out
        t = 0
        for b in self.blocks:
            w = t + (len(b) if length is None else length)
            if b.relevance >= 1:
                relevance[t: w] = 1
            t = w
        return relevance

def buffer_collate(batch): # does not collate
    return batch

if __name__ == "__main__":
    s = """I just recently realized that I am bisexual, and also just recently returned to religion, and have a good friend who has pointed out to me that homosexuality is a sin in the bible.  Well, I don't see how it could be considered a sin,
First of all as far as I know, only male homosexuality is explicitly
mentioned in the bibles, so you're off the hook there, I think. In
any event, there are *plenty* of people in many denominations who
do not consider a person's sexual identification of gay/lesbian/bisexual
as an "immoral lifestyle choice"
Also, I have always been a somewhat liberal feminist, and am pro-choice, and it seems that being pro-choice and being religious don't mix either.  I am told
This is another misconception. You are not being told the whole story.
My former minister is a lesbian, and I know personally and
professionally several openly gay and lesbian ministers. I am
a Unitarian-Universalist and like most others in my denomination,
am pro-choice. You needn't go looking to the Unitarian Universalists
(which is a liberal religion) for acceptance of your sexual
identification and pro-choice views, however; there are many of us
who believe in spirituality AND freedom of conscience.
Good Luck on your journey! ADDFSDFDE*(YT(*HO*E))DHF(NKLSHDFDFSFLFJDKSFKSHOFEINLIDS)*Y&(*&(23423534twer54324524)245)4353453777777777777777777777777777777777777777777777777777777777777777777777777777777
4353453777777777777777777777777777777777777777777777777777777777777777777777777777777
"""
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    Buffer.split_document_into_blocks(tokenizer.tokenize(s), tokenizer, hard=False)
