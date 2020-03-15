from fuzzywuzzy import fuzz, process as fuzzy_process
import re
import numpy as np
from bisect import bisect_left, bisect_right
import torch
import pdb

def dp(a, b):
    """A basic Dynamic programming for Edit-distance based fuzzy matching.
    
    Args:
        a (string): source.
        b (string): the long text.
    """
    f, start = np.zeros((len(a), len(b))), np.zeros((len(a), len(b)), dtype = np.int)
    for j in range(len(b)):
        f[0, j] = int(a[0] != b[j])
        if j > 0 and b[j - 1].isalnum():
            f[0, j] += 10 
        start[0, j] = j
    for i in range(1, len(a)):        
        for j in range(len(b)):
            # (0, i-1) + del(i) ~ (start[j], j)
            f[i, j] = f[i - 1, j] + 1
            start[i, j] = start[i - 1, j]
            if j == 0:
                continue
            if f[i, j] > f[i - 1, j - 1] + int(a[i] != b[j]):
                f[i, j] = f[i - 1, j - 1] + int(a[i] != b[j])
                start[i, j] = start[i-1, j - 1]
            if f[i, j] > f[i, j - 1] + 0.5:
                f[i, j] = f[i, j - 1] + 0.5
                start[i, j] = start[i, j - 1]
    r = np.argmin(f[len(a) - 1])
    ret = [start[len(a) - 1, r], r + 1]
    score = f[len(a) - 1, r] / len(a)
    return (ret, score)

def fuzzy_find(entities, sentence, ratio = 80):
    """Try to find as much entities in sentence precisely.
    Args:
        entities (list): Candidates.
        sentence (string): The sentence to examine.
    
    Returns:
        List of tuples: (entity, match span, start position, end position, score)
    """
    ret = []
    for entity in entities:
        item = re.sub(r' \(.*?\)$', '', entity).strip()
        if item == '':
            item = entity
            print(item)
        r, score = dp(item, sentence)
        if score < 0.5:
            matched = sentence[r[0]: r[1]].lower()
            final_word = item.split()[-1]
            retry = False
            while fuzz.partial_ratio(final_word.lower(), matched) < ratio:
                retry = True
                end = len(item) - len(final_word)
                while end > 0 and item[end - 1].isspace():
                    end -= 1
                if end == 0:
                    retry = False
                    score = 1
                    break
                item = item[:end]
                final_word = item.split()[-1]
            if retry:
                r, score = dp(item, sentence)
                score += 0.1
            if score >= 0.5:
                continue
            del final_word
            # from start
            retry = False
            first_word = item.split()[0]
            while fuzz.partial_ratio(first_word.lower(), matched) < ratio:
                retry = True
                start = len(first_word)
                while start < len(item) and item[start].isspace():
                    start += 1
                if start == len(item):
                    retry = False
                    score = 1
                    break
                item = item[start:]
                first_word = item.split()[0]
            if retry:
                r, score = dp(item, sentence)
                score = max(score, 1 - ((r[1] - r[0]) / len(entity)))
                score += 0.1
            if score < 0.5:
                if item.isdigit() and sentence[r[0]: r[1]] != item:
                    continue
                ret.append((entity, sentence[r[0]: r[1]], int(r[0]), int(r[1]), score))
    non_intersection = []
    for i in range(len(ret)):
        ok = True
        for j in range(len(ret)):
            if j != i:
                if not (ret[i][2] >= ret[j][3] or ret[j][2] >= ret[i][3]) and ret[j][4] < ret[i][4]:
                    ok = False
                    break
                if ret[i][4] > 0.2 and ret[j][4] < 0.1 and not ret[i][1][0].isupper() and len(ret[i][1].split()) <= 3:
                    ok = False
                    break
        if ok:
            non_intersection.append(ret[i][:4])
    return non_intersection

def find_start_end_after_tokenized(tokenizer, tokenized_text, spans: ['Obama Care', '2006']):
    """Find start and end positions of untokenized spans in tokenized text.
    
    Args:
        tokenizer (Tokenizer): Word-Piece tokenizer.
        tokenized_text (list): List of word pieces(string). 
        spans (list): list of untokenized spans(string).
    
    Returns:
        list: List of (start position, end position).
    """
    end_offset, ret = [], []
    for x in tokenized_text:
        offset = end_offset[-1] if len(end_offset) > 0 else -1
        if x != '<pad>':
            offset += len(x)
        end_offset.append(offset)
    text = ''.join(tokenized_text).replace('<pad>', '')
    for span in spans:
        t = ''.join(tokenizer.tokenize(span)).replace('<pad>', '')
        start = text.find(t)
        if start >= 0:
            end = start + len(t) - 1 # include end
        else:
            result = fuzzy_find([t], text)
            if len(result) == 0:
                return None # cannot find exact match
            _, _, start, end = result[0]
            end -= 1
        ret.append((bisect_left(end_offset, start), bisect_left(end_offset, end)))
    return ret

def find_start_end_before_tokenized(tokenizer, orig_text, span: 'Obama[unk]care'):
    """Find start and end positions of tokenized spans in untokenized text.
    
    Args:
        orig_text (string): Original text.
        spans (list): List of list of word pieces, as showed in example.
    
    Returns:
        list: List of (start position, end position).
    """
    if span.find(tokenizer.unk_token) > 0:
        span = span.replace(tokenizer.unk_token, '')
    # orig_text = orig_text.lower()
    start = orig_text.find(span)
    if start >= 0:
        end = start + len(span) # exclude end
    else:
        result = fuzzy_find([span], orig_text)
        if len(result) == 0:
            return None
        _, _, start, end = result[0]
    return start, end