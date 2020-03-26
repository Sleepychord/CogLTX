CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63 # The max length of an episode
BLOCK_MIN = 10 # The min length of an episode

def convert_caps(s):
    # ret = []
    # for word in s.split():
    #     if word[0].isupper():
    #         ret.append('<pad>')
    #     ret.append(word)
    # return ' '.join(ret).lower()    
    return s.lower()

import pdb
import sys
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin