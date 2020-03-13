CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63 # The max length of an episode
BLOCK_MIN = 10 # The min length of an episode

def convert_caps(s):
    ret = []
    for word in s.split():
        if word[0].isupper():
            ret.append('<pad>')
        ret.append(word)
    return ' '.join(ret).lower()    

