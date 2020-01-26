import torch
CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63 # The max length of an episode
BLOCK_MIN = 10 # The min length of an episode

import pickle


class SimpleListDataset(torch.utils.data.Dataset):
    def __init__(self, source):
        if isinstance(source, str):
            with open(source, 'rb') as fout:
                self.dataset = pickle.load()
        elif isinstance(source, list):
            self.dataset = source
        if not isinstance(self.dataset, list):
            raise ValueError('The source of SimpleListDataset is not a list.')
    def __getitem__(self, index):
        return self.dataset[index] 
    def __len__(self):
        return len(self.dataset)
