import torch
CAPACITY = 512 # Working Memory
DEFAULT_MODEL_NAME = 'roberta-base'
BLOCK_SIZE = 63 # The max length of an episode
BLOCK_MIN = 10 # The min length of an episode

from torch.optim.lr_scheduler import _LRScheduler
import pickle
class WarmupLinearLR(_LRScheduler):
    def __init__(self, optimizer, step_size, peak_percentage=0.1, min_lr=1e-5, last_epoch=-1):
        self.step_size = step_size
        self.peak_step = peak_percentage * step_size
        self.min_lr = min_lr
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ret = []
        for base_lr in self.base_lrs:
            if self.last_epoch <= self.peak_step:
                ret.append(self.min_lr + (base_lr - self.min_lr) * self.last_epoch / self.peak_step)
            else:
                ret.append(self.min_lr + max(0, (base_lr - self.min_lr) * (self.step_size - self.last_epoch) / (self.step_size - self.peak_step)))
        return ret

class QueryDocumentDataset(torch.utils.data.Dataset):
    def __init__(self, source):
        if isinstance(source, str):
            with open(source, 'rb') as fout:
                self.dataset = pickle.load()
        elif isinstance(source, list):
            self.dataset = source
    def __getitem__(self, index):
        return self.dataset[index] # (q, q_property, d, d_property)
    def __len__(self):
        return len(self.dataset)