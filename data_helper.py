
import pickle
import os
import re

import torch
from torch.utils.data import Dataset

from buffer import Buffer

class SimpleListDataset(Dataset):
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

class BlkPosInterface:
    def __init__(self, dataset):
        assert isinstance(dataset, SimpleListDataset)
        self.d = {}
        self.dataset = dataset
        for bufs in dataset:
            for buf in dataset:
                for blk in buf:
                    assert blk.pos not in self.d
                    self.d[blk.pos] = blk
    def set_property(self, pos, key, value=None):
        blk = self.d[pos]
        if value is not None:
            setattr(blk, key, value)
        elif hasattr(blk, key):
            delattr(blk, key)
    def apply_changes_from_file(self, filename):
        with open(filename, 'r') as fin:
            for line in fin:
                tmp = line.split()
                if tmp[-1].isdecimal():
                    tmp[-1] = int(tmp[-1])
                self.set_property(*tmp)
    def apply_changes_from_dir(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('changes_'):
                self.apply_changes_from_file(filename)
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))
    def build_buffer_dataset_from_file(self, filename, ret=None):
        if ret is None:
            ret = []
        with open(filename, 'r') as fin:
            for line in fin:
                buf = Buffer()
                buf.blocks = [self.d[int(s)] for s in line.split()]
                ret.append(buf)
        return ret
    def build_buffer_dataset_from_dir(self, tmp_dir):
        ret = []
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('buffers_'):
                ret = self.build_buffer_dataset_from_file(filename, ret)
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))
        return SimpleListDataset(ret)

def find_lastest_checkpoint(checkpoints_dir):
    lastest = (-1, filename)
    for shortname in os.listdir(checkpoints_dir):
        m = re.match(r'_ckpt_epoch_(\d+).+', shortname)
        if m is not None and int(m.group(1)) > lastest[0]:
            lastest = (int(m.group(1)), shortname)
    return os.path.join(checkpoints_dir, lastest[-1])