import os
import json
import logging
from argparse import ArgumentParser
import random
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

from utils import SimpleListDataset

def main_loop(config, retriever, working_memory):
    qd_dataset = SimpleListDataset(config.train_source)
    retriever.set_dataset(qd_dataset)
    retriever.set_dataset(qd_dataset, 'test')
    for epoch in range(config.num_epochs):
        # Infer relevant blocks by MemReplay
        # TODO implement it in retriever.test_loop
        buf_dataset = retriever.results 
        working_memory.set_dataset(buf_dataset)
        # TODO Train working_memory
        # TODO Train retriever
    

