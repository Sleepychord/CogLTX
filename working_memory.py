import os
import json
import logging
from argparse import ArgumentParser
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

from optimization import WarmupLinearLR
from models import Introspector, QAReasoner
from utils import CAPACITY
class WorkingMemory(pl.LightningModule):

    def __init__(self, config, reasoner):
        super(WorkingMemory, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.introspector = Introspector.from_pretrained(config.model_name)
        self.reasoner = reasoner

    def on_epoch_start(self):
        self.device = next(self.introspector.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {'params': self.query_encoder.parameters()},
                {'params': self.key_encoder.parameters(), 'lr': self.config.lr2, 'weight_decay': self.config.weight_decay2}
            ],
            lr=self.config.lr1,
            weight_decay=self.config.weight_decay1
            )
        scheduler = WarmupLinearLR(optimizer, self.config.step_size)

        return [optimizer], [scheduler]

    def set_dataset(self, dataset, mode='train'):
        if mode == 'train':
            self.train_dataset = dataset
        elif mode == 'val':
            self.val_dataset = dataset
        elif mode == 'test':
            self.test_dataset = dataset
        else:
            raise ValueError('No such dataset')

    def train_dataloader(self):
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        if isinstance(self.config.gpus, list) and len(self.config.gpus) > 1 or self.config.gpus > 1:
            train_sampler = DistributedSampler(self.train_dataset)
        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )
        logging.info('train_dataset reloaded.')
        return loader

    def training_step(self, bufs, batch_idx):
        # Make inputs for reasoner
        inputs = torch.zeros(4, len(bufs), CAPACITY, dtype=torch.long, device=self.device, requires_grad=self.config.latent)
        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        labels = reasoner.export_labels(bufs, device) # TODO A

        tensorboard_logs = {'loss': loss, 'denominator_loss': denominator_loss, 'numerator_loss': numerator_loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def backward(self, use_amp, loss, optimizer):
        
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

def train_introspector(introspector, bufs, device):
    # introspector is from {ids,att_masks,type_ids} to {(seq_len, 1) 0/1 tensor}

    max_len = max([buf.calc_size() for buf in bufs])
    inputs = torch.zeros(4, len(bufs), max_len, dtype=torch.long, device=device)
    for i, buf in enumerate(bufs):
        buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        buf.export_relevance(out=inputs[3, i])
    loss = introspector(*inputs)[0].mean()
    return loss

def train_QA_reasoner(reasoner, bufs, device):
    # reasoner is from {ids,att_masks,type_ids} to {(seq_len, 2) tensor}

    max_len = max([buf.calc_size() for buf in bufs])
    inputs = torch.zeros(5, len(bufs), max_len, dtype=torch.long, device=device)
    for i, buf in enumerate(bufs):
        buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        buf.export_start_end(out=(inputs[3, i], inputs[4, i]))
    loss = reasoner(*inputs)[0].mean()
    return loss