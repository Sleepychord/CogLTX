import os
import json
import logging
from argparse import ArgumentParser
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

from optimization import WarmupLinearLR
from models import Introspector, QAReasoner
from utils import CAPACITY
from buffer import buffer_collate
from memreplay import _score_blocks

class IntrospectorModule(pl.LightningModule):

    def __init__(self, config):
        super(IntrospectorModule, self).__init__()
        self.config = config
        self.hparams = deepcopy(config)
        if hasattr(self.hparams, 'gpus'):
            del self.hparams.gpus
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.introspector = Introspector.from_pretrained(config.model_name)

    def on_save_checkpoint(self, checkpoint): 
        # to fix the bug of pytorch-lightning 6.0.0, will remove for future versions
        checkpoint['epoch'] += 1
        checkpoint['global_step'] += 1
        print('saved introspector!')

    def validation_step(self, batch, batch_idx):
        pass
    def validation_end(self, outputs):
        return {'val_loss': -self.current_epoch}
    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            dataset=range(8),
            sampler=DistributedSampler(range(8)),
            batch_size=1,
            num_workers=0
        )
    def forward(self, x):
        pass

    def on_epoch_start(self):
        self.device = next(self.introspector.parameters()).device
        self._file = open(os.path.join(self.config.tmp_dir, 'estimations_{}.txt'.format(self.device)), 'w')

    def on_epoch_end(self):
        self._file.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.introspector.parameters(),
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

    @pl.data_loader
    def train_dataloader(self):
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = DistributedSampler(self.train_dataset)
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size_intro_per_gpu,
            shuffle=False,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=buffer_collate
        )
        logging.info('train_dataset reloaded in Introspector.')
        return loader

    def _write_estimation(self, buf, relevance_blk):
        for i, blk in enumerate(buf):
            self._file.write(f'{blk.pos} {relevance_blk[i].item()}\n')

    def training_step(self, bufs, batch_idx):
        # Make inputs for reasoner
        inputs = torch.zeros(4, len(bufs), CAPACITY, dtype=torch.long, device=self.device)
        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        # Train the introspector after labeling
        for i, buf in enumerate(bufs):
            buf.export_relevance(device=self.device, out=inputs[3, i])
        # Label the relevance by the current reasoner
        loss_introspector, logits = self.introspector(*inputs[:3], labels=inputs[3])
        for i, buf in enumerate(bufs):
            self._write_estimation(buf, _score_blocks(buf, torch.sigmoid(logits[i])))
        tensorboard_logs = {'loss': loss_introspector}
        return {'loss': loss_introspector, 'log': tensorboard_logs}

    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--lr1', type=float, default=1e-4, help='learning rate of introspector')
        parser.add_argument('--weight_decay1', type=float, default=0, help='weight decay of introspector')
        parser.add_argument('--batch_size_intro_per_gpu', type=int, default=4, help='gradient batch_size')

