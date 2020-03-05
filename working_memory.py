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

class WorkingMemory(pl.LightningModule):

    def __init__(self, config):
        super(WorkingMemory, self).__init__()
        self.config = config
        self.hparams = deepcopy(config)
        if hasattr(self.hparams, 'gpus'):
            del self.hparams.gpus
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.introspector = Introspector.from_pretrained(config.model_name)
        self.reasoner = eval(config.reasoner_cls_name).from_pretrained(config.model_name)

    def on_save_checkpoint(self, checkpoint): 
        # to fix the bug of pytorch-lightning 6.0.0, will remove for future versions
        checkpoint['epoch'] += 1
        checkpoint['global_step'] += 1
        print('saved working memory!')

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
        if config.latent:
            self._file = open(os.path.join(self.config.tmp_dir, 'changes_{}.tmp'.format(self.device)), 'w')

    def on_epoch_end(self):
        self._file.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {'params': self.introspector.parameters()},
                {'params': self.reasoner.parameters(), 'lr': self.config.lr4, 'weight_decay': self.config.weight_decay4}
            ],
            lr=self.config.lr3,
            weight_decay=self.config.weight_decay3
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
            batch_size=self.config.max_reason_num_per_gpu,
            shuffle=False,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=buffer_collate
        )
        logging.info('train_dataset reloaded in Working Memory.')
        return loader

    def _write_changes(self, blk, key, value):
        if value is None:
            if hasattr(blk, key):
                delattr(blk, key)
                self._file.write('{} {} {}\n'.format(blk.pos, key, value))
        else:
            if not hasattr(blk, key) or getattr(blk, key) != value:
                setattr(blk, key, value)
                self._file.write('{} {}\n'.format(blk.pos, key))

    def _intervention(self, bufs, labels, crucials, loss_reasoner):
        loss_reasoner = loss_reasoner.detach()
        with torch.no_grad():
            max_bs = self.config.max_reason_num_per_gpu * 2
            max_blk_num = max([len(buf) for buf in bufs])
            for i in range(len(bufs)):
                ids, attn_masks, type_ids = bufs[i].export(device=self.device)
                bs = len(bufs[i]) - len(crucial_blks[i])
                # Make inputs by expand with different attention masks
                ids = ids.view(1, -1).expand(bs, -1)
                type_ids = type_ids.view(1, -1).expand(bs, -1)
                attn_masks = attn_masks.view(1, -1).repeat(bs, 1)
                label = labels[i].view(1, -1).expand(bs, -1)
                blk_start, t = 0, 0
                for blk in bufs[i]:
                    blk_end = blk_start + len(blk)
                    if blk not in crucials[i]:
                        attn_masks[t, blk_start: blk_end].zero_()
                        t += 1
                    blk_start = blk_end
                assert t == bs
                # if bs > max_bs, we cannot feed the inputs directly.
                losses = []
                for j in range((bs - 1) // max_bs + 1): 
                    l, r = max_bs * i, min(bs, max_bs * (i + 1))
                    losses.append(self.reasoner(ids[l:r], attn_masks[l:r], type_ids[l:r], labels=label[l:r]))
                losses_delta = torch.cat(losses, dim=0) - loss_reasoner[i]
                # Label relevance
                t = 0
                for blk in bufs[i]:
                    if blk in crucials[i]:
                        self._write_changes(blk, 'relevance', 1)
                    else:
                        if losses_delta[t] >= self.config.relevance_threshold: # TODO topk
                            self._write_changes(blk, 'relevance', 1)
                        else:
                            self._write_changes(blk, 'relevance', None)
                        t += 1

    def training_step(self, bufs, batch_idx):
        # Make inputs for reasoner
        inputs = torch.zeros(4, len(bufs), CAPACITY, dtype=torch.long, device=self.device)
        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        # Train the introspector after labeling
        for i, buf in enumerate(bufs):
            buf.export_relevance(device=self.device, out=inputs[3, i])

        # Extract the labels for reasoner, e.g. start and end position for QA reasoner
        labels, crucials = self.reasoner.export_labels(bufs, self.device) # TODO A
        loss_reasoner = self.reasoner(*inputs[:3], labels=labels).mean()
        # Label the relevance by the current reasoner
        if self.config.latent:
            self._intervention(bufs, labels, crucials, loss_reasoner)

        loss_introspector = self.introspector(*inputs[:3], labels=inputs[3]) if self.config.introspect else 0
        loss = loss_introspector + loss_reasoner
        tensorboard_logs = {'loss': loss, 'loss_introspector': loss_introspector, 'loss_reasoner': loss_reasoner}
        return {'loss': loss, 'log': tensorboard_logs}

    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--lr3', type=float, default=1e-4, help='learning rate of introspector')
        parser.add_argument('--weight_decay3', type=float, default=0, help='weight decay of introspector')
        parser.add_argument('--lr4', type=float, default=1e-4, help='learning rate of reasoner')
        parser.add_argument('--weight_decay4', type=float, default=0, help='weight decay of reasoner')
        parser.add_argument('--max_reason_num_per_gpu', type=int, default=2, help='gradient batch_size')

