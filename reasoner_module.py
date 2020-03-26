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
from models import *
from utils import CAPACITY, ForkedPdb
from buffer import buffer_collate

class ReasonerModule(pl.LightningModule):

    def __init__(self, config):
        super(ReasonerModule, self).__init__()
        self.config = config
        self.hparams = deepcopy(config)
        if hasattr(self.hparams, 'gpus'):
            del self.hparams.gpus
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        reasnoer_config = dict([(k[16:], v) for k,v in config.__dict__.items() if k.startswith('reasoner_config_')])
        self.reasoner = eval(config.reasoner_cls_name).from_pretrained(config.model_name, **reasnoer_config)

    def on_save_checkpoint(self, checkpoint): 
        # to fix the bug of pytorch-lightning 6.0.0, will remove for future versions
        checkpoint['epoch'] += 1
        checkpoint['global_step'] += 1
        print('saved reasoner!')

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
        self.device = next(self.reasoner.parameters()).device
        self._file = open(os.path.join(self.config.tmp_dir, 'changes_{}.txt'.format(self.device)), 'w')

    def on_epoch_end(self):
        self._file.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.reasoner.parameters(),
            lr=self.config.lr2,
            weight_decay=self.config.weight_decay2
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
            batch_size=self.config.batch_size_reason_per_gpu,
            shuffle=False,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=buffer_collate
        )
        logging.info('train_dataset reloaded in Reasoner.')
        return loader

    def _write_changes(self, blk, key, value):
        self._file.write('{} {} {}\n'.format(blk.pos, key, value))

    def _intervention(self, bufs, labels, crucials, loss_reasoner):
        loss_reasoner = loss_reasoner.detach()
        with torch.no_grad():
            max_bs = self.config.batch_size_reason_per_gpu * 4
            max_blk_num = max([len(buf) for buf in bufs])
            for i in range(len(bufs)):
                ids, attn_masks, type_ids = bufs[i].export(device=self.device)
                bs = len(bufs[i]) - len(crucials[i])
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
                # ForkedPdb().set_trace()
                # if bs > max_bs, we cannot feed the inputs directly.
                losses = []
                for j in range((bs - 1) // max_bs + 1): 
                    l, r = max_bs * j, min(bs, max_bs * (j + 1))
                    result = self.reasoner(ids[l:r], attn_masks[l:r], type_ids[l:r], labels=label[l:r])
                    result = result[0] if isinstance(result, tuple) else result
                    losses.append(result)
                losses_delta = torch.cat(losses, dim=0) - loss_reasoner[i]
                # Label relevance
                t = 0
                for blk in bufs[i]:
                    if blk in crucials[i]:
                        pass
                        # self._write_changes(blk, 'relevance', 3)
                    else:
                        if losses_delta[t] >= self.config.levelup_threshold and blk.relevance < 2: # TODO topk
                            self._write_changes(blk, 'relevance', blk.relevance + 1)
                        elif losses_delta[t] <= self.config.leveldown_threshold and blk.relevance > -1:
                            self._write_changes(blk, 'relevance', blk.relevance - 1)
                        t += 1
                assert t == bs

    def training_step(self, bufs, batch_idx):
        # Make inputs for reasoner
        inputs = torch.zeros(3, len(bufs), CAPACITY, dtype=torch.long, device=self.device)
        for i, buf in enumerate(bufs):
            buf.export(out=(inputs[0, i], inputs[1, i], inputs[2, i]))
        # Extract the labels for reasoner, e.g. start and end position for QA reasoner
        labels, crucials = self.reasoner.export_labels(bufs, self.device) # TODO A
        result = self.reasoner(*inputs, labels=labels)
        result = result[0] if isinstance(result, tuple) else result
        loss_reasoner = result.mean()
        # Label the relevance by the current reasoner
        if self.config.latent:  
            self._intervention(bufs, labels, crucials, result)
        tensorboard_logs = {'loss': loss_reasoner}
        return {'loss': loss_reasoner, 'log': tensorboard_logs}

    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--lr2', type=float, default=1e-4, help='learning rate of reasoner')
        parser.add_argument('--weight_decay2', type=float, default=0, help='weight decay of reasoner')
        parser.add_argument('--batch_size_reason_per_gpu', type=int, default=4, help='gradient batch_size')
        parser.add_argument('--levelup_threshold', type=float, default=0.2, help='gradient batch_size')
        parser.add_argument('--leveldown_threshold', type=float, default=-0.05, help='gradient batch_size')
