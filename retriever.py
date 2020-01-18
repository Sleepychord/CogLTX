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

from optimization import WarmupLinearLR, mocofy
from memory_bank import MemoryBank

class Retriever(pl.LightningModule):

    def __init__(self, config):
        super(Retriever, self).__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.key_encoder = AutoModel.from_pretrained(config.model_name)
        self.query_encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.query_encoder.hidden_size

    def on_epoch_start(self):
        self.device = next(self.key_encoder.parameters()).device
        self.memory_bank = MemoryBank(self.config.memory_size, self.hidden_size, device=self.device)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.query_encoder.parameters(),
    #         lr=self.config.lr_retriever,
    #         weight_decay=self.config.weight_decay_retriever
    #         )
    #     optimizer = mocofy(optimizer, self.key_encoder.parameters(), momentum=self.config.momentum)
    #     # if change to mixed optimizer, use two param_groups
    #     scheduler = WarmupLinearLR(optimizer, self.config.step_size)
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {'params': self.query_encoder.parameters()},
                {'params': self.key_encoder.parameters(), 'lr': self.config.lr2, 'weight_decay': self.config.weight_decay2}
            ]
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
    def _add_contrastive_samples(self, buf):
        with torch.no_grad():
            max_bs = self.config.max_blk_num_per_gpu * 2
            a, b, c = buf.export_as_batch(device=self.device)
            for i in range((len(batch) - 1) // max_bs + 1):
                l, r = max_bs * i, min(len(batch), max_bs * (i + 1))
                addr = self.memory_bank.get_addr(r - l)
                addr[:] = F.normalize(self.key_encoder(a[l:r], b[l:r], c[l:r])[1], dim=1)

    def training_step(self, batch, batch_idx):
        qbuf, dbuf = batch
        pbuf, nbuf = dbuf.filtered(lambda blk, idx: hasattr(blk, 'relevance'), need_residue=True)
        # Clip qbuf + pbuf to CAPACITY
        qbuf.fill_(pbuf).sort_() # is_prior=lambda blk: hasattr(blk, 'start') or hasattr(blk, 'end'))
        # Mixed Contrastive Learning
        if len(dbuf) <= self.config.max_blk_num_per_gpu:
            grad_kbuf = dbuf
        else:
            if len(pbuf) < self.config.max_blk_num_per_gpu:
                pos_lucky = [blk.pos for blk in random.sample(nbuf.blocks, self.config.max_blk_num_per_gpu - len(pbuf))]
                grad_kbuf, ungrad_kbuf = dbuf.filtered(lambda blk, idx: hasattr(blk, 'relevance') or (blk.pos in pos_lucky), need_residue=True)
            elif self.config.max_blk_num_per_gpu <= len(pbuf):
                pos_lucky = [blk.pos for blk in random.sample(pbuf.blocks, self.config.max_blk_num_per_gpu)]
                grad_kbuf, ungrad_kbuf = dbuf.filtered(lambda blk, idx: blk.pos in pos_lucky, need_residue=True)
            self._add_contrastive_samples(ungrad_kbuf)
        # Tensors about keys:
        labels = grad_kbuf.export_relevance(device=self.device, length=1, dtype=torch.float) / labels.sum() # (len(grad_kbuf),)
        kids, kattn_masks, ktype_ids = grad_kbuf.export_as_batch(device=self.device) # each (len(grad_kbuf), hidden_size)
        keys = F.normalize(kenc(kids, kattn_masks, ktype_ids)[1], dim=1)
        # Tensors about queries:
        qids, qattn_masks, qtype_ids = qbuf.export(device=self.device) # (capacity)
        qids = qids.view(1, -1).expand(len(qbuf), -1)
        qtype_ids = qtype_ids.view(1, -1).expand(len(qbuf), -1)
        qattn_masks = torch.zeros(len(qbuf), len(qattn_masks), device=self.device)
        for i, t in enumerate(qbuf.block_ends()): # TODO check if mask and short seq are equivalent, infer_replay
            qattn_masks[i, :t] = 1
        queries = F.normalize(qenc(qids, qatt_masks, qtype_ids)[1], dim=1) # queries (len(qbuf), hidden_size)
        # Contrastive loss
        grad_products = queries.matmul(keys.t()) / self.config.temperature # (len(qbuf), len(grad_kbuf))
        ungrad_products = queries.matmul(self.memory_bank.data.t()) / self.config.temperature # (len(qbuf), memory_size)
        # Products are in [-1./temp, 1./temp], minus 1./temp if necessary
        denominator_loss = torch.log(grad_products.exp().sum(dim=1) + torch.exp().sum(dim=1)).mean()
        numerator_loss = -torch.sum(labels * grad_products.mean(dim=0))
        loss = denominator_loss + numerator_loss
        # Push grad_kbuf into the memory bank
        self.memory_bank.get_addr(len(keys))[:] = keys
        tensorboard_logs = {'loss': loss, 'denominator_loss': denominator_loss, 'numerator_loss': numerator_loss}
        return {'loss': loss, 'log': tensorboard_logs}

