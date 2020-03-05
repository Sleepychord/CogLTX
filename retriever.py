import os
import json
import logging
from argparse import ArgumentParser
import random
import sys
import pdb
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda._utils import _get_device_index
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel

from optimization import WarmupLinearLR, mocofy
from memory_bank import MemoryBank
from memreplay import mem_replay
from utils import CAPACITY, BLOCK_SIZE
from buffer import Buffer, Block, buffer_collate



class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class Retriever(pl.LightningModule):
    def __init__(self, config):
        super(Retriever, self).__init__()
        self.config = config
        self.hparams = deepcopy(config)
        if hasattr(self.hparams, 'gpus'):
            del self.hparams.gpus
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.key_encoder = AutoModel.from_pretrained(config.model_name)
        self.query_encoder = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.query_encoder.config.hidden_size

    def on_save_checkpoint(self, checkpoint): 
        # to fix the bug of pytorch-lightning 6.0.0, will remove for future versions
        checkpoint['epoch'] += 1
        checkpoint['global_step'] += 1
        print('saved retriever!')
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
            num_workers=0)
    def forward(self, x):
        pass

    def on_epoch_start(self):
        self.device = next(self.key_encoder.parameters()).device
        self.memory_bank = MemoryBank(self.config.memory_size, self.hidden_size, device=self.device)
        self._file = open(os.path.join(self.config.tmp_dir, 'buffers_{}.tmp'.format(self.device)), 'w')
        self.step_size = len(self.train_dataset) * self.config.num_epochs

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

    @pl.data_loader
    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset)
        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0,
            collate_fn=buffer_collate
        )
        logging.info('train_dataset reloaded in Retriever.')
        return loader

    def _add_contrastive_samples(self, buf):
        with torch.no_grad():
            max_bs = self.config.max_key_num_per_gpu * 2
            a, b, c = buf.export_as_batch(device=self.device, add_cls=True)
            poses = torch.tensor([blk.pos for blk in buf], dtype=torch.long, device='cpu')
            for i in range((len(buf) - 1) // max_bs + 1):
                l, r = max_bs * i, min(len(buf), max_bs * (i + 1))
                data_addr, pos_addr = self.memory_bank.get_addr(r - l)
                data_addr[:] = F.normalize(self.key_encoder(a[l:r], b[l:r], c[l:r])[1], dim=1)
                pos_addr[:] = poses[l:r]

    def _write_buffers(self, bufs):
        for buf in bufs:
            for blk in buf:
                self._file.write(f'{blk.pos} ')
            self._file.write('\n')
    def _construct_buffer_for_reasoning(self, base_buf, nbuf, bank_products, k=10):
        # base_buf = qbuf.clone().fill_(pbuf)
        if base_buf.calc_size() > CAPACITY - BLOCK_SIZE: # Full
            self._write_buffers([base_buf])
            return
        if self.config.easy_size > 0:
            bufs = base_buf.marry(nbuf, self.config.easy_size)
            self._write_buffers(bufs)
        if self.config.hard_size > 0:
            indices = bank_products[0].topk(k)[-1]
            # Only saved positions in the memory bank, construct temporal blocks
            tmp_nbuf = Buffer()
            scapegoat = torch.zeros(BLOCK_SIZE + 1, dtype=torch.long)
            for pos in self.memory_bank.pos[indices]:
                tmp_nbuf.insert(Block(scapegoat, pos))
            bufs = base_buf.marry(tmp_nbuf, self.config.hard_size)
            self._write_buffers(bufs)


    def training_step(self, batch, batch_idx):
        assert len(batch) == 1
        qbuf, dbuf = batch[0]
        pbuf, nbuf = dbuf.filtered(lambda blk, idx: hasattr(blk, 'relevance'), need_residue=True)
        # Clip qbuf + pbuf to CAPACITY
        qbuf.fill_(pbuf) # is_prior=lambda blk: hasattr(blk, 'start') or hasattr(blk, 'end'))
        # Mixed Contrastive Learning
        if len(dbuf) <= self.config.max_key_num_per_gpu:
            grad_kbuf = dbuf
        else:
            if len(pbuf) < self.config.max_key_num_per_gpu:
                pos_lucky = [blk.pos for blk in random.sample(nbuf.blocks, self.config.max_key_num_per_gpu - len(pbuf))]
                grad_kbuf, ungrad_kbuf = dbuf.filtered(lambda blk, idx: hasattr(blk, 'relevance') or (blk.pos in pos_lucky), need_residue=True)
            elif self.config.max_key_num_per_gpu <= len(pbuf):
                pos_lucky = [blk.pos for blk in random.sample(pbuf.blocks, self.config.max_key_num_per_gpu)]
                grad_kbuf, ungrad_kbuf = dbuf.filtered(lambda blk, idx: blk.pos in pos_lucky, need_residue=True)
            self._add_contrastive_samples(ungrad_kbuf)

        # Tensors about keys:
        labels = grad_kbuf.export_relevance(device=self.device, length=1, dtype=torch.float) # (len(grad_kbuf),)
        labels /= labels.sum()
        kids, kattn_masks, ktype_ids = grad_kbuf.export_as_batch(device=self.device, add_cls=True) # each (len(grad_kbuf), hidden_size)
        keys = F.normalize(self.key_encoder(kids, kattn_masks, ktype_ids)[1], dim=1)
        
        # Tensors about queries:
        qids, qattn_masks, qtype_ids = qbuf.export(device=self.device) # (capacity)
        ends, query_num = qbuf.block_ends(), len(qbuf)
        if query_num > self.config.max_query_num_per_gpu:
            query_num = self.config.max_query_num_per_gpu
            ends = random.sample(ends, query_num)
        qids = qids.view(1, -1).expand(query_num, -1)
        qtype_ids = qtype_ids.view(1, -1).expand(query_num, -1)
        qattn_masks = torch.zeros(query_num, len(qattn_masks), device=self.device)
        for i, t in enumerate(ends):
            # TODO check if mask and short seq are equivalent, infer_replay
            qattn_masks[i, :t] = 1
        queries = F.normalize(self.query_encoder(qids, qattn_masks, qtype_ids)[1], dim=1) # queries (query_num, hidden_size)
        # Contrastive loss
        grad_products = queries.matmul(keys.t()) / self.config.temperature # (query_num, len(grad_kbuf))
        ungrad_products = queries.matmul(self.memory_bank.data.t()) / self.config.temperature # (query_num, memory_size)
        # Products are in [-1./temp, 1./temp], minus 1./temp if necessary
        loss_denominator = torch.log(grad_products.exp().sum(dim=1) + ungrad_products.exp().sum(dim=1)).mean()
        loss_numerator = -torch.sum(labels * grad_products.mean(dim=0))
        loss = loss_denominator + loss_numerator
        # construct reasoning buffer:
        self._construct_buffer_for_reasoning(qbuf, nbuf, ungrad_products.detach()) # TODO hyperparam topk
        
        # saved for on_after_backward hook, cannot modify memory_bank to influence backward
        self.unpushed_states = (keys.detach(), grad_kbuf)

        tensorboard_logs = {'loss': loss, 'loss_denominator': loss_denominator, 'loss_numerator': loss_numerator}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def on_after_backward(self):
        # Push grad_kbuf into the memory bank
        keys, grad_kbuf = self.unpushed_states
        data_addr, pos_addr = self.memory_bank.get_addr(len(keys))
        data_addr[:] = keys
        pos_addr[:] = torch.tensor([blk.pos for blk in grad_kbuf], dtype=torch.long, device=self.device)

    @staticmethod
    def add_specific_args(parser):
        parser.add_argument('--memory_size', type=int, default=512, help="num blocks in memory bank")
        parser.add_argument('--lr1', type=float, default=1e-3, help='learning rate of query encoder')
        parser.add_argument('--weight_decay1', type=float, default=0, help='weight decay of query encoder')
        parser.add_argument('--lr2', type=float, default=5e-4, help='learning rate of key encoder')
        parser.add_argument('--weight_decay2', type=float, default=0, help='weight decay of key encoder')
        parser.add_argument('--max_key_num_per_gpu', type=int, default=4, help='gradient batch_size')
        parser.add_argument('--max_query_num_per_gpu', type=int, default=4, help='query batch_size')
        parser.add_argument('--temperature', type=float, default=1., help='temperature in softmax')
        parser.add_argument('--easy_size', type=int, default=1, help='num easy buffer for reasoning per qd sample')
        parser.add_argument('--hard_size', type=int, default=1, help='num hard buffer for reasoning per qd sample')