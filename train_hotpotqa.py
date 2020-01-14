import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from transformers import AutoTokenizer, AutoModel
from models import Introspector, QAReasoner
from utils import WarmupLinearLR, QueryDocumentDataset
from buffer import Buffer
from training import *
from cogqa_utils import find_start_end_before_tokenized
from hotpot_evaluate_utils import eval_func
import json

class QATask(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, config):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(QATask, self).__init__()
        self.config = config

        self.batch_size = config.batch_size
        self.step_size = config.step_size

        # if you specify an example input, the summary will show input/output for each layer
        # self.example_input_array = torch.rand(5, 28 * 28)

        # build model
        # TODO key input with Query
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.key_encoder = AutoModel.from_pretrained(config.key_encoder_name)
        self.query_encoder = AutoModel.from_pretrained(config.query_encoder_name)
        self.introspector = Introspector.from_pretrained(config.introspector_name)
        self.reasoner = QAReasoner.from_pretrained(config.reasoner_name)

    # ---------------------
    # TRAINING
    # ---------------------

    def training_step(self, batch, batch_idx, opt_idx):
        # --------------------- process batch ---------------------
        q, q_property, d, d_property = batch
        qbuf = Buffer.split_document_into_blocks(q, self.tokenizer, properties=q_property)
        dbuf = Buffer.split_document_into_blocks(d, self.tokenizer, properties=d_property)
        # --------------------- train different parts ---------------------
        if opt_idx == 0: # encoder
            loss = train_encoders(self.query_encoder, self.key_encoder, dbuf, qbuf, device=self.config.device)
            tqdm_dict = {'encoder_loss': loss}
        elif opt_idx == 1: # introspector
            # TODO hard marriage
            bufs = construct_introspect_batch(dbuf, qbuf, self.config.marriage_batch_size)
            loss = train_introspector(self.introspector, bufs, device=self.config.device)
            tqdm_dict = {'introspector_loss': loss}
        elif opt_idx == 2:
            bufs = construct_reasoning_batch(dbuf, qbuf, self.config.marriage_batch_size)
            loss = train_QA_reasoner(self.reasoner, bufs, device=self.config.device)
            tqdm_dict = {'reasoner_loss': loss}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output

    def validation_step(self, batch, batch_idx):
        q, q_property, d, d_property = batch
        qbuf = Buffer.split_document_into_blocks(q, self.tokenizer, properties=q_property)
        dbuf = Buffer.split_document_into_blocks(d, self.tokenizer, properties=d_property)

        buf = infer_replay(self.key_encoder, self.query_encoder, self.introspector, dbuf, qbuf, device=self.config.device)
        sp = infer_supporting_facts(self.introspector, buf, device=device)
        ids, origins = infer_QA_reason(self.reasoner, buf, device=device)
        ans = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids))

        for name, value in q_property[0]:
            if name == '_id':
                _id = value
        output = OrderedDict({
            '_id': _id,
            'ans': ans,
            'sp': sp,
        })
        
    def validation_end(self, outputs):

        sp, ans = {}, {}
        for data in outputs:
            sp[data['_id']] = data['sp']
            ans[data['_id']] = data['ans']
        with open(self.config.output_path, 'w') as fout:
            pred = {'answer': ans, 'sp': sp}
            json.dump(pred, fout)
        metrics = eval_func(pred, self.config.validation_file)
        return {'f1': metrics['f1'],
            'sp_f1': metrics['sp_f1'], 
            'joint_f1': metrics['joint_f1'],
            'log': metrics
        }

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizers, schedulers = [], []
        # TODO mirror moco optimizer
        for params in [self.key_encoder.parameters() + self.query_encoder.parameters(), 
                    self.introspector.parameters(),
                    self.reasoner.parameters()
                ]:
            optimizer = optim.Adam(params, lr=self.hparams.learning_rate)
            scheduler = WarmupLinearLR(optimizer, self.step_size)
            optimizers.append(optimizer)
            schedulers.append(scheduler)
        return optimizers, schedulers

    def __dataloader(self, source):
        # init data generators
        dataset = QueryDocumentDataset(source)
        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size = self.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        logging.info('training data loader called')
        return self.__dataloader(source=self.config.train_source)

    @pl.data_loader
    def val_dataloader(self):
        logging.info('val data loader called')
        return self.__dataloader(source=self.config.validate_source)

    @pl.data_loader
    def test_dataloader(self):
        logging.info('test data loader called')
        return self.__dataloader(source=self.config.test_source)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=10, type=int)
        # use 500 for CPU, 50000 for GPU to see speed difference
        parser.add_argument('--hidden_dim', default=50000, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=64, type=int)
        return parser