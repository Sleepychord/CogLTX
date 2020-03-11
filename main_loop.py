import os
import json
import logging
from argparse import ArgumentParser
import random
from tqdm import tqdm
import pdb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.logging import TensorBoardLogger

from data_helper import SimpleListDataset, BlkPosInterface, find_lastest_checkpoint
from introspector_module import IntrospectorModule
from reasoner_module import ReasonerModule
from memreplay import mem_replay


def main_loop(config):
    os.makedirs(config.tmp_dir, exist_ok=True)
    introspector = IntrospectorModule(config)
    reasoner = ReasonerModule(config)
    qd_dataset = SimpleListDataset(config.train_source)
    interface = BlkPosInterface(qd_dataset)
    logger_intro = TensorBoardLogger(config.log_dir, name='introspector', version=config.version)
    logger_reason = TensorBoardLogger(config.log_dir, name='reasoner', version=config.version)
    if config.latent:
        # TODO Label the initial relavance
        raise NotImplementedError
    def _create_new_trainer(epoch, logger):
        return Trainer(max_epochs=epoch, 
            gpus=config.gpus, 
            distributed_backend='ddp', 
            default_save_path=config.save_dir,
            logger=logger, 
            weights_summary=None,
            early_stop_callback=False,
            check_val_every_n_epoch=1,
        )
    if os.path.exists(os.path.join(config.save_dir, 'introspector', f'version_{config.version}')):
        min_epoch = min(find_lastest_checkpoint(os.path.join(config.save_dir, 'introspector', f'version_{config.version}', 'checkpoints'), epoch=True), find_lastest_checkpoint(os.path.join(config.save_dir, 'reasoner', f'version_{config.version}', 'checkpoints'), epoch=True)) + 1
    else:
        min_epoch = 0
    logging.info(f'Continue training at epoch {min_epoch}...')
    for epoch in range(min_epoch, config.num_epochs):
        intro_dataset = interface.build_random_buffer(num_samples=config.num_samples)
        introspector.set_dataset(intro_dataset)
        trainer = _create_new_trainer(epoch + 1, logger_intro)
        trainer.fit(introspector)

        interface.collect_estimations_from_dir(config.tmp_dir)
        reason_dataset = interface.build_promising_buffer(num_samples=config.num_samples)
        reasoner.set_dataset(reason_dataset)
        trainer = _create_new_trainer(epoch + 1, logger_reason)
        trainer.fit(reasoner)
        if config.latent:
            interface.apply_changes_from_dir(config.tmp_dir)

def prediction(config):
    device = f'cuda:{config.gpus[0]}'
    intro_model = IntrospectorModule.load_from_checkpoint(find_lastest_checkpoint(os.path.join(config.save_dir, 'introspector', f'version_{config.version}', 'checkpoints'))).to(device).eval()
    reason_model = ReasonerModule.load_from_checkpoint(find_lastest_checkpoint(os.path.join(config.save_dir, 'reasoner', f'version_{config.version}', 'checkpoints'))).to(device).eval()
    qd_dataset = SimpleListDataset(config.test_source)
    with torch.no_grad():
        for qbuf, dbuf in tqdm(qd_dataset):
            # pdb.set_trace()
            buf, relevance_score = mem_replay(intro_model.introspector, qbuf, dbuf, device=device) # TODO times hyperparam
            inputs = [t.unsqueeze(0) for t in buf.export(device=device)]
            output = reason_model.reasoner(*inputs)
            yield qbuf, dbuf, buf, relevance_score, inputs[0][0], output


def main_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), 'save_dir'), help="saving models")
    parser.add_argument("--tmp_dir", type=str, default=os.path.join(os.getcwd(), 'tmp_dir'), help="saving ddp tmp files")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), 'log_dir'), help="saving logs")
    parser.add_argument("--num_epochs", type=int, default=2, help="num epoch")
    parser.add_argument('--model_name', type=str, default='roberta-base', help='name of pretrained models')
    parser.add_argument('--version', type=int, default=0, help='the version to save or restore')
    parser.add_argument('--step_size', type=int, default=20000, help='the version to save or restore')

    parser.add_argument('--num_samples', type=str, default='2,1,3', help='num of continous, discrete random samples and promising samples')
    parser.add_argument('--batch_size_inference', type=int, default=8, help='batch_size in memreplay')
    parser.add_argument('--latent', action='store_true', help='without relevance labels')

    parser.add_argument("--gpus", type=int, nargs='+', required=True, help="available gpus")
    parser.add_argument('--train_source', type=str, help='training dataset')
    parser.add_argument('--test_source', type=str, help='test dataset')
    IntrospectorModule.add_specific_args(parser)
    ReasonerModule.add_specific_args(parser)
    return parser
    

