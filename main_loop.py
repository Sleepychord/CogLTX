import os
import json
import logging
from argparse import ArgumentParser
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from transformers import AutoTokenizer, AutoModel
from pytorch_lightning.logging import TensorBoardLogger

from data_helper import SimpleListDataset, BlkPosInterface, find_lastest_checkpoint
from retriever import Retriever
from working_memory import WorkingMemory
from memreplay import mem_replay

        
def main_loop(config, reasoner):
    os.makedirs(config.tmp_dir, exist_ok=True)
    retriever = Retriever(config)
    working_memory = WorkingMemory(config, reasoner)
    qd_dataset = SimpleListDataset(config.train_source)
    interface = BlkPosInterface(qd_dataset)
    retriever.set_dataset(qd_dataset)
    logger_retr = TensorBoardLogger(config.log_dir, name='retriever', version=config.version)
    logger_wkmm = TensorBoardLogger(config.log_dir, name='working_memory', version=config.version)
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

    for epoch in range(config.num_epochs):
        trainer = _create_new_trainer(epoch + 1, logger_retr)
        trainer.fit(retriever)
        buf_dataset = interface.build_buffer_dataset_from_dir(config.tmp_dir)
        working_memory.set_dataset(buf_dataset)
        # Train working_memory
        trainer = _create_new_trainer(epoch + 1, logger_wkmm)
        trainer.fit(working_memory)
        interface.apply_changes_from_dir(config.tmp_dir)

def prediction(config):
    retriever = Retriever.load_from_checkpoint(find_lastest_checkpoint(os.path.join(config.save_dir, 'retriever', f'version_{config.version}')))
    working_memory = WorkingMemory.load_from_checkpoint(find_lastest_checkpoint(os.path.join(config.save_dir, 'working_memory', f'version_{config.version}')))
    qd_dataset = SimpleListDataset(config.test_source)
    device = f'cuda:{config.gpus[0]}'
    with torch.no_grad():
        introspector = working_memory.introspector if config.introspect else None
        for qbuf, dbuf in tqdm(qd_dataset):
            buf, relevance_score = mem_replay(retriever.key_encoder, retriever.query_encoder, introspector, dbuf, qbuf, device=device) # TODO times hyperparam
            inputs = buf.export()
            output = working_memory.reasoner(*inputs)
            yield qbuf, dbuf, buf, relevance_score, inputs[0], output



def main_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), 'save_dir'), help="saving models")
    parser.add_argument("--tmp_dir", type=str, default=os.path.join(os.getcwd(), 'tmp_dir'), help="saving ddp tmp files")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.getcwd(), 'log_dir'), help="saving logs")
    parser.add_argument("--num_epochs", type=int, default=2, help="num epoch")
    parser.add_argument('--model_name', type=str, default='roberta-base', help='name of pretrained models')
    parser.add_argument('--version', type=int, default=0, help='the version to save or restore')
    parser.add_argument('--step_size', type=int, default=50000, help='the version to save or restore')


    parser.add_argument('--latent', action='store_true', help='without relevance labels')
    parser.add_argument('--introspect', action='store_true', help='with introspection')

    parser.add_argument("--gpus", type=int, nargs='+', required=True, help="available gpus")
    parser.add_argument('--train_source', type=str, help='training dataset')
    parser.add_argument('--test_source', type=str, help='test dataset')
    Retriever.add_specific_args(parser)
    WorkingMemory.add_specific_args(parser)
    return parser
    

