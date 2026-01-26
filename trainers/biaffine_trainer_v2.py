import os
from datetime import timedelta, datetime
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR

from utils.logs import logger
from utils.util_deep_learning import device
from transforms.conll import CoNLL, progress_bar
from utils.sp_data import Dataset
from models.graph_based.biaffine_parser_v2 import BiaffineParserV2
from utils.sp_metric import Metric, AttachmentMetric
from utils.sp_field import Field, SubwordField
from utils.constants import pad, unk, bos
from utils.sp_parallel import DistributedDataParallel as DDP, is_master
from utils.sp_embedding import Embedding

class BiaffineTrainerV2:
    def __init__(self, parser, corpus):
        self.parser = parser
        self.corpus = corpus

    def train(
        self, base_path: Union[Path, str],
        fix_len=20,
        min_freq=2,
        buckets=1000,
        batch_size=5000,
        lr=2e-3,
        bert_lr=2e-5, # Specific LR for BERT
        mu=.9,
        nu=.9,
        epsilon=1e-12,
        clip=5.0,
        decay=.75,
        decay_steps=5000,
        patience=100,
        max_epochs=20,
        wandb=None
    ):
        feat = self.parser.feat
        embed = self.parser.embed
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        logger.info("Building the fields")
        
        # 1. WORD Field
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        
        # 2. FEAT Field (BERT or Char)
        if feat == 'char':
            FEAT = SubwordField('chars', pad=pad, unk=unk, bos=bos, fix_len=fix_len)
        elif feat == 'bert':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.parser.bert)
            FEAT = SubwordField('bert',
                                pad=tokenizer.pad_token,
                                unk=tokenizer.unk_token,
                                bos=tokenizer.bos_token or tokenizer.cls_token,
                                fix_len=fix_len,
                                tokenize=tokenizer.tokenize)
            FEAT.vocab = tokenizer.get_vocab()
        else:
            FEAT = Field('tags', bos=bos) # Fallback, though V2 focuses on BERT

        # 3. TAG Field (Always used in V2 for BERT)
        TAG = Field('tags', bos=bos)
        
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)

        # Config Transform to include TAGS
        if feat == 'bert':
            # Structure: FORM -> (WORD, BERT), CPOS -> TAG, ...
            transform = CoNLL(FORM=(WORD, FEAT), CPOS=TAG, HEAD=ARC, DEPREL=REL)
        elif feat == 'char':
             transform = CoNLL(FORM=(WORD, FEAT), CPOS=TAG, HEAD=ARC, DEPREL=REL)
        else:
            # Simple Tag only mode? Not main focus but support it
            transform = CoNLL(FORM=WORD, CPOS=TAG, HEAD=ARC, DEPREL=REL)


        train = Dataset(transform, self.corpus.train)
        WORD.build(train, min_freq, (Embedding.load(embed, unk) if self.parser.embed else None))
        if feat != 'bert': # Bert vocab is pre-built
            FEAT.build(train)
        TAG.build(train)
        REL.build(train)
        
        n_words = WORD.vocab.n_init
        n_feats = len(FEAT.vocab)
        n_tags = len(TAG.vocab)
        n_rels = len(REL.vocab)
        
        parser = BiaffineParserV2(
            n_words=n_words,
            n_feats=n_feats,
            n_tags=n_tags,
            n_rels=n_rels,
            pad_index=WORD.pad_index,
            unk_index=WORD.unk_index,
            feat_pad_index=FEAT.pad_index,
            tag_pad_index=TAG.pad_index,
            transform=transform,
            feat=self.parser.feat,
            bert=self.parser.bert,
            n_feat_embed=100
        )
        
        parser.embeddings = self.parser.embeddings
        if hasattr(parser, 'load_pretrained'):
             parser.load_pretrained(WORD.embed).to(device)

        # OPTIMIZER SETUP WITH DIFFERENTIAL LEARNING RATES
        # Separate BERT params from the rest
        bert_params = []
        rest_params = []
        for name, param in parser.named_parameters():
             if 'feat_embed.bert' in name or 'model.bert' in name: # Check naming in BertEmbedding
                 bert_params.append(param)
             else:
                 rest_params.append(param)
        
        optimizer = Adam([
            {'params': rest_params, 'lr': lr},
            {'params': bert_params, 'lr': bert_lr}
        ], betas=(mu, nu), eps=epsilon)
        
        scheduler = ExponentialLR(optimizer, decay ** (1 / decay_steps))

        ################################################################################################################
        # TRAIN LOOP
        ################################################################################################################
        
        parser.transform.train()
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()

        train = Dataset(parser.transform, self.corpus.train)
        dev = Dataset(parser.transform, self.corpus.dev)
        test = Dataset(parser.transform, self.corpus.test)
        
        train.build(batch_size, buckets, True, dist.is_initialized())
        dev.build(batch_size, buckets)
        test.build(batch_size, buckets)
        
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")
        
        if dist.is_initialized():
            parser = DDP(parser, device_ids=[dist.get_rank()], find_unused_parameters=True)

        elapsed = timedelta()
        best_e, best_metric = 1, Metric()

        for epoch in range(1, max_epochs + 1):
            start = datetime.now()
            logger.info(f'Epoch {epoch} / {max_epochs}:')

            parser.train()
            bar = progress_bar(train.loader)
            metric = AttachmentMetric()
            
            # Note: The loader yields items based on transform fields order
            # If transform has (WORD, FEAT), TAG, ARC, REL
            # it should yield 5 tensors
            for batch in bar:
                if len(batch) == 5:
                     words, feats, tags, arcs, rels = batch
                else:
                    # Fallback or error if setup is wrong
                     raise ValueError(f"Expected 5 items from loader, got {len(batch)}")
                
                optimizer.zero_grad()
                mask = words.ne(parser.pad_index)
                mask[:, 0] = 0
                
                s_arc, s_rel = parser.forward(words, feats, tags)
                loss = parser.forward_loss(s_arc, s_rel, arcs, rels, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(parser.parameters(), clip)
                optimizer.step()
                scheduler.step()

                arc_preds, rel_preds = parser.decode(s_arc, s_rel, mask)
                metric(arc_preds, rel_preds, arcs, rels, mask)
                bar.set_postfix_str(f'lr: {scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}')

            # Evaluate
            dev_loss, dev_metric = parser.evaluate(dev.loader)
            logger.info(f"{'dev:':6} - loss: {dev_loss:.4f} - {dev_metric}")
            test_loss, test_metric = parser.evaluate(test.loader)
            logger.info(f"{'test:':6} - loss: {test_loss:.4f} - {test_metric}")
            if wandb:
                wandb.log({"test_loss": test_loss})
                wandb.log({"test_metric_uas": test_metric.uas})
                wandb.log({"test_metric_las": test_metric.las})
            
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    parser.save(base_path)
                logger.info(f'Saved best model at epoch {epoch}')
            
            elapsed += datetime.now() - start
            if epoch - best_e >= patience:
                break
        
        logger.info(f"Training Done. Best Dev: {best_metric}")
