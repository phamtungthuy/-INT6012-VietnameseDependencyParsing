# -*- coding: utf-8 -*-
"""
Trainer for Triaffine Joint Tagger REAL (with Sibling Features)

This trainer handles the TriaffineJointTaggerReal model which has:
- Joint POS Tagger Branch
- Sibling features with Triaffine scoring
"""

import os
from pathlib import Path
from typing import Union
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from utils.logs import logger
from utils.util_deep_learning import device
from transforms.conll import CoNLL, progress_bar
from utils.sp_data import Dataset
from utils.sp_field import Field, SubwordField
from utils.constants import pad, unk, bos
from utils.sp_metric import AttachmentMetric
from utils.sp_parallel import DistributedDataParallel as DDP, is_master
from models.graph_based.ablation.triaffine_jointtagger_real import TriaffineJointTaggerReal


class JointTaggerRealTrainer:
    """Trainer for Triaffine with Joint POS Tagger + Sibling Features."""
    
    def __init__(self, corpus):
        self.corpus = corpus
    
    def train(
        self,
        base_path: Union[Path, str],
        bert='vinai/phobert-base',
        embed=None,
        fix_len=20,
        min_freq=2,
        buckets=1000,
        batch_size=3000,
        lr=2e-3,
        bert_lr=5e-5,
        mu=0.9,
        nu=0.9,
        epsilon=1e-12,
        clip=5.0,
        decay=0.75,
        decay_steps=5000,
        patience=100,
        max_epochs=20,
        use_sibling=True  # NEW: Toggle sibling features
    ):
        """Train the joint tagger parser with sibling features."""
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        logger.info(f"[JointTaggerReal] Building the fields")
        logger.info(f"[JointTaggerReal] use_sibling={use_sibling}")
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert)
        
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        FEAT = SubwordField('bert',
                           pad=tokenizer.pad_token,
                           unk=tokenizer.unk_token,
                           bos=tokenizer.bos_token or tokenizer.cls_token,
                           fix_len=fix_len,
                           tokenize=tokenizer.tokenize)
        FEAT.vocab = tokenizer.get_vocab()
        
        TAG = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        
        transform = CoNLL(FORM=(WORD, FEAT), CPOS=TAG, HEAD=ARC, DEPREL=REL)
        
        # Build vocabularies
        if embed:
            from utils.sp_embedding import Embedding
            vectors = Embedding.load(embed)
        else:
            vectors = None

        train = Dataset(transform, self.corpus.train)
        WORD.build(train, min_freq, vectors)
        TAG.build(train)
        REL.build(train)
        
        n_words = len(WORD.vocab)
        n_tags = len(TAG.vocab)
        n_rels = len(REL.vocab)
        
        logger.info(f"[JointTaggerReal] Vocab - Words: {n_words}, Tags: {n_tags}, Rels: {n_rels}")
        
        # Initialize model
        parser = TriaffineJointTaggerReal(
            n_words=n_words,
            n_tags=n_tags,
            n_rels=n_rels,
            pad_index=WORD.pad_index,
            unk_index=WORD.unk_index,
            bert=bert,
            feat_pad_index=FEAT.pad_index,
            use_sibling=use_sibling,  # Pass sibling toggle
            transform=transform
        ).to(device)
        
        # Optimizer with different LR for BERT
        bert_params, other_params = [], []
        for name, param in parser.named_parameters():
            if 'bert_embed' in name:
                bert_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = AdamW([
            {'params': other_params, 'lr': lr},
            {'params': bert_params, 'lr': bert_lr}
        ], betas=(mu, nu), eps=epsilon)
        
        scheduler = ExponentialLR(optimizer, decay ** (1 / decay_steps))
        
        # Data loaders
        transform.train()
        if dist.is_initialized():
            batch_size = batch_size // dist.get_world_size()
        
        train = Dataset(transform, self.corpus.train)
        dev = Dataset(transform, self.corpus.dev)
        test = Dataset(transform, self.corpus.test)
        
        train.build(batch_size, buckets, True, dist.is_initialized())
        dev.build(batch_size, buckets)
        test.build(batch_size, buckets)
        
        logger.info(f"\n{'train:':6} {train}\n{'dev:':6} {dev}\n{'test:':6} {test}\n")
        
        if dist.is_initialized():
            parser = DDP(parser, device_ids=[dist.get_rank()], find_unused_parameters=True)
        
        # Training loop
        best_e, best_metric = 1, AttachmentMetric()
        
        for epoch in range(1, max_epochs + 1):
            logger.info(f'[JointTaggerReal] Epoch {epoch} / {max_epochs}:')
            
            parser.train()
            bar = progress_bar(train.loader)
            train_loss, n_batches = 0, 0
            
            for batch in bar:
                words, feats, tags, arcs, rels = batch
                
                mask = words.ne(parser.pad_index)
                mask[:, 0] = 0
                
                optimizer.zero_grad()
                # Joint forward: returns s_tag, s_arc, s_rel
                s_tag, s_arc, s_rel = parser.forward(words, feats, tags, use_gold_tags=True)
                loss = parser.forward_loss(s_tag, s_arc, s_rel, tags, arcs, rels, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(parser.parameters(), clip)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                n_batches += 1
                bar.set_postfix_str(f"loss: {loss:.4f}")
            
            logger.info(f"Train Loss: {train_loss / n_batches:.4f}")
            
            # Evaluate
            dev_loss, dev_tag_acc, dev_metric = self._evaluate(parser, dev.loader)
            logger.info(f"dev: loss={dev_loss:.4f} tag_acc={dev_tag_acc:.2%} - {dev_metric}")
            
            test_loss, test_tag_acc, test_metric = self._evaluate(parser, test.loader)
            logger.info(f"test: loss={test_loss:.4f} tag_acc={test_tag_acc:.2%} - {test_metric}")
            
            if dev_metric > best_metric:
                best_e, best_metric = epoch, dev_metric
                if is_master():
                    parser.save(base_path)
                logger.info(f'Saved best at epoch {epoch}')
            
            if epoch - best_e >= patience:
                break
        
        logger.info(f"[JointTaggerReal] Done. Best Dev: {best_metric}")
        logger.info(f"[JointTaggerReal] Test: UAS={test_metric.uas:.2%}, LAS={test_metric.las:.2%}")
    
    def _evaluate(self, parser, loader):
        parser.eval()
        total_loss = 0
        tag_correct, tag_total = 0, 0
        metric = AttachmentMetric()
        
        with torch.no_grad():
            for batch in loader:
                words, feats, tags, arcs, rels = batch
                mask = words.ne(parser.pad_index)
                mask[:, 0] = 0
                
                s_tag, s_arc, s_rel = parser.forward(words, feats, use_gold_tags=False)
                loss = parser.forward_loss(s_tag, s_arc, s_rel, tags, arcs, rels, mask)
                
                # Tag accuracy
                tag_preds = s_tag.argmax(-1)
                tag_correct += (tag_preds[mask] == tags[mask]).sum().item()
                tag_total += mask.sum().item()
                
                # Arc/Rel
                arc_preds, rel_preds = parser.decode(s_arc, s_rel, mask)
                
                total_loss += loss.item()
                metric(arc_preds, rel_preds, arcs, rels, mask)
        
        tag_acc = tag_correct / tag_total if tag_total > 0 else 0
        return total_loss / len(loader), tag_acc, metric
