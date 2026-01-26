# -*- coding: utf-8 -*-
"""
Trainer for Neural Transition-based Dependency Parser
Based on Chen & Manning 2014
"""

import os
from pathlib import Path
from typing import Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils.logs import logger
from utils.util_deep_learning import device
from utils.sp_data import Dataset
from utils.sp_field import Field
from utils.sp_metric import AttachmentMetric
from utils.constants import pad, unk, bos
from transforms.conll import CoNLL, progress_bar
from models.transition_based.neural_parser import NeuralTransitionParser, NeuralParserTrainer


class ActionVocab:
    """Vocabulary for parsing actions (picklable)"""
    def __init__(self, actions):
        self.itos = actions
        self.stoi = {a: i for i, a in enumerate(actions)}
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.itos[key]
        return self.stoi.get(key, 0)


class NeuralTransitionTrainer:
    """
    Trainer for Neural Transition-based Dependency Parser
    """
    
    def __init__(self, parser: NeuralTransitionParser, corpus):
        self.parser = parser
        self.corpus = corpus
    
    def train(
        self,
        base_path: Union[Path, str],
        min_freq: int = 2,
        batch_size: int = 1000,
        lr: float = 0.01,
        epochs: int = 10,
        dropout: float = 0.5,
        hidden_dim: int = 200,
        word_embed_dim: int = 50,
        tag_embed_dim: int = 50,
        rel_embed_dim: int = 50,
        wandb=None
    ):
        """Train the neural transition parser"""
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        
        logger.info("Building fields and vocabularies...")
        
        # Build fields
        WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
        TAG = Field('tags', bos=bos)
        ARC = Field('arcs', bos=bos, use_vocab=False, fn=CoNLL.get_arcs)
        REL = Field('rels', bos=bos)
        
        transform = CoNLL(FORM=WORD, CPOS=TAG, HEAD=ARC, DEPREL=REL)
        
        # Build vocabularies
        train_dataset = Dataset(transform, self.corpus.train)
        WORD.build(train_dataset, min_freq)
        TAG.build(train_dataset)
        REL.build(train_dataset)
        
        # Build action vocabulary
        action_vocab = self._build_action_vocab(REL.vocab)
        
        n_words = len(WORD.vocab)
        n_tags = len(TAG.vocab)
        n_rels = len(REL.vocab)
        n_actions = len(action_vocab)
        
        logger.info(f"Vocab sizes - Words: {n_words}, Tags: {n_tags}, Rels: {n_rels}, Actions: {n_actions}")
        
        # Initialize parser
        parser = NeuralTransitionParser(
            n_words=n_words,
            n_tags=n_tags,
            n_rels=n_rels,
            n_actions=n_actions,
            word_embed_dim=word_embed_dim,
            tag_embed_dim=tag_embed_dim,
            rel_embed_dim=rel_embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        ).to(device)
        
        # Set vocab mappings
        parser.word_vocab = WORD.vocab.stoi
        parser.tag_vocab = TAG.vocab.stoi
        parser.rel_vocab = REL.vocab.stoi
        parser.action_vocab = action_vocab
        parser.transform = transform
        
        # Trainer helper
        trainer_helper = NeuralParserTrainer(parser)
        
        # Generate training instances
        logger.info("Generating training instances...")
        train_instances = self._generate_all_instances(
            train_dataset, trainer_helper, WORD, TAG, REL, action_vocab
        )
        
        logger.info(f"Generated {len(train_instances)} training instances")
        
        # Create data loader
        word_data = torch.tensor([inst[0] for inst in train_instances], dtype=torch.long)
        tag_data = torch.tensor([inst[1] for inst in train_instances], dtype=torch.long)
        rel_data = torch.tensor([inst[2] for inst in train_instances], dtype=torch.long)
        action_data = torch.tensor([inst[3] for inst in train_instances], dtype=torch.long)
        
        dataset = TensorDataset(word_data, tag_data, rel_data, action_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.AdamW(parser.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_uas = 0
        for epoch in range(1, epochs + 1):
            parser.train()
            total_loss = 0
            correct = 0
            total = 0
            
            bar = progress_bar(loader)
            for word_batch, tag_batch, rel_batch, action_batch in bar:
                word_batch = word_batch.to(device)
                tag_batch = tag_batch.to(device)
                rel_batch = rel_batch.to(device)
                action_batch = action_batch.to(device)
                
                optimizer.zero_grad()
                scores = parser(word_batch, tag_batch, rel_batch)
                loss = criterion(scores, action_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = scores.argmax(dim=-1)
                correct += (preds == action_batch).sum().item()
                total += action_batch.size(0)
                
                bar.set_postfix_str(f"loss: {loss.item():.4f}, acc: {correct/total:.2%}")
            
            avg_loss = total_loss / len(loader)
            accuracy = correct / total
            logger.info(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
            
            # Evaluate on dev
            dev_dataset = Dataset(transform, self.corpus.dev)
            dev_uas, dev_las = self._evaluate(parser, dev_dataset, WORD, TAG, REL)
            logger.info(f"Dev - UAS: {dev_uas:.2%}, LAS: {dev_las:.2%}")
            
            if dev_uas > best_uas:
                best_uas = dev_uas
                self._save_model(parser, base_path, WORD, TAG, REL, action_vocab)
                logger.info(f"Saved best model at epoch {epoch}")
            
            if wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_acc": accuracy,
                    "dev_uas": dev_uas,
                    "dev_las": dev_las
                })
        
        # Final test evaluation
        logger.info("Evaluating on test set...")
        test_dataset = Dataset(transform, self.corpus.test)
        test_uas, test_las = self._evaluate(parser, test_dataset, WORD, TAG, REL)
        logger.info(f"Test - UAS: {test_uas:.2%}, LAS: {test_las:.2%}")
    
    def _build_action_vocab(self, rel_vocab):
        """Build action vocabulary from relation vocabulary"""
        actions = ['SHIFT']
        for rel in rel_vocab.itos:
            if rel not in [pad, unk, bos]:
                actions.append(f'LEFT-ARC#{rel}')
                actions.append(f'RIGHT-ARC#{rel}')
        
        return ActionVocab(actions)
    
    def _generate_all_instances(self, dataset, trainer, WORD, TAG, REL, action_vocab):
        """Generate training instances from all sentences"""
        all_instances = []
        
        for sentence in dataset.sentences:
            # Get word, tag, arc, rel data
            words = [WORD.vocab.stoi.get(w.lower(), WORD.unk_index) 
                    for w in [bos] + list(sentence.words)]
            tags = [TAG.vocab.stoi.get(t, 0) 
                   for t in [bos] + list(sentence.tags)]
            
            gold_heads = [0] + list(CoNLL.get_arcs(sentence.arcs))
            gold_rels = [bos] + list(sentence.rels)
            
            # Generate instances
            instances = trainer.generate_training_data(words, tags, gold_heads, gold_rels)
            
            for word_feats, tag_feats, rel_feats, action in instances:
                action_idx = action_vocab.stoi.get(action, 0)
                all_instances.append((word_feats, tag_feats, rel_feats, action_idx))
        
        return all_instances
    
    def _evaluate(self, parser, dataset, WORD, TAG, REL):
        """Evaluate parser on a dataset"""
        parser.eval()
        
        total_arcs = 0
        correct_arcs = 0
        correct_rels = 0
        
        for sentence in dataset.sentences:
            words = [WORD.vocab.stoi.get(w.lower(), WORD.unk_index) 
                    for w in [bos] + list(sentence.words)]
            tags = [TAG.vocab.stoi.get(t, 0) 
                   for t in [bos] + list(sentence.tags)]
            
            gold_heads = [0] + list(CoNLL.get_arcs(sentence.arcs))
            gold_rels = [bos] + list(sentence.rels)
            
            pred_arcs, pred_rels = parser.parse(words, tags)
            
            # Compare predictions with gold
            for i in range(1, len(words)):
                total_arcs += 1
                pred_head = pred_arcs.get(i, -1)
                pred_rel = pred_rels.get(i, '')
                
                if pred_head == gold_heads[i]:
                    correct_arcs += 1
                    if pred_rel == gold_rels[i]:
                        correct_rels += 1
        
        uas = correct_arcs / total_arcs if total_arcs > 0 else 0
        las = correct_rels / total_arcs if total_arcs > 0 else 0
        
        return uas, las
    
    def _save_model(self, parser, path, WORD, TAG, REL, action_vocab):
        """Save model checkpoint"""
        state = {
            'state_dict': parser.state_dict(),
            'word_vocab': WORD.vocab,
            'tag_vocab': TAG.vocab,
            'rel_vocab': REL.vocab,
            'action_vocab': action_vocab,
            'transform': parser.transform,
            'args': {
                'n_words': parser.n_words,
                'n_tags': parser.n_tags,
                'n_rels': parser.n_rels,
                'n_actions': parser.n_actions
            }
        }
        torch.save(state, path)
