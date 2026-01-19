"""
Trainer for Transition-based Parser (Chen & Manning 2014)
"""

from typing import Dict, List, Tuple
from tqdm import tqdm

import torch

from trainers.base_trainer import BaseTrainer


class TransitionTrainer(BaseTrainer):
    """Trainer for Transition-based Dependency Parser"""
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f'Epoch {epoch}',
            leave=False
        )
        
        for batch in progress_bar:
            words = batch['words'].to(self.device)
            pos_tags = batch['pos_tags'].to(self.device)
            heads = batch['heads'].to(self.device)
            rels = batch['rels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            # Forward (returns dummy scores for transition parser)
            arc_scores, label_scores = self.model(words, pos_tags, lengths)
            
            # Compute loss using oracle
            arc_loss, label_loss = self.model.loss(
                arc_scores, label_scores, heads, rels, lengths,
                words=words, pos_tags=pos_tags
            )
            loss = arc_loss + label_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss, avg_loss / 2, avg_loss / 2
    
    def evaluate(self, data_loader=None):
        """Override evaluate to pass words and pos_tags to decode"""
        if data_loader is None:
            data_loader = self.validation_loader
        
        return self.evaluator.evaluate_transition_model(
            self.model, 
            data_loader, 
            device=self.device
        )

