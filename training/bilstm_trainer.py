"""
Trainer for Graph-based BiLSTM Parser (Dozat & Manning 2017)
"""

from typing import Dict, List, Tuple
from tqdm import tqdm

import torch

from training.base_trainer import BaseTrainer


class BiLSTMTrainer(BaseTrainer):
    """Trainer for BiLSTM + Biaffine Attention Parser"""
    
    def train_epoch(self, epoch: int) -> Tuple[float, float, float]:
        self.model.train()
        total_arc_loss = 0.0
        total_label_loss = 0.0
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
            
            # Forward pass
            arc_scores, label_scores = self.model(words, pos_tags, lengths)
            
            # Compute loss
            arc_loss, label_loss = self.model.loss(
                arc_scores, label_scores, heads, rels, lengths
            )
            loss = arc_loss + label_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()
            
            total_arc_loss += arc_loss.item()
            total_label_loss += label_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'arc': f'{arc_loss.item():.4f}',
                'lbl': f'{label_loss.item():.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_arc_loss = total_arc_loss / num_batches
        avg_label_loss = total_label_loss / num_batches
        
        return avg_loss, avg_arc_loss, avg_label_loss
