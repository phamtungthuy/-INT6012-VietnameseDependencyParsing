"""
Chen & Manning (2014) - A Fast and Accurate Dependency Parser using Neural Networks

Key characteristics:
- Greedy local training with cross-entropy loss
- Static oracle
- Simple MLP classifier
- Fast inference (no beam search)

Reference: https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural_transition.base_transition_parser import BaseTransitionParser
from models.neural_transition.transition_system import ParserState


class ChenManningParser(BaseTransitionParser):
    """
    Neural Transition-based Parser (Chen & Manning 2014).
    
    Architecture:
    - Input: Concatenated embeddings from stack (s0, s1, s2) and buffer (b0, b1, b2)
    - Hidden: One hidden layer with cube activation (or ReLU)
    - Output: Softmax over actions
    
    Training: Greedy local, maximize P(action | state)
    """
    
    def __init__(
        self,
        vocab_size: int,
        pos_size: int,
        num_labels: int,
        embedding_dim: int = 50,
        pos_dim: int = 50,
        hidden_dim: int = 200,
        dropout: float = 0.5,
        num_stack: int = 3,
        num_buffer: int = 3,
    ):
        super().__init__(
            vocab_size=vocab_size,
            pos_size=pos_size,
            num_labels=num_labels,
            embedding_dim=embedding_dim,
            pos_dim=pos_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_stack=num_stack,
            num_buffer=num_buffer,
        )
        
        # Feature dimension
        num_features = num_stack + num_buffer
        feature_dim = num_features * (embedding_dim + pos_dim)
        
        # MLP classifier (cube activation in original, we use ReLU for stability)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_actions),
        )
    
    def forward(
        self,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns dummy scores (actual computation in loss/decode)"""
        batch_size, seq_len = words.shape
        device = words.device
        
        arc_scores = torch.zeros(batch_size, seq_len, seq_len, device=device)
        label_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_labels, device=device)
        
        return arc_scores, label_scores
    
    def loss(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        lengths: torch.Tensor,
        words: Optional[torch.Tensor] = None,
        pos_tags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy local training: maximize log P(gold_action | state).
        
        Uses static oracle to get gold action sequence.
        """
        if words is None or pos_tags is None:
            raise ValueError("words and pos_tags required for ChenManningParser")
        
        batch_size = words.shape[0]
        device = words.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_actions = 0
        
        for i in range(batch_size):
            length = int(lengths[i].item())
            
            # Get oracle action sequence
            states, actions = self.oracle.get_oracle_sequence(
                sentence_length=length,
                gold_heads=heads[i, :length].tolist(),
                gold_labels=rels[i, :length].tolist(),
            )
            
            if len(actions) == 0:
                continue
            
            # Collect features and targets
            batch_features = []
            batch_targets = []
            batch_masks = []
            
            for state, action in zip(states, actions):
                feat = self.extract_features(state, words[i], pos_tags[i])
                batch_features.append(feat)
                
                target = self.transition_system.get_action_id(action)
                batch_targets.append(target)
                
                valid_mask = self.transition_system.get_valid_action_mask(state)
                batch_masks.append(valid_mask)
            
            if len(batch_features) == 0:
                continue
            
            # Stack tensors
            features = torch.stack(batch_features)
            targets = torch.tensor(batch_targets, dtype=torch.long, device=device)
            masks = torch.tensor(batch_masks, dtype=torch.bool, device=device)
            
            # Forward
            logits = self.classifier(features)
            logits = logits.masked_fill(~masks, -1e9)
            
            # Cross entropy loss
            loss = F.cross_entropy(logits, targets, reduction='sum')
            total_loss = total_loss + loss
            num_actions += len(batch_targets)
        
        if num_actions == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero
        
        avg_loss = total_loss / num_actions
        return avg_loss / 2, avg_loss / 2
    
    def decode(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        lengths: torch.Tensor,
        words: Optional[torch.Tensor] = None,
        pos_tags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Greedy decoding: select best valid action at each step."""
        if words is None or pos_tags is None:
            raise ValueError("words and pos_tags required for ChenManningParser")
        
        batch_size, seq_len = words.shape
        device = words.device
        
        pred_heads = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        pred_labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        self.eval()
        with torch.no_grad():
            for i in range(batch_size):
                length = int(lengths[i].item())
                state = ParserState.initial(length)
                
                max_steps = 2 * length + 10
                for _ in range(max_steps):
                    if state.is_terminal():
                        break
                    
                    feat = self.extract_features(state, words[i], pos_tags[i])
                    logits = self.classifier(feat.unsqueeze(0))
                    
                    valid_mask = self.transition_system.get_valid_action_mask(state)
                    valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)
                    logits = logits.masked_fill(~valid_mask.unsqueeze(0), -1e9)
                    
                    action_id = logits.argmax(dim=-1).item()
                    action = self.transition_system.get_action_from_id(action_id)
                    state = self.transition_system.apply(state, action)
                
                for j in range(length):
                    pred_heads[i, j] = max(0, state.heads[j])
                    pred_labels[i, j] = max(0, state.labels[j])
        
        return pred_heads, pred_labels
