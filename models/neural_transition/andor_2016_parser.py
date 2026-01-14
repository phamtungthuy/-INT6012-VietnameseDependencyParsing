"""
Andor et al. (2016) - Globally Normalized Transition-Based Neural Networks

Key characteristics:
- Global normalization over entire action sequence
- CRF-style training
- Beam search with global scoring
- State-of-the-art transition-based results

Reference: https://aclanthology.org/P16-1231.pdf
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural_transition.base_transition_parser import BaseTransitionParser
from models.neural_transition.transition_system import ParserState, Action


@dataclass
class BeamItem:
    """Beam item with cumulative score"""
    state: ParserState
    score: float  # Cumulative unnormalized score
    actions: List[Action]


class Andor2016Parser(BaseTransitionParser):
    """
    Globally Normalized Transition-Based Parser (Andor et al. 2016).
    
    Key differences from local training:
    - Scores are unnormalized at each step
    - Global normalization: P(y|x) = exp(score(y)) / Z(x)
    - Training maximizes log P(gold sequence)
    - Requires computing partition function (approximated with beam)
    
    Training: Global normalization with beam search
    """
    
    def __init__(
        self,
        vocab_size: int,
        pos_size: int,
        num_labels: int,
        embedding_dim: int = 64,
        pos_dim: int = 32,
        hidden_dim: int = 1024,
        dropout: float = 0.5,
        num_stack: int = 3,
        num_buffer: int = 3,
        beam_size: int = 32,
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
        
        self.beam_size = beam_size
        
        # Feature dimension
        num_features = num_stack + num_buffer
        feature_dim = num_features * (embedding_dim + pos_dim)
        
        # Deep network for scoring
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_actions),
        )
    
    def forward(
        self,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns dummy scores"""
        batch_size, seq_len = words.shape
        device = words.device
        
        arc_scores = torch.zeros(batch_size, seq_len, seq_len, device=device)
        label_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_labels, device=device)
        
        return arc_scores, label_scores
    
    def _get_action_scores(
        self,
        state: ParserState,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
    ) -> torch.Tensor:
        """Get unnormalized action scores"""
        feat = self.extract_features(state, words, pos_tags)
        scores = self.scorer(feat.unsqueeze(0)).squeeze(0)
        
        valid_mask = self.transition_system.get_valid_action_mask(state)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=scores.device)
        scores = scores.masked_fill(~valid_mask, -1e9)
        
        return scores
    
    def _beam_search_all(
        self,
        length: int,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
    ) -> List[BeamItem]:
        """Beam search returning all final beam items"""
        initial_state = ParserState.initial(length)
        beam = [BeamItem(state=initial_state, score=0.0, actions=[])]
        
        max_steps = 2 * length + 10
        for _ in range(max_steps):
            if all(item.state.is_terminal() for item in beam):
                break
            
            candidates = []
            
            for item in beam:
                if item.state.is_terminal():
                    candidates.append(item)
                    continue
                
                scores = self._get_action_scores(item.state, words, pos_tags)
                valid_actions = self.transition_system.get_valid_actions(item.state)
                
                for action in valid_actions:
                    action_id = self.transition_system.get_action_id(action)
                    new_score = item.score + scores[action_id].item()
                    new_state = self.transition_system.apply(item.state, action)
                    new_actions = item.actions + [action]
                    
                    candidates.append(BeamItem(
                        state=new_state,
                        score=new_score,
                        actions=new_actions,
                    ))
            
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:self.beam_size]
        
        return beam
    
    def _compute_gold_score(
        self,
        length: int,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
        gold_heads: List[int],
        gold_labels: List[int],
    ) -> torch.Tensor:
        """Compute score of gold sequence"""
        gold_states, gold_actions = self.oracle.get_oracle_sequence(
            sentence_length=length,
            gold_heads=gold_heads,
            gold_labels=gold_labels,
        )
        
        if len(gold_actions) == 0:
            return torch.tensor(0.0, device=words.device)
        
        total_score = torch.tensor(0.0, device=words.device, requires_grad=True)
        
        for state, action in zip(gold_states, gold_actions):
            scores = self._get_action_scores(state, words, pos_tags)
            action_id = self.transition_system.get_action_id(action)
            total_score = total_score + scores[action_id]
        
        return total_score
    
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
        Global normalization loss.
        
        Loss = -log P(gold) = -score(gold) + log Z
        where Z ≈ sum over beam items
        """
        if words is None or pos_tags is None:
            raise ValueError("words and pos_tags required")
        
        batch_size = words.shape[0]
        device = words.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_sentences = 0
        
        for i in range(batch_size):
            length = int(lengths[i].item())
            
            # Gold score
            gold_score = self._compute_gold_score(
                length=length,
                words=words[i],
                pos_tags=pos_tags[i],
                gold_heads=heads[i, :length].tolist(),
                gold_labels=rels[i, :length].tolist(),
            )
            
            # Approximate partition function with beam
            beam = self._beam_search_all(length, words[i], pos_tags[i])
            
            if len(beam) == 0:
                continue
            
            # Log-sum-exp of beam scores
            beam_scores = torch.tensor([item.score for item in beam], device=device)
            log_z = torch.logsumexp(beam_scores, dim=0)
            
            # Negative log-likelihood
            loss = -gold_score + log_z
            total_loss = total_loss + loss
            num_sentences += 1
        
        if num_sentences == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, zero
        
        avg_loss = total_loss / num_sentences
        return avg_loss / 2, avg_loss / 2
    
    def decode(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        lengths: torch.Tensor,
        words: Optional[torch.Tensor] = None,
        pos_tags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Beam search decoding with global scoring"""
        if words is None or pos_tags is None:
            raise ValueError("words and pos_tags required")
        
        batch_size, seq_len = words.shape
        device = words.device
        
        pred_heads = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        pred_labels = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        self.eval()
        with torch.no_grad():
            for i in range(batch_size):
                length = int(lengths[i].item())
                
                beam = self._beam_search_all(length, words[i], pos_tags[i])
                
                if beam:
                    best = max(beam, key=lambda x: x.score)
                    state = best.state
                    
                    for j in range(length):
                        pred_heads[i, j] = max(0, state.heads[j])
                        pred_labels[i, j] = max(0, state.labels[j])
        
        return pred_heads, pred_labels
