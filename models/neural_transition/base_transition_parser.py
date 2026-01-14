"""
Base class for Neural Transition-based Dependency Parsers
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from models.neural_transition.transition_system import (
    ArcStandardTransitionSystem,
    Oracle,
    ParserState,
)


class BaseTransitionParser(nn.Module, ABC):
    """
    Abstract base class for neural transition-based parsers.
    
    Common components:
    - Transition system (Arc-Standard)
    - Oracle for training
    - Feature extraction from parser state
    
    Subclasses implement different training objectives:
    - Local (greedy): Chen & Manning 2014
    - Structured (beam): Weiss et al. 2015  
    - Global: Andor et al. 2016
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
        super().__init__()
        
        self.vocab_size = vocab_size
        self.pos_size = pos_size
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.num_stack = num_stack
        self.num_buffer = num_buffer
        
        # Transition system
        self.transition_system = ArcStandardTransitionSystem(num_labels)
        self.oracle = Oracle(self.transition_system)
        self.num_actions = self.transition_system.num_actions
        
        # Embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_size, pos_dim, padding_idx=0)
        
        # NULL embeddings for empty stack/buffer positions
        self.null_word_emb = nn.Parameter(torch.zeros(embedding_dim))
        self.null_pos_emb = nn.Parameter(torch.zeros(pos_dim))
        nn.init.normal_(self.null_word_emb, std=0.01)
        nn.init.normal_(self.null_pos_emb, std=0.01)
        
        self.dropout = nn.Dropout(dropout)
    
    def _get_token_embedding(
        self,
        idx: Optional[int],
        words: torch.Tensor,
        pos_tags: torch.Tensor,
    ) -> torch.Tensor:
        """Get embedding for a token, or NULL embedding if invalid"""
        if idx is None or idx < 0:
            return torch.cat([self.null_word_emb, self.null_pos_emb])
        word_emb = self.word_embedding(words[idx])
        pos_emb = self.pos_embedding(pos_tags[idx])
        return torch.cat([word_emb, pos_emb])
    
    def extract_features(
        self,
        state: ParserState,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
    ) -> torch.Tensor:
        """Extract feature vector from parser state"""
        features = []
        
        # Stack features (top num_stack elements)
        for i in range(self.num_stack):
            idx = state.get_stack_top(i)
            features.append(self._get_token_embedding(idx, words, pos_tags))
        
        # Buffer features (first num_buffer elements)
        for i in range(self.num_buffer):
            idx = state.get_buffer_front(i)
            features.append(self._get_token_embedding(idx, words, pos_tags))
        
        return torch.cat(features)
    
    @abstractmethod
    def forward(
        self,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - returns arc_scores and label_scores (may be dummy)"""
        raise NotImplementedError
    
    @abstractmethod
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
        """Compute training loss"""
        raise NotImplementedError
    
    @abstractmethod
    def decode(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        lengths: torch.Tensor,
        words: Optional[torch.Tensor] = None,
        pos_tags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode to get predicted heads and labels"""
        raise NotImplementedError
