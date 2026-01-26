# -*- coding: utf-8 -*-
"""
Neural Network-based Transition Dependency Parser
Based on: Chen & Manning 2014 - "A Fast and Accurate Dependency Parser using Neural Networks"
https://aclanthology.org/D14-1082.pdf

This implements:
- Arc-Standard transition system (SHIFT, LEFT-ARC, RIGHT-ARC)
- Feedforward neural network with cube activation
- Word, POS tag, and arc label embeddings as features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from utils.util_deep_learning import device


class NeuralTransitionParser(nn.Module):
    """
    Neural Network Transition-based Dependency Parser
    
    Architecture (Chen & Manning 2014):
    1. Extract features from stack and buffer positions
    2. Look up embeddings for words, POS tags, and arc labels
    3. Concatenate all embeddings
    4. Pass through hidden layer with cube activation
    5. Softmax over transition actions
    
    Features extracted:
    - Stack: s1, s2, s3 (top 3 words on stack)
    - Buffer: b1, b2, b3 (first 3 words in buffer)
    - Children: lc1(si), rc1(si), lc2(si), rc2(si) for i in {1,2}
    """
    
    def __init__(
        self,
        n_words: int,
        n_tags: int,
        n_rels: int,
        n_actions: int,
        word_embed_dim: int = 50,
        tag_embed_dim: int = 50,
        rel_embed_dim: int = 50,
        hidden_dim: int = 200,
        dropout: float = 0.5,
        pretrained_embed: Optional[torch.Tensor] = None
    ):
        super(NeuralTransitionParser, self).__init__()
        
        self.n_words = n_words
        self.n_tags = n_tags
        self.n_rels = n_rels
        self.n_actions = n_actions
        
        # Feature dimensions (as in Chen & Manning)
        # Words: s1, s2, s3, b1, b2, b3, lc1(s1), rc1(s1), lc1(s2), rc1(s2), 
        #        lc2(s1), rc2(s1), lc1(b1), rc1(b1), lc1(lc1(s1)), lc1(rc1(s1))
        # = 18 word features
        self.n_word_feats = 18
        self.n_tag_feats = 18  # Same positions for POS tags
        self.n_rel_feats = 12  # Arc labels for children positions
        
        # Embedding layers
        self.word_embed = nn.Embedding(n_words, word_embed_dim)
        self.tag_embed = nn.Embedding(n_tags, tag_embed_dim)
        self.rel_embed = nn.Embedding(n_rels + 1, rel_embed_dim)  # +1 for NULL label
        
        # Load pretrained embeddings if provided
        if pretrained_embed is not None:
            self.word_embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        
        # Calculate input dimension
        input_dim = (self.n_word_feats * word_embed_dim + 
                     self.n_tag_feats * tag_embed_dim + 
                     self.n_rel_feats * rel_embed_dim)
        
        # Hidden layer
        self.hidden = nn.Linear(input_dim, hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, n_actions)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Vocab mappings (set during training)
        self.word_vocab = None
        self.tag_vocab = None
        self.rel_vocab = None
        self.action_vocab = None
        
        # Special indices
        self.null_idx = 0  # NULL token for missing features
        self.root_idx = 1  # ROOT token
        self.unk_idx = 2   # Unknown token
        
    def _init_weights(self):
        """Initialize weights following Chen & Manning"""
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def cube_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Cube activation function: f(x) = x^3"""
        return x ** 3
    
    def forward(
        self,
        word_feats: torch.Tensor,
        tag_feats: torch.Tensor,
        rel_feats: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            word_feats: [batch_size, n_word_feats] - Word indices
            tag_feats: [batch_size, n_tag_feats] - POS tag indices
            rel_feats: [batch_size, n_rel_feats] - Arc label indices
            
        Returns:
            action_scores: [batch_size, n_actions] - Scores for each action
        """
        # Look up embeddings
        word_embeds = self.word_embed(word_feats)  # [batch, n_word_feats, embed_dim]
        tag_embeds = self.tag_embed(tag_feats)
        rel_embeds = self.rel_embed(rel_feats)
        
        # Flatten embeddings
        word_embeds = word_embeds.view(word_embeds.size(0), -1)
        tag_embeds = tag_embeds.view(tag_embeds.size(0), -1)
        rel_embeds = rel_embeds.view(rel_embeds.size(0), -1)
        
        # Concatenate all features
        x = torch.cat([word_embeds, tag_embeds, rel_embeds], dim=-1)
        
        # Hidden layer with cube activation
        x = self.dropout(x)
        h = self.hidden(x)
        h = self.cube_activation(h)
        h = self.dropout(h)
        
        # Output layer
        action_scores = self.output(h)
        
        return action_scores
    
    def extract_features(
        self,
        stack: List[int],
        buffer: List[int],
        arcs: Dict[int, int],
        rels: Dict[int, str],
        words: List[int],
        tags: List[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Extract features from current parsing state.
        
        Following Chen & Manning 2014:
        - s1, s2, s3: Top 3 on stack
        - b1, b2, b3: First 3 in buffer
        - lc1(si), rc1(si): Leftmost/rightmost child of stack[i]
        - lc2(si), rc2(si): 2nd leftmost/rightmost child
        
        Args:
            stack: Current stack (list of word indices)
            buffer: Current buffer (list of word indices)
            arcs: Dict mapping dependent -> head
            rels: Dict mapping dependent -> relation label
            words: Word indices for the sentence
            tags: POS tag indices for the sentence
            
        Returns:
            word_feats, tag_feats, rel_feats: Lists of feature indices
        """
        def get_word(idx):
            if idx is None or idx < 0 or idx >= len(words):
                return self.null_idx
            return words[idx]
        
        def get_tag(idx):
            if idx is None or idx < 0 or idx >= len(tags):
                return self.null_idx
            return tags[idx]
        
        def get_rel(idx):
            if idx is None or idx not in rels:
                return 0  # NULL relation
            rel_str = rels[idx]
            if self.rel_vocab and rel_str in self.rel_vocab:
                return self.rel_vocab[rel_str]
            return 0
        
        def get_children(head_idx, arcs):
            """Get sorted children of a head"""
            children = [dep for dep, h in arcs.items() if h == head_idx]
            return sorted(children)
        
        def get_left_child(head_idx, arcs, n=1):
            """Get nth leftmost child"""
            children = get_children(head_idx, arcs)
            left_children = [c for c in children if c < head_idx]
            if len(left_children) >= n:
                return left_children[n-1]
            return None
        
        def get_right_child(head_idx, arcs, n=1):
            """Get nth rightmost child"""
            children = get_children(head_idx, arcs)
            right_children = [c for c in children if c > head_idx]
            if len(right_children) >= n:
                return right_children[-n]
            return None
        
        # Stack positions
        s1 = stack[-1] if len(stack) >= 1 else None
        s2 = stack[-2] if len(stack) >= 2 else None
        s3 = stack[-3] if len(stack) >= 3 else None
        
        # Buffer positions
        b1 = buffer[0] if len(buffer) >= 1 else None
        b2 = buffer[1] if len(buffer) >= 2 else None
        b3 = buffer[2] if len(buffer) >= 3 else None
        
        # Children of stack elements
        lc1_s1 = get_left_child(s1, arcs, 1) if s1 is not None else None
        rc1_s1 = get_right_child(s1, arcs, 1) if s1 is not None else None
        lc1_s2 = get_left_child(s2, arcs, 1) if s2 is not None else None
        rc1_s2 = get_right_child(s2, arcs, 1) if s2 is not None else None
        
        lc2_s1 = get_left_child(s1, arcs, 2) if s1 is not None else None
        rc2_s1 = get_right_child(s1, arcs, 2) if s1 is not None else None
        
        # Children of buffer top
        lc1_b1 = get_left_child(b1, arcs, 1) if b1 is not None else None
        rc1_b1 = get_right_child(b1, arcs, 1) if b1 is not None else None
        
        # Grandchildren
        lc1_lc1_s1 = get_left_child(lc1_s1, arcs, 1) if lc1_s1 is not None else None
        lc1_rc1_s1 = get_left_child(rc1_s1, arcs, 1) if rc1_s1 is not None else None
        
        # Word features (18 positions)
        word_positions = [s1, s2, s3, b1, b2, b3,
                         lc1_s1, rc1_s1, lc1_s2, rc1_s2,
                         lc2_s1, rc2_s1, lc1_b1, rc1_b1,
                         lc1_lc1_s1, lc1_rc1_s1,
                         None, None]  # Padding to 18
        
        word_feats = [get_word(p) for p in word_positions]
        tag_feats = [get_tag(p) for p in word_positions]
        
        # Arc label features (12 positions - for children)
        rel_positions = [lc1_s1, rc1_s1, lc1_s2, rc1_s2,
                        lc2_s1, rc2_s1, lc1_b1, rc1_b1,
                        lc1_lc1_s1, lc1_rc1_s1,
                        None, None]  # Padding to 12
        
        rel_feats = [get_rel(p) for p in rel_positions]
        
        return word_feats, tag_feats, rel_feats
    
    def get_valid_actions(self, stack: List[int], buffer: List[int]) -> List[str]:
        """Get valid actions based on current state"""
        valid = []
        if len(buffer) > 0:
            valid.append('SHIFT')
        if len(stack) >= 2:
            if stack[-2] != 0:  # Can't LEFT-ARC on ROOT
                valid.append('LEFT-ARC')
            valid.append('RIGHT-ARC')
        return valid
    
    def apply_action(
        self,
        action: str,
        stack: List[int],
        buffer: List[int],
        arcs: Dict[int, int],
        rels: Dict[int, str]
    ):
        """Apply a transition action to the current state"""
        parts = action.split('#')
        action_type = parts[0]
        label = parts[1] if len(parts) > 1 else 'dep'
        
        if action_type == 'SHIFT':
            if buffer:
                stack.append(buffer.pop(0))
        elif action_type == 'LEFT-ARC':
            if len(stack) >= 2:
                dep = stack.pop(-2)  # s2 is dependent
                head = stack[-1]     # s1 is head
                arcs[dep] = head
                rels[dep] = label
        elif action_type == 'RIGHT-ARC':
            if len(stack) >= 2:
                dep = stack.pop(-1)  # s1 is dependent
                head = stack[-1]     # s2 is head
                arcs[dep] = head
                rels[dep] = label
    
    def parse(
        self,
        words: List[int],
        tags: List[int]
    ) -> Tuple[Dict[int, int], Dict[int, str]]:
        """
        Parse a sentence using greedy decoding.
        
        Args:
            words: Word indices including ROOT at position 0
            tags: POS tag indices
            
        Returns:
            arcs: Dict mapping dependent index -> head index
            rels: Dict mapping dependent index -> relation label
        """
        self.eval()
        
        stack = [0]  # Start with ROOT
        buffer = list(range(1, len(words)))
        arcs = {}
        rels = {}
        
        max_steps = len(words) * 3
        step = 0
        
        while buffer or len(stack) > 1:
            step += 1
            if step > max_steps:
                break
            
            valid_actions = self.get_valid_actions(stack, buffer)
            if not valid_actions:
                break
            
            # Extract features
            word_feats, tag_feats, rel_feats = self.extract_features(
                stack, buffer, arcs, rels, words, tags
            )
            
            # Convert to tensors
            word_tensor = torch.tensor([word_feats], dtype=torch.long, device=device)
            tag_tensor = torch.tensor([tag_feats], dtype=torch.long, device=device)
            rel_tensor = torch.tensor([rel_feats], dtype=torch.long, device=device)
            
            # Get action scores
            with torch.no_grad():
                scores = self.forward(word_tensor, tag_tensor, rel_tensor)
            
            # Select best valid action
            scores = scores.squeeze(0)
            best_action = None
            best_score = float('-inf')
            
            for action_idx in range(self.n_actions):
                if self.action_vocab:
                    action = self.action_vocab.itos[action_idx]
                else:
                    action = str(action_idx)
                
                base_action = action.split('#')[0]
                if base_action in valid_actions and scores[action_idx] > best_score:
                    best_score = scores[action_idx]
                    best_action = action
            
            if best_action is None:
                best_action = valid_actions[0]
            
            self.apply_action(best_action, stack, buffer, arcs, rels)
        
        return arcs, rels


class NeuralParserTrainer:
    """Training utilities for Neural Transition Parser"""
    
    def __init__(self, parser: NeuralTransitionParser):
        self.parser = parser
    
    def get_oracle_action(
        self,
        stack: List[int],
        buffer: List[int],
        gold_heads: List[int],
        gold_rels: List[str],
        arcs: Dict[int, int]
    ) -> str:
        """Get oracle (gold) action for training"""
        valid = self.parser.get_valid_actions(stack, buffer)
        if not valid:
            return 'SHIFT'
        
        if len(stack) >= 2:
            s1 = stack[-1]
            s2 = stack[-2]
            
            # LEFT-ARC: s2 <- s1
            if 'LEFT-ARC' in valid and gold_heads[s2] == s1:
                # Check if s2 has collected all its dependents
                all_deps = all(i in arcs for i, h in enumerate(gold_heads) if h == s2)
                if all_deps:
                    label = gold_rels[s2]
                    return f'LEFT-ARC#{label}'
            
            # RIGHT-ARC: s1 <- s2
            if 'RIGHT-ARC' in valid and gold_heads[s1] == s2:
                all_deps = all(i in arcs for i, h in enumerate(gold_heads) if h == s1)
                if all_deps:
                    label = gold_rels[s1]
                    return f'RIGHT-ARC#{label}'
        
        return 'SHIFT' if 'SHIFT' in valid else valid[0]
    
    def generate_training_data(
        self,
        words: List[int],
        tags: List[int],
        gold_heads: List[int],
        gold_rels: List[str]
    ) -> List[Tuple]:
        """Generate training instances from a gold parse tree"""
        stack = [0]
        buffer = list(range(1, len(words)))
        arcs = {}
        rels = {}
        
        instances = []
        max_steps = len(words) * 3
        step = 0
        
        while buffer or len(stack) > 1:
            step += 1
            if step > max_steps:
                break
            
            valid = self.parser.get_valid_actions(stack, buffer)
            if not valid:
                break
            
            # Extract features
            word_feats, tag_feats, rel_feats = self.parser.extract_features(
                stack, buffer, arcs, rels, words, tags
            )
            
            # Get oracle action
            gold_action = self.get_oracle_action(
                stack, buffer, gold_heads, gold_rels, arcs
            )
            
            instances.append((word_feats, tag_feats, rel_feats, gold_action))
            
            # Apply action
            self.parser.apply_action(gold_action, stack, buffer, arcs, rels)
        
        return instances
