"""
Scaled Biaffine Attention với Score Normalization

Reference:
    - Bhatt et al. (2024): End-to-end Parsing of Procedural Text into Flow Graphs
      https://aclanthology.org/2024.lrec-main.517/
    - Dozat & Manning (2017): Deep Biaffine Attention for Neural Dependency Parsing
"""

import math

import torch
import torch.nn as nn


class ScaledBiaffineAttention(nn.Module):
    """
    Biaffine Attention với Score Normalization
    
    score(x, y) = (x^T W y + U^T x + V^T y + b) * scale
    
    where scale = 1/√d (d = input dimension)
    
    Điều này giúp ổn định gradient khi dimension lớn.
    """
    
    def __init__(
        self, 
        head_dim: int, 
        dep_dim: int, 
        out_features: int = 1,
        use_head_bias: bool = True,
        use_dep_bias: bool = True,
        use_score_norm: bool = True
    ):
        super().__init__()
        
        self.head_dim = head_dim
        self.dep_dim = dep_dim
        self.out_features = out_features
        self.use_score_norm = use_score_norm
        
        # Scale factor: a = 1/√d
        self.scale = 1.0 / math.sqrt(head_dim) if use_score_norm else 1.0
        
        # Bilinear weight: W ∈ R^{out × dep × head}
        self.weight = nn.Parameter(torch.Tensor(out_features, dep_dim, head_dim))
        
        # Linear weights for head and dependent
        self.head_bias = nn.Parameter(torch.Tensor(out_features, head_dim)) if use_head_bias else None
        self.dep_bias = nn.Parameter(torch.Tensor(out_features, dep_dim)) if use_dep_bias else None
        
        # Scalar bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        if self.head_bias is not None:
            nn.init.zeros_(self.head_bias)
        if self.dep_bias is not None:
            nn.init.zeros_(self.dep_bias)
    
    def forward(self, head: torch.Tensor, dep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            head: [batch, seq_len, head_dim] - head representations
            dep: [batch, seq_len, dep_dim] - dependent representations
        
        Returns:
            scores: [batch, seq_len (dep), seq_len (head), out_features]
        """
        batch_size, seq_len, _ = head.shape
        
        # Bilinear: dep^T W head
        # [batch, seq_dep, seq_head, out]
        scores = torch.einsum('bdi,odi,bhj->bdhj', dep, self.weight, head)
        
        # Add head bias: U^T head
        if self.head_bias is not None:
            head_scores = torch.einsum('bhi,oi->bho', head, self.head_bias)
            scores = scores + head_scores.unsqueeze(1)
        
        # Add dependent bias: V^T dep
        if self.dep_bias is not None:
            dep_scores = torch.einsum('bdi,oi->bdo', dep, self.dep_bias)
            scores = scores + dep_scores.unsqueeze(2)
        
        # Add scalar bias
        scores = scores + self.bias
        
        # Apply score normalization
        scores = scores * self.scale
        
        return scores
