"""Base Embedding Module"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseEmbedding(nn.Module, ABC):
    """Abstract base class for embedding layers"""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns:
            embeddings: [batch, seq_len, output_dim]
        """
        raise NotImplementedError

