from abc import abstractmethod

import torch.nn as nn

class BaseAttentionParser(nn.Module):
    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward method")
    
    @abstractmethod
    def loss(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement loss method")
    
    @abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement decode method")