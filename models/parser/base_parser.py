from abc import abstractmethod

import torch.nn as nn

class BaseParser(nn.Module):
    @abstractmethod
    def forward(self, words, pos_tags, lengths):
        raise NotImplementedError("Subclasses must implement forward method")
    
    @abstractmethod
    def loss(self, arc_scores, label_scores, heads, rels, lengths):
        raise NotImplementedError("Subclasses must implement loss method")
    
    @abstractmethod
    def decode(self, arc_scores, label_scores, lengths):
        raise NotImplementedError("Subclasses must implement decode method")