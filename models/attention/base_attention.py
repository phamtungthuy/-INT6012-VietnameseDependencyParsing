from abc import abstractmethod

import torch
import torch.nn as nn

class BaseAttention(nn.Module):
    def __init__(self, in_features, out_features, bias=(True, True)):
        super(BaseAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        # Weight matrix: [out_features, in_features, in_features]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features, in_features))
        
        # Bias terms
        if bias[0]:
            self.bias_x = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.register_parameter('bias_x', None)
            
        if bias[1]:
            self.bias_y = nn.Parameter(torch.Tensor(out_features, in_features))
        else:
            self.register_parameter('bias_y', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias_x is not None:
            nn.init.zeros_(self.bias_x)
        if self.bias_y is not None:
            nn.init.zeros_(self.bias_y)
            
    @abstractmethod
    def forward(self, x, y):
        raise NotImplementedError("Subclasses must implement forward method")