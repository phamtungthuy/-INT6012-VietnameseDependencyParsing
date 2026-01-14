import math

import torch
import torch.nn as nn

class BiaffineAttention(nn.Module):
    """
    Biaffine Attention: score = (x^T W y + bias_x^T x + bias_y^T y) * scale
    
    With optional score normalization (1/√d) from paper:
    "Score Normalization for Biaffine Attention" (Gajo et al., 2025)
    """
    
    def __init__(self, in_features, out_features, bias=(True, True), scale=True):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension (1 for arc, num_labels for label)
            bias: Tuple (bias_x, bias_y) whether to use bias terms
            scale: Whether to apply score normalization (1/√d)
        """
        self.scale = scale
        self.scale_factor = 1.0 / math.sqrt(in_features) if scale else 1.0
        
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
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, len_x, in_features] (dependent)
            y: [batch, len_y, in_features] (head)
        
        Returns:
            [batch, len_x, len_y, out_features]
        """
        # Biaffine transformation: x^T W y
        out = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        
        # Add bias terms
        if self.bias_x is not None:
            out = out + torch.einsum('bxi,oi->box', x, self.bias_x).unsqueeze(3)
        
        if self.bias_y is not None:
            out = out + torch.einsum('byj,oj->boy', y, self.bias_y).unsqueeze(2)
        
        # Score normalization: a = 1/√d
        out = out * self.scale_factor
        
        # Transpose to [batch, len_x, len_y, out]
        return out.permute(0, 2, 3, 1)