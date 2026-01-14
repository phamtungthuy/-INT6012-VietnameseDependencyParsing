"""
Các module con cho BERT Biaffine Parser

Bao gồm:
    - TaggerModule: Joint POS Tagging (BiLSTM φ + MLP)
    - ParserEncoder: BiLSTM ψ × N layers
    - EdgeScorer: MLPs + Biaffine Attention

Reference:
    - Bhatt et al. (2024): End-to-end Parsing of Procedural Text into Flow Graphs
      https://aclanthology.org/2024.lrec-main.517/
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.neural_attention.bert_biaffine.attention import ScaledBiaffineAttention


class TaggerModule(nn.Module):
    """
    Tagger Module cho Joint POS Tagging
    
    Architecture:
        input → BiLSTM φ → MLP tag classifier → tags
                         ↓
                    tag-to-dense MLP → tag embeddings
    
    Tagger output được concatenate với BERT output trước khi đưa vào parser.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_tags: int = 50,
        tag_embedding_dim: int = 64,
        dropout: float = 0.33
    ):
        super().__init__()
        
        self.num_tags = num_tags
        self.tag_embedding_dim = tag_embedding_dim
        
        # BiLSTM φ for tagging
        self.tagger_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # MLP tag classifier
        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.tag_classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags)
        )
        
        # Tag-to-dense MLP: converts predicted tag distribution to dense embedding
        # This allows gradients to flow through tagging decisions
        self.tag_to_dense = nn.Sequential(
            nn.Linear(num_tags, tag_embedding_dim),
            nn.LayerNorm(tag_embedding_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable tag embeddings (used during training with gold tags)
        self.tag_embeddings = nn.Embedding(num_tags, tag_embedding_dim, padding_idx=0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
        gold_tags: Optional[torch.Tensor] = None,
        use_gold: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_dim] - input features (from BERT)
            lengths: [batch] - sequence lengths
            gold_tags: [batch, seq_len] - gold POS tags (optional, for training)
            use_gold: whether to use gold tags during training
        
        Returns:
            tag_logits: [batch, seq_len, num_tags] - tag prediction logits
            tag_embeddings: [batch, seq_len, tag_embedding_dim] - dense tag embeddings
            lstm_output: [batch, seq_len, hidden_dim*2] - tagger BiLSTM output
        """
        batch_size, seq_len, _ = x.shape
        
        # Pack and run BiLSTM
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.tagger_lstm(packed)
        lstm_output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        lstm_output = self.dropout(lstm_output)
        
        # Tag classification
        tag_logits = self.tag_classifier(lstm_output)  # [batch, seq_len, num_tags]
        
        # Get tag embeddings
        if use_gold and gold_tags is not None:
            # During training: use gold tag embeddings
            tag_emb = self.tag_embeddings(gold_tags)
        else:
            # During inference: use soft predictions (Gumbel-Softmax or straight-through)
            tag_probs = F.softmax(tag_logits, dim=-1)
            tag_emb = self.tag_to_dense(tag_probs)
        
        return tag_logits, tag_emb, lstm_output


class ParserEncoder(nn.Module):
    """
    Parser Encoder: BiLSTM ψ × N layers
    
    Processes the concatenated BERT + tagger output
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.33
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_dim = hidden_dim * 2  # bidirectional
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim]
            lengths: [batch]
        
        Returns:
            output: [batch, seq_len, hidden_dim * 2]
        """
        seq_len = x.size(1)
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        
        return output


class EdgeScorer(nn.Module):
    """
    Edge Scorer với 4 MLPs: MLP_{e^h, e^d, r^h, r^d}
    
    - e^h, e^d: representations for arc prediction (edge existence)
    - r^h, r^d: representations for relation/label prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        arc_dim: int = 512,
        label_dim: int = 128,
        num_labels: int = 50,
        dropout: float = 0.33,
        use_score_norm: bool = True
    ):
        super().__init__()
        
        self.num_labels = num_labels
        
        # MLPs for arc prediction (edge existence)
        self.arc_head_mlp = self._make_mlp(input_dim, arc_dim, dropout)  # e^h
        self.arc_dep_mlp = self._make_mlp(input_dim, arc_dim, dropout)   # e^d
        
        # MLPs for label prediction (edge labels/relations)
        self.label_head_mlp = self._make_mlp(input_dim, label_dim, dropout)  # r^h
        self.label_dep_mlp = self._make_mlp(input_dim, label_dim, dropout)   # r^d
        
        # Biaffine attention for arc scoring
        self.arc_attention = ScaledBiaffineAttention(
            head_dim=arc_dim,
            dep_dim=arc_dim,
            out_features=1,
            use_head_bias=True,
            use_dep_bias=False,
            use_score_norm=use_score_norm
        )
        
        # Biaffine attention for label scoring
        self.label_attention = ScaledBiaffineAttention(
            head_dim=label_dim,
            dep_dim=label_dim,
            out_features=num_labels,
            use_head_bias=True,
            use_dep_bias=True,
            use_score_norm=use_score_norm
        )
    
    def _make_mlp(self, input_dim: int, output_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, encoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_output: [batch, seq_len, input_dim]
        
        Returns:
            arc_scores: [batch, seq_len, seq_len] - score of token i having head j
            label_scores: [batch, seq_len, seq_len, num_labels]
        """
        # Arc representations
        arc_head = self.arc_head_mlp(encoder_output)  # [batch, seq_len, arc_dim]
        arc_dep = self.arc_dep_mlp(encoder_output)    # [batch, seq_len, arc_dim]
        
        # Label representations
        label_head = self.label_head_mlp(encoder_output)  # [batch, seq_len, label_dim]
        label_dep = self.label_dep_mlp(encoder_output)    # [batch, seq_len, label_dim]
        
        # Arc scores: [batch, seq_len, seq_len, 1] -> [batch, seq_len, seq_len]
        arc_scores = self.arc_attention(arc_head, arc_dep).squeeze(-1)
        
        # Label scores: [batch, seq_len, seq_len, num_labels]
        label_scores = self.label_attention(label_head, label_dep)
        
        return arc_scores, label_scores
