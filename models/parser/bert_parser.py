"""
BERT-based Dependency Parser with Multi-task Learning

Architecture from "Score Normalization for Biaffine Attention" (Gajo et al., 2025):
- BERT encoder (shared)
- Tagger branch: BiLSTM → MLP → POS tags → tag-to-dense
- Parser branch: BiLSTM × N → MLPs → Biaffine scorer (with 1/√d normalization)
- Multi-task: POS tagging + Dependency Parsing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from models.parser.base_parser import BaseParser
from models import BiaffineAttention


class BERTParser(BaseParser):
    """
    BERT + Multi-task (Tagger + Parser) with Score Normalization
    
    Architecture:
        Input → BERT → ⊕ (concat) → Parser BiLSTM → MLPs → Biaffine → Decoder
                  ↓
              Tagger BiLSTM → MLP → tags → tag-to-dense ──────────────↑
    """
    
    def __init__(
        self,
        bert_model: str = "vinai/phobert-base-v2",
        num_pos_tags: int = 20,
        num_labels: int = 50,
        # Tagger branch
        tagger_hidden_dim: int = 256,
        tag_dense_dim: int = 128,
        # Parser branch
        parser_hidden_dim: int = 400,
        parser_num_layers: int = 3,
        arc_dim: int = 500,
        label_dim: int = 100,
        # Common
        dropout: float = 0.33,
        freeze_bert: bool = False,
        use_score_norm: bool = True,
    ):
        super().__init__()
        
        self.num_pos_tags = num_pos_tags
        self.num_labels = num_labels
        self.use_score_norm = use_score_norm
        
        # ============ BERT Encoder ============
        try:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(bert_model)
            self.bert_dim = self.bert.config.hidden_size  # 768 for base
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # ============ Tagger Branch ============
        # BiLSTM φ for tagging
        self.tagger_lstm = nn.LSTM(
            input_size=self.bert_dim,
            hidden_size=tagger_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # MLP tag classifier
        self.tag_classifier = nn.Sequential(
            nn.Linear(tagger_hidden_dim * 2, tagger_hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(tagger_hidden_dim, num_pos_tags)
        )
        
        # Tag-to-dense MLP (convert predicted tags to dense representation)
        self.tag_embedding = nn.Embedding(num_pos_tags, tag_dense_dim)
        self.tag_to_dense = nn.Sequential(
            nn.Linear(tag_dense_dim, tag_dense_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # ============ Parser Branch ============
        # Input: BERT output ⊕ tag dense features
        parser_input_dim = self.bert_dim + tag_dense_dim
        
        # BiLSTM ψ × N for parsing
        self.parser_lstm = nn.LSTM(
            input_size=parser_input_dim,
            hidden_size=parser_hidden_dim,
            num_layers=parser_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if parser_num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # MLP_{e^h, e^d, r^h, r^d} - 4 MLPs for arc/label head/dep
        lstm_output_dim = parser_hidden_dim * 2
        
        # Arc MLPs (e^h, e^d)
        self.arc_head_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, arc_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, arc_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Label MLPs (r^h, r^d)
        self.label_head_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, label_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.label_dep_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, label_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Biaffine scorer with score normalization (1/√d)
        self.arc_attention = BiaffineAttention(
            arc_dim, 1, bias=(True, False), scale=use_score_norm
        )
        self.label_attention = BiaffineAttention(
            label_dim, num_labels, bias=(True, True), scale=use_score_norm
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
        gold_tags: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [batch, seq_len] - BERT input tokens
            attention_mask: [batch, seq_len] - attention mask
            lengths: [batch] - actual lengths
            gold_tags: [batch, seq_len] - gold POS tags (for training)
        
        Returns:
            arc_scores: [batch, seq_len, seq_len]
            label_scores: [batch, seq_len, seq_len, num_labels]
            tag_logits: [batch, seq_len, num_pos_tags]
        """
        batch_size, seq_len = input_ids.shape
        
        # ============ BERT Encoding ============
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_hidden = bert_output.last_hidden_state  # [batch, seq_len, bert_dim]
        
        # ============ Tagger Branch ============
        # BiLSTM φ
        tagger_out, _ = self.tagger_lstm(bert_hidden)  # [batch, seq_len, tagger_hidden*2]
        
        # MLP tag classifier
        tag_logits = self.tag_classifier(tagger_out)  # [batch, seq_len, num_pos_tags]
        
        # Get tags (gold during training, predicted during inference)
        if gold_tags is not None and self.training:
            tags = gold_tags
        else:
            tags = tag_logits.argmax(dim=-1)  # [batch, seq_len]
        
        # Tag-to-dense: convert tags to dense representation
        tag_emb = self.tag_embedding(tags)  # [batch, seq_len, tag_dense_dim]
        tag_features = self.tag_to_dense(tag_emb)  # [batch, seq_len, tag_dense_dim]
        
        # ============ Parser Branch ============
        # Concatenate BERT output ⊕ tag features
        parser_input = torch.cat([bert_hidden, tag_features], dim=-1)
        parser_input = self.dropout(parser_input)
        
        # Pack sequence for efficiency
        packed_input = nn.utils.rnn.pack_padded_sequence(
            parser_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM ψ × N
        packed_output, _ = self.parser_lstm(packed_input)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        lstm_out = self.dropout(lstm_out)
        
        # MLP_{e^h, e^d, r^h, r^d}
        arc_head = self.arc_head_mlp(lstm_out)
        arc_dep = self.arc_dep_mlp(lstm_out)
        label_head = self.label_head_mlp(lstm_out)
        label_dep = self.label_dep_mlp(lstm_out)
        
        # Biaffine scorer (with 1/√d normalization)
        arc_scores = self.arc_attention(arc_dep, arc_head).squeeze(-1)
        label_scores = self.label_attention(label_dep, label_head)
        
        return arc_scores, label_scores, tag_logits
    
    def loss(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        tag_logits: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        gold_tags: torch.Tensor,
        lengths: torch.Tensor,
        tag_loss_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-task loss: L = L_arc + L_label + λ * L_tag
        
        Returns:
            arc_loss, label_loss, tag_loss
        """
        batch_size, seq_len = heads.shape
        
        # Create mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=heads.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        
        # ============ Arc Loss ============
        arc_scores_masked = arc_scores.masked_fill(~mask.unsqueeze(1), -1e9)
        arc_loss = F.cross_entropy(
            arc_scores_masked.reshape(batch_size * seq_len, seq_len),
            heads.reshape(-1),
            ignore_index=0,
            reduction='sum'
        ) / mask.sum()
        
        # ============ Label Loss ============
        batch_indices = torch.arange(batch_size, device=label_scores.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=label_scores.device).unsqueeze(0).expand(batch_size, -1)
        selected_label_scores = label_scores[batch_indices, seq_indices, heads]
        
        label_loss = F.cross_entropy(
            selected_label_scores.reshape(batch_size * seq_len, -1),
            rels.reshape(-1),
            ignore_index=0,
            reduction='sum'
        ) / mask.sum()
        
        # ============ Tag Loss (Multi-task) ============
        tag_loss = F.cross_entropy(
            tag_logits.reshape(batch_size * seq_len, -1),
            gold_tags.reshape(-1),
            ignore_index=0,
            reduction='sum'
        ) / mask.sum()
        
        return arc_loss, label_loss, tag_loss * tag_loss_weight
    
    def decode(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding (can be replaced with MST for better results)
        """
        batch_size, seq_len, _ = arc_scores.shape
        
        pred_heads = arc_scores.argmax(dim=-1)
        
        batch_indices = torch.arange(batch_size, device=label_scores.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=label_scores.device).unsqueeze(0).expand(batch_size, -1)
        
        selected_label_scores = label_scores[batch_indices, seq_indices, pred_heads]
        pred_rels = selected_label_scores.argmax(dim=-1)
        
        for i, length in enumerate(lengths):
            pred_heads[i, length:] = 0
            pred_rels[i, length:] = 0
        
        return pred_heads, pred_rels

