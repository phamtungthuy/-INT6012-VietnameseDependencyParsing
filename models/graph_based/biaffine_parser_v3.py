# -*- coding: utf-8 -*-
"""
Biaffine Dependency Parser V3 - Joint POS Tagging + Parsing
Based on the architecture diagram with:
- Tagger branch: BERT → BiLSTM → MLP → POS Tags
- Parser branch: (BERT ⊕ Tag Dense) → BiLSTM → MLP → Biaffine → Edges
"""

import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.nn import Model
from transforms.conll import CoNLL
from modules.base import BiLSTM, SharedDropout, MLP, Biaffine, IndependentDropout
from modules.bert import BertEmbedding
from utils.sp_fn import ispunct
from utils.util_deep_learning import device
from utils.logs import logger
from utils.sp_data import Dataset
from utils.sp_alg import eisner, mst
from utils.constants import PRETRAINED
from utils.sp_metric import AttachmentMetric


class BiaffineParserV3(Model):
    """
    Joint POS Tagger + Biaffine Dependency Parser
    
    Architecture (from diagram):
    ┌───────────────────────────────────────────────────────────────────┐
    │                              INPUT                                │
    │                                ↓                                  │
    │                              BERT                                 │
    │                         ↓           ↓                             │
    │  ┌─────────────────────────┐   ┌─────────────────────────────┐   │
    │  │      TAGGER BRANCH      │   │       PARSER BRANCH          │   │
    │  │                         │   │                               │   │
    │  │  tagger BiLSTM φ       │   │  BiLSTM ψ × N                │   │
    │  │         ↓               │   │         ↓                     │   │
    │  │  MLP tag classifier    │   │  MLP (arc_h, arc_d, rel_h,   │   │
    │  │         ↓               │   │       rel_d)                  │   │
    │  │       tags             │   │         ↓                     │   │
    │  │         ↓               │   │  Biaffine Scorer             │   │
    │  │  tag-to-dense MLP ─────┼──→│         ↓                     │   │
    │  │                         │   │  Score Decoder               │   │
    │  └─────────────────────────┘   │         ↓    ↓               │   │
    │                                │      edges  labels           │   │
    │                                └─────────────────────────────┘   │
    └───────────────────────────────────────────────────────────────────┘
    """
    
    NAME = 'biaffine-dependency-v3-joint'
    
    def __init__(
        self,
        n_words=None,
        n_tags=None,
        n_rels=None,
        pad_index=0,
        unk_index=1,
        bert='vinai/phobert-base',
        n_bert_layers=4,
        bert_fine_tune=True,
        # Tagger branch params
        n_tagger_lstm_hidden=256,
        n_tagger_lstm_layers=1,
        tagger_dropout=0.33,
        # Parser branch params
        n_parser_lstm_hidden=400,
        n_parser_lstm_layers=3,
        parser_dropout=0.33,
        # MLP params
        n_tag_dense=128,  # tag-to-dense MLP output
        n_mlp_arc=500,
        n_mlp_rel=100,
        mlp_dropout=0.33,
        # Others
        embed_dropout=0.33,
        feat_pad_index=0,
        transform=None,
        init_pre_train=False
    ):
        super(BiaffineParserV3, self).__init__()
        
        self.bert_name = bert
        self.n_tags = n_tags
        self.n_rels = n_rels
        
        self.args = {
            'n_words': n_words,
            'n_tags': n_tags,
            'n_rels': n_rels,
            'pad_index': pad_index,
            'unk_index': unk_index,
            'bert': bert,
            'n_bert_layers': n_bert_layers,
            'n_tagger_lstm_hidden': n_tagger_lstm_hidden,
            'n_parser_lstm_hidden': n_parser_lstm_hidden,
            'n_tag_dense': n_tag_dense,
            'feat_pad_index': feat_pad_index
        }
        
        if init_pre_train:
            return
        
        # =====================================================================
        # SHARED: BERT Embedding
        # =====================================================================
        self.bert_embed = BertEmbedding(
            model=bert,
            n_layers=n_bert_layers,
            n_out=None,  # Use BERT's original hidden size
            pad_index=feat_pad_index,
            dropout=0.0,
            requires_grad=bert_fine_tune
        )
        bert_hidden_size = self.bert_embed.n_out
        
        # =====================================================================
        # TAGGER BRANCH (Left side of diagram)
        # =====================================================================
        
        # tagger BiLSTM φ
        self.tagger_lstm = BiLSTM(
            input_size=bert_hidden_size,
            hidden_size=n_tagger_lstm_hidden,
            num_layers=n_tagger_lstm_layers,
            dropout=tagger_dropout
        )
        self.tagger_lstm_dropout = SharedDropout(p=tagger_dropout)
        
        # MLP tag classifier
        self.tag_classifier = nn.Sequential(
            nn.Linear(n_tagger_lstm_hidden * 2, n_tagger_lstm_hidden),
            nn.ReLU(),
            nn.Dropout(tagger_dropout),
            nn.Linear(n_tagger_lstm_hidden, n_tags)
        )
        
        # tag-to-dense MLP (sends info to parser branch)
        self.tag_to_dense = nn.Sequential(
            nn.Linear(n_tags, n_tag_dense),
            nn.ReLU(),
            nn.Dropout(tagger_dropout)
        )
        
        # =====================================================================
        # PARSER BRANCH (Right side of diagram)
        # =====================================================================
        
        # Input: BERT ⊕ tag_dense
        parser_input_size = bert_hidden_size + n_tag_dense
        
        # BiLSTM ψ × N (N layers)
        self.parser_lstm = BiLSTM(
            input_size=parser_input_size,
            hidden_size=n_parser_lstm_hidden,
            num_layers=n_parser_lstm_layers,
            dropout=parser_dropout
        )
        self.parser_lstm_dropout = SharedDropout(p=parser_dropout)
        
        # MLP layers for arc and rel
        self.mlp_arc_d = MLP(n_in=n_parser_lstm_hidden * 2,
                            n_out=n_mlp_arc,
                            dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=n_parser_lstm_hidden * 2,
                            n_out=n_mlp_arc,
                            dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_parser_lstm_hidden * 2,
                            n_out=n_mlp_rel,
                            dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=n_parser_lstm_hidden * 2,
                            n_out=n_mlp_rel,
                            dropout=mlp_dropout)
        
        # Biaffine scorer
        self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        
        # Score normalization: α = 1/√d (as shown in diagram)
        self.score_scale = (n_mlp_arc ** -0.5)
        
        # Loss functions
        self.tag_criterion = nn.CrossEntropyLoss()
        self.arc_criterion = nn.CrossEntropyLoss()
        self.rel_criterion = nn.CrossEntropyLoss()
        
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.transform = transform
    
    def forward(self, words, feats, gold_tags=None, use_gold_tags=False):
        """
        Forward pass through the joint model.
        
        Args:
            words: [batch_size, seq_len] - Word indices (for masking)
            feats: [batch_size, seq_len, fix_len] - BERT subword indices
            gold_tags: [batch_size, seq_len] - Gold POS tags (for training)
            use_gold_tags: If True, use gold tags instead of predicted tags
            
        Returns:
            s_tag: [batch_size, seq_len, n_tags] - Tag logits
            s_arc: [batch_size, seq_len, seq_len] - Arc scores
            s_rel: [batch_size, seq_len, seq_len, n_rels] - Rel scores
        """
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        
        # =====================================================================
        # BERT Embedding (Shared)
        # =====================================================================
        bert_out = self.bert_embed(feats)  # [batch, seq_len, bert_hidden]
        
        # =====================================================================
        # TAGGER BRANCH
        # =====================================================================
        # tagger BiLSTM φ
        tagger_x = pack_padded_sequence(bert_out, lens.cpu(), batch_first=True, enforce_sorted=False)
        tagger_x, _ = self.tagger_lstm(tagger_x)
        tagger_x, _ = pad_packed_sequence(tagger_x, batch_first=True, total_length=seq_len)
        tagger_x = self.tagger_lstm_dropout(tagger_x)
        
        # MLP tag classifier → tag logits
        s_tag = self.tag_classifier(tagger_x)  # [batch, seq_len, n_tags]
        
        # Get tag probabilities for tag-to-dense
        if use_gold_tags and gold_tags is not None:
            # Use one-hot of gold tags during training (teacher forcing)
            tag_probs = torch.zeros_like(s_tag).scatter_(-1, gold_tags.unsqueeze(-1), 1.0)
        else:
            # Use softmax of predictions
            tag_probs = torch.softmax(s_tag, dim=-1)
        
        # tag-to-dense MLP → dense tag representation
        tag_dense = self.tag_to_dense(tag_probs)  # [batch, seq_len, n_tag_dense]
        
        # =====================================================================
        # PARSER BRANCH
        # =====================================================================
        # Concatenate BERT ⊕ tag_dense
        parser_input = torch.cat([bert_out, tag_dense], dim=-1)
        
        # BiLSTM ψ × N
        parser_x = pack_padded_sequence(parser_input, lens.cpu(), batch_first=True, enforce_sorted=False)
        parser_x, _ = self.parser_lstm(parser_x)
        parser_x, _ = pad_packed_sequence(parser_x, batch_first=True, total_length=seq_len)
        parser_x = self.parser_lstm_dropout(parser_x)
        
        # MLP layers
        arc_d = self.mlp_arc_d(parser_x)
        arc_h = self.mlp_arc_h(parser_x)
        rel_d = self.mlp_rel_d(parser_x)
        rel_h = self.mlp_rel_h(parser_x)
        
        # Biaffine scorer
        s_arc = self.arc_attn(arc_d, arc_h)  # [batch, seq_len, seq_len]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)  # [batch, seq_len, seq_len, n_rels]
        
        # Score normalization: α = 1/√d
        s_arc = s_arc * self.score_scale
        
        # Mask invalid positions
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        
        return s_tag, s_arc, s_rel
    
    def forward_loss(self, s_tag, s_arc, s_rel, gold_tags, arcs, rels, mask):
        """
        Compute joint loss for both tagging and parsing.
        """
        # Tag loss
        s_tag_flat = s_tag[mask]
        gold_tags_flat = gold_tags[mask]
        tag_loss = self.tag_criterion(s_tag_flat, gold_tags_flat)
        
        # Arc loss
        s_arc_masked = s_arc[mask]
        arcs_masked = arcs[mask]
        arc_loss = self.arc_criterion(s_arc_masked, arcs_masked)
        
        # Rel loss
        s_rel_masked = s_rel[mask]
        rels_masked = rels[mask]
        s_rel_selected = s_rel_masked[torch.arange(len(arcs_masked)), arcs_masked]
        rel_loss = self.rel_criterion(s_rel_selected, rels_masked)
        
        # Joint loss (can add weights if needed)
        total_loss = tag_loss + arc_loss + rel_loss
        
        return total_loss, tag_loss, arc_loss, rel_loss
    
    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """Decode arc and relation predictions."""
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        
        # Ensure tree structure if needed
        bad = [not CoNLL.istree(seq[1:i + 1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        
        return arc_preds, rel_preds
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate on a data loader."""
        self.eval()
        
        total_loss = 0
        tag_correct, tag_total = 0, 0
        arc_metric = AttachmentMetric()
        
        for batch in loader:
            words, feats, tags, arcs, rels = batch
            
            mask = words.ne(self.pad_index)
            mask[:, 0] = 0  # Ignore ROOT
            
            # Forward (use predicted tags during eval)
            s_tag, s_arc, s_rel = self.forward(words, feats, use_gold_tags=False)
            
            # Loss
            loss, _, _, _ = self.forward_loss(s_tag, s_arc, s_rel, tags, arcs, rels, mask)
            total_loss += loss.item()
            
            # Tag accuracy
            tag_preds = s_tag.argmax(-1)
            tag_correct += (tag_preds[mask] == tags[mask]).sum().item()
            tag_total += mask.sum().item()
            
            # Arc/Rel accuracy
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
            arc_metric(arc_preds, rel_preds, arcs, rels, mask)
        
        avg_loss = total_loss / len(loader)
        tag_acc = tag_correct / tag_total if tag_total > 0 else 0
        
        return avg_loss, tag_acc, arc_metric
    
    def save(self, path):
        """Save model checkpoint."""
        model = self.module if hasattr(self, 'module') else self
        
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        state = {
            'name': self.NAME,
            'args': self.args,
            'state_dict': state_dict,
            'transform': self.transform
        }
        torch.save(state, path)
    
    @classmethod
    def load(cls, path):
        """Load model from checkpoint."""
        state = torch.load(path, map_location=device, weights_only=False)
        args = state['args']
        
        model = cls(
            n_words=args['n_words'],
            n_tags=args['n_tags'],
            n_rels=args['n_rels'],
            pad_index=args['pad_index'],
            unk_index=args['unk_index'],
            bert=args['bert'],
            n_bert_layers=args.get('n_bert_layers', 4),
            n_tagger_lstm_hidden=args.get('n_tagger_lstm_hidden', 256),
            n_parser_lstm_hidden=args.get('n_parser_lstm_hidden', 400),
            n_tag_dense=args.get('n_tag_dense', 128),
            feat_pad_index=args.get('feat_pad_index', 0),
            transform=state['transform']
        )
        
        model.load_state_dict(state['state_dict'], strict=False)
        model.eval()
        model.to(device)
        
        return model
