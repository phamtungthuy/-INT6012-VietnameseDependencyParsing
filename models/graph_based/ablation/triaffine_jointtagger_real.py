# -*- coding: utf-8 -*-
"""
Triaffine Parser with Joint POS Tagging (REAL Implementation)

This is the REAL Triaffine + Joint Tagger that includes:
- Joint POS Tagger Branch (like triaffine_jointtagger.py)
- Sibling features with Triaffine scoring (like triaffine_parser.py)

Architecture:
    - Tagger Branch: BERT → BiLSTM × 1 → MLP → POS Tags
    - Parser Branch: (BERT ⊕ Tag Dense ⊕ Word Embed) → BiLSTM × 3 → 
                     [MLP_arc^h, MLP_arc^d, MLP_arc^s] → Triaffine → arc_score
                     [MLP_rel^h, MLP_rel^d, MLP_rel^s] → Triaffine → rel_score
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.nn import Model
from transforms.conll import CoNLL
from modules.base import BiLSTM, SharedDropout, MLP, Biaffine
from modules.bert import BertEmbedding
from utils.util_deep_learning import device
from utils.logs import logger
from utils.sp_alg import eisner, mst
from utils.sp_metric import AttachmentMetric


class Triaffine(nn.Module):
    """
    Triaffine Attention for second-order scoring.
    
    Computes: score(i, j, k) = h_i^T W h_j + h_i^T U h_k + h_j^T V h_k + ...
    
    Where:
    - i = head position
    - j = dependent position  
    - k = sibling position (another child of the same head)
    """
    
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Triaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        # Biaffine component (head-dep)
        self.biaffine_hd = Biaffine(n_in, n_out, bias_x, bias_y)
        
        # Biaffine component (head-sib)
        self.biaffine_hs = Biaffine(n_in, n_out, bias_x, bias_y)
        
        # Biaffine component (dep-sib)
        self.biaffine_ds = Biaffine(n_in, n_out, bias_x, bias_y)
        
    def forward(self, h_head, h_dep, h_sib=None):
        """
        Args:
            h_head: [batch, seq_len, n_in] - Head representations
            h_dep: [batch, seq_len, n_in] - Dependent representations
            h_sib: [batch, seq_len, n_in] - Sibling representations (optional)
            
        Returns:
            scores: [batch, seq_len, seq_len]
        """
        # Standard biaffine (head-dep): [batch, seq, seq]
        s_hd = self.biaffine_hd(h_dep, h_head)
        
        if h_sib is None:
            return s_hd
        
        # Head-sibling interaction: [batch, seq, seq]
        s_hs = self.biaffine_hs(h_sib, h_head)
        
        # Dep-sibling interaction: [batch, seq, seq]
        s_ds = self.biaffine_ds(h_sib, h_dep)
        
        # Aggregate sibling influence (mean over potential sibling positions)
        sib_bonus = (s_hs + s_ds).mean(dim=-1, keepdim=True)
        
        return s_hd + sib_bonus.expand_as(s_hd)


class TriaffineJointTaggerReal(Model):
    """
    REAL Triaffine Parser with Joint POS Tagger.
    
    Features:
    1. Joint Tagger Branch: Predicts POS tags from BERT
    2. Parser Branch: Uses BERT + predicted tag info for parsing
    3. Sibling features: MLP_arc^s, MLP_rel^s with Triaffine scoring
    """
    
    NAME = 'triaffine-jointtagger-real'
    
    def __init__(
        self,
        n_words=None,
        n_tags=None,
        n_rels=None,
        n_embed=100,
        pad_index=0,
        unk_index=1,
        bert='vinai/phobert-base',
        n_bert_layers=4,
        bert_fine_tune=True,
        # Tagger branch params
        n_tagger_lstm_hidden=256,
        n_tagger_lstm_layers=1,
        tagger_dropout=0.33,
        n_tag_dense=128,
        # Parser branch params
        n_parser_lstm_hidden=400,
        n_parser_lstm_layers=3,
        parser_dropout=0.33,
        n_mlp_arc=500,
        n_mlp_rel=100,
        n_mlp_sib=100,  # NEW: Sibling MLP dimension
        mlp_dropout=0.33,
        feat_pad_index=0,
        use_sibling=True,  # NEW: Toggle sibling features
        transform=None,
        init_pre_train=False
    ):
        super(TriaffineJointTaggerReal, self).__init__()
        
        self.bert_name = bert
        self.n_tags = n_tags
        self.use_sibling = use_sibling
        
        self.args = {
            'n_words': n_words,
            'n_tags': n_tags,
            'n_rels': n_rels,
            'n_embed': n_embed,
            'pad_index': pad_index,
            'unk_index': unk_index,
            'bert': bert,
            'n_bert_layers': n_bert_layers,
            'n_tagger_lstm_hidden': n_tagger_lstm_hidden,
            'n_parser_lstm_hidden': n_parser_lstm_hidden,
            'n_tag_dense': n_tag_dense,
            'n_mlp_arc': n_mlp_arc,
            'n_mlp_rel': n_mlp_rel,
            'n_mlp_sib': n_mlp_sib,
            'use_sibling': use_sibling,
            'feat_pad_index': feat_pad_index
        }
        
        if init_pre_train:
            return
        
        logger.info(f"[TriaffineJointTaggerReal] Joint POS Tagger + Sibling Features")
        logger.info(f"  - n_tags: {n_tags}, use_sibling: {use_sibling}")
        
        # =====================================================================
        # SHARED: BERT Embedding
        # =====================================================================
        self.bert_embed = BertEmbedding(
            model=bert,
            n_layers=n_bert_layers,
            n_out=None,
            pad_index=feat_pad_index,
            dropout=0.0,
            requires_grad=bert_fine_tune
        )
        bert_hidden_size = self.bert_embed.n_out
        
        # =====================================================================
        # TAGGER BRANCH
        # =====================================================================
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
        
        # tag-to-dense MLP
        self.tag_to_dense = nn.Sequential(
            nn.Linear(n_tags, n_tag_dense),
            nn.ReLU(),
            nn.Dropout(tagger_dropout)
        )
        
        # =====================================================================
        # PARSER BRANCH
        # =====================================================================
        parser_input_size = bert_hidden_size + n_tag_dense
        
        # Word embedding (optional)
        if n_words is not None and n_embed > 0:
            self.word_embed = nn.Embedding(n_words, n_embed)
            parser_input_size += n_embed
        else:
            self.word_embed = None
        
        # BiLSTM
        self.parser_lstm = BiLSTM(
            input_size=parser_input_size,
            hidden_size=n_parser_lstm_hidden,
            num_layers=n_parser_lstm_layers,
            dropout=parser_dropout
        )
        self.parser_lstm_dropout = SharedDropout(p=parser_dropout)
        
        # MLP for Arc (head, dependent, sibling)
        self.mlp_arc_h = MLP(n_in=n_parser_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=n_parser_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        if use_sibling:
            self.mlp_arc_s = MLP(n_in=n_parser_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        
        # MLP for Rel (head, dependent, sibling)
        self.mlp_rel_h = MLP(n_in=n_parser_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_parser_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        if use_sibling:
            self.mlp_rel_s = MLP(n_in=n_parser_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        
        # Triaffine/Biaffine Attention
        if use_sibling:
            self.arc_attn = Triaffine(n_in=n_mlp_arc, n_out=1, bias_x=True, bias_y=False)
            self.rel_attn = Triaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        else:
            self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
            self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        
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
        
        Returns:
            s_tag: [batch, seq_len, n_tags] - Tag logits
            s_arc: [batch, seq_len, seq_len] - Arc scores
            s_rel: [batch, seq_len, seq_len, n_rels] - Rel scores
        """
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        
        # BERT Embedding
        bert_out = self.bert_embed(feats)
        
        # =====================================================================
        # TAGGER BRANCH
        # =====================================================================
        tagger_x = pack_padded_sequence(bert_out, lens.cpu(), batch_first=True, enforce_sorted=False)
        tagger_x, _ = self.tagger_lstm(tagger_x)
        tagger_x, _ = pad_packed_sequence(tagger_x, batch_first=True, total_length=seq_len)
        tagger_x = self.tagger_lstm_dropout(tagger_x)
        
        # Tag logits
        s_tag = self.tag_classifier(tagger_x)
        
        # Get tag probabilities for tag-to-dense
        if use_gold_tags and gold_tags is not None:
            tag_probs = torch.zeros_like(s_tag).scatter_(-1, gold_tags.unsqueeze(-1), 1.0)
        else:
            tag_probs = torch.softmax(s_tag, dim=-1)
        
        # tag-to-dense
        tag_dense = self.tag_to_dense(tag_probs)
        
        # =====================================================================
        # PARSER BRANCH
        # =====================================================================
        parser_input = torch.cat([bert_out, tag_dense], dim=-1)
        
        if self.word_embed is not None:
            parser_input = torch.cat([parser_input, self.word_embed(words)], dim=-1)
        
        # BiLSTM
        parser_x = pack_padded_sequence(parser_input, lens.cpu(), batch_first=True, enforce_sorted=False)
        parser_x, _ = self.parser_lstm(parser_x)
        parser_x, _ = pad_packed_sequence(parser_x, batch_first=True, total_length=seq_len)
        parser_x = self.parser_lstm_dropout(parser_x)
        
        # MLP
        arc_h = self.mlp_arc_h(parser_x)
        arc_d = self.mlp_arc_d(parser_x)
        rel_h = self.mlp_rel_h(parser_x)
        rel_d = self.mlp_rel_d(parser_x)
        
        # Triaffine or Biaffine
        if self.use_sibling:
            arc_s = self.mlp_arc_s(parser_x)
            rel_s = self.mlp_rel_s(parser_x)
            s_arc = self.arc_attn(arc_h, arc_d, arc_s)
            s_rel = self.rel_attn(rel_h, rel_d, rel_s).permute(0, 2, 3, 1)
        else:
            s_arc = self.arc_attn(arc_d, arc_h)
            s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        
        # Mask
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        
        return s_tag, s_arc, s_rel
    
    def forward_loss(self, s_tag, s_arc, s_rel, gold_tags, arcs, rels, mask):
        """Compute joint loss: tag_loss + arc_loss + rel_loss."""
        # Tag loss
        tag_loss = self.tag_criterion(s_tag[mask], gold_tags[mask])
        
        # Arc loss
        arc_loss = self.arc_criterion(s_arc[mask], arcs[mask])
        
        # Rel loss
        s_rel_masked = s_rel[mask]
        arcs_masked = arcs[mask]
        rels_masked = rels[mask]
        s_rel_selected = s_rel_masked[torch.arange(len(arcs_masked)), arcs_masked]
        rel_loss = self.rel_criterion(s_rel_selected, rels_masked)
        
        return tag_loss + arc_loss + rel_loss
    
    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """Decode arc and rel predictions."""
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        
        bad = [not CoNLL.istree(seq[1:i + 1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds
    
    def decode_tags(self, s_tag):
        """Decode tag predictions."""
        return s_tag.argmax(-1)
    
    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate on a data loader."""
        self.eval()
        
        total_loss = 0
        metric = AttachmentMetric()
        tag_correct, tag_total = 0, 0
        
        for batch in loader:
            words, feats, tags, arcs, rels = batch
            
            mask = words.ne(self.pad_index)
            mask[:, 0] = 0
            
            s_tag, s_arc, s_rel = self.forward(words, feats)
            loss = self.forward_loss(s_tag, s_arc, s_rel, tags, arcs, rels, mask)
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
            tag_preds = self.decode_tags(s_tag)
            
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
            
            # Tag accuracy
            tag_correct += (tag_preds[mask] == tags[mask]).sum().item()
            tag_total += mask.sum().item()
        
        tag_acc = tag_correct / tag_total if tag_total > 0 else 0
        return total_loss / len(loader), metric, tag_acc
    
    def save(self, path):
        """Save model."""
        model = self.module if hasattr(self, 'module') else self
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({
            'name': self.NAME,
            'args': self.args,
            'state_dict': state_dict,
            'transform': self.transform
        }, path)
    
    @classmethod
    def load(cls, path):
        """Load model."""
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(**state['args'], transform=state['transform'])
        model.load_state_dict(state['state_dict'], strict=False)
        model.eval()
        return model.to(device)
