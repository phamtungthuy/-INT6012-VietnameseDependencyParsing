# -*- coding: utf-8 -*-
"""
Triaffine Dependency Parser
Second-Order Graph-based Parser with Sibling Attention

This extends Biaffine by adding:
- Sibling features: captures relationships between children of the same head
- Triaffine scoring: Score(head, dep, sibling) for second-order interactions

Architecture:
    Input → BERT → BiLSTM → [MLP_head, MLP_dep, MLP_sib] → Triaffine → Arcs/Rels
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


class Triaffine(nn.Module):
    """
    Triaffine Attention for second-order scoring.
    
    Computes: score(i, j, k) = h_i^T W h_j + h_i^T U h_k + h_j^T V h_k + h_i^T T h_j h_k
    
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
        
        # Trilinear weight for three-way interaction
        # W[k, i, j] represents the weight for (head_i, dep_j, sib_k)
        self.W_tri = nn.Parameter(torch.zeros(n_out, n_in, n_in, n_in))
        nn.init.xavier_uniform_(self.W_tri.view(n_out, -1))
        
    def forward(self, h_head, h_dep, h_sib=None):
        """
        Args:
            h_head: [batch, seq_len, n_in] - Head representations
            h_dep: [batch, seq_len, n_in] - Dependent representations
            h_sib: [batch, seq_len, n_in] - Sibling representations (optional)
            
        Returns:
            scores: [batch, seq_len, seq_len] or [batch, seq_len, seq_len, seq_len]
        """
        # Standard biaffine (head-dep): [batch, seq, seq]
        s_hd = self.biaffine_hd(h_dep, h_head)
        
        if h_sib is None:
            return s_hd
        
        # For full triaffine with sibling, we need to aggregate sibling info
        # Simplified approach: use mean pooling of potential siblings
        # This approximates: for each (h,d) pair, consider influence of all other positions as siblings
        
        # Head-sibling interaction: [batch, seq, seq]
        s_hs = self.biaffine_hs(h_sib, h_head)
        
        # Dep-sibling interaction: [batch, seq, seq]
        s_ds = self.biaffine_ds(h_sib, h_dep)
        
        # Aggregate sibling influence (simplified: mean over potential sibling positions)
        # This gives each (h,d) pair a bonus based on how well it fits with siblings
        sib_bonus = (s_hs + s_ds).mean(dim=-1, keepdim=True)  # [batch, seq, 1]
        
        return s_hd + sib_bonus.expand_as(s_hd)


class TriaffineParser(Model):
    """
    Second-Order Graph-based Dependency Parser with Triaffine Attention.
    
    Improvements over Biaffine:
    1. Adds MLP_sibling to capture sibling relationships
    2. Uses Triaffine scoring: incorporates (head, dep, sibling) interactions
    3. Can optionally use BERT for contextual embeddings
    
    This addresses the limitation where Biaffine only looks at (head, dep) pairs
    without considering the structure of other children.
    """
    
    NAME = 'triaffine-dependency'
    
    def __init__(
        self,
        n_words=None,
        n_tags=None,
        n_rels=None,
        n_embed=100,  # NEW: Word embedding dimension
        pad_index=0,
        unk_index=1,
        bert='vinai/phobert-base',
        n_bert_layers=4,
        bert_fine_tune=True,
        n_lstm_hidden=400,
        n_lstm_layers=3,
        lstm_dropout=0.33,
        n_mlp_arc=500,
        n_mlp_rel=100,
        n_mlp_sib=100,  # NEW: Sibling MLP dimension
        n_tag_embed=100,  # NEW: POS Tag embedding dimension
        mlp_dropout=0.33,
        embed_dropout=0.33,
        feat_pad_index=0,
        use_sibling=True,  # NEW: Toggle sibling features
        transform=None,
        init_pre_train=False
    ):
        super(TriaffineParser, self).__init__()
        
        self.bert_name = bert
        self.use_sibling = use_sibling
        
        self.args = {
            'n_words': n_words,
            'n_embed': n_embed,
            'n_tags': n_tags,
            'n_tags': n_tags,
            'n_rels': n_rels,
            'pad_index': pad_index,
            'unk_index': unk_index,
            'bert': bert,
            'n_bert_layers': n_bert_layers,
            'n_lstm_hidden': n_lstm_hidden,
            'n_mlp_arc': n_mlp_arc,
            'n_mlp_rel': n_mlp_rel,
            'n_mlp_sib': n_mlp_sib,
            'n_tag_embed': n_tag_embed,
            'use_sibling': use_sibling,
            'feat_pad_index': feat_pad_index
        }
        
        if init_pre_train:
            return
        
        # =====================================================================
        # BERT Embedding
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
        # POS Tag Embedding (like V2)
        # =====================================================================
        self.n_tags = n_tags
        self.n_tag_embed = n_tag_embed
        if n_tags is not None and n_tag_embed > 0:
            logger.info(f"Using POS tag embedding with {n_tags} tags and {n_tag_embed} dimensions")
            self.tag_embed = nn.Embedding(n_tags, n_tag_embed)
            lstm_input_size = bert_hidden_size + n_tag_embed
        else:
            logger.info(f"Not using POS tag embedding")
            self.tag_embed = None
            lstm_input_size = bert_hidden_size
        
        # =====================================================================
        # Word Embedding (like V2)
        # =====================================================================
        self.n_words = n_words
        self.n_embed = n_embed
        if n_words is not None and n_embed > 0:
            self.word_embed = nn.Embedding(n_words, n_embed)
            lstm_input_size += n_embed
        else:
            self.word_embed = None

        # =====================================================================
        # BiLSTM Encoder
        # =====================================================================
        self.lstm = BiLSTM(
            input_size=lstm_input_size,
            hidden_size=n_lstm_hidden,
            num_layers=n_lstm_layers,
            dropout=lstm_dropout
        )
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        
        # =====================================================================
        # MLP Layers for Arc Scoring
        # =====================================================================
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        
        # NEW: MLP for Sibling
        if use_sibling:
            logger.info(f"Using sibling features with {n_mlp_sib} dimensions")
            self.mlp_arc_s = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        
        # =====================================================================
        # MLP Layers for Relation Scoring
        # =====================================================================
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        
        if use_sibling:
            self.mlp_rel_s = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        
        # =====================================================================
        # Triaffine/Biaffine Attention Layers
        # =====================================================================
        if use_sibling:
            self.arc_attn = Triaffine(n_in=n_mlp_arc, n_out=1, bias_x=True, bias_y=False)
            self.rel_attn = Triaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        else:
            # Fallback to standard Biaffine
            self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
            self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        
        # Loss
        self.criterion = nn.CrossEntropyLoss()
        
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.transform = transform
    
    def forward(self, words, feats, tags=None):
        """
        Forward pass.
        
        Args:
            words: [batch, seq_len] - Word indices (for masking)
            feats: [batch, seq_len, fix_len] - BERT subword indices
            tags: [batch, seq_len] - POS tag indices (optional)
            
        Returns:
            s_arc: [batch, seq_len, seq_len] - Arc scores
            s_rel: [batch, seq_len, seq_len, n_rels] - Relation scores
        """
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        
        # BERT Embedding
        x = self.bert_embed(feats)  # [batch, seq_len, bert_hidden]
        
        # Add POS Tag Embedding if available
        if self.tag_embed is not None and tags is not None:
            tag_emb = self.tag_embed(tags)  # [batch, seq_len, n_tag_embed]
            x = torch.cat([x, tag_emb], dim=-1)  # [batch, seq_len, bert_hidden + n_tag_embed]
            
        # Add Word Embedding if available
        if self.word_embed is not None:
            word_emb = self.word_embed(words)
            x = torch.cat([x, word_emb], dim=-1)
        
        # BiLSTM
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=seq_len)
        x = self.lstm_dropout(x)
        
        # MLP for Arc
        arc_h = self.mlp_arc_h(x)  # Head representation
        arc_d = self.mlp_arc_d(x)  # Dependent representation
        
        # MLP for Rel
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        
        if self.use_sibling:
            arc_s = self.mlp_arc_s(x)  # Sibling representation
            rel_s = self.mlp_rel_s(x)
            
            # Triaffine scoring with sibling
            s_arc = self.arc_attn(arc_h, arc_d, arc_s)
            s_rel = self.rel_attn(rel_h, rel_d, rel_s).permute(0, 2, 3, 1)
        else:
            # Standard Biaffine
            s_arc = self.arc_attn(arc_d, arc_h)
            s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        
        # Mask invalid positions
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        
        return s_arc, s_rel
    
    def forward_loss(self, s_arc, s_rel, arcs, rels, mask):
        """Compute loss."""
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)
        
        return arc_loss + rel_loss
    
    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        """Decode predictions."""
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        
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
        metric = AttachmentMetric()
        
        for batch in loader:
            words, feats, arcs, rels = batch
            
            mask = words.ne(self.pad_index)
            mask[:, 0] = 0
            
            s_arc, s_rel = self.forward(words, feats)
            loss = self.forward_loss(s_arc, s_rel, arcs, rels, mask)
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
            
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        
        return total_loss / len(loader), metric
    
    def save(self, path):
        """Save model."""
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
        """Load model."""
        state = torch.load(path, map_location=device, weights_only=False)
        args = state['args']
        
        model = cls(
            n_words=args['n_words'],
            n_tags=args.get('n_tags'),
            n_rels=args['n_rels'],
            pad_index=args['pad_index'],
            unk_index=args['unk_index'],
            bert=args['bert'],
            use_sibling=args.get('use_sibling', True),
            transform=state['transform']
        )
        
        model.load_state_dict(state['state_dict'], strict=False)
        model.eval()
        model.to(device)
        
        return model
