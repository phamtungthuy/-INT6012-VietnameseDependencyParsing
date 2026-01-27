# -*- coding: utf-8 -*-
"""
Triaffine Parser WITHOUT BERT (Ablation Study)

Base: triaffine_parser.py
Ablation: Replace BERT with CharLSTM + Word Embeddings

Architecture:
    Input → [CharLSTM + Word Embed + POS Tag Embed] → BiLSTM → 
    [MLP_head, MLP_dep, MLP_sib] → Triaffine → Arcs/Rels

Purpose: Measure how much BERT contributes to parsing performance
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.nn import Model
from transforms.conll import CoNLL
from modules.base import BiLSTM, SharedDropout, MLP, Biaffine, CharLSTM
from utils.util_deep_learning import device
from utils.logs import logger
from utils.sp_alg import eisner, mst
from utils.sp_metric import AttachmentMetric


class Triaffine(nn.Module):
    """Triaffine Attention for second-order scoring."""
    
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Triaffine, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        self.biaffine_hd = Biaffine(n_in, n_out, bias_x, bias_y)
        self.biaffine_hs = Biaffine(n_in, n_out, bias_x, bias_y)
        self.biaffine_ds = Biaffine(n_in, n_out, bias_x, bias_y)
        
    def forward(self, h_head, h_dep, h_sib=None):
        s_hd = self.biaffine_hd(h_dep, h_head)
        
        if h_sib is None:
            return s_hd
        
        s_hs = self.biaffine_hs(h_sib, h_head)
        s_ds = self.biaffine_ds(h_sib, h_dep)
        sib_bonus = (s_hs + s_ds).mean(dim=-1, keepdim=True)
        
        return s_hd + sib_bonus.expand_as(s_hd)


class TriaffineNoBert(Model):
    """
    Triaffine Parser WITHOUT BERT - uses CharLSTM + Word Embeddings.
    
    This is an ablation model to compare BERT vs traditional embeddings.
    """
    
    NAME = 'triaffine-nobert'
    
    def __init__(
        self,
        n_words=None,
        n_chars=None,
        n_tags=None,
        n_rels=None,
        n_embed=100,          # Word embedding dimension
        n_char_embed=50,      # Character embedding dimension
        n_char_hidden=100,    # CharLSTM hidden dimension
        n_tag_embed=100,      # POS Tag embedding dimension
        pad_index=0,
        unk_index=1,
        n_lstm_hidden=400,
        n_lstm_layers=3,
        lstm_dropout=0.33,
        n_mlp_arc=500,
        n_mlp_rel=100,
        mlp_dropout=0.33,
        embed_dropout=0.33,
        use_sibling=True,
        transform=None,
        init_pre_train=False
    ):
        super(TriaffineNoBert, self).__init__()
        
        self.use_sibling = use_sibling
        
        self.args = {
            'n_words': n_words,
            'n_chars': n_chars,
            'n_tags': n_tags,
            'n_rels': n_rels,
            'n_embed': n_embed,
            'n_char_embed': n_char_embed,
            'n_char_hidden': n_char_hidden,
            'n_tag_embed': n_tag_embed,
            'pad_index': pad_index,
            'unk_index': unk_index,
            'n_lstm_hidden': n_lstm_hidden,
            'n_mlp_arc': n_mlp_arc,
            'n_mlp_rel': n_mlp_rel,
            'use_sibling': use_sibling
        }
        
        if init_pre_train:
            return
        
        logger.info(f"[TriaffineNoBert] Using CharLSTM + Word + Tag (NO BERT)")
        
        # =====================================================================
        # Word Embedding (with pretrained option)
        # =====================================================================
        self.word_embed = nn.Embedding(n_words, n_embed)
        
        # =====================================================================
        # Character-level LSTM
        # =====================================================================
        self.char_embed = CharLSTM(
            n_chars=n_chars,
            n_embed=n_char_embed,
            n_out=n_char_hidden,
            pad_index=pad_index
        )
        
        # =====================================================================
        # POS Tag Embedding
        # =====================================================================
        self.tag_embed = nn.Embedding(n_tags, n_tag_embed)
        
        # =====================================================================
        # Input Dropout
        # =====================================================================
        self.embed_dropout = SharedDropout(p=embed_dropout)
        
        # Total input size: word + char + tag
        lstm_input_size = n_embed + n_char_hidden + n_tag_embed
        
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
        if use_sibling:
            self.mlp_arc_s = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        
        # =====================================================================
        # MLP Layers for Relation Scoring
        # =====================================================================
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        if use_sibling:
            self.mlp_rel_s = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        
        # =====================================================================
        # Triaffine/Biaffine Attention
        # =====================================================================
        if use_sibling:
            self.arc_attn = Triaffine(n_in=n_mlp_arc, n_out=1, bias_x=True, bias_y=False)
            self.rel_attn = Triaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        else:
            self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
            self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.transform = transform
    
    def load_pretrained(self, embed):
        """Load pretrained word embeddings."""
        if embed is not None:
            self.word_embed = nn.Embedding.from_pretrained(embed, freeze=False)
        return self
    
    def forward(self, words, chars, tags):
        """
        Forward pass.
        
        Args:
            words: [batch, seq_len] - Word indices
            chars: [batch, seq_len, fix_len] - Character indices
            tags: [batch, seq_len] - POS tag indices
            
        Returns:
            s_arc: [batch, seq_len, seq_len] - Arc scores
            s_rel: [batch, seq_len, seq_len, n_rels] - Relation scores
        """
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        
        # Word embeddings
        word_emb = self.word_embed(words)
        
        # Character-level embeddings
        char_emb = self.char_embed(chars)
        
        # Tag embeddings
        tag_emb = self.tag_embed(tags)
        
        # Concatenate all embeddings
        x = torch.cat([word_emb, char_emb, tag_emb], dim=-1)
        x = self.embed_dropout(x)
        
        # BiLSTM
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=seq_len)
        x = self.lstm_dropout(x)
        
        # MLP for Arc
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        
        # MLP for Rel
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        
        if self.use_sibling:
            arc_s = self.mlp_arc_s(x)
            rel_s = self.mlp_rel_s(x)
            s_arc = self.arc_attn(arc_h, arc_d, arc_s)
            s_rel = self.rel_attn(rel_h, rel_d, rel_s).permute(0, 2, 3, 1)
        else:
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
            words, chars, tags, arcs, rels = batch
            
            mask = words.ne(self.pad_index)
            mask[:, 0] = 0
            
            s_arc, s_rel = self.forward(words, chars, tags)
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
        
        model = cls(**args, transform=state['transform'])
        model.load_state_dict(state['state_dict'], strict=False)
        model.eval()
        model.to(device)
        
        return model
