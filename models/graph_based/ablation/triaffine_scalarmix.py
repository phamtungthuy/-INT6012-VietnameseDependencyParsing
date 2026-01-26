# -*- coding: utf-8 -*-
"""
Triaffine Parser with Scalar Mix (Ablation Study)

Base: triaffine_parser.py (BiLSTM encoder)
Enhancement: Use ALL 12 BERT layers with learnable Scalar Mix weights
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


class TriaffineScalarMix(Model):
    """
    Triaffine Parser with Scalar Mix.
    Uses ALL 12 BERT layers instead of just the last 4.
    """
    
    NAME = 'triaffine-scalarmix'
    
    def __init__(
        self,
        n_words=None,
        n_tags=None,
        n_rels=None,
        n_embed=100,
        n_tag_embed=100,
        pad_index=0,
        unk_index=1,
        bert='vinai/phobert-base',
        n_bert_layers=12,  # USE ALL LAYERS (Key difference!)
        bert_fine_tune=True,
        n_lstm_hidden=400,
        n_lstm_layers=3,
        lstm_dropout=0.33,
        n_mlp_arc=500,
        n_mlp_rel=100,
        mlp_dropout=0.33,
        feat_pad_index=0,
        transform=None,
        init_pre_train=False
    ):
        super(TriaffineScalarMix, self).__init__()
        
        self.bert_name = bert
        
        self.args = {
            'n_words': n_words,
            'n_embed': n_embed,
            'n_tags': n_tags,
            'n_rels': n_rels,
            'pad_index': pad_index,
            'unk_index': unk_index,
            'bert': bert,
            'n_bert_layers': n_bert_layers,
            'n_lstm_hidden': n_lstm_hidden,
            'n_mlp_arc': n_mlp_arc,
            'n_mlp_rel': n_mlp_rel,
            'feat_pad_index': feat_pad_index
        }
        
        if init_pre_train:
            return
        
        # BERT Embedding with Scalar Mix (ALL 12 LAYERS)
        logger.info(f"[ScalarMix] Using Scalar Mix with {n_bert_layers} BERT layers")
        self.bert_embed = BertEmbedding(
            model=bert,
            n_layers=n_bert_layers,  # Use all 12 layers
            n_out=None,
            pad_index=feat_pad_index,
            dropout=0.0,
            requires_grad=bert_fine_tune
        )
        bert_hidden_size = self.bert_embed.n_out
        
        # POS Tag Embedding
        self.n_tags = n_tags
        self.n_tag_embed = n_tag_embed
        lstm_input_size = bert_hidden_size
        
        if n_tags is not None and n_tag_embed > 0:
            self.tag_embed = nn.Embedding(n_tags, n_tag_embed)
            lstm_input_size += n_tag_embed
        else:
            self.tag_embed = None
        
        # Word Embedding
        if n_words is not None and n_embed > 0:
            self.word_embed = nn.Embedding(n_words, n_embed)
            lstm_input_size += n_embed
        else:
            self.word_embed = None
        
        # BiLSTM Encoder
        self.lstm = BiLSTM(
            input_size=lstm_input_size,
            hidden_size=n_lstm_hidden,
            num_layers=n_lstm_layers,
            dropout=lstm_dropout
        )
        self.lstm_dropout = SharedDropout(p=lstm_dropout)
        
        # MLP for Arc
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_arc, dropout=mlp_dropout)
        
        # MLP for Rel
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden * 2, n_out=n_mlp_rel, dropout=mlp_dropout)
        
        # Standard Biaffine (no multi-head)
        self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.transform = transform
    
    def forward(self, words, feats, tags=None):
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        
        x = self.bert_embed(feats)
        
        if self.tag_embed is not None and tags is not None:
            x = torch.cat([x, self.tag_embed(tags)], dim=-1)
        
        if self.word_embed is not None:
            x = torch.cat([x, self.word_embed(words)], dim=-1)
        
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=seq_len)
        x = self.lstm_dropout(x)
        
        arc_h = self.mlp_arc_h(x)
        arc_d = self.mlp_arc_d(x)
        rel_h = self.mlp_rel_h(x)
        rel_d = self.mlp_rel_d(x)
        
        s_arc = self.arc_attn(arc_d, arc_h)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))
        
        return s_arc, s_rel
    
    def forward_loss(self, s_arc, s_rel, arcs, rels, mask):
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        
        return self.criterion(s_arc, arcs) + self.criterion(s_rel, rels)
    
    def decode(self, s_arc, s_rel, mask, tree=False, proj=False):
        lens = mask.sum(1)
        arc_preds = s_arc.argmax(-1)
        
        bad = [not CoNLL.istree(seq[1:i + 1], proj)
               for i, seq in zip(lens.tolist(), arc_preds.tolist())]
        if tree and any(bad):
            alg = eisner if proj else mst
            arc_preds[bad] = alg(s_arc[bad], mask[bad])
        
        rel_preds = s_rel.argmax(-1).gather(-1, arc_preds.unsqueeze(-1)).squeeze(-1)
        return arc_preds, rel_preds
    
    def save(self, path):
        model = self.module if hasattr(self, 'module') else self
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({'name': self.NAME, 'args': self.args, 'state_dict': state_dict, 'transform': self.transform}, path)
    
    @classmethod
    def load(cls, path):
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(**state['args'], transform=state['transform'])
        model.load_state_dict(state['state_dict'], strict=False)
        model.eval()
        return model.to(device)
