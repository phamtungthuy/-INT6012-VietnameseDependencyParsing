# -*- coding: utf-8 -*-
"""
Triaffine Parser with Character-level LSTM Embedding (Ablation Study)

Base: triaffine_parser.py (BiLSTM encoder)
Enhancement: Add Character-level LSTM to handle rare words and morphology
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from modules.nn import Model
from transforms.conll import CoNLL
from modules.base import BiLSTM, SharedDropout, MLP, Biaffine
from modules.bert import BertEmbedding
from utils.util_deep_learning import device
from utils.logs import logger
from utils.sp_alg import eisner, mst
from utils.sp_metric import AttachmentMetric


class CharLSTM(nn.Module):
    """
    Character-level LSTM for capturing morphological features.
    Useful for handling rare words, OOV, and typos.
    """
    
    def __init__(self, n_chars, n_embed=50, n_out=100, pad_index=0):
        super(CharLSTM, self).__init__()
        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_out = n_out
        self.pad_index = pad_index
        
        self.embed = nn.Embedding(n_chars, n_embed)
        self.lstm = nn.LSTM(n_embed, n_out // 2, batch_first=True, bidirectional=True)
    
    def forward(self, chars):
        """
        Args:
            chars: [batch, seq_len, max_char_len] - Character indices
        Returns:
            char_embed: [batch, seq_len, n_out] - Character-level embeddings
        """
        batch, seq_len, max_char_len = chars.shape
        
        # Flatten to [batch * seq_len, max_char_len]
        chars_flat = chars.view(-1, max_char_len)
        
        # Compute lengths (non-padding chars)
        mask = chars_flat.ne(self.pad_index)
        lens = mask.sum(dim=1).clamp(min=1)
        
        # Embed characters
        x = self.embed(chars_flat)  # [batch*seq, max_char, n_embed]
        
        # Pack and run LSTM
        x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        # h: [2, batch*seq, n_out//2]
        h = torch.cat([h[0], h[1]], dim=-1)  # [batch*seq, n_out]
        
        # Reshape back
        char_embed = h.view(batch, seq_len, -1)
        
        return char_embed


class TriaffineCharLSTM(Model):
    """
    Triaffine Parser with Character-level LSTM Embedding.
    Adds character-level features for better handling of rare words.
    """
    
    NAME = 'triaffine-charlstm'
    
    def __init__(
        self,
        n_words=None,
        n_chars=None,  # NEW: Number of characters
        n_tags=None,
        n_rels=None,
        n_embed=100,
        n_tag_embed=100,
        n_char_embed=50,   # NEW: Character embedding dimension
        n_char_out=100,    # NEW: CharLSTM output dimension
        pad_index=0,
        unk_index=1,
        char_pad_index=0,  # NEW
        bert='vinai/phobert-base',
        n_bert_layers=4,
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
        super(TriaffineCharLSTM, self).__init__()
        
        self.bert_name = bert
        
        self.args = {
            'n_words': n_words,
            'n_chars': n_chars,
            'n_embed': n_embed,
            'n_tags': n_tags,
            'n_rels': n_rels,
            'n_char_embed': n_char_embed,
            'n_char_out': n_char_out,
            'char_pad_index': char_pad_index,
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
        
        # BERT Embedding
        self.bert_embed = BertEmbedding(
            model=bert,
            n_layers=n_bert_layers,
            n_out=None,
            pad_index=feat_pad_index,
            dropout=0.0,
            requires_grad=bert_fine_tune
        )
        bert_hidden_size = self.bert_embed.n_out
        lstm_input_size = bert_hidden_size
        
        # Character-level LSTM Embedding (ENHANCEMENT)
        self.n_chars = n_chars
        self.char_pad_index = char_pad_index
        if n_chars is not None and n_char_out > 0:
            logger.info(f"[CharLSTM] Using Character LSTM with {n_chars} chars, embed={n_char_embed}, out={n_char_out}")
            self.char_lstm = CharLSTM(n_chars, n_char_embed, n_char_out, char_pad_index)
            lstm_input_size += n_char_out
        else:
            self.char_lstm = None
        
        # POS Tag Embedding
        self.n_tags = n_tags
        self.n_tag_embed = n_tag_embed
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
        
        # Standard Biaffine
        self.arc_attn = Biaffine(n_in=n_mlp_arc, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel, n_out=n_rels, bias_x=True, bias_y=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index
        self.transform = transform
    
    def forward(self, words, feats, tags=None, chars=None):
        """
        Args:
            words: [batch, seq_len] - Word indices
            feats: [batch, seq_len, fix_len] - BERT subword indices
            tags: [batch, seq_len] - POS tag indices (optional)
            chars: [batch, seq_len, max_char_len] - Character indices (optional)
        """
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        
        x = self.bert_embed(feats)
        
        # Add Character-level embedding
        if self.char_lstm is not None and chars is not None:
            char_emb = self.char_lstm(chars)
            x = torch.cat([x, char_emb], dim=-1)
        
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
