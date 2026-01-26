import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.nn import Model
from transforms.conll import CoNLL
from modules.base import (
    CharLSTM, BiLSTM, SharedDropout, 
    MLP, Biaffine, IndependentDropout
)
from modules.bert import BertEmbedding
from utils.sp_fn import ispunct
from utils.util_deep_learning import device

from utils.logs import logger
from utils.sp_data import Dataset
from utils.sp_alg import eisner, mst
from utils.constants import PRETRAINED
from utils.sp_metric import AttachmentMetric
from utils.sp_field import Field

class BiaffineParserV2(Model):
    r"""
    The implementation of Biaffine Dependency Parser V2.
    Enhanced with POS Tag embeddings and Fine-tuning capabilities for BERT.
    """
    NAME = 'biaffine-dependency-v2'

    def __init__(
            self,
            n_words=None,
            pad_index=0,
            unk_index=1,
            n_feats=None,
            n_tags=None,
            n_rels=None,
            feat='char',
            n_embed=50,
            n_feat_embed=100,
            n_tag_embed=64, # Default 64 as in the notebook
            n_char_embed=50,
            bert=None,
            n_bert_layers=4,
            embed_dropout=.33,
            max_len=None,
            mix_dropout=.0,
            embeddings=[],
            embed=False,
            n_lstm_hidden=400,
            n_lstm_layers=3,
            lstm_dropout=.33,
            n_mlp_arc=500,
            n_mlp_rel=100,
            mlp_dropout=.33,
            feat_pad_index=0,
            tag_pad_index=0,
            init_pre_train=False,
            transform=None
    ):
        super(BiaffineParserV2, self).__init__()
        self.embed = embed
        self.feat = feat
        self.bert = bert
        self.embeddings = embeddings
        self.args = {
            "n_words": n_words,
            'pad_index': pad_index,
            'unk_index': unk_index,
            "n_feats": n_feats,
            "n_tags": n_tags,
            "n_rels": n_rels,
            'feat_pad_index': feat_pad_index,
            'tag_pad_index': tag_pad_index,
            "feat": feat,
            'tree': False,
            'proj': False,
            'punct': False,
            'bert': bert
        }

        if init_pre_train:
            return

        # 1. Word Embedding
        self.word_embed = nn.Embedding(num_embeddings=n_words,
                                       embedding_dim=n_embed)
        
        # 2. Feat Embedding (Char or BERT)
        self.use_bert = False
        if feat == 'char':
            self.feat_embed = CharLSTM(n_chars=n_feats,
                                       n_embed=n_char_embed,
                                       n_out=n_feat_embed,
                                       pad_index=feat_pad_index)
        elif feat == 'bert':
            self.use_bert = True
            self.feat_embed = BertEmbedding(model=bert,
                                            n_layers=n_bert_layers,
                                            n_out=n_feat_embed,
                                            pad_index=feat_pad_index,
                                            max_len=max_len,
                                            dropout=mix_dropout,
                                            requires_grad=True) # Enable fine-tuning
            self.n_feat_embed = self.feat_embed.n_out
        elif feat == 'tag':
            self.feat_embed = nn.Embedding(num_embeddings=n_feats,
                                           embedding_dim=n_feat_embed)
        else:
            raise RuntimeError("The feat type should be in ['char', 'bert', 'tag'].")

        # 3. POS Tag Embedding (Specifically for V2 improvement)
        # Always enable tag embedding if n_tags is provided, or specifically if feat='bert' logic requires it
        if n_tags is not None:
             self.tag_embed = nn.Embedding(num_embeddings=n_tags,
                                           embedding_dim=n_tag_embed,
                                           padding_idx=tag_pad_index)
             self.n_tag_embed = n_tag_embed
        else:
             self.tag_embed = None
             self.n_tag_embed = 0

        self.embed_dropout = IndependentDropout(p=embed_dropout)

        # Calculate total input size for LSTM
        lstm_input_size = n_embed + self.n_feat_embed + self.n_tag_embed

        # the lstm layer
        self.lstm = BiLSTM(input_size=lstm_input_size,
                           hidden_size=n_lstm_hidden,
                           num_layers=n_lstm_layers,
                           dropout=lstm_dropout)
        self.lstm_dropout = SharedDropout(p=lstm_dropout)

        # the MLP layers
        self.mlp_arc_d = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_arc_h = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_arc,
                             dropout=mlp_dropout)
        self.mlp_rel_d = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)
        self.mlp_rel_h = MLP(n_in=n_lstm_hidden * 2,
                             n_out=n_mlp_rel,
                             dropout=mlp_dropout)

        # the Biaffine layers
        self.arc_attn = Biaffine(n_in=n_mlp_arc,
                                 bias_x=True,
                                 bias_y=False)
        self.rel_attn = Biaffine(n_in=n_mlp_rel,
                                 n_out=n_rels,
                                 bias_x=True,
                                 bias_y=True)
        self.criterion = nn.CrossEntropyLoss()
        self.pad_index = pad_index
        self.unk_index = unk_index

        self.transform = transform
        
        # Helper variables for decoding/eval
        self.puncts = None
        if self.transform is not None:
             # Just for punctuation mask logic
             # This part might need adjustment depending on how transform is set up in V2
             pass

    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.word_embed.weight)
        return self

    def forward(self, words, feats, tags):
        r"""
        Args:
            words: [batch_size, seq_len]
            feats: [batch_size, seq_len] or [batch_size, seq_len, fix_len] (if bert/char)
            tags: [batch_size, seq_len] (POS tags)
        """
        batch_size, seq_len = words.shape
        mask = words.ne(self.pad_index)
        ext_words = words
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.unk_index)

        # 1. Word Embed
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            word_embed += self.pretrained(words)
        
        # 2. Feat Embed (Char/BERT)
        feat_embed = self.feat_embed(feats)
        
        # 3. Tag Embed
        if self.tag_embed is not None:
            tag_embed = self.tag_embed(tags)
            # Concatenate all three
            word_embed, feat_embed, tag_embed = self.embed_dropout(word_embed, feat_embed, tag_embed)
            embed = torch.cat((word_embed, feat_embed, tag_embed), -1)
        else:
            word_embed, feat_embed = self.embed_dropout(word_embed, feat_embed)
            embed = torch.cat((word_embed, feat_embed), -1)

        x = pack_padded_sequence(embed, mask.sum(1), True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        arc_d = self.mlp_arc_d(x)
        arc_h = self.mlp_arc_h(x)
        rel_d = self.mlp_rel_d(x)
        rel_h = self.mlp_rel_h(x)

        s_arc = self.arc_attn(arc_d, arc_h)
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))

        return s_arc, s_rel

    def forward_loss(self, s_arc, s_rel, arcs, rels, mask):
        s_arc, arcs = s_arc[mask], arcs[mask]
        s_rel, rels = s_rel[mask], rels[mask]
        s_rel = s_rel[torch.arange(len(arcs)), arcs]
        arc_loss = self.criterion(s_arc, arcs)
        rel_loss = self.criterion(s_rel, rels)

        return arc_loss + rel_loss

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

    @torch.no_grad()
    def evaluate(self, loader):
        self.eval()
        total_loss, metric = 0, AttachmentMetric()

        for words, feats, tags, arcs, rels in loader:
            mask = words.ne(self.pad_index)
            mask[:, 0] = 0
            s_arc, s_rel = self.forward(words, feats, tags)
            loss = self.forward_loss(s_arc, s_rel, arcs, rels, mask)
            arc_preds, rel_preds = self.decode(s_arc, s_rel, mask)
            
            # Punctuation masking logic (simplified for now)
            # if not self.args['punct']: ...
            
            total_loss += loss.item()
            metric(arc_preds, rel_preds, arcs, rels, mask)
        total_loss /= len(loader)

        return total_loss, metric
    
    # Missing methods like load, save etc. can be inherited or copied if we didn't inherit from Model
    # Since we inherit form Model (modules.nn), we likely need to implement _get_state_dict/load methods if they were custom in V1.
    # In V1 they were specialized. Let's copy them but ensure they work for V2.

    def save(self, path):
        model = self
        if hasattr(self, 'module'):
            model = self.module

        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {
            'name': self.NAME,
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained,
            'transform': self.transform,
            'embeddings': self.embeddings
        }
        torch.save(state, path)

    @staticmethod
    def _init_model_with_state_dict(state):
        args = state['args']
        transform = state['transform']
        
        # We need to reconstruct the model with correct args
        model = BiaffineParserV2(
            n_words=args['n_words'],
            n_feats=args['n_feats'],
            n_tags=args.get('n_tags', None), # Handle backward compatibility if needed, though V2 is new
            n_rels=args['n_rels'],
            pad_index=args['pad_index'],
            unk_index=args['unk_index'],
            feat_pad_index=args['feat_pad_index'],
            tag_pad_index=args.get('tag_pad_index', 0),
            transform=transform,
            feat=args['feat'], # Pass feat type
            bert=args.get('bert', "vinai/phobert-base")
        )
        if 'bert' in args: # If we save it
             pass 
        
        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        return model

