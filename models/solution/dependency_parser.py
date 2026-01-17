import torch.nn as nn

class DependencyParser(nn.Module):
    r"""
    The implementation of Biaffine Dependency Parser.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """
    


    def __init__(
            self,
            n_words=None,
            pad_index=0,
            unk_index=1,
            n_feats=None,
            n_rels=None,
            feat='char',
            n_embed=50,
            n_feat_embed=100,
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
            init_pre_train=False,
            transform=None
    ):
        pass
