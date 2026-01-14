"""
BERT-based Biaffine Parser with Joint POS Tagging

Module structure:
    - attention.py: ScaledBiaffineAttention
    - modules.py: TaggerModule, ParserEncoder, EdgeScorer
    - decoder.py: ScoreDecoder (Greedy, Eisner, MST)
    - parser.py: BertBiaffineParser

Reference:
    - Bhatt et al. (2024): End-to-end Parsing of Procedural Text into Flow Graphs
      https://aclanthology.org/2024.lrec-main.517/
    - Dozat & Manning (2017): Deep Biaffine Attention for Neural Dependency Parsing
"""

from models.neural_attention.bert_biaffine.attention import ScaledBiaffineAttention
from models.neural_attention.bert_biaffine.modules import TaggerModule, ParserEncoder, EdgeScorer
from models.neural_attention.bert_biaffine.decoder import ScoreDecoder
from models.neural_attention.bert_biaffine.parser import BertBiaffineParser, create_bert_biaffine_parser

__all__ = [
    'ScaledBiaffineAttention',
    'TaggerModule',
    'ParserEncoder',
    'EdgeScorer',
    'ScoreDecoder',
    'BertBiaffineParser',
    'create_bert_biaffine_parser'
]
