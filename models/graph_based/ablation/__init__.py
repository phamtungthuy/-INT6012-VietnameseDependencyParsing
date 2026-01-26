# -*- coding: utf-8 -*-
"""
Ablation Study Variants for Triaffine Parser

These variants are based on triaffine_parser.py with specific enhancements
isolated for ablation study purposes.
"""

from models.graph_based.ablation.triaffine_multihead import TriaffineMultiHead
from models.graph_based.ablation.triaffine_scalarmix import TriaffineScalarMix
from models.graph_based.ablation.triaffine_charlstm import TriaffineCharLSTM
from models.graph_based.ablation.triaffine_multihead_scalarmix import TriaffineMultiHeadScalarMix

__all__ = [
    'TriaffineMultiHead',
    'TriaffineScalarMix', 
    'TriaffineCharLSTM',
    'TriaffineMultiHeadScalarMix'
]
