from .base_parser import BaseParser
from .bilstm_parser import BiLSTMParser

try:
    from .bert_parser import BERTParser
    __all__ = ['BaseParser', 'BiLSTMParser', 'BERTParser']
except ImportError:
    __all__ = ['BaseParser', 'BiLSTMParser']