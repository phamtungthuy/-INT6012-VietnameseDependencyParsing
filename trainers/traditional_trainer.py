"""
TraditionalTrainer - Trainer for models/traditional parsers.

These are the fast, simple traditional parsers.
"""

import os
from datetime import timedelta, datetime
from pathlib import Path
from typing import Union, List, Dict, Any

from tqdm import tqdm

from utils.logs import logger

from models.traditional.malt_parser_old import MaltParser
from models.traditional.mst_parser import MSTParser  
from models.traditional.turbo_parser import TurboParser


class TraditionalTrainer:
    """
    Trainer for models/traditional parsers.
    """
    
    PARSER_CLASSES = {
        'malt': MaltParser,
        'mst': MSTParser,
        'turbo': TurboParser
    }
    
    def __init__(self, parser_type: str, corpus, **parser_kwargs):
        if parser_type not in self.PARSER_CLASSES:
            raise ValueError(f"Unknown parser type: {parser_type}")
        
        self.parser_type = parser_type
        self.parser_class = self.PARSER_CLASSES[parser_type]
        self.corpus = corpus
        self.parser_kwargs = parser_kwargs
    
    def _load_conll(self, path: str) -> List[Dict[str, Any]]:
        """Load CoNLL-U file as list of sentences."""
        sentences = []
        
        with open(path, 'r', encoding='utf-8') as f:
            words = []
            pos_tags = []
            heads = []
            rels = []
            
            for line in f:
                line = line.strip()
                
                if not line:
                    if words:
                        sentences.append({
                            'words': words,
                            'pos_tags': pos_tags,
                            'heads': heads,
                            'rels': rels
                        })
                        words, pos_tags, heads, rels = [], [], [], []
                elif not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 8 and '-' not in parts[0] and '.' not in parts[0]:
                        words.append(parts[1].lower())  # FORM
                        pos_tags.append(parts[3])       # UPOS
                        heads.append(int(parts[6]))     # HEAD
                        rels.append(parts[7])           # DEPREL
            
            if words:
                sentences.append({
                    'words': words,
                    'pos_tags': pos_tags,
                    'heads': heads,
                    'rels': rels
                })
        
        return sentences
    
    def train(
        self,
        save_path: str = None,
        patience: int = 10,
        max_epochs: int = 20,
        verbose: bool = True,
        **kwargs
    ):
        """Train the parser."""
        # Load data
        logger.info(f"Loading data for {self.parser_type.upper()} parser (tmp)")
        train_sents = self._load_conll(self.corpus.train)
        dev_sents = self._load_conll(self.corpus.dev)
        test_sents = self._load_conll(self.corpus.test)
        
        logger.info(f"train: {len(train_sents)} sentences")
        logger.info(f"dev: {len(dev_sents)} sentences")
        logger.info(f"test: {len(test_sents)} sentences")
        
        # Create parser
        parser = self.parser_class()
        
        logger.info(f"\nTraining: {self.parser_type.upper()}")
        logger.info(f"Calling fit() with {max_epochs} epochs...")
        
        start = datetime.now()
        
        # Call fit() ONCE with all epochs - this trains both structure and labels
        parser.fit(train_sents, epochs=max_epochs, verbose=True)
        
        elapsed = datetime.now() - start
        
        # Final evaluation
        logger.info(f'\n=== Final Results ===')
        
        dev_result = parser.evaluate(dev_sents)
        test_result = parser.evaluate(test_sents)
        
        logger.info(f"Dev  UAS: {dev_result['uas']:.2f}% | LAS: {dev_result['las']:.2f}%")
        logger.info(f"Test UAS: {test_result['uas']:.2f}% | LAS: {test_result['las']:.2f}%")
        logger.info(f'{elapsed}s elapsed')
        
        return parser

