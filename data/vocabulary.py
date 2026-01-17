from typing import List, Tuple
from collections import Counter

from utils.logs import logger

class Vocabulary:
    """The format example of a sentence:
        [{'id': 1, 'form': 'Tôi', 'upos': 'PRON', 'head': 2, 'deprel': 'nsubj'}, ...]
    """
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<ROOT>': 2}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<ROOT>'}
        self.pos2idx = {'<PAD>': 0, '<UNK>': 1, '<ROOT>': 2}
        self.idx2pos = {0: '<PAD>', 1: '<UNK>', 2: '<ROOT>'}
        self.rel2idx = {'<PAD>': 0, '<ROOT>': 1}
        self.idx2rel = {0: '<PAD>', 1: '<ROOT>'}
        
        self.word_counter = Counter()
        self.pos_counter = Counter()
        self.rel_counter = Counter()
        
    def build_vocab(self, sentences: List[List[dict]], min_freq = 2) -> None:
        # Count frequencies
        for sent in sentences:
            for word_info in sent:
                self.word_counter[word_info['form'].lower()] += 1
                self.pos_counter[word_info['upos']] += 1
                self.rel_counter[word_info['deprel']] += 1
        
        # Build word vocabulary
        word_idx = len(self.word2idx)
        for word, count in self.word_counter.most_common():
            if count >= min_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = word_idx
                    self.idx2word[word_idx] = word
                    word_idx += 1
        
        # Build POS vocabulary
        pos_idx = len(self.pos2idx)
        for pos in self.pos_counter:
            if pos not in self.pos2idx:
                self.pos2idx[pos] = pos_idx
                self.idx2pos[pos_idx] = pos
                pos_idx += 1
        
        # Build relation vocabulary
        rel_idx = len(self.rel2idx)
        for rel in self.rel_counter:
            if rel not in self.rel2idx:
                self.rel2idx[rel] = rel_idx
                self.idx2rel[rel_idx] = rel
                rel_idx += 1
                
        logger.info(f"Vocabulary built: {len(self.word2idx)} words, {len(self.pos2idx)} POS tags, {len(self.rel2idx)} relations")
    
    def encode_sentence(self, sentence: List[dict]) -> Tuple[List[int], List[int], List[int], List[int]]:
        words = [self.word2idx['<ROOT>']]
        pos_tags = [self.pos2idx['<ROOT>']]
        heads = [0] 
        rels = [self.rel2idx['<ROOT>']]
        
        for word_info in sentence:
            word = word_info['form'].lower()
            words.append(self.word2idx.get(word, self.word2idx['<UNK>']))
            pos_tags.append(self.pos2idx.get(word_info['upos'], self.pos2idx['<UNK>']))
            heads.append(word_info['head'])
            rels.append(self.rel2idx.get(word_info['deprel'], self.rel2idx['<PAD>']))
        
        return words, pos_tags, heads, rels