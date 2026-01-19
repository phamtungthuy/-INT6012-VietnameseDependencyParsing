from typing import Tuple
from pathlib import Path
from typing import List
from conllu import parse_incr
from conllu.exceptions import ParseException

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from utils.logs import logger
from data.vocabulary import Vocabulary
from data.dependency import DependencyDataset

class CoNLLUDataset:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.sentences = []
        self.load_data()
    
    def load_data(self):
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for sentence_tokenlist in parse_incr(f):
                    sentence = []
                    
                    for token in sentence_tokenlist:
                        if isinstance(token['id'], Tuple):
                            continue
                        
                        if isinstance(token['id'], float):
                            continue
                        
                        word_info = {
                            'id': int(token['id']),
                            'form': token['form'],
                            'lemma': token.get('lemma') or '_',
                            'upos': token.get('upos') or '_',
                            'xpos': token.get('xpos') or '_',
                            'head': int(token['head']) if token.get('head') else 0,
                            'deprel': token.get('deprel') or '_'
                        }
                        
                        sentence.append(word_info)
                        
                    if sentence:
                        self.sentences.append(sentence)
            logger.info(f"Loaded {len(self.sentences)} sentences from {self.file_path}")
        except ParseException as e:
            logger.error(f"Error parsing file {self.file_path}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise e
                    
    def get_sentences(self):
        return self.sentences

def collate_fn(batch: List[dict]) -> dict[str, torch.Tensor]:
    words_list = [item['words'] for item in batch]
    pos_tags_list = [item['pos_tags'] for item in batch]
    heads_list = [item['heads'] for item in batch]
    rels_list = [item['rels'] for item in batch]
    lengths = torch.LongTensor([item['length'] for item in batch])
    
    words = pad_sequence(words_list, batch_first=True, padding_value=0)
    pos_tags = pad_sequence(pos_tags_list, batch_first=True, padding_value=0)
    heads = pad_sequence(heads_list, batch_first=True, padding_value=0)
    rels = pad_sequence(rels_list, batch_first=True, padding_value=0)
    
    return {
        'words': words,
        'pos_tags': pos_tags,
        'heads': heads,
        'rels': rels,
        'lengths': lengths
    }
    
def get_data_loaders(train_path: Path, validation_path: Path, test_path: Path, batch_size: int, min_freq: int):
    train_dataset = CoNLLUDataset(train_path)
    validation_dataset = CoNLLUDataset(validation_path)
    test_dataset = CoNLLUDataset(test_path)
    
    vocab = Vocabulary()
    vocab.build_vocab(train_dataset.get_sentences(), min_freq=min_freq)
    
    train_data = DependencyDataset(train_dataset.get_sentences(), vocab)
    validation_data = DependencyDataset(validation_dataset.get_sentences(), vocab)
    test_data = DependencyDataset(test_dataset.get_sentences(), vocab)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, validation_loader, test_loader, vocab