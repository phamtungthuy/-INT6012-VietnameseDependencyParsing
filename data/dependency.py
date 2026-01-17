import torch
from torch.utils.data import Dataset

class DependencyDataset(Dataset):
    def __init__(self, sentences, vocab):
        self.sentences = sentences
        self.vocab = vocab
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        words, pos_tags, heads, rels = self.vocab.encode_sentence(sentence)
        
        return {
            'words': torch.LongTensor(words),
            'pos_tags': torch.LongTensor(pos_tags),
            'heads': torch.LongTensor(heads),
            'rels': torch.LongTensor(rels),
            'length': len(words)
        }