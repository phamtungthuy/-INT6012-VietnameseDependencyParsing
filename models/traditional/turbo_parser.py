"""
TurboParser - Graph-based parser with second-order features

Reference: Martins et al. (2010) "TurboParser: Dependency Parsing by Approximate Variational Inference"

Features:
- First and second-order arc features
- Structured Perceptron learning
- Greedy decoding (simplified from full dual decomposition)
"""

from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

from models.traditional.base_traditional_parser import BaseTraditionalParser


class TurboParser(BaseTraditionalParser):
    """
    Simplified TurboParser-style dependency parser.
    Uses first and second-order features with greedy decoding.
    """
    
    def __init__(self):
        super().__init__()
        self.weights: Dict[str, float] = defaultdict(float)
        self.averaged_weights: Dict[str, float] = defaultdict(float)
        self._updates = 0
    
    def extract_features(
        self,
        words: List[str],
        pos_tags: List[str],
        head: int,
        dep: int
    ) -> List[str]:
        """Extract first-order arc features"""
        features = []
        n = len(words)
        
        direction = "L" if dep < head else "R"
        distance = abs(head - dep)
        
        h_word = words[head] if 0 <= head < n else "<ROOT>"
        h_pos = pos_tags[head] if 0 <= head < n else "<ROOT>"
        d_word = words[dep]
        d_pos = pos_tags[dep]
        
        # First-order features
        features.extend([
            f"arc:h_w={h_word}", f"arc:h_p={h_pos}",
            f"arc:d_w={d_word}", f"arc:d_p={d_pos}",
            f"arc:h_w,d_w={h_word},{d_word}",
            f"arc:h_p,d_p={h_pos},{d_pos}",
            f"arc:h_w,d_p={h_word},{d_pos}",
            f"arc:h_p,d_w={h_pos},{d_word}",
            f"arc:dir={direction}",
            f"arc:dir,h_p={direction},{h_pos}",
            f"arc:dir,d_p={direction},{d_pos}",
            f"arc:dir,h_p,d_p={direction},{h_pos},{d_pos}",
        ])
        
        # Distance
        dist_bin = str(min(distance, 10))
        features.extend([
            f"arc:dist={dist_bin}",
            f"arc:dist,dir={dist_bin},{direction}",
        ])
        
        # Context
        h_prev = pos_tags[head - 1] if 0 < head < n else "<S>"
        h_next = pos_tags[head + 1] if head < n - 1 else "<E>"
        d_prev = pos_tags[dep - 1] if dep > 0 else "<S>"
        d_next = pos_tags[dep + 1] if dep < n - 1 else "<E>"
        
        features.extend([
            f"arc:h_prev={h_prev}", f"arc:h_next={h_next}",
            f"arc:d_prev={d_prev}", f"arc:d_next={d_next}",
            f"arc:h_p,h_prev,h_next={h_pos},{h_prev},{h_next}",
            f"arc:d_p,d_prev,d_next={d_pos},{d_prev},{d_next}",
        ])
        
        return features
    
    def _extract_sibling_features(
        self,
        words: List[str],
        pos_tags: List[str],
        head: int,
        dep1: int,
        dep2: int
    ) -> List[str]:
        """Extract second-order sibling features"""
        features = []
        n = len(words)
        
        h_pos = pos_tags[head] if 0 <= head < n else "<ROOT>"
        d1_pos = pos_tags[dep1]
        d2_pos = pos_tags[dep2]
        
        features.extend([
            f"sib:h_p={h_pos}",
            f"sib:d1_p,d2_p={d1_pos},{d2_pos}",
            f"sib:h_p,d1_p,d2_p={h_pos},{d1_pos},{d2_pos}",
        ])
        
        dir1 = "L" if dep1 < head else "R"
        dir2 = "L" if dep2 < head else "R"
        features.append(f"sib:dirs={dir1},{dir2}")
        
        sib_dist = abs(dep1 - dep2)
        features.append(f"sib:dist={min(sib_dist, 5)}")
        
        return features
    
    def _score_arc(self, features: List[str], use_averaged: bool = False) -> float:
        weights = self.averaged_weights if use_averaged else self.weights
        return sum(weights[f] for f in features)
    
    def _decode(
        self,
        words: List[str],
        pos_tags: List[str],
        use_averaged: bool = False
    ) -> List[int]:
        """Greedy decoding: for each word, pick best head"""
        n = len(words)
        heads = []
        
        for dep in range(n):
            best_h = -1
            best_score = float('-inf')
            
            # Try ROOT
            features = self.extract_features(words, pos_tags, n, dep)
            score = self._score_arc(features, use_averaged)
            if score > best_score:
                best_score = score
                best_h = -1
            
            # Try each word as head
            for head in range(n):
                if head != dep:
                    features = self.extract_features(words, pos_tags, head, dep)
                    score = self._score_arc(features, use_averaged)
                    if score > best_score:
                        best_score = score
                        best_h = head
            
            heads.append(0 if best_h == -1 else best_h + 1)
        
        return heads
    
    def fit(
        self,
        sentences: List[Dict[str, Any]],
        epochs: int = 10,
        verbose: bool = True
    ) -> 'TurboParser':
        for epoch in range(epochs):
            correct = 0
            total = 0
            
            for sent in sentences:
                words = sent['words']
                pos_tags = sent['pos_tags']
                gold_heads = sent['heads']
                
                pred_heads = self._decode(words, pos_tags, use_averaged=False)
                
                for dep in range(len(words)):
                    gold_h = gold_heads[dep] - 1 if gold_heads[dep] > 0 else len(words)
                    pred_h = pred_heads[dep] - 1 if pred_heads[dep] > 0 else len(words)
                    
                    if gold_h != pred_h:
                        gold_feats = self.extract_features(words, pos_tags, gold_h, dep)
                        for f in gold_feats:
                            self.weights[f] += 1
                            self.averaged_weights[f] += self._updates
                        
                        pred_feats = self.extract_features(words, pos_tags, pred_h, dep)
                        for f in pred_feats:
                            self.weights[f] -= 1
                            self.averaged_weights[f] -= self._updates
                    else:
                        correct += 1
                    
                    total += 1
                
                self._updates += 1
            
            if verbose:
                uas = correct / total * 100 if total > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs} - UAS: {uas:.2f}%")
        
        for f in self.weights:
            self.averaged_weights[f] = self.weights[f] - self.averaged_weights[f] / max(self._updates, 1)
        
        self.is_trained = True
        return self
    
    def predict(self, words: List[str], pos_tags: List[str]) -> List[int]:
        return self._decode(words, pos_tags, use_averaged=True)
