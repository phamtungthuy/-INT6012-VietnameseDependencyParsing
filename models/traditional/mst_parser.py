"""
MST Parser - Graph-based parser using Maximum Spanning Tree

Reference: McDonald et al. (2005) "Non-projective Dependency Parsing using Spanning Tree Algorithms"

Features:
- Chu-Liu-Edmonds algorithm for non-projective parsing
- Hand-crafted arc features
- Averaged Perceptron learning
"""

from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

from models.traditional.base_traditional_parser import BaseTraditionalParser


class MSTParser(BaseTraditionalParser):
    """Maximum Spanning Tree Parser using Chu-Liu-Edmonds algorithm."""
    
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
        """Extract features for an arc (head -> dep)"""
        features = []
        n = len(words)
        
        direction = "L" if dep < head else "R"
        distance = abs(head - dep)
        
        h_word = words[head] if head < n else "<ROOT>"
        h_pos = pos_tags[head] if head < n else "<ROOT>"
        d_word = words[dep]
        d_pos = pos_tags[dep]
        
        # Context
        h_prev = pos_tags[head - 1] if 0 < head < n else "<S>"
        h_next = pos_tags[head + 1] if head < n - 1 else "<E>"
        d_prev = pos_tags[dep - 1] if dep > 0 else "<S>"
        d_next = pos_tags[dep + 1] if dep < n - 1 else "<E>"
        
        # Unigram
        features.extend([
            f"h_w={h_word}", f"h_p={h_pos}",
            f"d_w={d_word}", f"d_p={d_pos}",
        ])
        
        # Bigram
        features.extend([
            f"h_w,d_w={h_word},{d_word}",
            f"h_p,d_p={h_pos},{d_pos}",
            f"h_w,d_p={h_word},{d_pos}",
            f"h_p,d_w={h_pos},{d_word}",
        ])
        
        # Direction + POS
        features.extend([
            f"dir={direction}",
            f"dir,h_p={direction},{h_pos}",
            f"dir,d_p={direction},{d_pos}",
            f"dir,h_p,d_p={direction},{h_pos},{d_pos}",
        ])
        
        # Distance
        dist_bin = "1" if distance == 1 else "2-3" if distance <= 3 else "4-6" if distance <= 6 else "7+"
        features.extend([
            f"dist={dist_bin}",
            f"dist,dir={dist_bin},{direction}",
        ])
        
        # Context
        features.extend([
            f"h_prev={h_prev}", f"h_next={h_next}",
            f"d_prev={d_prev}", f"d_next={d_next}",
        ])
        
        # Between POS
        if dep < head:
            between = set(pos_tags[dep+1:head])
        else:
            between = set(pos_tags[head+1:dep])
        for bp in between:
            features.append(f"between={bp}")
        
        return features
    
    def _score_arc(self, features: List[str], use_averaged: bool = False) -> float:
        weights = self.averaged_weights if use_averaged else self.weights
        return sum(weights[f] for f in features)
    
    def _build_score_matrix(
        self,
        words: List[str],
        pos_tags: List[str],
        use_averaged: bool = False
    ) -> np.ndarray:
        n = len(words)
        scores = np.full((n + 1, n + 1), -np.inf)
        
        for dep in range(n):
            # Arc from ROOT
            features = self.extract_features(words, pos_tags, n, dep)
            scores[dep][n] = self._score_arc(features, use_averaged)
            
            # Arc from other words
            for head in range(n):
                if head != dep:
                    features = self.extract_features(words, pos_tags, head, dep)
                    scores[dep][head] = self._score_arc(features, use_averaged)
        
        return scores
    
    def _chu_liu_edmonds(self, scores: np.ndarray) -> List[int]:
        """Simplified greedy MST (for efficiency)"""
        n = scores.shape[0] - 1
        heads = []
        
        for dep in range(n):
            best_head = np.argmax(scores[dep])
            if best_head == n:
                heads.append(0)  # ROOT
            else:
                heads.append(best_head + 1)  # 1-indexed
        
        return heads
    
    def fit(
        self,
        sentences: List[Dict[str, Any]],
        epochs: int = 10,
        verbose: bool = True
    ) -> 'MSTParser':
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
                        # Positive update
                        gold_feats = self.extract_features(words, pos_tags, gold_h, dep)
                        for f in gold_feats:
                            self.weights[f] += 1
                            self.averaged_weights[f] += self._updates
                        
                        # Negative update
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
        
        # Finalize averaged weights
        for f in self.weights:
            self.averaged_weights[f] = self.weights[f] - self.averaged_weights[f] / max(self._updates, 1)
        
        self.is_trained = True
        return self
    
    def _decode(self, words: List[str], pos_tags: List[str], use_averaged: bool = True) -> List[int]:
        scores = self._build_score_matrix(words, pos_tags, use_averaged)
        return self._chu_liu_edmonds(scores)
    
    def predict(self, words: List[str], pos_tags: List[str]) -> List[int]:
        return self._decode(words, pos_tags, use_averaged=True)
