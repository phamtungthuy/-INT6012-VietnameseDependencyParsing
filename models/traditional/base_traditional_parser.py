"""
Base class for Traditional (non-neural) Dependency Parsers
"""

import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional


class BaseTraditionalParser(ABC):
    """
    Abstract base class for traditional (non-neural) dependency parsers.
    
    Unlike neural parsers (BaseParser), traditional parsers use:
    - Hand-crafted features
    - Linear classifiers (Perceptron, SVM)
    - No forward() or loss() methods - training is handled differently
    
    Main methods:
    - extract_features(): Feature extraction from parser state
    - fit(): Train the parser
    - predict(): Parse a sentence (heads + labels)
    """
    
    def __init__(self):
        self.weights: Dict[str, float] = {}
        self.label_weights: Dict[str, float] = {}  # For label prediction
        self.is_trained = False
    
    @abstractmethod
    def extract_features(self, *args, **kwargs) -> List[str]:
        """Extract features from parser state."""
        raise NotImplementedError
    
    def extract_label_features(
        self,
        words: List[str],
        pos_tags: List[str],
        head: int,
        dep: int
    ) -> List[str]:
        """Extract features for label classification (head, dep) -> label"""
        features = []
        n = len(words)
        
        h_word = words[head] if 0 <= head < n else "<ROOT>"
        h_pos = pos_tags[head] if 0 <= head < n else "<ROOT>"
        d_word = words[dep]
        d_pos = pos_tags[dep]
        
        direction = "L" if dep < head else "R"
        distance = abs(head - dep) if head < n else dep + 1
        
        features.extend([
            f"lbl:h_w={h_word}",
            f"lbl:h_p={h_pos}",
            f"lbl:d_w={d_word}",
            f"lbl:d_p={d_pos}",
            f"lbl:h_p,d_p={h_pos},{d_pos}",
            f"lbl:h_w,d_w={h_word},{d_word}",
            f"lbl:dir={direction}",
            f"lbl:dist={min(distance, 5)}",
            f"lbl:h_p,d_p,dir={h_pos},{d_pos},{direction}",
        ])
        
        return features
    
    @abstractmethod
    def fit(
        self,
        sentences: List[Dict[str, Any]],
        epochs: int = 10,
        verbose: bool = True
    ) -> 'BaseTraditionalParser':
        """Train the parser on a list of sentences."""
        raise NotImplementedError
    
    def fit_labels(
        self,
        sentences: List[Dict[str, Any]],
        epochs: int = 5,
        verbose: bool = True
    ):
        """Train label classifier separately."""
        # Collect all labels
        all_labels = set()
        for sent in sentences:
            if 'rels' in sent:
                all_labels.update(sent['rels'])
        self.labels = sorted(all_labels)
        
        if not self.labels:
            return
        
        # Train with perceptron
        updates = 0
        for epoch in range(epochs):
            correct = 0
            total = 0
            
            for sent in sentences:
                words = sent['words']
                pos_tags = sent['pos_tags']
                heads = sent['heads']
                rels = sent.get('rels', [None] * len(words))
                
                for dep in range(len(words)):
                    gold_label = rels[dep]
                    if gold_label is None:
                        continue
                    
                    head = heads[dep] - 1 if heads[dep] > 0 else len(words)
                    features = self.extract_label_features(words, pos_tags, head, dep)
                    
                    # Predict
                    scores = {lbl: sum(self.label_weights.get(f"{f}_{lbl}", 0) for f in features) 
                              for lbl in self.labels}
                    pred_label = max(scores, key=lambda x: scores[x])
                    
                    if pred_label != gold_label:
                        for f in features:
                            self.label_weights[f"{f}_{gold_label}"] = \
                                self.label_weights.get(f"{f}_{gold_label}", 0) + 1
                            self.label_weights[f"{f}_{pred_label}"] = \
                                self.label_weights.get(f"{f}_{pred_label}", 0) - 1
                    else:
                        correct += 1
                    
                    total += 1
                    updates += 1
            
            if verbose:
                acc = correct / total * 100 if total > 0 else 0
                print(f"  Label Epoch {epoch + 1}/{epochs} - Acc: {acc:.2f}%")
    
    def predict_label(
        self,
        words: List[str],
        pos_tags: List[str],
        head: int,
        dep: int
    ) -> Optional[str]:
        """Predict label for an arc."""
        if not hasattr(self, 'labels') or not self.labels:
            return None
        
        features = self.extract_label_features(words, pos_tags, head, dep)
        scores = {lbl: sum(self.label_weights.get(f"{f}_{lbl}", 0) for f in features) 
                  for lbl in self.labels}
        return max(scores, key=lambda x: scores[x])
    
    @abstractmethod
    def predict(
        self,
        words: List[str],
        pos_tags: List[str]
    ) -> List[int]:
        """Parse a sentence and return head indices (0 = ROOT, 1-indexed)."""
        raise NotImplementedError
    
    def predict_with_labels(
        self,
        words: List[str],
        pos_tags: List[str]
    ) -> Tuple[List[int], List[Optional[str]]]:
        """Parse and return both heads and labels."""
        heads = self.predict(words, pos_tags)
        labels = []
        
        for dep, head in enumerate(heads):
            head_idx = head - 1 if head > 0 else len(words)
            label = self.predict_label(words, pos_tags, head_idx, dep)
            labels.append(label)
        
        return heads, labels
    
    def evaluate(
        self,
        sentences: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate parser with UAS, LAS, and speed.
        
        Returns:
            Dict with 'uas', 'las', 'speed' (sent/s)
        """
        uas_correct = 0
        las_correct = 0
        total = 0
        
        start_time = time.time()
        
        for sent in sentences:
            words = sent['words']
            pos_tags = sent['pos_tags']
            gold_heads = sent['heads']
            gold_rels = sent.get('rels', [None] * len(words))
            
            pred_heads, pred_rels = self.predict_with_labels(words, pos_tags)
            
            for i in range(len(words)):
                if pred_heads[i] == gold_heads[i]:
                    uas_correct += 1
                    if pred_rels[i] == gold_rels[i]:
                        las_correct += 1
                total += 1
        
        elapsed = time.time() - start_time
        num_sentences = len(sentences)
        
        uas = (uas_correct / total * 100) if total > 0 else 0.0
        las = (las_correct / total * 100) if total > 0 else 0.0
        speed = num_sentences / elapsed if elapsed > 0 else 0.0
        
        return {
            'uas': uas,
            'las': las,
            'speed': speed,
            'uas_correct': uas_correct,
            'las_correct': las_correct,
            'total': total,
            'num_sentences': num_sentences,
            'elapsed_seconds': elapsed,
        }
