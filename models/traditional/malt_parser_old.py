"""
MaltParser - Transition-based parser with linear classifier

Reference: Nivre et al. (2006) "MaltParser: A Data-Driven Parser-Generator for Dependency Parsing"

Features:
- Arc-Standard transition system
- Hand-crafted features
- Averaged Perceptron classifier
"""

from typing import List, Dict, Tuple, Any
from collections import defaultdict

from models.traditional.base_traditional_parser import BaseTraditionalParser


def _nested_float_dict():
    """Helper function for picklable nested defaultdict"""
    return defaultdict(float)


class MaltParser(BaseTraditionalParser):
    """
    MaltParser-style transition-based dependency parser.
    
    Arc-Standard transitions:
    - SHIFT: Move buffer front to stack
    - LEFT-ARC: Create arc stack[-2] <- stack[-1], pop stack[-2]
    - RIGHT-ARC: Create arc stack[-2] -> stack[-1], pop stack[-1]
    """
    
    def __init__(self):
        super().__init__()
        self.weights: Dict[str, Dict[str, float]] = defaultdict(_nested_float_dict)
        self.averaged_weights: Dict[str, Dict[str, float]] = defaultdict(_nested_float_dict)
        self._updates = 0
        self.actions = ['SHIFT', 'LEFT-ARC', 'RIGHT-ARC']
    
    def extract_features(
        self,
        stack: List[int],
        buffer: List[int],
        words: List[str],
        pos_tags: List[str],
        arcs: Dict[int, int]
    ) -> List[str]:
        """Extract features from parser state"""
        features = []
        n = len(words)
        
        def get_word(idx):
            if idx is None or idx < 0 or idx >= n:
                return "<NULL>"
            return words[idx]
        
        def get_pos(idx):
            if idx is None or idx < 0 or idx >= n:
                return "<NULL>"
            return pos_tags[idx]
        
        # Stack positions
        s0 = stack[-1] if len(stack) > 0 else None
        s1 = stack[-2] if len(stack) > 1 else None
        s2 = stack[-3] if len(stack) > 2 else None
        
        # Buffer positions
        b0 = buffer[0] if len(buffer) > 0 else None
        b1 = buffer[1] if len(buffer) > 1 else None
        b2 = buffer[2] if len(buffer) > 2 else None
        
        # Stack word/POS
        s0_w, s0_p = get_word(s0), get_pos(s0)
        s1_w, s1_p = get_word(s1), get_pos(s1)
        s2_p = get_pos(s2)
        
        # Buffer word/POS
        b0_w, b0_p = get_word(b0), get_pos(b0)
        b1_w, b1_p = get_word(b1), get_pos(b1)
        
        # Unigram features
        features.extend([
            f"s0_w={s0_w}", f"s0_p={s0_p}",
            f"s1_w={s1_w}", f"s1_p={s1_p}",
            f"b0_w={b0_w}", f"b0_p={b0_p}",
            f"b1_w={b1_w}", f"b1_p={b1_p}",
        ])
        
        # Bigram features
        features.extend([
            f"s0_w,s1_w={s0_w},{s1_w}",
            f"s0_p,s1_p={s0_p},{s1_p}",
            f"s0_w,b0_w={s0_w},{b0_w}",
            f"s0_p,b0_p={s0_p},{b0_p}",
            f"s1_p,b0_p={s1_p},{b0_p}",
            f"b0_p,b1_p={b0_p},{b1_p}",
        ])
        
        # Trigram features
        features.extend([
            f"s0_p,s1_p,b0_p={s0_p},{s1_p},{b0_p}",
            f"s0_p,b0_p,b1_p={s0_p},{b0_p},{b1_p}",
            f"s0_p,s1_p,s2_p={s0_p},{s1_p},{s2_p}",
        ])
        
        # Word + POS combinations
        features.extend([
            f"s0_w,s0_p={s0_w},{s0_p}",
            f"s1_w,s1_p={s1_w},{s1_p}",
            f"b0_w,b0_p={b0_w},{b0_p}",
        ])
        
        # Distance features
        if s0 is not None and s1 is not None:
            dist = abs(s0 - s1)
            dist_bin = "1" if dist == 1 else "2-3" if dist <= 3 else "4+"
            features.append(f"s0_s1_dist={dist_bin}")
        
        # Stack/buffer size
        features.extend([
            f"stack_size={min(len(stack), 5)}",
            f"buffer_size={min(len(buffer), 5)}",
        ])
        
        return features
    
    def _score_action(self, features: List[str], action: str, use_averaged: bool = False) -> float:
        weights = self.averaged_weights if use_averaged else self.weights
        return sum(weights[f][action] for f in features)
    
    def _get_valid_actions(self, stack: List[int], buffer: List[int]) -> List[str]:
        valid = []
        if len(buffer) > 0:
            valid.append('SHIFT')
        if len(stack) >= 2:
            if stack[-2] != -1:  # Can't have ROOT as dependent
                valid.append('LEFT-ARC')
            valid.append('RIGHT-ARC')
        return valid
    
    def _apply_action(
        self,
        action: str,
        stack: List[int],
        buffer: List[int],
        arcs: Dict[int, int]
    ) -> Tuple[List[int], List[int], Dict[int, int]]:
        stack = list(stack)
        buffer = list(buffer)
        arcs = dict(arcs)
        
        if action == 'SHIFT':
            stack.append(buffer.pop(0))
        elif action == 'LEFT-ARC':
            dep = stack.pop(-2)
            head = stack[-1]
            arcs[dep] = head
        elif action == 'RIGHT-ARC':
            dep = stack.pop(-1)
            head = stack[-1]
            arcs[dep] = head
        
        return stack, buffer, arcs
    
    def _get_oracle_action(
        self,
        stack: List[int],
        buffer: List[int],
        gold_heads: List[int],
        arcs: Dict[int, int]
    ) -> str:
        valid = self._get_valid_actions(stack, buffer)
        if not valid:
            return 'SHIFT'  # Fallback
        
        if len(stack) >= 2:
            s0, s1 = stack[-1], stack[-2]
            
            # LEFT-ARC: s1's gold head is s0
            if 'LEFT-ARC' in valid and s1 >= 0 and gold_heads[s1] == s0:
                all_deps_attached = all(
                    i in arcs for i, h in enumerate(gold_heads) if h == s1
                )
                if all_deps_attached:
                    return 'LEFT-ARC'
            
            # RIGHT-ARC: s0's gold head is s1
            if 'RIGHT-ARC' in valid and s0 >= 0 and gold_heads[s0] == s1:
                all_deps_attached = all(
                    i in arcs for i, h in enumerate(gold_heads) if h == s0
                )
                if all_deps_attached:
                    return 'RIGHT-ARC'
        
        return 'SHIFT' if 'SHIFT' in valid else valid[0]
    
    def fit(
        self,
        sentences: List[Dict[str, Any]],
        epochs: int = 10,
        verbose: bool = True
    ) -> 'MaltParser':
        # Collect all labels first
        all_labels = set()
        for sent in sentences:
            if 'rels' in sent:
                all_labels.update(sent['rels'])
        self.labels = sorted(all_labels) if all_labels else []
        
        for epoch in range(epochs):
            correct = 0
            total = 0
            label_correct = 0
            label_total = 0
            
            for sent in sentences:
                words = sent['words']
                pos_tags = sent['pos_tags']
                gold_heads_0idx = [h - 1 for h in sent['heads']]  # Convert to 0-indexed
                gold_rels = sent.get('rels', [])
                
                stack = [-1]  # ROOT
                buffer = list(range(len(words)))
                arcs: Dict[int, int] = {}
                
                while buffer or len(stack) > 1:
                    oracle = self._get_oracle_action(stack, buffer, gold_heads_0idx, arcs)
                    features = self.extract_features(stack, buffer, words, pos_tags, arcs)
                    valid = self._get_valid_actions(stack, buffer)
                    
                    if valid:
                        predicted = max(valid, key=lambda a: self._score_action(features, a, False))
                        
                        if predicted != oracle:
                            for f in features:
                                self.weights[f][oracle] += 1
                                self.averaged_weights[f][oracle] += self._updates
                                self.weights[f][predicted] -= 1
                                self.averaged_weights[f][predicted] -= self._updates
                        else:
                            correct += 1
                        total += 1
                    
                    stack, buffer, arcs = self._apply_action(oracle, stack, buffer, arcs)
                    self._updates += 1
                
                # Train labels online
                if gold_rels and self.labels:
                    for dep in range(len(words)):
                        gold_label = gold_rels[dep]
                        head = sent['heads'][dep] - 1 if sent['heads'][dep] > 0 else len(words)
                        lbl_features = self.extract_label_features(words, pos_tags, head, dep)
                        
                        scores = {lbl: sum(self.label_weights.get(f"{f}_{lbl}", 0) for f in lbl_features) 
                                  for lbl in self.labels}
                        pred_label = max(scores, key=lambda x: scores[x])
                        
                        if pred_label != gold_label:
                            for f in lbl_features:
                                self.label_weights[f"{f}_{gold_label}"] = \
                                    self.label_weights.get(f"{f}_{gold_label}", 0) + 1
                                self.label_weights[f"{f}_{pred_label}"] = \
                                    self.label_weights.get(f"{f}_{pred_label}", 0) - 1
                        else:
                            label_correct += 1
                        label_total += 1
            
            if verbose:
                acc = correct / total * 100 if total > 0 else 0
                las = label_correct / label_total * 100 if label_total > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs} - Action Acc: {acc:.2f}% | Label Acc: {las:.2f}%")
        
        # Finalize averaged weights
        for f in self.weights:
            for a in self.weights[f]:
                self.averaged_weights[f][a] = self.weights[f][a] - self.averaged_weights[f][a] / max(self._updates, 1)
        
        self.is_trained = True
        return self
    
    def predict(self, words: List[str], pos_tags: List[str]) -> List[int]:
        n = len(words)
        stack = [-1]
        buffer = list(range(n))
        arcs: Dict[int, int] = {}
        
        max_steps = 2 * n + 10
        for _ in range(max_steps):
            if not buffer and len(stack) <= 1:
                break
            
            features = self.extract_features(stack, buffer, words, pos_tags, arcs)
            valid = self._get_valid_actions(stack, buffer)
            
            if not valid:
                break
            
            best = max(valid, key=lambda a: self._score_action(features, a, True))
            stack, buffer, arcs = self._apply_action(best, stack, buffer, arcs)
        
        heads = [0] * n
        for dep, head in arcs.items():
            heads[dep] = 0 if head == -1 else head + 1
        
        return heads
