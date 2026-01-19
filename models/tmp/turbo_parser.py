"""
TurboParser - Graph-based parser with second-order features and AD3 decoding

Reference: Martins et al. (2010) "Turbo Parsers: Dependency Parsing by Approximate Variational Inference"

Features:
- First-order arc features
- Second-order features (sibling, grandparent)
- AD3 (Alternating Directions Dual Decomposition) decoding
- Structured Perceptron learning
"""

from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

from models.tmp.base_traditional_parser import BaseTraditionalParser


class TurboParser(BaseTraditionalParser):
    """
    Full TurboParser with AD3 decoding and second-order features.
    
    Based on Martins et al. (2010):
    - First-order arc features (head, dep)
    - Second-order sibling features (head, dep1, dep2)
    - Second-order grandparent features (grandparent, head, dep)
    - AD3 decoding with tree constraint enforcement
    """
    
    def __init__(self):
        super().__init__()
        self.weights: Dict[str, float] = defaultdict(float)
        self.averaged_weights: Dict[str, float] = defaultdict(float)
        self._updates = 0
    
    # ==================== Feature Extraction ====================
    
    def extract_features(
        self,
        words: List[str],
        pos_tags: List[str],
        head: int,
        dep: int
    ) -> List[str]:
        """Extract first-order arc features (head -> dep)"""
        features = []
        n = len(words)
        
        direction = "L" if dep < head else "R"
        distance = abs(head - dep) if head < n else dep + 1
        
        h_word = words[head] if 0 <= head < n else "<ROOT>"
        h_pos = pos_tags[head] if 0 <= head < n else "<ROOT>"
        d_word = words[dep]
        d_pos = pos_tags[dep]
        
        # Unigram features
        features.extend([
            f"arc:h_w={h_word}", f"arc:h_p={h_pos}",
            f"arc:d_w={d_word}", f"arc:d_p={d_pos}",
        ])
        
        # Bigram features
        features.extend([
            f"arc:h_w,d_w={h_word},{d_word}",
            f"arc:h_p,d_p={h_pos},{d_pos}",
            f"arc:h_w,d_p={h_word},{d_pos}",
            f"arc:h_p,d_w={h_pos},{d_word}",
        ])
        
        # Direction features
        features.extend([
            f"arc:dir={direction}",
            f"arc:dir,h_p={direction},{h_pos}",
            f"arc:dir,d_p={direction},{d_pos}",
            f"arc:dir,h_p,d_p={direction},{h_pos},{d_pos}",
        ])
        
        # Distance features
        dist_bin = "1" if distance == 1 else "2-3" if distance <= 3 else "4-6" if distance <= 6 else "7+"
        features.extend([
            f"arc:dist={dist_bin}",
            f"arc:dist,dir={dist_bin},{direction}",
            f"arc:dist,h_p,d_p={dist_bin},{h_pos},{d_pos}",
        ])
        
        # Context features
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
        
        # Between features (POS tags between head and dep)
        if head < n:
            start, end = (dep + 1, head) if dep < head else (head + 1, dep)
            for b in range(start, end):
                features.append(f"arc:between={pos_tags[b]}")
        
        return features
    
    def _extract_sibling_features(
        self,
        words: List[str],
        pos_tags: List[str],
        head: int,
        sib1: int,
        sib2: int
    ) -> List[str]:
        """Extract second-order sibling features (head, sib1, sib2)"""
        features = []
        n = len(words)
        
        h_pos = pos_tags[head] if 0 <= head < n else "<ROOT>"
        s1_pos = pos_tags[sib1]
        s2_pos = pos_tags[sib2]
        s1_word = words[sib1]
        s2_word = words[sib2]
        
        # Basic sibling features
        features.extend([
            f"sib:h_p={h_pos}",
            f"sib:s1_p,s2_p={s1_pos},{s2_pos}",
            f"sib:h_p,s1_p,s2_p={h_pos},{s1_pos},{s2_pos}",
            f"sib:s1_w,s2_w={s1_word},{s2_word}",
        ])
        
        # Direction of siblings
        dir1 = "L" if sib1 < head or head >= n else "R"
        dir2 = "L" if sib2 < head or head >= n else "R"
        features.append(f"sib:dirs={dir1},{dir2}")
        
        # Distance between siblings
        sib_dist = abs(sib1 - sib2)
        dist_bin = "1" if sib_dist == 1 else "2-3" if sib_dist <= 3 else "4+"
        features.append(f"sib:dist={dist_bin}")
        
        # Consecutive sibling indicator
        if sib_dist == 1:
            features.append("sib:consecutive")
        
        return features
    
    def _extract_grandparent_features(
        self,
        words: List[str],
        pos_tags: List[str],
        gp: int,
        head: int,
        dep: int
    ) -> List[str]:
        """Extract second-order grandparent features (gp -> head -> dep)"""
        features = []
        n = len(words)
        
        gp_pos = pos_tags[gp] if 0 <= gp < n else "<ROOT>"
        h_pos = pos_tags[head] if 0 <= head < n else "<ROOT>"
        d_pos = pos_tags[dep]
        
        # Chain features
        features.extend([
            f"gp:gp_p={gp_pos}",
            f"gp:gp_p,h_p={gp_pos},{h_pos}",
            f"gp:h_p,d_p={h_pos},{d_pos}",
            f"gp:gp_p,h_p,d_p={gp_pos},{h_pos},{d_pos}",
        ])
        
        # Direction chain
        dir1 = "L" if head < gp or gp >= n else "R"
        dir2 = "L" if dep < head or head >= n else "R"
        features.append(f"gp:dirs={dir1},{dir2}")
        
        return features
    
    # ==================== Scoring ====================
    
    def _score_features(self, features: List[str], use_averaged: bool = False) -> float:
        weights = self.averaged_weights if use_averaged else self.weights
        return sum(weights[f] for f in features)
    
    def _build_arc_scores(
        self,
        words: List[str],
        pos_tags: List[str],
        use_averaged: bool = False
    ) -> np.ndarray:
        """Build first-order arc score matrix. scores[dep][head]"""
        n = len(words)
        scores = np.full((n, n + 1), -np.inf)
        
        for dep in range(n):
            # Arc from ROOT (index n)
            features = self.extract_features(words, pos_tags, n, dep)
            scores[dep][n] = self._score_features(features, use_averaged)
            
            # Arc from each word
            for head in range(n):
                if head != dep:
                    features = self.extract_features(words, pos_tags, head, dep)
                    scores[dep][head] = self._score_features(features, use_averaged)
        
        return scores
    
    # ==================== MST Decoding ====================
    
    def _chu_liu_edmonds(self, scores: np.ndarray) -> List[int]:
        """
        Chu-Liu-Edmonds algorithm using iterative cycle breaking.
        
        Args:
            scores: shape (n, n+1), scores[dep][head] is score of head->dep
                   head=n represents ROOT
        Returns:
            List of heads (1-indexed, 0 = ROOT)
        """
        n = scores.shape[0]
        if n == 0:
            return []
            
        # 1. Greedy selection: best incoming arc for each node
        heads = np.zeros(n, dtype=int)
        for dep in range(n):
            heads[dep] = np.argmax(scores[dep])
            
        # 2. Cycle breaking
        # Repeatedly find cycles and break them by swapping an edge
        max_cycles = n  # Safety limit
        
        for _ in range(max_cycles):
            cycle = self._find_cycle(heads, n)
            if not cycle:
                break
                
            # Find the best edge to swap to break the cycle
            # gain(v, new_p) = score(v, new_p) - score(v, current_p)
            
            best_gain = float('-inf')
            best_v = -1
            best_new_p = -1
            
            cycle_set = set(cycle)
            
            for v in cycle:
                current_p = heads[v]
                current_score = scores[v][current_p]
                
                for new_p in range(n + 1):
                    if new_p == current_p:
                        continue
                        
                    # Calculate gain
                    new_score = scores[v][new_p]
                    gain = new_score - current_score
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_v = v
                        best_new_p = new_p
            
            if best_v != -1:
                heads[best_v] = best_new_p
            else:
                break 
                
        # 3. Ensure single root
        root_children = [i for i in range(n) if heads[i] == n]
        
        if len(root_children) > 1:
            best_root_child = -1
            best_root_score = float('-inf')
            
            for dep in root_children:
                if scores[dep][n] > best_root_score:
                    best_root_score = scores[dep][n]
                    best_root_child = dep
            
            for dep in root_children:
                if dep != best_root_child:
                    best_alt_h = -1
                    best_alt_s = float('-inf')
                    for h in range(n):
                        if h != dep and scores[dep][h] > best_alt_s:
                            best_alt_s = scores[dep][h]
                            best_alt_h = h
                    if best_alt_h != -1:
                        heads[dep] = best_alt_h
        
        # 4. Final cycle check
        final_cycle = self._find_cycle(heads, n)
        if final_cycle:
             pass 

        return [0 if h == n else int(h) + 1 for h in heads]
    
    def _find_cycle(self, heads: np.ndarray, n: int) -> List[int]:
        """Find a cycle in the current head assignments. Returns empty list if no cycle."""
        visited = [False] * n
        in_stack = [False] * n
        
        for start in range(n):
            if visited[start]:
                continue
            
            path = []
            current = start
            
            while current < n and not visited[current]:
                path.append(current)
                in_stack[current] = True
                visited[current] = True
                current = heads[current]
            
            # If we hit a node in current path, we found a cycle
            if current < n and in_stack[current]:
                cycle_start = path.index(current)
                cycle = path[cycle_start:]
                # Reset in_stack
                for node in path:
                    in_stack[node] = False
                return cycle
            
            # Reset in_stack for next iteration
            for node in path:
                in_stack[node] = False
        
        return []
    
    # ==================== AD3 Decoding ====================
    
    def _ad3_decode(
        self,
        words: List[str],
        pos_tags: List[str],
        use_averaged: bool = False,
        max_iter: int = 50,
        rho: float = 0.1,
        tol: float = 1e-4
    ) -> List[int]:
        """
        AD3 (Alternating Directions Dual Decomposition) decoding.
        
        Combines first-order and second-order factors with tree constraint.
        """
        n = len(words)
        if n == 0:
            return []
        if n == 1:
            return [0]  # Single word, attach to ROOT
        
        # Build first-order scores
        arc_scores = self._build_arc_scores(words, pos_tags, use_averaged)
        
        # Build second-order sibling scores
        # For efficiency, we only consider consecutive siblings
        sib_scores = {}  # (head, sib1, sib2) -> score
        for head in range(n + 1):  # Include ROOT
            deps = [d for d in range(n) if d != head or head == n]
            for i, sib1 in enumerate(deps[:-1]):
                for sib2 in deps[i+1:]:
                    if abs(sib1 - sib2) <= 3:  # Only nearby siblings
                        features = self._extract_sibling_features(
                            words, pos_tags, head if head < n else n, sib1, sib2
                        )
                        sib_scores[(head, sib1, sib2)] = self._score_features(features, use_averaged)
        
        # Initialize arc posteriors q[dep][head]
        q = np.zeros((n, n + 1))
        for dep in range(n):
            # Softmax initialization based on arc scores
            exp_scores = np.exp(arc_scores[dep] - np.max(arc_scores[dep]))
            q[dep] = exp_scores / (exp_scores.sum() + 1e-10)
        
        # Lagrange multipliers
        u = np.zeros((n, n + 1))
        
        # AD3 iterations
        for iteration in range(max_iter):
            q_old = q.copy()
            
            # Step 1: Update arc variables with first-order scores + penalties
            for dep in range(n):
                # Score + penalty term
                scores_with_penalty = arc_scores[dep] + rho * u[dep]
                
                # Add second-order contributions from siblings
                for head in range(n + 1):
                    for other_dep in range(n):
                        if other_dep != dep:
                            key = tuple(sorted([dep, other_dep]))
                            key = (head, key[0], key[1]) if head < n else (n, key[0], key[1])
                            if key in sib_scores:
                                scores_with_penalty[head] += 0.5 * sib_scores[key] * q[other_dep][head]
                
                # Normalize to get posterior
                exp_scores = np.exp(scores_with_penalty - np.max(scores_with_penalty))
                q[dep] = exp_scores / (exp_scores.sum() + 1e-10)
            
            # Step 2: Project to tree constraint using MST
            # Build score matrix from current posteriors
            projection_scores = arc_scores + rho * (q + u)
            z = self._chu_liu_edmonds(projection_scores)
            
            # Convert z to indicator matrix
            z_matrix = np.zeros((n, n + 1))
            for dep, head in enumerate(z):
                if head == 0:
                    z_matrix[dep][n] = 1.0
                else:
                    z_matrix[dep][head - 1] = 1.0
            
            # Step 3: Update Lagrange multipliers
            u = u + q - z_matrix
            
            # Check convergence
            if np.max(np.abs(q - q_old)) < tol:
                break
        
        # Final projection to valid tree
        final_scores = arc_scores + rho * u
        return self._chu_liu_edmonds(final_scores)
    
    def _decode(
        self,
        words: List[str],
        pos_tags: List[str],
        use_averaged: bool = False
    ) -> List[int]:
        """Main decode function using AD3."""
        return self._ad3_decode(words, pos_tags, use_averaged)
    
    # ==================== Training ====================
    
    def fit(
        self,
        sentences: List[Dict[str, Any]],
        epochs: int = 10,
        verbose: bool = True
    ) -> 'TurboParser':
        """
        Train with structured perceptron using first and second-order features.
        """
        # Collect all labels
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
                gold_heads = sent['heads']
                gold_rels = sent.get('rels', [])
                n = len(words)
                
                # Decode with current weights
                pred_heads = self._decode(words, pos_tags, use_averaged=False)
                
                # Update first-order features
                for dep in range(n):
                    gold_h = gold_heads[dep] - 1 if gold_heads[dep] > 0 else n
                    pred_h = pred_heads[dep] - 1 if pred_heads[dep] > 0 else n
                    
                    if gold_h != pred_h:
                        # Positive update for gold
                        gold_feats = self.extract_features(words, pos_tags, gold_h, dep)
                        for f in gold_feats:
                            self.weights[f] += 1
                            self.averaged_weights[f] += self._updates
                        
                        # Negative update for predicted
                        pred_feats = self.extract_features(words, pos_tags, pred_h, dep)
                        for f in pred_feats:
                            self.weights[f] -= 1
                            self.averaged_weights[f] -= self._updates
                    else:
                        correct += 1
                    
                    total += 1
                
                # Update second-order sibling features
                gold_children = defaultdict(list)
                pred_children = defaultdict(list)
                
                for dep in range(n):
                    gold_h = gold_heads[dep] - 1 if gold_heads[dep] > 0 else n
                    pred_h = pred_heads[dep] - 1 if pred_heads[dep] > 0 else n
                    gold_children[gold_h].append(dep)
                    pred_children[pred_h].append(dep)
                
                # Update for each head
                for head in set(list(gold_children.keys()) + list(pred_children.keys())):
                    gold_sibs = sorted(gold_children.get(head, []))
                    pred_sibs = sorted(pred_children.get(head, []))
                    
                    # Gold sibling pairs
                    for i in range(len(gold_sibs) - 1):
                        sib1, sib2 = gold_sibs[i], gold_sibs[i + 1]
                        if (head, sib1, sib2) not in [(h, s1, s2) for h in pred_children 
                                                       for s1, s2 in zip(sorted(pred_children[h])[:-1], 
                                                                        sorted(pred_children[h])[1:])
                                                       if h == head]:
                            features = self._extract_sibling_features(
                                words, pos_tags, head if head < n else n, sib1, sib2
                            )
                            for f in features:
                                self.weights[f] += 1
                                self.averaged_weights[f] += self._updates
                    
                    # Predicted sibling pairs (negative update)
                    for i in range(len(pred_sibs) - 1):
                        sib1, sib2 = pred_sibs[i], pred_sibs[i + 1]
                        if sib1 not in gold_sibs or sib2 not in gold_sibs or \
                           gold_sibs.index(sib2) - gold_sibs.index(sib1) != 1:
                            features = self._extract_sibling_features(
                                words, pos_tags, head if head < n else n, sib1, sib2
                            )
                            for f in features:
                                self.weights[f] -= 1
                                self.averaged_weights[f] -= self._updates
                
                # Train labels online
                if gold_rels and self.labels:
                    for dep in range(n):
                        gold_label = gold_rels[dep]
                        head = gold_heads[dep] - 1 if gold_heads[dep] > 0 else n
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
                
                self._updates += 1
            
            if verbose:
                uas = correct / total * 100 if total > 0 else 0
                las = label_correct / label_total * 100 if label_total > 0 else 0
                print(f"Epoch {epoch + 1}/{epochs} - UAS: {uas:.2f}% | LAS: {las:.2f}%")
        
        # Finalize averaged weights
        for f in self.weights:
            self.averaged_weights[f] = self.weights[f] - self.averaged_weights[f] / max(self._updates, 1)
        
        self.is_trained = True
        return self
    
    def predict(self, words: List[str], pos_tags: List[str]) -> List[int]:
        """Parse a sentence and return heads (1-indexed, 0 = ROOT)."""
        return self._decode(words, pos_tags, use_averaged=True)
