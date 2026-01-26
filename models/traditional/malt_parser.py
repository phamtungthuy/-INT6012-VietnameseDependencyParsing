from typing import Optional, Dict, List
from collections import defaultdict

from transforms.conll import CoNLL

def _nested_float_dict():
    """Helper function for picklable nested defaultdict"""
    return defaultdict(float)

class MaltParser:
    def __init__(self,
    transform: Optional[CoNLL] = None):
        self.transform = transform
        if self.transform:
            self.WORD, self.FEAT = self.transform.FORM, self.transform.CPOS
            self.ARC, self.REL = self.transform.HEAD, self.transform.DEPREL
            self.vocab_rels = self.transform.DEPREL.vocab
        self.weights: Dict[str, Dict[str, float]] = defaultdict(_nested_float_dict)
        self.avg_weights: Dict[str, Dict[str, float]] = defaultdict(_nested_float_dict)
        self.steps = 1

    def _get_valid_actions(self, stack: List[int], buffer: List[int]) -> List[str]:
        """Trả về danh sách các hành động hợp lệ dựa trên trạng thái hiện tại."""
        valid = []
        if len(buffer) > 0:
            valid.append('SHIFT')
        if len(stack) >= 2:
            # Không cho LEFT-ARC nếu s1 là ROOT (index 0)
            if stack[-2] != 0:
                valid.append('LEFT-ARC')
            valid.append('RIGHT-ARC')
        return valid

    def update(self, words: List[int], feats: List[int], gold_heads: List[int], gold_rels: List[int]):
        """
        Args:
            words: List of IDs of words [ROOT, w1, w2, ...]
            feats: List of IDs of features
            gold_heads: List of IDs of gold heads
            gold_rels: List of IDs of gold relations
        """
        stack: List[int] = [0]
        buffer: List[int] = list(range(1, len(words)))
        
        arcs = {}
        rels = {}
        
        gold_rels_str = [self.vocab_rels.itos[i] for i in gold_rels] if hasattr(self, 'vocab_rels') else [str(i) for i in gold_rels]

        # Helper to get word and feature
        def get_w(idx): 
            if idx >= len(words): return "<UNK>"
            return str(words[idx])
            
        def get_p(idx): 
            if idx >= len(feats): return "<UNK>"
            return str(feats[idx])

        # Vòng lặp Parsing
        while buffer or len(stack) > 1:
            valid_actions = self._get_valid_actions(stack, buffer)
            if not valid_actions:
                break  # Không có hành động hợp lệ -> thoát
            
            # a. Features
            features = self.extract_features(stack, buffer, get_w, get_p)
            
            # b. Oracle (Đáp án)
            gold_action = self.get_oracle_action(stack, buffer, gold_heads, gold_rels_str, arcs)
            
            # c. Predict (Dự đoán) - chỉ từ danh sách hợp lệ
            pred_action = self.predict(features, valid_actions)
            
            # d. Update Weights (Học)
            if pred_action != gold_action:
                for f in features:
                    self.weights[f][gold_action] += 1
                    self.avg_weights[f][gold_action] += self.steps
                    
                    self.weights[f][pred_action] -= 1
                    self.avg_weights[f][pred_action] -= self.steps
            
            self.apply_action(gold_action, stack, buffer, arcs, rels)
            self.steps += 1
            
        return arcs, rels

    def extract_features(self, stack: List[int], buffer: List[int], get_w, get_p) -> List[str]:
        features = []
        
        # Stack Top (S0) & Second (S1)
        s0 = stack[-1] if len(stack) > 0 else None
        s1 = stack[-2] if len(stack) > 1 else None
        
        # Buffer Top (B0) & Second (B1)
        b0 = buffer[0] if len(buffer) > 0 else None
        b1 = buffer[1] if len(buffer) > 1 else None
        
        # Unigram Features (Đơn)
        if s0 is not None:
            features.append(f"s0w={get_w(s0)}")
            features.append(f"s0p={get_p(s0)}")
        if s1 is not None:
            features.append(f"s1w={get_w(s1)}")
            features.append(f"s1p={get_p(s1)}")
        if b0 is not None:
            features.append(f"b0w={get_w(b0)}")
            features.append(f"b0p={get_p(b0)}")
        if b1 is not None:
            features.append(f"b1p={get_p(b1)}")
            
        # Bigram Features (Cặp)
        if s0 is not None and b0 is not None:
            features.append(f"s0p_b0p={get_p(s0)}_{get_p(b0)}")
            features.append(f"s0w_b0p={get_w(s0)}_{get_p(b0)}")
            
        if s0 is not None and s1 is not None:
            features.append(f"s1p_s0p={get_p(s1)}_{get_p(s0)}")
            features.append(f"s1w_s0w={get_w(s1)}_{get_w(s0)}")

        # Trigram Features (Ba)
        if s0 is not None and s1 is not None and b0 is not None:
            features.append(f"s1p_s0p_b0p={get_p(s1)}_{get_p(s0)}_{get_p(b0)}")
            
        return features

    def get_oracle_action(self, stack: List[int], buffer: List[int], gold_heads: List[int], gold_rels: List[str], arcs: Dict[int, int]):
        """Arc-Standard Oracle với kiểm tra điều kiện đầy đủ."""
        valid = self._get_valid_actions(stack, buffer)
        if not valid:
            return 'SHIFT'  # Fallback (sẽ không xảy ra vì đã check ở ngoài)
        
        if len(stack) >= 2:
            s0 = stack[-1]
            s1 = stack[-2]
            
            # LEFT-ARC: S1 <- S0 (S0 làm cha, S1 làm con)
            if 'LEFT-ARC' in valid and gold_heads[s1] == s0:
                # Kiểm tra s1 đã thu thập hết con chưa
                all_deps_attached = all(
                    i in arcs for i, h in enumerate(gold_heads) if h == s1
                )
                if all_deps_attached:
                    label = gold_rels[s1] 
                    return f'LEFT-ARC#{label}'
            
            # RIGHT-ARC: S0 <- S1 (S1 làm cha, S0 làm con)
            if 'RIGHT-ARC' in valid and gold_heads[s0] == s1:
                # Kiểm tra s0 đã thu thập hết con chưa
                all_deps_attached = all(
                    i in arcs for i, h in enumerate(gold_heads) if h == s0
                )
                if all_deps_attached:
                    label = gold_rels[s0] 
                    return f'RIGHT-ARC#{label}'
        
        # Fallback: Ưu tiên SHIFT, nếu không thì chọn action hợp lệ đầu tiên
        return 'SHIFT' if 'SHIFT' in valid else valid[0]

    def predict(self, features: List[str], valid_actions: Optional[List[str]] = None):
        """Dự đoán action tốt nhất từ danh sách hợp lệ."""
        scores = defaultdict(float)
        for f in features:
            for action, weight in self.weights[f].items():
                scores[action] += weight
        
        if valid_actions:
            # Chỉ xét các action hợp lệ
            valid_scores = {a: scores.get(a, 0) for a in valid_actions}
            # Thêm các typed actions (LEFT-ARC#xxx, RIGHT-ARC#yyy)
            for action in scores:
                base_action = action.split('#')[0]
                if base_action in valid_actions:
                    if action not in valid_scores or valid_scores[action] < scores[action]:
                        valid_scores[action] = scores[action]
            
            if valid_scores:
                return max(valid_scores, key=lambda k: valid_scores[k])
            return valid_actions[0]  # Fallback
        
        if not scores: 
            return 'SHIFT'
        return max(scores, key=lambda k: scores[k])

    def apply_action(self, action_str: str, stack: List[int], buffer: List[int], arcs=None, rels=None):
        parts = action_str.split('#')
        action = parts[0]
        label = parts[1] if len(parts) > 1 else None

        if action == 'SHIFT':
            if buffer:
                stack.append(buffer.pop(0))
                
        elif action == 'LEFT-ARC':
            if len(stack) > 1:
                dep = stack.pop(-2) # S1
                head = stack[-1]    # S0
                if arcs is not None: arcs[dep] = head
                if rels is not None and label: rels[dep] = label
                
        elif action == 'RIGHT-ARC':
            if len(stack) > 1:
                dep = stack.pop(-1) # S0
                head = stack[-1]    # S1
                if arcs is not None: arcs[dep] = head
                if rels is not None and label: rels[dep] = label

    def parse(self, words: List[int], feats: List[int]):
        """Dự đoán cây cú pháp cho 1 câu (Inference mode)."""
        stack: List[int] = [0]
        buffer: List[int] = list(range(1, len(words)))
        
        arcs = {}
        rels = {}

        # Helper to get word and feature
        def get_w(idx): 
            if idx >= len(words): return "<UNK>"
            return str(words[idx])
        def get_p(idx): 
            if idx >= len(feats): return "<UNK>"
            return str(feats[idx])

        # Giới hạn số bước để tránh infinite loop
        max_steps = len(words) * 3
        step_count = 0

        while buffer or len(stack) > 1:
            step_count += 1
            if step_count > max_steps:
                break  # Thoát nếu quá nhiều bước
            
            valid_actions = self._get_valid_actions(stack, buffer)
            if not valid_actions:
                break  # Không có hành động hợp lệ -> thoát
            
            features = self.extract_features(stack, buffer, get_w, get_p)
            
            # Dự đoán từ danh sách hợp lệ
            pred_action_str = self.predict(features, valid_actions)
            
            # Thực thi hành động dự đoán
            self.apply_action(pred_action_str, stack, buffer, arcs, rels)

        return arcs, rels

