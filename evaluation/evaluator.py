"""
Evaluator class cho Dependency Parsing
Dùng chung cho cả model-based và baseline evaluation
"""
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import torch
from tqdm import tqdm

from evaluation.metrics import (
    compute_uas, 
    compute_las, 
    compute_metrics_by_relation,
    compute_metrics_by_distance
)



class Evaluator:
    """
    Evaluator class để đánh giá dependency parsing models
    
    Hỗ trợ:
    - Đánh giá neural models (BiLSTM, Transformer...)
    - Đánh giá baseline models
    - Tính UAS, LAS
    - Phân tích lỗi theo relation, distance
    """
    
    def __init__(self, vocab: Optional[Any] = None):
        """
        Args:
            vocab: Vocabulary object (optional, dùng cho error analysis)
        """
        self.vocab = vocab
        self.reset()
    
    def reset(self):
        """Reset các counters"""
        self.total_uas_correct = 0
        self.total_las_correct = 0
        self.total_tokens = 0
        
        self.all_pred_heads = []
        self.all_pred_rels = []
        self.all_gold_heads = []
        self.all_gold_rels = []
    
    def evaluate_model(self, model: any, data_loader: torch.utils.data.DataLoader,
                       device: str = 'cpu', return_predictions: bool = False) -> Dict:
        """
        Đánh giá neural model trên data loader
        
        Args:
            model: PyTorch model
            data_loader: DataLoader chứa test data
            device: 'cpu' hoặc 'cuda'
            return_predictions: Có trả về predictions không
        
        Returns:
            Dict chứa UAS, LAS và predictions (nếu có)
        """
        self.reset()
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                # Move to device
                words = batch['words'].to(device)
                pos_tags = batch['pos_tags'].to(device)
                heads = batch['heads'].to(device)
                rels = batch['rels'].to(device)
                lengths = batch['lengths'].to(device)
                
                # Forward pass
                arc_scores, label_scores = model(words, pos_tags, lengths)
                
                # Decode
                pred_heads, pred_rels = model.decode(arc_scores, label_scores, lengths)
                
                # Tính metrics cho từng sentence trong batch
                for i in range(len(lengths)):
                    length = lengths[i].item()
                    
                    pred_h = pred_heads[i, :length].cpu().tolist()
                    pred_r = pred_rels[i, :length].cpu().tolist()
                    gold_h = heads[i, :length].cpu().tolist()
                    gold_r = rels[i, :length].cpu().tolist()
                    
                    # Tính UAS
                    uas_correct, uas_total = compute_uas(pred_h, gold_h)
                    self.total_uas_correct += uas_correct
                    
                    # Tính LAS
                    las_correct, las_total = compute_las(pred_h, pred_r, gold_h, gold_r)
                    self.total_las_correct += las_correct
                    
                    self.total_tokens += uas_total
                    
                    # Lưu predictions
                    if return_predictions:
                        self.all_pred_heads.append(pred_h)
                        self.all_pred_rels.append(pred_r)
                        self.all_gold_heads.append(gold_h)
                        self.all_gold_rels.append(gold_r)
        
        results = self._compute_results()
        
        if return_predictions:
            results['predictions'] = {
                'pred_heads': self.all_pred_heads,
                'pred_rels': self.all_pred_rels,
                'gold_heads': self.all_gold_heads,
                'gold_rels': self.all_gold_rels
            }
        
        return results
    
    def evaluate_transition_model(self, model: any, data_loader: torch.utils.data.DataLoader,
                                   device: str = 'cpu', return_predictions: bool = False) -> Dict:
        self.reset()
        model.eval()
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Evaluating'):
                words = batch['words'].to(device)
                pos_tags = batch['pos_tags'].to(device)
                heads = batch['heads'].to(device)
                rels = batch['rels'].to(device)
                lengths = batch['lengths'].to(device)
                
                # Forward pass (returns dummy scores for transition parser)
                arc_scores, label_scores = model(words, pos_tags, lengths)
                
                # Decode - transition parser needs words and pos_tags
                pred_heads, pred_rels = model.decode(
                    arc_scores, label_scores, lengths,
                    words=words, pos_tags=pos_tags
                )
                
                # Tính metrics cho từng sentence trong batch
                for i in range(len(lengths)):
                    length = int(lengths[i].item())
                    
                    pred_h = pred_heads[i, :length].cpu().tolist()
                    pred_r = pred_rels[i, :length].cpu().tolist()
                    gold_h = heads[i, :length].cpu().tolist()
                    gold_r = rels[i, :length].cpu().tolist()
                    
                    uas_correct, uas_total = compute_uas(pred_h, gold_h)
                    self.total_uas_correct += uas_correct
                    
                    las_correct, las_total = compute_las(pred_h, pred_r, gold_h, gold_r)
                    self.total_las_correct += las_correct
                    
                    self.total_tokens += uas_total
                    
                    if return_predictions:
                        self.all_pred_heads.append(pred_h)
                        self.all_pred_rels.append(pred_r)
                        self.all_gold_heads.append(gold_h)
                        self.all_gold_rels.append(gold_r)
        
        results = self._compute_results()
        
        if return_predictions:
            results['predictions'] = {
                'pred_heads': self.all_pred_heads,
                'pred_rels': self.all_pred_rels,
                'gold_heads': self.all_gold_heads,
                'gold_rels': self.all_gold_rels
            }
        
        return results
    
    def evaluate_predictions(self, pred_heads: List[List[int]], gold_heads: List[List[int]],
                             pred_rels: Optional[List[List[int]]] = None,
                             gold_rels: Optional[List[List[int]]] = None) -> Dict:
        """
        Đánh giá từ predictions có sẵn
        
        Args:
            pred_heads: List các predicted heads cho mỗi sentence
            gold_heads: List các gold heads cho mỗi sentence
            pred_rels: List các predicted relations (optional)
            gold_rels: List các gold relations (optional)
        
        Returns:
            Dict chứa UAS và LAS (nếu có relations)
        """
        self.reset()
        
        for i in range(len(pred_heads)):
            pred_h = pred_heads[i]
            gold_h = gold_heads[i]
            
            # Tính UAS
            uas_correct, uas_total = compute_uas(pred_h, gold_h, skip_root=False)
            self.total_uas_correct += uas_correct
            self.total_tokens += uas_total
            
            # Tính LAS nếu có relations
            if pred_rels is not None and gold_rels is not None:
                las_correct, _ = compute_las(
                    pred_h, pred_rels[i], 
                    gold_h, gold_rels[i], 
                    skip_root=False
                )
                self.total_las_correct += las_correct
        
        return self._compute_results()
    
    def _compute_results(self) -> Dict:
        """Tính toán kết quả cuối cùng"""
        uas = self.total_uas_correct / self.total_tokens * 100 if self.total_tokens > 0 else 0.0
        las = self.total_las_correct / self.total_tokens * 100 if self.total_tokens > 0 else 0.0
        
        return {
            'uas': uas,
            'las': las,
            'total_tokens': self.total_tokens,
            'uas_correct': self.total_uas_correct,
            'las_correct': self.total_las_correct
        }
    
    def analyze_errors(self, pred_heads: List[List[int]], pred_rels: List[List[int]],
                       gold_heads: List[List[int]], gold_rels: List[List[int]],
                       print_results: bool = True) -> Dict:
        """
        Phân tích lỗi chi tiết
        
        Returns:
            Dict chứa error analysis
        """
        total_tokens = 0
        arc_errors = 0
        label_errors = 0
        both_errors = 0
        
        error_by_distance = defaultdict(int)
        error_by_relation = defaultdict(int)
        
        for pred_h, pred_r, gold_h, gold_r in zip(pred_heads, pred_rels, gold_heads, gold_rels):
            for i in range(1, len(gold_h)):
                total_tokens += 1
                
                arc_correct = (pred_h[i] == gold_h[i])
                label_correct = (pred_r[i] == gold_r[i])
                
                if not arc_correct:
                    arc_errors += 1
                    distance = abs(gold_h[i] - i)
                    error_by_distance[distance] += 1
                
                if not label_correct:
                    label_errors += 1
                    if self.vocab:
                        rel_name = self.vocab.idx2rel.get(gold_r[i], '<UNK>')
                    else:
                        rel_name = str(gold_r[i])
                    error_by_relation[rel_name] += 1
                
                if not arc_correct and not label_correct:
                    both_errors += 1
        
        results = {
            'total_tokens': total_tokens,
            'arc_errors': arc_errors,
            'label_errors': label_errors,
            'both_errors': both_errors,
            'arc_error_rate': arc_errors / total_tokens * 100 if total_tokens > 0 else 0,
            'label_error_rate': label_errors / total_tokens * 100 if total_tokens > 0 else 0,
            'error_by_distance': dict(error_by_distance),
            'error_by_relation': dict(error_by_relation)
        }
        
        if print_results:
            self._print_error_analysis(results, arc_errors, label_errors)
        
        return results
    
    def _print_error_analysis(self, results: Dict, arc_errors: int, label_errors: int):
        """In kết quả error analysis"""
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80 + "\n")
        
        print(f"Total tokens analyzed: {results['total_tokens']}")
        print(f"Arc errors: {arc_errors} ({results['arc_error_rate']:.2f}%)")
        print(f"Label errors: {label_errors} ({results['label_error_rate']:.2f}%)")
        print(f"Both errors: {results['both_errors']} ({results['both_errors']/results['total_tokens']*100:.2f}%)")
        
        print("\n" + "-"*80)
        print("Top 10 Error Distances (head-dependent):")
        print("-"*80)
        sorted_distances = sorted(
            results['error_by_distance'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for distance, count in sorted_distances:
            if arc_errors > 0:
                print(f"  Distance {distance}: {count} errors ({count/arc_errors*100:.2f}%)")
        
        print("\n" + "-"*80)
        print("Top 10 Error Relations:")
        print("-"*80)
        sorted_relations = sorted(
            results['error_by_relation'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        for rel, count in sorted_relations:
            if label_errors > 0:
                print(f"  {rel}: {count} errors ({count/label_errors*100:.2f}%)")
        
        print("\n" + "="*80 + "\n")

