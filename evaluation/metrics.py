"""
Các metrics đánh giá cho Dependency Parsing
"""
from typing import List, Tuple, Dict
from collections import defaultdict


def compute_uas(pred_heads: List[int], gold_heads: List[int], 
                skip_root: bool = True) -> Tuple[int, int]:
    """
    Tính Unlabeled Attachment Score (UAS)
    
    Args:
        pred_heads: List các predicted heads
        gold_heads: List các gold heads
        skip_root: Bỏ qua ROOT token (index 0)
    
    Returns:
        Tuple (correct, total)
    """
    correct = 0
    total = 0
    
    start_idx = 1 if skip_root else 0
    
    for i in range(start_idx, len(gold_heads)):
        if pred_heads[i] == gold_heads[i]:
            correct += 1
        total += 1
    
    return correct, total


def compute_las(pred_heads: List[int], pred_rels: List[int],
                gold_heads: List[int], gold_rels: List[int],
                skip_root: bool = True) -> Tuple[int, int]:
    """
    Tính Labeled Attachment Score (LAS)
    LAS yêu cầu cả head và relation đều đúng
    
    Args:
        pred_heads: List các predicted heads
        pred_rels: List các predicted relations
        gold_heads: List các gold heads
        gold_rels: List các gold relations
        skip_root: Bỏ qua ROOT token (index 0)
    
    Returns:
        Tuple (correct, total)
    """
    correct = 0
    total = 0
    
    start_idx = 1 if skip_root else 0
    
    for i in range(start_idx, len(gold_heads)):
        if pred_heads[i] == gold_heads[i] and pred_rels[i] == gold_rels[i]:
            correct += 1
        total += 1
    
    return correct, total


def compute_metrics_by_relation(pred_heads: List[int], pred_rels: List[int],
                                 gold_heads: List[int], gold_rels: List[int],
                                 idx2rel: Dict[int, str]) -> Dict[str, Dict]:
    """
    Tính metrics cho từng loại relation
    
    Returns:
        Dict: {relation: {'correct': int, 'total': int, 'precision': float}}
    """
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i in range(1, len(gold_heads)):
        rel_name = idx2rel.get(gold_rels[i], '<UNK>')
        stats[rel_name]['total'] += 1
        
        if pred_heads[i] == gold_heads[i] and pred_rels[i] == gold_rels[i]:
            stats[rel_name]['correct'] += 1
    
    # Tính precision cho từng relation
    for rel, data in stats.items():
        if data['total'] > 0:
            data['precision'] = data['correct'] / data['total'] * 100
        else:
            data['precision'] = 0.0
    
    return dict(stats)


def compute_metrics_by_distance(pred_heads: List[int], gold_heads: List[int]) -> Dict[int, Dict]:
    """
    Tính metrics theo khoảng cách head-dependent
    
    Returns:
        Dict: {distance: {'correct': int, 'total': int, 'accuracy': float}}
    """
    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i in range(1, len(gold_heads)):
        distance = abs(gold_heads[i] - i)
        stats[distance]['total'] += 1
        
        if pred_heads[i] == gold_heads[i]:
            stats[distance]['correct'] += 1
    
    # Tính accuracy cho từng distance
    for dist, data in stats.items():
        if data['total'] > 0:
            data['accuracy'] = data['correct'] / data['total'] * 100
        else:
            data['accuracy'] = 0.0
    
    return dict(stats)

