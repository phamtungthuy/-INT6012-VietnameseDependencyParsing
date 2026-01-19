
import os
import sys
import collections
import statistics
import json
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transforms.conll import CoNLL

def compute_stats(sentences, name):
    num_sents = len(sentences)
    num_tokens = 0
    token_lengths = []
    
    pos_counts = collections.Counter()
    deprel_counts = collections.Counter()
    
    dep_distances = []
    non_projective_count = 0
    
    for sent in sentences:
        # sent is CoNLLSentence.
        # fields: ID, FORM, LEMMA, CPOS, POS, FEATS, HEAD, DEPREL...
        # sent.values contains standard columns.
        # We can access attributes if mapped in CoNLL.
        
        # Accessing columns
        # sent.POS is list of POS tags
        # sent.HEAD is list of heads (str ids)
        # sent.DEPREL is list of relations
        
        # sent.values is a list of columns (tuples), transposed from rows.
        # CoNLL fields: ID, FORM, LEMMA, CPOS, POS, FEATS, HEAD, DEPREL, ...
        # Indices: 0, 1, 2, 3, 4, 5, 6, 7
        
        forms = sent.values[1]
        pos_tags = sent.values[3] # CPOS (UPOS)
        heads = sent.values[6]
        deprels = sent.values[7]
        
        # Convert heads to int (handles 0 for root)
        # Heads in CoNLLSentence are strings initially
        heads_int = [int(h) for h in heads]
        
        seq_len = len(forms)
        num_tokens += seq_len
        token_lengths.append(seq_len)
        
        # POS Stats
        pos_counts.update(pos_tags)
        
        # DepRel Stats
        deprel_counts.update(deprels)
        
        # Dependency Distance (Head - Dep)
        # i is 1-based index of dependent. head is 1-based index of head (0=root).
        current_sent_distances = []
        for i, h in enumerate(heads_int, 1):
            if h == 0:
                dist = 0 # or distance to root? Usually ignored or treated specially. 
                # Let's count distance to Root as '0' or 'i'?
                # Nivre typically measures dependency length for non-root arcs.
                pass
            else:
                dist = h - i
                current_sent_distances.append(abs(dist))
        
        dep_distances.extend(current_sent_distances)
        
        # Non-projectivity
        # heads_int is list of heads. CoNLL.isprojective expects list or similar.
        if not CoNLL.isprojective(heads_int):
            non_projective_count += 1
            
    stats = {
        "split": name,
        "sentences": num_sents,
        "tokens": num_tokens,
        "avg_len": statistics.mean(token_lengths) if token_lengths else 0,
        "max_len": max(token_lengths) if token_lengths else 0,
        "min_len": min(token_lengths) if token_lengths else 0,
        "pos_dist": dict(pos_counts.most_common()),
        "deprel_dist": dict(deprel_counts.most_common()),
        "avg_dep_dist": statistics.mean(dep_distances) if dep_distances else 0,
        "non_projective_sents": non_projective_count,
        "non_projective_pct": (non_projective_count / num_sents * 100) if num_sents else 0
    }
    return stats

def main():
    conll = CoNLL()
    
    datasets_dir = Path('datasets')
    files = {
        'train': datasets_dir / 'vi_vtb-ud-train.conllu',
        'dev': datasets_dir / 'vi_vtb-ud-dev.conllu',
        'test': datasets_dir / 'vi_vtb-ud-test.conllu'
    }
    
    all_stats = {}
    
    for split, path in files.items():
        if not path.exists():
            print(f"File {path} not found.")
            continue
            
        print(f"Loading {split} from {path}...")
        # Use CoNLL load
        sentences = conll.load(str(path))
        print(f"Analyzing {split} ({len(sentences)} sentences)...")
        stats = compute_stats(sentences, split)
        all_stats[split] = stats
        
    # Output results
    print(json.dumps(all_stats, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
