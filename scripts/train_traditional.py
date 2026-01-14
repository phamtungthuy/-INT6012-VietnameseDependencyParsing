"""
Training script for Traditional (non-neural) Dependency Parsers
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from tqdm import tqdm

from data_processing.loader import CoNLLUDataset
from models.traditional_parser import MaltParser, MSTParser, TurboParser
from utils.constants import (
    TRAIN_FILE_PATH,
    VALIDATION_FILE_PATH,
    TEST_FILE_PATH,
)


PARSER_CLASSES = {
    'malt': MaltParser,
    'mst': MSTParser,
    'turbo': TurboParser,
}


def print_banner(parser_type: str):
    parser_info = {
        'malt': 'MaltParser - Transition-based (Nivre et al. 2006)',
        'mst': 'MSTParser - Graph-based MST (McDonald et al. 2005)',
        'turbo': 'TurboParser - Graph-based (Martins et al. 2010)',
    }
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   📊 Traditional Dependency Parser (Non-Neural Baseline)                     ║
║   {parser_info.get(parser_type, parser_type):<68} ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def load_sentences(file_path: Path) -> List[Dict[str, Any]]:
    """Load CoNLL-U file and convert to list of sentence dicts"""
    dataset = CoNLLUDataset(file_path)
    sentences = []
    
    for sent in tqdm(dataset.sentences, desc=f"Loading {Path(file_path).name}"):
        words = [token['form'] for token in sent]
        pos_tags = [token['upos'] for token in sent]
        heads = [token['head'] for token in sent]
        rels = [token['deprel'] for token in sent]
        
        sentences.append({
            'words': words,
            'pos_tags': pos_tags,
            'heads': heads,
            'rels': rels,
        })
    
    return sentences


def run_traditional_training(
    parser_type: str = 'malt',
    epochs: int = 10,
    save_dir: str = 'checkpoints/traditional',
):
    print_banner(parser_type)
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n📂 Loading data...")
    
    train_sentences = load_sentences(TRAIN_FILE_PATH)
    dev_sentences = load_sentences(VALIDATION_FILE_PATH)
    test_sentences = load_sentences(TEST_FILE_PATH)
    
    print(f"  Train: {len(train_sentences)} sentences")
    print(f"  Dev: {len(dev_sentences)} sentences")
    print(f"  Test: {len(test_sentences)} sentences")
    
    # =========================================================================
    # Create Parser
    # =========================================================================
    print(f"\n🔧 Creating {parser_type.upper()} parser...")
    
    parser_class = PARSER_CLASSES.get(parser_type)
    if parser_class is None:
        raise ValueError(f"Unknown parser type: {parser_type}. Choose from: {list(PARSER_CLASSES.keys())}")
    
    parser = parser_class()
    
    # =========================================================================
    # Train Structure (Heads)
    # =========================================================================
    print(f"\n🚀 Training structure for {epochs} epochs...")
    
    parser.fit(train_sentences, epochs=epochs, verbose=True)
    
    # =========================================================================
    # Train Labels
    # =========================================================================
    print(f"\n🏷️  Training label classifier...")
    
    parser.fit_labels(train_sentences, epochs=5, verbose=True)
    
    # =========================================================================
    # Evaluate
    # =========================================================================
    print("\n📊 Evaluating...")
    
    dev_results = parser.evaluate(dev_sentences)
    test_results = parser.evaluate(test_sentences)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION RESULTS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Dataset    │    UAS    │    LAS    │   Speed (sent/s)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  Dev        │  {dev_results['uas']:6.2f}%  │  {dev_results['las']:6.2f}%  │  {dev_results['speed']:8.1f}                          │
│  Test       │  {test_results['uas']:6.2f}%  │  {test_results['las']:6.2f}%  │  {test_results['speed']:8.1f}                          │
└─────────────────────────────────────────────────────────────────────────────┘
    """)
    
    # =========================================================================
    # Save
    # =========================================================================
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = save_path / f"{parser_type}_{timestamp}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(parser, f)
    
    # Save results
    results = {
        'parser_type': parser_type,
        'epochs': epochs,
        'dev_uas': dev_results['uas'],
        'dev_las': dev_results['las'],
        'dev_speed': dev_results['speed'],
        'test_uas': test_results['uas'],
        'test_las': test_results['las'],
        'test_speed': test_results['speed'],
        'timestamp': timestamp,
    }
    
    results_path = save_path / f"{parser_type}_{timestamp}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TRAINING COMPLETED!                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Parser: {parser_type:<67} ║
║  Test UAS: {test_results['uas']:5.2f}%  |  Test LAS: {test_results['las']:5.2f}%  |  Speed: {test_results['speed']:.1f} sent/s{' ' * 16}║
║  Model saved: {str(model_path):<60} ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    return results
