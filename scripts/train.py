import json
import random
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from data_processing.loader import get_data_loaders
from models.parser_factory import ParserFactory, ParserType
from training import TrainConfig, TrainerFactory
from utils.constants import (
    ROOT_PATH,
    TRAIN_FILE_PATH,
    VALIDATION_FILE_PATH,
    TEST_FILE_PATH,
    CONFIG_FILE,
)
from utils.logs import train_logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


PARSER_DESCRIPTIONS = {
    'biaffine': 'BiLSTM + Biaffine Attention (Dozat & Manning 2017)',
    'chen_manning_2014': 'Transition-based Greedy (Chen & Manning 2014)',
    'weiss_2015': 'Transition-based Structured (Weiss et al. 2015)',
    'andor_2016': 'Transition-based Global (Andor et al. 2016)',
}


def print_banner(parser_type: str):
    parser_name = PARSER_DESCRIPTIONS.get(parser_type, parser_type)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🌳 Vietnamese Dependency Parser                                            ║
║   {parser_name:<68} ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def save_results(
    config: TrainConfig,
    results: dict,
    history: list,
    vocab_data: dict,
    results_dir: Path
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir / f"run_{config.parser_type}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(CONFIG_FILE, run_dir / "config.yaml")
    
    config_dict = asdict(config)
    config_dict['vocab_size'] = vocab_data['vocab_size']
    config_dict['pos_size'] = vocab_data['pos_size']
    config_dict['num_labels'] = vocab_data['num_labels']
    config_dict['timestamp'] = timestamp
    
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    with open(run_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(run_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    torch.save(vocab_data['vocab'], run_dir / "vocab.pt")
    
    train_logger.info(f"Saved all results to {run_dir}")
    
    return run_dir


def run_training(config: TrainConfig):
    print_banner(config.parser_type)
    
    # Device setup
    device = config.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    set_seed(config.seed)
    train_logger.info(f"Parser type: {config.parser_type}")
    train_logger.info(f"Random seed: {config.seed}")
    train_logger.info(f"Device: {device}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    train_logger.info("Loading data...")
    
    train_loader, dev_loader, test_loader, vocab = get_data_loaders(
        train_path=TRAIN_FILE_PATH,
        validation_path=VALIDATION_FILE_PATH,
        test_path=TEST_FILE_PATH,
        batch_size=config.batch_size,
        min_freq=config.min_freq
    )
    
    train_logger.info(f"Dataset statistics:")
    train_logger.info(f"  Train batches: {len(train_loader)}")
    train_logger.info(f"  Dev batches: {len(dev_loader)}")
    train_logger.info(f"  Test batches: {len(test_loader)}")
    train_logger.info(f"  Vocabulary size: {len(vocab.word2idx)}")
    train_logger.info(f"  POS tags: {len(vocab.pos2idx)}")
    train_logger.info(f"  Relations: {len(vocab.rel2idx)}")
    
    # =========================================================================
    # Create Model using Factory
    # =========================================================================
    train_logger.info("Creating model...")
    
    parser_type = ParserFactory.from_string(config.parser_type)
    
    # Common kwargs for all parsers
    model_kwargs = {
        'vocab_size': len(vocab.word2idx),
        'pos_size': len(vocab.pos2idx),
        'num_labels': len(vocab.rel2idx),
        'embedding_dim': config.embedding_dim,
        'dropout': config.dropout,
    }
    
    # Parser-specific kwargs
    if parser_type == ParserType.BIAFFINE:
        # Biaffine attention parser (Dozat & Manning 2017)
        model_kwargs.update({
            'pos_dim': config.pos_dim,
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'arc_dim': config.arc_dim,
            'label_dim': config.label_dim,
        })
    elif parser_type == ParserType.CHEN_MANNING_2014:
        # Chen & Manning 2014 - Greedy transition parser
        model_kwargs.update({
            'pos_dim': config.pos_dim,
            'hidden_dim': config.hidden_dim,
            'num_stack': 3,
            'num_buffer': 3,
        })
    elif parser_type == ParserType.WEISS_2015:
        # Weiss et al. 2015 - Structured training with beam
        model_kwargs.update({
            'pos_dim': config.pos_dim,
            'hidden_dim': 1024,  # Larger network
            'num_stack': 3,
            'num_buffer': 3,
            'beam_size': 8,
        })
    elif parser_type == ParserType.ANDOR_2016:
        # Andor et al. 2016 - Globally normalized
        model_kwargs.update({
            'pos_dim': config.pos_dim,
            'hidden_dim': 1024,  # Larger network
            'num_stack': 3,
            'num_buffer': 3,
            'beam_size': 32,
        })
    
    model = ParserFactory.create_parser(parser_type, **model_kwargs)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_logger.info(f"Total parameters: {total_params:,}")
    train_logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # =========================================================================
    # Create Trainer using Factory
    # =========================================================================
    trainer_type = TrainerFactory.from_parser_type(config.parser_type)
    trainer = TrainerFactory.create_trainer(
        trainer_type=trainer_type,
        model=model,
        train_loader=train_loader,
        validation_loader=dev_loader,
        vocab=vocab,
        device=device,
        lr=config.lr,
        weight_decay=config.weight_decay,
        save_dir=config.save_dir,
        save_every=config.save_every
    )
    
    # Resume from checkpoint if specified
    if config.resume:
        start_epoch = trainer.load(config.resume)
        train_logger.info(f"Resumed from epoch {start_epoch}")
    
    # =========================================================================
    # Train
    # =========================================================================
    train_history = trainer.train(config.num_epochs)
    
    # =========================================================================
    # Final Evaluation on Test Set
    # =========================================================================
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         FINAL EVALUATION ON TEST SET                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Load best model
    best_model_path = Path(config.save_dir) / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        train_logger.info(f"Loaded best model from {best_model_path}")
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_loader)
    
    train_logger.info(f"Test Results:")
    train_logger.info(f"  UAS (Unlabeled Attachment Score): {test_results['uas']:.2f}%")
    train_logger.info(f"  LAS (Labeled Attachment Score): {test_results['las']:.2f}%")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    results = {
        'parser_type': config.parser_type,
        'test_uas': test_results['uas'],
        'test_las': test_results['las'],
        'best_dev_uas': trainer.best_uas,
        'best_dev_las': trainer.best_las,
        'best_epoch': trainer.best_epoch,
        'total_epochs': config.num_epochs,
    }
    
    vocab_data = {
        'vocab_size': len(vocab.word2idx),
        'pos_size': len(vocab.pos2idx),
        'num_labels': len(vocab.rel2idx),
        'vocab': {
            'word2idx': vocab.word2idx,
            'idx2word': vocab.idx2word,
            'pos2idx': vocab.pos2idx,
            'idx2pos': vocab.idx2pos,
            'rel2idx': vocab.rel2idx,
            'idx2rel': vocab.idx2rel,
        }
    }
    
    results_dir = ROOT_PATH / config.results_dir
    run_dir = save_results(
        config=config,
        results=results,
        history=train_history,
        vocab_data=vocab_data,
        results_dir=results_dir
    )
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      TRAINING AND EVALUATION COMPLETED!                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Parser: {config.parser_type:<67} ║
║  Results saved to: {str(run_dir):<56} ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    return results
