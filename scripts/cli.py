import click
from pathlib import Path

from utils.constants import config

# All available neural parser types
NEURAL_PARSERS = [
    # Attention-based
    'biaffine',
    # Transition-based
    'chen_manning_2014', 'weiss_2015', 'andor_2016',
]

# Feature types for biaffine parser
FEAT_TYPES = ['char', 'bert', 'tag']


@click.command()
@click.option('--parser', '-p', default=None, 
              type=click.Choice(NEURAL_PARSERS, case_sensitive=False),
              help='Parser type (overrides config.yaml)')
@click.option('--resume', default=None, type=str, help='Resume from checkpoint path')
def train(parser: str, resume: str):
    """Train the Vietnamese Dependency Parser model.
    
    Available parsers:
    
    \b
    ATTENTION-BASED:
      biaffine         - BiLSTM + Biaffine Attention (Dozat & Manning 2017)
    
    \b
    TRANSITION-BASED:
      chen_manning_2014 - Greedy local training (Chen & Manning 2014)
      weiss_2015        - Structured training with beam (Weiss et al. 2015)
      andor_2016        - Globally normalized (Andor et al. 2016)
    """
    from scripts.train import run_training
    from trainers.train_config import TrainConfig
    
    # Parser type: CLI option > config.yaml
    parser_type = parser or config.get('parser_type', 'biaffine')
    
    train_config = TrainConfig(
        # Parser type
        parser_type=parser_type,
        # Model
        embedding_dim=config['model']['embedding_dim'],
        pos_dim=config['model']['pos_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        arc_dim=config['model']['arc_dim'],
        label_dim=config['model']['label_dim'],
        dropout=config['model']['dropout'],
        # Training
        batch_size=config['training']['batch_size'],
        num_epochs=config['training']['num_epochs'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        min_freq=config['training']['min_freq'],
        seed=config['training']['seed'],
        save_every=config['training']['save_every'],
        # Paths
        save_dir=config['paths']['save_dir'],
        results_dir=config['paths']['results_dir'],
        # Device
        device=config['device'],
        resume=resume,
    )
    
    run_training(train_config)


@click.command()
@click.option('--host', '-h', default='127.0.0.1', help='Host to bind to')
@click.option('--port', '-p', default=5000, type=int, help='Port to bind to')
@click.option('--debug/--no-debug', default=True, help='Enable debug mode')
def visualize(host: str, port: int, debug: bool):
    """Launch interactive dependency tree visualization server."""
    from scripts.visualization import run_visualization
    run_visualization(host=host, port=port, debug=debug)


@click.command()
@click.option('--host', '-h', default='127.0.0.1', help='Host to bind to')
@click.option('--port', '-p', default=5001, type=int, help='Port to bind to')
@click.option('--debug/--no-debug', default=True, help='Enable debug mode')
@click.option('--model', '-m', default='checkpoints/best_model.pt', help='Model checkpoint path')
@click.option('--vocab', '-v', default=None, help='Vocab path (auto-detect from results if not specified)')
def demo(host: str, port: int, debug: bool, model: str, vocab: Path):
    """Launch demo server to compare ground truth vs predictions."""
    from scripts.demo import run_demo
    
    run_demo(host=host, port=port, debug=debug, model_path=model, vocab_path=vocab)


@click.command()
def analyze():
    """Analyze dataset to recommend optimal hyperparameters (e.g., min_freq)."""
    from data.analyzer import Analyzer
    
    analyzer = Analyzer()
    recommended = analyzer.print_analysis_report()
    click.echo(f"\n🎯 Suggested min_freq for config.yaml: {recommended}")


@click.command('train-biaffine')
@click.option('--feat', '-f', default='char', type=click.Choice(FEAT_TYPES, case_sensitive=False),
              help='Feature type: char (CharLSTM), bert (BERT embeddings), tag (POS tags)')
@click.option('--bert', '-b', default='bert-base-multilingual-cased',
              help='BERT model name (only used when --feat=bert)')
@click.option('--embed', '-e', default=None, type=str,
              help='Path to pretrained word embeddings (e.g., word2vec, fastText)')
@click.option('--epochs', default=100, type=int, help='Maximum number of epochs')
@click.option('--batch-size', default=5000, type=int, help='Batch size (in tokens)')
@click.option('--lr', default=2e-3, type=float, help='Learning rate')
@click.option('--save-path', '-s', default='checkpoints/biaffine_model.pt',
              help='Path to save the best model')
@click.option('--min-freq', default=2, type=int, help='Minimum word frequency')
@click.option('--patience', default=100, type=int, help='Early stopping patience')
def train_biaffine(feat: str, bert: str, embed: str, epochs: int, batch_size: int,
                   lr: float, save_path: str, min_freq: int, patience: int):
    """Train Biaffine Dependency Parser (Dozat & Manning 2017).
    
    This uses the DependencyParserTrainer from the solution module,
    which provides a complete training pipeline with advanced features.
    
    \b
    FEATURE TYPES:
      char  - Character-level LSTM embeddings (default, recommended)
      bert  - BERT contextual embeddings (requires transformers)
      tag   - POS tag embeddings only
    
    \b
    EXAMPLES:
      # Train with character embeddings (default)
      parsing train-biaffine
      
      # Train with BERT embeddings
      parsing train-biaffine --feat bert --bert vinai/phobert-base
      
      # Train with pretrained word embeddings
      parsing train-biaffine --embed path/to/embeddings.txt
      
      # Custom training settings
      parsing train-biaffine --epochs 50 --batch-size 3000 --lr 1e-3
    """
    from dataclasses import dataclass
    
    # Import corpus directly to avoid loader dependency issues
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.dependency_parser_trainer import DependencyParserTrainer
    
    from models.solution.dependency_parser_v2 import DependencyParser
    from modules.embeddings import CharacterEmbeddings, FieldEmbeddings
    embeddings = [
        FieldEmbeddings()
    ]
    parser = DependencyParser(
        embeddings=embeddings, init_pre_train=True,
        feat='bert',
        bert='vinai/phobert-base',
        n_feat_embed=768)
    # Load Vietnamese Treebank corpus
    corpus = ViVTBCorpus()
    
    click.echo(f"📊 Configuration:")
    click.echo(f"   Feature type: {feat}")
    if feat.lower() == 'bert':
        click.echo(f"   BERT model: {bert}")
    if embed:
        click.echo(f"   Pretrained embeddings: {embed}")
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   Learning rate: {lr}")
    click.echo(f"   Min frequency: {min_freq}")
    click.echo(f"   Patience: {patience}")
    click.echo(f"   Save path: {save_path}")
    click.echo()
    
    
    # Create trainer and start training
    trainer = DependencyParserTrainer(parser=parser, corpus=corpus)
    
    trainer.train(
        base_path=save_path,
        min_freq=min_freq,
        batch_size=batch_size,
        lr=lr,
        max_epochs=epochs,
        patience=patience,
        
    )
    
    click.echo(f"\n✅ Training complete! Model saved to: {save_path}")


@click.group()
def cli():
    """Vietnamese Dependency Parser CLI"""


cli.add_command(train)
cli.add_command(train_biaffine)
cli.add_command(visualize)
cli.add_command(demo)
cli.add_command(analyze)

def main():
    cli()


if __name__ == "__main__":
    main()
