import click
from pathlib import Path

from utils.constants import config

# Feature types for biaffine parser
FEAT_TYPES = ['char', 'bert', 'tag']


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
    
    from models.biaffine.dependency_parser import DependencyParser
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


# === Tmp trainer commands ===

@click.command('train-malt-parser')
@click.option('--epochs', default=20, type=int, help='Maximum number of epochs')
@click.option('--save-path', '-s', default='checkpoints/malt_parser', help='Path to save model')
def train_malt_parser(epochs: int, save_path: str):
    """Train MaltParser from models/traditional (Arc-Standard)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.malt_trainer import MaltTrainer
    from models.traditional.malt_parser import MaltParser
    
    malt_parser = MaltParser()
    corpus = ViVTBCorpus()
    click.echo(f"📊 MaltParser (Arc-Standard) - Epochs: {epochs}")
    # MaltTrainer creates its own parser inside train(), so we pass None
    trainer = MaltTrainer(parser=malt_parser, corpus=corpus) 
    trainer.train(base_path=save_path, max_epochs=epochs)


@click.command('train-mst-parser')
@click.option('--epochs', default=20, type=int, help='Maximum number of epochs')
def train_mst_parser(epochs: int):
    """Train MSTParser from models/traditional (fast version)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.traditional_trainer import TraditionalTrainer
    
    corpus = ViVTBCorpus()
    click.echo(f"📊 MSTParser - Epochs: {epochs}")
    trainer = TraditionalTrainer(parser_type='mst', corpus=corpus)
    trainer.train(max_epochs=epochs)


@click.command('train-turbo-parser')
@click.option('--epochs', default=20, type=int, help='Maximum number of epochs')
def train_turbo_parser(epochs: int):
    """Train TurboParser from models/traditional (fast version)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.traditional_trainer import TraditionalTrainer
    
    corpus = ViVTBCorpus()
    click.echo(f"📊 TurboParser - Epochs: {epochs}")
    trainer = TraditionalTrainer(parser_type='turbo', corpus=corpus)
    trainer.train(max_epochs=epochs)


cli.add_command(train_biaffine)
cli.add_command(train_malt_parser)
cli.add_command(train_mst_parser)
cli.add_command(train_turbo_parser)
cli.add_command(visualize)
cli.add_command(demo)
cli.add_command(analyze)

def main():
    cli()


if __name__ == "__main__":
    main()
