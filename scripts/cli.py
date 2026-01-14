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
    from training import TrainConfig
    
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
    from data_processing.analyzer import Analyzer
    
    analyzer = Analyzer()
    recommended = analyzer.print_analysis_report()
    click.echo(f"\n🎯 Suggested min_freq for config.yaml: {recommended}")


@click.command('train-baseline')
@click.option('--parser', '-p', default='malt', type=click.Choice(['malt', 'mst', 'turbo']),
              help='Traditional parser type')
@click.option('--epochs', '-e', default=10, type=int, help='Number of training epochs')
@click.option('--save-dir', '-s', default='checkpoints/traditional', help='Save directory')
def train_baseline(parser: str, epochs: int, save_dir: str):
    """Train traditional (non-neural) dependency parsers.
    
    \b
    TRADITIONAL PARSERS:
      malt  - MaltParser (Nivre et al. 2006) - Transition-based
      mst   - MSTParser (McDonald et al. 2005) - Graph-based MST
      turbo - TurboParser (Martins et al. 2010) - Dual decomposition
    """
    from scripts.train_traditional import run_traditional_training
    
    run_traditional_training(
        parser_type=parser,
        epochs=epochs,
        save_dir=save_dir,
    )


@click.command('list-parsers')
def list_parsers():
    """List all available parser types."""
    click.echo("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     Available Dependency Parsers                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🧠 NEURAL ATTENTION-BASED (use: train -p <name>)                            ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  biaffine           BiLSTM + Biaffine (Dozat & Manning 2017)                 ║
║                                                                              ║
║  🔄 NEURAL TRANSITION-BASED (use: train -p <name>)                           ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  chen_manning_2014  Greedy local training (Chen & Manning 2014)              ║
║  weiss_2015         Structured + Beam search (Weiss et al. 2015)             ║
║  andor_2016         Globally normalized (Andor et al. 2016)                  ║
║                                                                              ║
║  📊 TRADITIONAL (non-neural) (use: train-baseline -p <name>)                 ║
║  ────────────────────────────────────────────────────────────────────────    ║
║  malt               MaltParser - Transition-based (Nivre et al. 2006)        ║
║  mst                MSTParser - Maximum Spanning Tree (McDonald et al. 2005) ║
║  turbo              TurboParser - Dual decomposition (Martins et al. 2010)   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


@click.group()
def cli():
    """Vietnamese Dependency Parser CLI"""


cli.add_command(train)
cli.add_command(train_baseline)
cli.add_command(visualize)
cli.add_command(demo)
cli.add_command(analyze)
cli.add_command(list_parsers)


def main():
    cli()


if __name__ == "__main__":
    main()
