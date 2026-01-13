import click
from pathlib import Path

from utils.constants import config


@click.command()
@click.option('--resume', default=None, type=str, help='Resume from checkpoint path')
def train(resume: str):
    """Train the Vietnamese Dependency Parser model."""
    from scripts.train import run_training
    from training import TrainConfig
    
    train_config = TrainConfig(
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

@click.group()
def cli():
    """Vietnamese Dependency Parser CLI"""

cli.add_command(train)
cli.add_command(visualize)
cli.add_command(demo)
cli.add_command(analyze)

def main():
    cli()

if __name__ == "__main__":
    main()
