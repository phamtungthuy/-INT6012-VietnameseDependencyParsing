import click
from pathlib import Path

# Feature types for biaffine parser
FEAT_TYPES = ['char', 'bert', 'tag']

@click.command('train-biaffine')
@click.option('--feat', '-f', default='char', type=click.Choice(FEAT_TYPES, case_sensitive=False),
              help='Feature type: char (CharLSTM), bert (BERT embeddings), tag (POS tags)')
@click.option('--bert', '-b', default='bert-base-multilingual-cased',
              help='BERT model name (only used when --feat=bert)')
@click.option('--embed', '-e', default=None, type=str,
              help='Path to pretrained word embeddings (e.g., word2vec, fastText)')
@click.option('--epochs', default=20, type=int, help='Maximum number of epochs')
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
    from trainers.biaffine_trainer import BiaffineTrainer
    
    from models.graph_based.biaffine_parser import BiaffineParser
    from modules.embeddings import CharacterEmbeddings, FieldEmbeddings
    embeddings = [
        FieldEmbeddings()
    ]
    parser = BiaffineParser(
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
    trainer = BiaffineTrainer(parser=parser, corpus=corpus)
    
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
    from models.transition_based.malt_parser import MaltParser
    
    malt_parser = MaltParser()
    corpus = ViVTBCorpus()
    click.echo(f"📊 MaltParser (Arc-Standard) - Epochs: {epochs}")
    # MaltTrainer creates its own parser inside train(), so we pass None
    trainer = MaltTrainer(parser=malt_parser, corpus=corpus) 
    trainer.train(base_path=save_path, max_epochs=epochs)


@click.command('train-biaffine-v2')
@click.option('--bert', '-b', default='vinai/phobert-base',
              help='BERT model name (default: vinai/phobert-base)')
@click.option('--embed', '-e', default=None, type=str,
              help='Path to pretrained word embeddings (e.g., cc.vi.300.vec recommended)')
@click.option('--epochs', default=20, type=int, help='Maximum number of epochs')
@click.option('--batch-size', default=3000, type=int, help='Batch size (in tokens). Lower batch size recommended for BERT.')
@click.option('--lr', default=2e-3, type=float, help='Learning rate for Non-BERT layers')
@click.option('--bert-lr', default=2e-5, type=float, help='Learning rate for BERT layers (fine-tuning)')
@click.option('--save-path', '-s', default='checkpoints/biaffine_model_v2.pt',
              help='Path to save the best model')
@click.option('--min-freq', default=2, type=int, help='Minimum word frequency')
@click.option('--patience', default=20, type=int, help='Early stopping patience')
def train_biaffine_v2(bert: str, embed: str, epochs: int, batch_size: int,
                      lr: float, bert_lr: float, save_path: str, min_freq: int, patience: int):
    """Train Enhanced Biaffine Parser V2 (BERT Fine-tune + POS Tag + FastText).
    
    This V2 implementation matches the high-performance architecture (80%+ UAS):
    1. Uses Fine-tuned BERT (PhoBERT)
    2. Incorporates POS Tag embeddings
    3. Uses FastText word embeddings (if provided)
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.biaffine_trainer_v2 import BiaffineTrainerV2
    from models.graph_based.biaffine_parser_v2 import BiaffineParserV2
    from modules.embeddings import FieldEmbeddings
    
    # Initialize V2 Parser
    # Note: V2 parser logic is slightly different, it doesn't need 'embeddings' list wrapper 
    # as much as V1 if we pass args directly, but let's keep it simple.
    
    parser = BiaffineParserV2(
        feat='bert', # V2 focuses on BERT
        bert=bert,
        n_feat_embed=768, # PhoBERT base dim
        init_pre_train=True, # Skip immediate init, let Trainer handle it
        embed=embed # Pass embedding path for loading
    )
    
    # Load Vietnamese Treebank corpus
    corpus = ViVTBCorpus()
    
    click.echo(f"🚀 Training Biaffine Parser V2 (Enhanced Architecture)")
    click.echo(f"📊 Configuration:")
    click.echo(f"   Structure: Word (FastText) + BERT (Fine-tune) + POS Tag")
    click.echo(f"   BERT model: {bert}")
    if embed:
        click.echo(f"   Pretrained embeddings: {embed}")
    else:
        click.echo(f"   Pretrained embeddings: None (Random init - Recommended to use FastText!)")
        
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   LR (General): {lr}")
    click.echo(f"   LR (BERT): {bert_lr} (Fine-tuning)")
    click.echo(f"   Save path: {save_path}")
    click.echo()
    
    trainer = BiaffineTrainerV2(parser=parser, corpus=corpus)
    
    trainer.train(
        base_path=save_path,
        min_freq=min_freq,
        batch_size=batch_size,
        lr=lr,
        bert_lr=bert_lr,
        max_epochs=epochs,
        patience=patience
    )
    
    click.echo(f"\n✅ Training V2 complete! Model saved to: {save_path}")

@click.command('test-biaffine-v2')
@click.option('--model-path', '-m', required=True, help='Path to saved model checkpoint')
@click.option('--batch-size', default=3000, type=int, help='Batch size (in tokens)')
def test_biaffine_v2(model_path: str, batch_size: int):
    """Evaluate trained Enhanced Biaffine Parser V2 on Test set."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from models.graph_based.biaffine_parser_v2 import BiaffineParserV2
    from utils.sp_data import Dataset
    
    click.echo(f"📁 Loading model from: {model_path}")
    parser = BiaffineParserV2.load(model_path)
    
    click.echo("📊 Loading Test Corpus...")
    corpus = ViVTBCorpus()
    test = Dataset(parser.transform, corpus.test)
    test.build(batch_size, 1000)
    
    click.echo("🚀 Starting Evaluation...")
    loss, metric = parser.evaluate(test.loader)
    
    click.echo(f"\n✅ Test Result:")
    click.echo(f"   Loss: {loss:.4f}")
    click.echo(f"   UAS: {metric.uas:.2%}")
    click.echo(f"   LAS: {metric.las:.2%}")


@click.command('train-neural-parser')
@click.option('--epochs', default=10, type=int, help='Number of epochs')
@click.option('--batch-size', default=1000, type=int, help='Batch size')
@click.option('--lr', default=0.01, type=float, help='Learning rate')
@click.option('--hidden-dim', default=200, type=int, help='Hidden layer dimension')
@click.option('--word-embed-dim', default=50, type=int, help='Word embedding dimension')
@click.option('--save-path', '-s', default='checkpoints/neural_parser.pt',
              help='Path to save the model')
def train_neural_parser(epochs: int, batch_size: int, lr: float, 
                        hidden_dim: int, word_embed_dim: int, save_path: str):
    """Train Neural Transition-based Parser (Chen & Manning 2014).
    
    This parser uses:
    - Arc-Standard transition system (SHIFT, LEFT-ARC, RIGHT-ARC)
    - Feedforward neural network with cube activation
    - Word, POS tag, and arc label embeddings
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.neural_trainer import NeuralTransitionTrainer
    from models.transition_based.neural_parser import NeuralTransitionParser
    
    click.echo("🧠 Training Neural Transition Parser (Chen & Manning 2014)")
    click.echo(f"📊 Configuration:")
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   Learning rate: {lr}")
    click.echo(f"   Hidden dim: {hidden_dim}")
    click.echo(f"   Word embed dim: {word_embed_dim}")
    click.echo(f"   Save path: {save_path}")
    click.echo()
    
    corpus = ViVTBCorpus()
    parser = NeuralTransitionParser(
        n_words=1, n_tags=1, n_rels=1, n_actions=1  # Placeholders, set by trainer
    )
    
    trainer = NeuralTransitionTrainer(parser=parser, corpus=corpus)
    trainer.train(
        base_path=save_path,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        hidden_dim=hidden_dim,
        word_embed_dim=word_embed_dim
    )
    
    click.echo(f"\n✅ Training complete! Model saved to: {save_path}")


@click.command('train-biaffine-v3')
@click.option('--epochs', default=20, type=int, help='Number of epochs')
@click.option('--batch-size', default=3000, type=int, help='Batch size (in tokens)')
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--save-path', '-s', default='checkpoints/biaffine_model_v3.pt',
              help='Path to save the model')
def train_biaffine_v3(epochs: int, batch_size: int, lr: float, bert_lr: float, save_path: str):
    """Train Joint POS Tagger + Biaffine Parser V3.
    
    This is the most advanced architecture with:
    - Tagger branch: BERT → BiLSTM → MLP → POS Tags
    - Parser branch: (BERT ⊕ Tag) → BiLSTM → Biaffine → Parse Tree
    - Joint training with scheduled sampling
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.biaffine_trainer_v3 import BiaffineTrainerV3
    from models.graph_based.biaffine_parser_v3 import BiaffineParserV3
    
    click.echo("🧬 Training Joint Tagger + Biaffine Parser V3")
    click.echo(f"📊 Configuration:")
    click.echo(f"   Architecture: BERT → Tagger Branch + Parser Branch (Joint)")
    click.echo(f"   BERT model: vinai/phobert-base")
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   LR (General): {lr}")
    click.echo(f"   LR (BERT): {bert_lr}")
    click.echo(f"   Save path: {save_path}")
    click.echo()
    
    corpus = ViVTBCorpus()
    parser = BiaffineParserV3(
        n_tags=1, n_rels=1,  # Placeholders
        init_pre_train=True
    )
    parser.bert_name = 'vinai/phobert-base'
    
    trainer = BiaffineTrainerV3(parser=parser, corpus=corpus)
    trainer.train(
        base_path=save_path,
        max_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        bert_lr=bert_lr
    )
    
    click.echo(f"\n✅ Training V3 complete! Model saved to: {save_path}")



@click.command('train-triaffine')
@click.option('--epochs', default=20, type=int, help='Number of epochs')
@click.option('--batch-size', default=3000, type=int, help='Batch size (in tokens)')
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--embed', default=None, help='Path to pretrained embeddings (e.g. cc.vi.300.vec)')
@click.option('--save-path', '-s', default='checkpoints/triaffine_model.pt',
              help='Path to save the model')
def train_triaffine(epochs: int, batch_size: int, lr: float, bert_lr: float, embed: str, save_path: str):
    """Train Triaffine Parser (Second-Order with Sibling Attention).
    
    This is the most advanced architecture with:
    - Global context via BiLSTM
    - Sibling features via mlp_sibling + Triaffine scoring
    - BERT for contextual embeddings
    - Optional Word Embeddings (FastText) if --embed is provided
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.triaffine_trainer import TriaffineTrainer
    from models.graph_based.triaffine_parser import TriaffineParser
    
    click.echo("🔺 Training Triaffine Parser (Second-Order + Sibling Attention)")
    click.echo(f"📊 Configuration:")
    click.echo(f"   Architecture: BERT → BiLSTM → [MLP_head, MLP_dep, MLP_sib] → Triaffine")
    click.echo(f"   BERT model: vinai/phobert-base")
    click.echo(f"   Embeddings: {embed if embed else 'None (Pure BERT)'}")
    click.echo(f"   Epochs: {epochs}")
    click.echo(f"   Batch size: {batch_size}")
    click.echo(f"   LR (General): {lr}")
    click.echo(f"   LR (BERT): {bert_lr}")
    click.echo(f"   Save path: {save_path}")
    click.echo()
    
    corpus = ViVTBCorpus()
    parser = TriaffineParser(
        n_rels=1,
        init_pre_train=True
    )
    parser.bert_name = 'vinai/phobert-base'
    
    trainer = TriaffineTrainer(parser=parser, corpus=corpus)
    trainer.train(
        base_path=save_path,
        embed=embed,
        max_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        bert_lr=bert_lr
    )
    
    click.echo(f"\n✅ Training Triaffine complete! Model saved to: {save_path}")


cli.add_command(train_biaffine)
cli.add_command(train_biaffine_v2)
cli.add_command(test_biaffine_v2)
cli.add_command(train_biaffine_v3)
cli.add_command(train_triaffine)
cli.add_command(train_malt_parser)
cli.add_command(train_neural_parser)


# ============================================================================
# ABLATION STUDY COMMANDS
# ============================================================================

@click.command('train-ablation-multihead')
@click.option('--epochs', default=20, type=int)
@click.option('--batch-size', default=3000, type=int)
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--embed', default=None, help='Path to pretrained embeddings')
@click.option('--save-path', '-s', default='checkpoints/ablation_multihead.pt')
def train_ablation_multihead(epochs, batch_size, lr, bert_lr, embed, save_path):
    """Ablation: Multi-Head Biaffine (4 heads)"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.ablation_trainer import AblationTrainer
    from models.graph_based.ablation.triaffine_multihead import TriaffineMultiHead
    
    click.echo("🔬 Ablation Study: Multi-Head Biaffine")
    corpus = ViVTBCorpus()
    trainer = AblationTrainer(TriaffineMultiHead, corpus, 'MultiHead')
    trainer.train(base_path=save_path, embed=embed, max_epochs=epochs, batch_size=batch_size, lr=lr, bert_lr=bert_lr)
    click.echo(f"✅ Done! Model saved to: {save_path}")


@click.command('train-ablation-scalarmix')
@click.option('--epochs', default=20, type=int)
@click.option('--batch-size', default=3000, type=int)
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--embed', default=None, help='Path to pretrained embeddings')
@click.option('--save-path', '-s', default='checkpoints/ablation_scalarmix.pt')
def train_ablation_scalarmix(epochs, batch_size, lr, bert_lr, embed, save_path):
    """Ablation: Scalar Mix (all 12 BERT layers)"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.ablation_trainer import AblationTrainer
    from models.graph_based.ablation.triaffine_scalarmix import TriaffineScalarMix
    
    click.echo("🔬 Ablation Study: Scalar Mix (12 BERT layers)")
    corpus = ViVTBCorpus()
    trainer = AblationTrainer(TriaffineScalarMix, corpus, 'ScalarMix')
    trainer.train(base_path=save_path, embed=embed, max_epochs=epochs, batch_size=batch_size, lr=lr, bert_lr=bert_lr)
    click.echo(f"✅ Done! Model saved to: {save_path}")


@click.command('train-ablation-charlstm')
@click.option('--epochs', default=20, type=int)
@click.option('--batch-size', default=3000, type=int)
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--embed', default=None, help='Path to pretrained embeddings')
@click.option('--save-path', '-s', default='checkpoints/ablation_charlstm.pt')
def train_ablation_charlstm(epochs, batch_size, lr, bert_lr, embed, save_path):
    """Ablation: Character-level LSTM"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.ablation_trainer import AblationTrainer
    from models.graph_based.ablation.triaffine_charlstm import TriaffineCharLSTM
    
    click.echo("🔬 Ablation Study: Character LSTM")
    corpus = ViVTBCorpus()
    trainer = AblationTrainer(TriaffineCharLSTM, corpus, 'CharLSTM')
    trainer.train(base_path=save_path, embed=embed, max_epochs=epochs, batch_size=batch_size, lr=lr, bert_lr=bert_lr)
    click.echo(f"✅ Done! Model saved to: {save_path}")


@click.command('train-ablation-combined')
@click.option('--epochs', default=20, type=int)
@click.option('--batch-size', default=3000, type=int)
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--embed', default=None, help='Path to pretrained embeddings')
@click.option('--save-path', '-s', default='checkpoints/ablation_combined.pt')
def train_ablation_combined(epochs, batch_size, lr, bert_lr, embed, save_path):
    """Ablation: Multi-Head + Scalar Mix combined"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.ablation_trainer import AblationTrainer
    from models.graph_based.ablation.triaffine_multihead_scalarmix import TriaffineMultiHeadScalarMix
    
    click.echo("🔬 Ablation Study: Multi-Head + Scalar Mix")
    corpus = ViVTBCorpus()
    trainer = AblationTrainer(TriaffineMultiHeadScalarMix, corpus, 'MultiHead+ScalarMix')
    trainer.train(base_path=save_path, embed=embed, max_epochs=epochs, batch_size=batch_size, lr=lr, bert_lr=bert_lr)
    click.echo(f"✅ Done! Model saved to: {save_path}")


@click.command('train-ablation-noword')
@click.option('--epochs', default=20, type=int)
@click.option('--batch-size', default=3000, type=int)
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--save-path', '-s', default='checkpoints/ablation_noword.pt')
def train_ablation_noword(epochs, batch_size, lr, bert_lr, save_path):
    """Ablation: No Word Embedding (BERT + Tag only)"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.ablation_trainer import AblationTrainer
    from models.graph_based.ablation.triaffine_noword import TriaffineNoWord
    
    click.echo("🔬 Ablation Study: No Word Embedding (BERT + Tag only)")
    corpus = ViVTBCorpus()
    trainer = AblationTrainer(TriaffineNoWord, corpus, 'NoWord')
    trainer.train(base_path=save_path, max_epochs=epochs, batch_size=batch_size, lr=lr, bert_lr=bert_lr)
    click.echo(f"✅ Done! Model saved to: {save_path}")


@click.command('train-ablation-jointtagger')
@click.option('--epochs', default=20, type=int)
@click.option('--batch-size', default=3000, type=int)
@click.option('--lr', default=2e-3, type=float, help='Learning rate (general)')
@click.option('--bert-lr', default=5e-5, type=float, help='Learning rate (BERT)')
@click.option('--embed', default=None, help='Path to pretrained embeddings')
@click.option('--save-path', '-s', default='checkpoints/ablation_jointtagger.pt')
def train_ablation_jointtagger(epochs, batch_size, lr, bert_lr, embed, save_path):
    """Ablation: Joint POS Tagger (like V3)"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from datasets import ViVTBCorpus
    from trainers.jointtagger_trainer import JointTaggerTrainer
    
    click.echo("🔬 Ablation Study: Joint POS Tagger")
    corpus = ViVTBCorpus()
    trainer = JointTaggerTrainer(corpus)
    trainer.train(base_path=save_path, embed=embed, max_epochs=epochs, batch_size=batch_size, lr=lr, bert_lr=bert_lr)
    click.echo(f"✅ Done! Model saved to: {save_path}")


cli.add_command(train_ablation_multihead)
cli.add_command(train_ablation_scalarmix)
cli.add_command(train_ablation_charlstm)
cli.add_command(train_ablation_combined)
cli.add_command(train_ablation_noword)
cli.add_command(train_ablation_jointtagger)


@click.command('test-jointtagger')
@click.option('--model-path', '-m', required=True, help='Path to saved joint tagger model')
def test_jointtagger(model_path):
    """Test a saved Joint Tagger model on test set (no gold POS tags)"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import torch
    from datasets import ViVTBCorpus
    from models.graph_based.ablation.triaffine_jointtagger import TriaffineJointTagger
    from utils.sp_data import Dataset
    from utils.sp_metric import AttachmentMetric
    
    click.echo(f"📊 Loading model from: {model_path}")
    parser = TriaffineJointTagger.load(model_path)
    
    click.echo("📊 Loading test data...")
    corpus = ViVTBCorpus()
    test = Dataset(parser.transform, corpus.test)
    test.build(3000, 32)
    
    click.echo("🔬 Evaluating (without gold POS tags)...")
    parser.eval()
    
    tag_correct, tag_total = 0, 0
    metric = AttachmentMetric()
    
    with torch.no_grad():
        for batch in test.loader:
            words, feats, tags, arcs, rels = batch
            mask = words.ne(parser.pad_index)
            mask[:, 0] = 0
            
            # Forward WITHOUT gold tags
            s_tag, s_arc, s_rel = parser.forward(words, feats, use_gold_tags=False)
            
            # Tag accuracy
            tag_preds = s_tag.argmax(-1)
            tag_correct += (tag_preds[mask] == tags[mask]).sum().item()
            tag_total += mask.sum().item()
            
            # Arc/Rel
            arc_preds, rel_preds = parser.decode(s_arc, s_rel, mask)
            metric(arc_preds, rel_preds, arcs, rels, mask)
    
    tag_acc = tag_correct / tag_total if tag_total > 0 else 0
    click.echo(f"\n📊 Test Results (NO gold POS tags):")
    click.echo(f"   POS Tag Accuracy: {tag_acc:.2%}")
    click.echo(f"   {metric}")


cli.add_command(test_jointtagger)


def main():
    cli()


if __name__ == "__main__":
    main()

