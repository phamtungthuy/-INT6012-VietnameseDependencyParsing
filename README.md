# Vietnamese Dependency Parsing

BiLSTM + Biaffine Attention model for Vietnamese Dependency Parsing.

## 📁 Project Structure

```
vietnamese_dependency_parsing/
├── configs/                 # Training configuration
│   └── config.yaml
├── data/                    # CoNLL-U data files
│   ├── vi_vtb-ud-train.conllu
│   ├── vi_vtb-ud-dev.conllu
│   └── vi_vtb-ud-test.conllu
├── data_processing/         # Data processing modules
│   ├── loader.py           # CoNLL-U file loader
│   ├── vocabulary.py       # Vocabulary builder
│   ├── dependency.py       # PyTorch Dataset
│   └── analyzer.py         # Data analysis tools
├── models/                  # Model definitions
│   ├── attention/          # Attention layers
│   │   ├── base_attention.py
│   │   └── biaffine_attention.py
│   └── parser/             # Parser models
│       ├── base_parser.py
│       └── bilstm_parser.py
├── training/               # Training logic
│   ├── trainer.py
│   └── train_config.py
├── evaluation/             # Evaluation metrics
│   └── evaluator.py
├── scripts/                # CLI scripts
│   ├── cli.py             # CLI entry points
│   ├── train.py           # Training script
│   ├── visualization.py   # Visualization server
│   └── demo.py            # Demo server
├── templates/              # HTML templates
│   ├── visualization.html
│   └── demo.html
├── checkpoints/            # Saved model checkpoints
├── results/                # Training results
└── logs/                   # Log files
```

## 🚀 Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Package

```bash
# Install in editable mode
pip install -e .
```

This will:
- Install all dependencies (torch, numpy, tqdm, flask, ...)
- Register CLI commands: `train`, `visualize`, `demo`, `analyze`

## 💻 Usage

### Data Analysis

Analyze dataset to find optimal `min_freq` value:

```bash
analyze
```

Output:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MIN_FREQ ANALYSIS REPORT                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 Dataset Statistics:
   Train sentences: 1,400
   Dev sentences:   1,123
   Test sentences:  800

🔍 Min_freq Analysis:
   ┌─────────┬────────────┬───────────┬──────────┬────────────┐
   │ min_freq│ Vocab Size │ Coverage  │ OOV Rate │ Unique OOV │
   ├─────────┼────────────┼───────────┼──────────┼────────────┤
   │       1 │      3,401 │    100.0% │    15.7% │      3,191 │
   │       2 │      1,649 │     91.3% │    21.0% │      3,884 │
   ...
```

### Training

```bash
# Train with default config from configs/config.yaml
train

# Resume from checkpoint
train --resume checkpoints/checkpoint_epoch_10.pt
```

Training configuration in `configs/config.yaml`:

```yaml
model:
  embedding_dim: 100
  pos_dim: 50
  hidden_dim: 400
  num_layers: 3
  arc_dim: 500
  label_dim: 100
  dropout: 0.33

training:
  batch_size: 32
  num_epochs: 30
  lr: 0.002
  weight_decay: 0.0001
  min_freq: 2
  seed: 42
  save_every: 5

paths:
  save_dir: checkpoints
  results_dir: results

device: cuda  # or cpu
```

### Visualization

Launch web server to visualize dependency trees:

```bash
# Default: http://127.0.0.1:5000
visualize

# Custom host/port
visualize --host 0.0.0.0 --port 8000
```

Features:
- Select dataset (train/dev/test)
- Navigate through sentences
- Jump to specific sentence
- Interactive SVG dependency tree

### Demo

Compare ground truth vs model predictions:

```bash
# Default: http://127.0.0.1:5001
demo

# Custom model checkpoint
demo --model checkpoints/best_model.pt

# Custom host/port
demo --host 0.0.0.0 --port 8001
```

Features:
- Side-by-side ground truth and prediction comparison
- Error highlighting (incorrect arcs in red)
- UAS/LAS statistics
- Custom sentence parsing

## 📊 Results

Training results are saved in `results/run_YYYYMMDD_HHMMSS/`:

```
results/run_20260113_001411/
├── config.yaml      # Configuration used
├── config.json      # Configuration (JSON format)
├── vocab.pt         # Built vocabulary
├── history.json     # Training history (loss, accuracy per epoch)
└── results.json     # Final results (UAS, LAS)
```

## 📈 Metrics

- **UAS (Unlabeled Attachment Score)**: Percentage of tokens with correct head prediction
- **LAS (Labeled Attachment Score)**: Percentage of tokens with correct head and relation prediction

## 🔧 Dependencies

- Python >= 3.10
- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- tqdm >= 4.50.0
- conllu == 6.0.0
- loguru == 0.7.3
- PyYAML == 6.0.3
- Flask >= 3.0.0

## 📚 References

1. Dozat, T., & Manning, C. D. (2017). **Deep Biaffine Attention for Neural Dependency Parsing**. *ICLR 2017*.
2. Nguyen, P. T., Vu, X. L., Nguyen, T. M. H., Nguyen, V. H., & Le, H. P. (2009). **Building a Large Syntactically-Annotated Corpus of Vietnamese**. *Proceedings of the Third Linguistic Annotation Workshop (LAW III)*, pages 182-185. [[ACL Anthology]](https://aclanthology.org/W09-3035/)

## 🙏 Acknowledgements

This project is built upon the following works:

### Core Architecture
- **Biaffine Attention**: Dozat & Manning (2017) - "Deep Biaffine Attention for Neural Dependency Parsing"

### Data
- **Vietnamese Treebank (VTB)**: Nguyen et al. (2009) - "Building a Large Syntactically-Annotated Corpus of Vietnamese"

### Other References
<!-- Add more references as needed -->
<!--
- Author et al. (Year). "Paper Title". Conference/Journal.
-->

## 📝 License

MIT License
