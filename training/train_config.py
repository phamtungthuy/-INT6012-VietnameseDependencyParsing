from typing import Optional

from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Parser type: bilstm, transition
    parser_type: str = 'bilstm'
    
    # Model hyperparameters
    embedding_dim: int = 100
    pos_dim: int = 50
    hidden_dim: int = 400
    num_layers: int = 3
    arc_dim: int = 500
    label_dim: int = 100
    dropout: float = 0.33
    
    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 30
    lr: float = 2e-3
    weight_decay: float = 1e-4
    min_freq: int = 2
    seed: int = 42
    save_every: int = 5
    
    # Paths
    save_dir: str = 'checkpoints'
    results_dir: str = 'results'
    
    # Device & resume
    device: Optional[str] = None
    resume: Optional[str] = None
