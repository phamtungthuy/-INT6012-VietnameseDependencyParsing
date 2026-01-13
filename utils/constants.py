import os
import torch
from pathlib import Path

import yaml


def get_root_path():
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT_PATH = get_root_path()

# Data paths
DATA_PATH = ROOT_PATH / "data"
VALIDATION_FILE_PATH = DATA_PATH / "vi_vtb-ud-dev.conllu"
TEST_FILE_PATH = DATA_PATH / "vi_vtb-ud-test.conllu"
TRAIN_FILE_PATH = DATA_PATH / "vi_vtb-ud-train.conllu"

# Template paths
TEMPLATES_PATH = ROOT_PATH / "templates"

# Config path
CONFIG_PATH = ROOT_PATH / "configs"
CONFIG_FILE = CONFIG_PATH / "config.yaml"

# Load config
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Results path
RESULTS_PATH = ROOT_PATH / config['paths']['results_dir']
CHECKPOINTS_PATH = ROOT_PATH / config['paths']['save_dir']

def get_device():
    if config['device'] is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return config['device']