import os
import torch
from pathlib import Path

import yaml


def get_root_path():
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ROOT_PATH = get_root_path()

# Data paths
DATA_PATH = ROOT_PATH / "datasets"
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

PRETRAINED = {
    'vi-dp-v1': 'https://github.com/undertheseanlp/underthesea/releases/download/resources/vi-dp-v1.zip',
    'vi-dp-v1a0': 'https://github.com/undertheseanlp/underthesea/releases/download/resources/vi-dp-v1a0.zip',
    'vi-dp-v1a1': 'https://github.com/undertheseanlp/underthesea/releases/download/resources/vi-dp-v1a1.zip',
    'biaffine-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.char.zip',
    'biaffine-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.biaffine.dependency.char.zip',
    'biaffine-dep-bert-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.biaffine.dependency.bert.zip',
    'biaffine-dep-bert-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.biaffine.dependency.bert.zip',
    'crfnp-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crfnp.dependency.char.zip',
    'crfnp-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crfnp.dependency.char.zip',
    'crf-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.dependency.char.zip',
    'crf-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf.dependency.char.zip',
    'crf2o-dep-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf2o.dependency.char.zip',
    'crf2o-dep-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf2o.dependency.char.zip',
    'crf-con-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.constituency.char.zip',
    'crf-con-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf.constituency.char.zip',
    'crf-con-bert-en': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ptb.crf.constituency.bert.zip',
    'crf-con-bert-zh': 'https://github.com/yzhangcs/parser/releases/download/v1.0.0/ctb7.crf.constituency.bert.zip'
}

pad = '<pad>'
unk = '<unk>'
bos = '<bos>'
eos = '<eos>'
