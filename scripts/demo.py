import torch
from pathlib import Path
from flask import Flask, render_template, jsonify, request

from data_processing.loader import CoNLLUDataset
from data_processing import Vocabulary
from models import BiLSTMParser

from utils.constants import (
    TRAIN_FILE_PATH,
    VALIDATION_FILE_PATH, 
    TEST_FILE_PATH,
    config,
    TEMPLATES_PATH,
    get_device,
)
from utils.logs import logger

DATASETS = {
    "train": TRAIN_FILE_PATH,
    "dev": VALIDATION_FILE_PATH,
    "test": TEST_FILE_PATH,
}

# Cache
_dataset_cache: dict[str, CoNLLUDataset] = {}
_model = None
_vocab = None
_device = "cpu"


def get_dataset(name: str) -> CoNLLUDataset:
    if name not in _dataset_cache:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        _dataset_cache[name] = CoNLLUDataset(DATASETS[name])
    return _dataset_cache[name]


def load_model(model_path: str, vocab_path: Path, device: str):
    global _model, _vocab
    
    if _model is not None:
        return _model, _vocab
    
    # Load vocabulary
    vocab_data = torch.load(vocab_path, map_location=device)
    _vocab = Vocabulary()
    _vocab.word2idx = vocab_data['word2idx']
    _vocab.idx2word = vocab_data['idx2word']
    _vocab.pos2idx = vocab_data['pos2idx']
    _vocab.idx2pos = vocab_data['idx2pos']
    _vocab.rel2idx = vocab_data['rel2idx']
    _vocab.idx2rel = vocab_data['idx2rel']
    
    # Create model
    model_config = config['model']
    _model = BiLSTMParser(
        vocab_size=len(_vocab.word2idx),
        pos_size=len(_vocab.pos2idx),
        embedding_dim=model_config['embedding_dim'],
        pos_dim=model_config['pos_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        arc_dim=model_config['arc_dim'],
        label_dim=model_config['label_dim'],
        num_labels=len(_vocab.rel2idx),
        dropout=model_config['dropout']
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.to(device)
    _model.eval()
    
    return _model, _vocab


def parse_sentence(words: list, pos_tags: list, model, vocab, device: str):
    word_ids = [vocab.word2idx['<ROOT>']]
    pos_ids = [vocab.pos2idx['<ROOT>']]
    
    for word, pos in zip(words, pos_tags):
        word_ids.append(vocab.word2idx.get(word.lower(), vocab.word2idx['<UNK>']))
        pos_ids.append(vocab.pos2idx.get(pos, vocab.pos2idx['<UNK>']))
    
    # Convert to tensors
    word_tensor = torch.LongTensor(word_ids).unsqueeze(0).to(device)
    pos_tensor = torch.LongTensor(pos_ids).unsqueeze(0).to(device)
    length = torch.LongTensor([len(word_ids)]).to(device)
    
    # Forward
    with torch.no_grad():
        arc_scores, label_scores = model(word_tensor, pos_tensor, length)
        pred_heads, pred_rels = model.decode(arc_scores, label_scores, length)
    
    # Decode (skip ROOT token at index 0)
    pred_heads = pred_heads[0].cpu().numpy()[1:].tolist()
    pred_rels = pred_rels[0].cpu().numpy()[1:].tolist()
    
    relations = [vocab.idx2rel.get(rel, '<UNK>') for rel in pred_rels]
    
    return pred_heads, relations


app = Flask(__name__, template_folder=str(TEMPLATES_PATH))


@app.route('/')
def index():
    return render_template('demo.html')


@app.route('/api/dataset/<name>/info')
def dataset_info(name: str):
    try:
        dataset = get_dataset(name)
        return jsonify({'name': name, 'total': len(dataset.sentences)})
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/compare/<dataset_name>/<int:index>')
def compare_sentence(dataset_name: str, index: int):
    """Get ground truth and prediction for a sentence"""
    global _model, _vocab, _device
    
    try:
        dataset = get_dataset(dataset_name)
        
        if index < 0 or index >= len(dataset.sentences):
            return jsonify({'error': 'Index out of range'}), 404
        
        sentence = dataset.sentences[index]
        
        # Extract words and POS
        words = [t['form'] for t in sentence]
        pos_tags = [t['upos'] for t in sentence]
        
        # Get predictions
        pred_heads, pred_rels = parse_sentence(words, pos_tags, _model, _vocab, _device)
        
        return jsonify({
            'index': index,
            'tokens': sentence,
            'pred_heads': pred_heads,
            'pred_rels': pred_rels,
            'hasGold': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/parse', methods=['POST'])
def parse_custom():
    """Parse a custom sentence"""
    global _model, _vocab, _device
    
    data = request.json
    words = data.get('words', [])
    pos_tags = data.get('pos_tags', [])
    
    if not words:
        return jsonify({'error': 'No words provided'}), 400
    
    pred_heads, pred_rels = parse_sentence(words, pos_tags, _model, _vocab, _device)
    
    tokens = [
        {'id': i+1, 'form': w, 'upos': pos_tags[i] if i < len(pos_tags) else 'X', 
         'head': 0, 'deprel': '_'}
        for i, w in enumerate(words)
    ]
    
    return jsonify({
        'tokens': tokens,
        'pred_heads': pred_heads,
        'pred_rels': pred_rels,
        'hasGold': False
    })

def run_demo(
    host: str = '127.0.0.1',
    port: int = 5001,
    debug: bool = True,
    model_path: str = 'checkpoints/best_model.pt',
    vocab_path: Path = Path()
):
    global _device, _model, _vocab
    
    _device = get_device()
    
    logger.info(f"Loading model from {model_path}...")
    _model, _vocab = load_model(model_path, vocab_path, _device)
    logger.info(f"Model loaded! Device: {_device}")
    
    logger.info(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🔍 Vietnamese Dependency Parser - Demo                         ║
║                                                                  ║
║   Server running at: http://{host}:{port}                        ║
║                                                                  ║
║   Features:                                                      ║
║   • Compare ground truth vs predictions                          ║
║   • Parse custom sentences                                       ║
║   • Token-by-token accuracy analysis                             ║
║                                                                  ║
║   Press Ctrl+C to stop                                           ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_demo()
