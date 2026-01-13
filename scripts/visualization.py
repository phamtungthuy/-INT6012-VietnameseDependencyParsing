"""
Vietnamese Dependency Parsing - Interactive Visualization
A professional web-based visualization tool for exploring dependency trees.
"""

from pathlib import Path
from flask import Flask, render_template, jsonify

from data_processing.loader import CoNLLUDataset
from utils.constants import (
    TRAIN_FILE_PATH, VALIDATION_FILE_PATH, 
    TEST_FILE_PATH, ROOT_PATH, TEMPLATES_PATH
)

DATASETS = {
    "train": TRAIN_FILE_PATH,
    "dev": VALIDATION_FILE_PATH,
    "test": TEST_FILE_PATH,
}

# Cache loaded datasets
_dataset_cache: dict[str, CoNLLUDataset] = {}

def get_dataset(name: str) -> CoNLLUDataset:
    if name not in _dataset_cache:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        _dataset_cache[name] = CoNLLUDataset(DATASETS[name])
    return _dataset_cache[name]


app = Flask(__name__, template_folder=str(TEMPLATES_PATH))


@app.route('/')
def index():
    """Serve main visualization page"""
    return render_template('visualization.html')


@app.route('/api/dataset/<name>/info')
def dataset_info(name: str):
    """Get dataset information"""
    try:
        dataset = get_dataset(name)
        return jsonify({
            'name': name,
            'total': len(dataset.sentences)
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/dataset/<name>/sentence/<int:index>')
def get_sentence(name: str, index: int):
    """Get a specific sentence from dataset"""
    try:
        dataset = get_dataset(name)
        
        if index < 0 or index >= len(dataset.sentences):
            return jsonify({'error': 'Index out of range'}), 404
        
        sentence = dataset.sentences[index]
        
        return jsonify({
            'index': index,
            'tokens': sentence
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

def run_visualization(host: str = '127.0.0.1', port: int = 5000, debug: bool = True):
    """Start the visualization server"""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🌳 Vietnamese Dependency Tree Visualizer                       ║
║                                                                  ║
║   Server running at: http://{host}:{port}                        ║
║                                                                  ║
║   Keyboard shortcuts:                                            ║
║   ← →     Navigate between sentences                             ║
║   G       Go to specific sentence                                ║
║   ESC     Close dialogs                                          ║
║                                                                  ║
║   Press Ctrl+C to stop the server                                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_visualization()
