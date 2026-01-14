"""
BERT-based Biaffine Parser with Joint POS Tagging

Kiến trúc chi tiết (theo Bhatt et al., 2024):
    - BERT encoder (PhoBERT cho tiếng Việt)
    - Tagger module: BiLSTM φ + MLP tag classifier + tag-to-dense MLP
    - Parser module: BiLSTM ψ × N layers + MLP_{e^h, e^d, r^h, r^d}
    - Biaffine scorer với score normalization (a = 1/√d)
    - Score decoder: Eisner's algorithm hoặc greedy decoding

Full architecture (theo diagram):

    input
      ↓
    BERT ──────────────────────⊕──→ BiLSTM ψ × N
      │                        ↑         ↓
      │                        │    MLP_{e^h, e^d, r^h, r^d}
      ↓                        │         ↓
 tagger BiLSTM φ              │    biaffine scorer ←── score norm (a=1/√d)
      ↓                        │         ↓
 MLP tag classifier           │    score decoder
      ↓                        │         ↓
    tags                       │    edges, edge labels
      ↓                        │
 tag-to-dense MLP ─────────────┘

References:
    - Bhatt et al. (2024): End-to-end Parsing of Procedural Text into Flow Graphs
      https://aclanthology.org/2024.lrec-main.517/
    - Dozat & Manning (2017): Deep Biaffine Attention for Neural Dependency Parsing
"""

from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural_attention.base_attention_parser import BaseAttentionParser
from models.neural_attention.bert_biaffine.modules import TaggerModule, ParserEncoder, EdgeScorer
from models.neural_attention.bert_biaffine.decoder import ScoreDecoder


class BertBiaffineParser(BaseAttentionParser):
    """
    BERT + Biaffine Parser với Joint POS Tagging
    
    Features:
    - Joint POS tagging và dependency parsing
    - Score normalization (1/√d) for stable training
    - Support cho nhiều decoding algorithms
    - Multi-task learning với tag prediction
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_pos_tags: int,
        num_labels: int,
        # BERT/Embedding settings
        embedding_dim: int = 768,
        use_pretrained_bert: bool = False,
        bert_model_name: str = "vinai/phobert-base",
        # Tagger settings
        tagger_hidden_dim: int = 256,
        tagger_num_layers: int = 1,
        tag_embedding_dim: int = 64,
        # Parser settings
        parser_hidden_dim: int = 512,
        parser_num_layers: int = 3,
        arc_dim: int = 512,
        label_dim: int = 128,
        # Training settings
        dropout: float = 0.33,
        use_score_norm: bool = True,
        tag_loss_weight: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_pos_tags = num_pos_tags
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.use_pretrained_bert = use_pretrained_bert
        self.tag_loss_weight = tag_loss_weight
        
        # Input embedding
        if use_pretrained_bert:
            try:
                from transformers import AutoModel
                self.bert = AutoModel.from_pretrained(bert_model_name)
                self.embedding_dim = self.bert.config.hidden_size
                embedding_dim = self.embedding_dim
            except ImportError:
                print("transformers library not found. Using standard embeddings.")
                self.bert = None
                self._init_standard_embeddings(vocab_size, embedding_dim)
        else:
            self.bert = None
            self._init_standard_embeddings(vocab_size, embedding_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Tagger module
        self.tagger = TaggerModule(
            input_dim=embedding_dim,
            hidden_dim=tagger_hidden_dim,
            num_layers=tagger_num_layers,
            num_tags=num_pos_tags,
            tag_embedding_dim=tag_embedding_dim,
            dropout=dropout
        )
        
        # Parser encoder: takes BERT output + tag embeddings
        parser_input_dim = embedding_dim + tag_embedding_dim
        self.parser_encoder = ParserEncoder(
            input_dim=parser_input_dim,
            hidden_dim=parser_hidden_dim,
            num_layers=parser_num_layers,
            dropout=dropout
        )
        
        # Edge scorer
        self.edge_scorer = EdgeScorer(
            input_dim=parser_hidden_dim * 2,  # bidirectional
            arc_dim=arc_dim,
            label_dim=label_dim,
            num_labels=num_labels,
            dropout=dropout,
            use_score_norm=use_score_norm
        )
    
    def _init_standard_embeddings(self, vocab_size: int, embedding_dim: int):
        """Initialize standard word embeddings when BERT is not used"""
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(self.num_pos_tags, 50, padding_idx=0)
        
        # Project embeddings to embedding_dim
        self.embed_proj = nn.Linear(embedding_dim + 50, embedding_dim)
    
    def _get_embeddings(
        self, 
        words: torch.Tensor, 
        pos_tags: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get input embeddings from BERT or standard embeddings"""
        if self.bert is not None:
            outputs = self.bert(words, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
        else:
            word_emb = self.word_embedding(words)
            if pos_tags is not None:
                pos_emb = self.pos_embedding(pos_tags)
                embeddings = self.embed_proj(torch.cat([word_emb, pos_emb], dim=-1))
            else:
                # Pad with zeros if no POS tags provided
                pos_emb = torch.zeros(*words.shape, 50, device=words.device)
                embeddings = self.embed_proj(torch.cat([word_emb, pos_emb], dim=-1))
        
        return self.dropout(embeddings)
    
    def forward(
        self,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_gold_tags: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            words: [batch, seq_len] - word indices or BERT input_ids
            pos_tags: [batch, seq_len] - POS tag indices (gold tags for training)
            lengths: [batch] - sequence lengths
            attention_mask: [batch, seq_len] - attention mask for BERT
            use_gold_tags: whether to use gold POS tags (True during training)
        
        Returns:
            dict with:
                - arc_scores: [batch, seq_len, seq_len]
                - label_scores: [batch, seq_len, seq_len, num_labels]
                - tag_logits: [batch, seq_len, num_pos_tags]
        """
        batch_size, seq_len = words.shape
        
        # 1. Get input embeddings (BERT or standard)
        embeddings = self._get_embeddings(words, pos_tags if not self.use_pretrained_bert else None, attention_mask)
        
        # 2. Tagger module: BiLSTM φ → MLP tag classifier → tag-to-dense MLP
        tag_logits, tag_embeddings, _ = self.tagger(
            embeddings, 
            lengths, 
            gold_tags=pos_tags,
            use_gold=use_gold_tags
        )
        
        # 3. Concatenate BERT output with tag embeddings
        parser_input = torch.cat([embeddings, tag_embeddings], dim=-1)
        
        # 4. Parser encoder: BiLSTM ψ × N
        encoder_output = self.parser_encoder(parser_input, lengths)
        
        # 5. Edge scorer: MLPs + Biaffine attention
        arc_scores, label_scores = self.edge_scorer(encoder_output)
        
        return {
            'arc_scores': arc_scores,
            'label_scores': label_scores,
            'tag_logits': tag_logits
        }
    
    def loss(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        tag_logits: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        pos_tags: torch.Tensor,
        lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-task loss
        
        Args:
            arc_scores: [batch, seq_len, seq_len]
            label_scores: [batch, seq_len, seq_len, num_labels]
            tag_logits: [batch, seq_len, num_pos_tags]
            heads: [batch, seq_len] - gold head indices
            rels: [batch, seq_len] - gold relation indices
            pos_tags: [batch, seq_len] - gold POS tags
            lengths: [batch]
        
        Returns:
            total_loss: scalar tensor
            loss_dict: dictionary with individual losses
        """
        batch_size, seq_len = heads.shape
        device = heads.device
        
        # Create mask for valid positions (exclude padding)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            mask[i, 1:length] = True  # Exclude ROOT (position 0) from loss
        
        num_valid = mask.sum().float()
        
        # 1. Arc loss (cross-entropy)
        arc_loss = F.cross_entropy(
            arc_scores[mask],
            heads[mask],
            reduction='sum'
        ) / num_valid
        
        # 2. Label loss (using gold heads)
        # Get label scores for gold head positions
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        seq_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        selected_label_scores = label_scores[batch_idx, seq_idx, heads]  # [batch, seq_len, num_labels]
        
        label_loss = F.cross_entropy(
            selected_label_scores[mask],
            rels[mask],
            reduction='sum'
        ) / num_valid
        
        # 3. Tag loss (POS tagging)
        tag_loss = F.cross_entropy(
            tag_logits[mask],
            pos_tags[mask],
            reduction='sum'
        ) / num_valid
        
        # Total loss with weighting
        total_loss = arc_loss + label_loss + self.tag_loss_weight * tag_loss
        
        loss_dict = {
            'arc_loss': arc_loss,
            'label_loss': label_loss,
            'tag_loss': tag_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def decode(
        self,
        arc_scores: torch.Tensor,
        label_scores: torch.Tensor,
        lengths: torch.Tensor,
        algorithm: str = 'greedy'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode predictions from scores
        
        Args:
            arc_scores: [batch, seq_len, seq_len]
            label_scores: [batch, seq_len, seq_len, num_labels]
            lengths: [batch]
            algorithm: 'greedy' or 'eisner'
        
        Returns:
            pred_heads: [batch, seq_len]
            pred_rels: [batch, seq_len]
        """
        batch_size, seq_len, _ = arc_scores.shape
        device = arc_scores.device
        
        # Decode heads
        if algorithm == 'eisner':
            pred_heads = ScoreDecoder.eisner_decode(arc_scores, lengths)
        else:
            pred_heads = ScoreDecoder.greedy_decode(arc_scores, lengths)
        
        # Decode labels based on predicted heads
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
        seq_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        selected_label_scores = label_scores[batch_idx, seq_idx, pred_heads]
        pred_rels = selected_label_scores.argmax(dim=-1)
        
        # Mask padding
        for i, length in enumerate(lengths):
            pred_heads[i, length:] = 0
            pred_rels[i, length:] = 0
        
        return pred_heads, pred_rels
    
    def predict(
        self,
        words: torch.Tensor,
        pos_tags: torch.Tensor,
        lengths: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decode_algorithm: str = 'greedy'
    ) -> Dict[str, torch.Tensor]:
        """
        Full prediction pipeline
        
        Args:
            words: [batch, seq_len]
            pos_tags: [batch, seq_len]
            lengths: [batch]
            attention_mask: [batch, seq_len]
            decode_algorithm: 'greedy' or 'eisner'
        
        Returns:
            dict with predicted heads, relations, and tags
        """
        # Forward pass without using gold tags
        outputs = self.forward(
            words, pos_tags, lengths, 
            attention_mask=attention_mask,
            use_gold_tags=False
        )
        
        # Decode
        pred_heads, pred_rels = self.decode(
            outputs['arc_scores'],
            outputs['label_scores'],
            lengths,
            algorithm=decode_algorithm
        )
        
        # Predict tags
        pred_tags = outputs['tag_logits'].argmax(dim=-1)
        
        return {
            'heads': pred_heads,
            'rels': pred_rels,
            'tags': pred_tags,
            'arc_scores': outputs['arc_scores'],
            'label_scores': outputs['label_scores'],
            'tag_logits': outputs['tag_logits']
        }


def create_bert_biaffine_parser(
    vocab_size: int,
    num_pos_tags: int = 50,
    num_labels: int = 50,
    use_bert: bool = False,
    **kwargs
) -> BertBiaffineParser:
    """
    Factory function to create BertBiaffineParser
    
    Args:
        vocab_size: vocabulary size
        num_pos_tags: number of POS tags
        num_labels: number of dependency relation labels
        use_bert: whether to use pretrained BERT
        **kwargs: additional arguments
    
    Returns:
        BertBiaffineParser instance
    """
    default_config = {
        'embedding_dim': 768 if use_bert else 256,
        'use_pretrained_bert': use_bert,
        'tagger_hidden_dim': 256,
        'tagger_num_layers': 1,
        'tag_embedding_dim': 64,
        'parser_hidden_dim': 512,
        'parser_num_layers': 3,
        'arc_dim': 512,
        'label_dim': 128,
        'dropout': 0.33,
        'use_score_norm': True,
        'tag_loss_weight': 0.1
    }
    default_config.update(kwargs)
    
    return BertBiaffineParser(
        vocab_size=vocab_size,
        num_pos_tags=num_pos_tags,
        num_labels=num_labels,
        **default_config
    )


if __name__ == "__main__":
    # Test the parser
    torch.manual_seed(42)
    
    # Create parser
    parser = create_bert_biaffine_parser(
        vocab_size=10000,
        num_pos_tags=20,
        num_labels=40,
        use_bert=False
    )
    
    print(f"Model parameters: {sum(p.numel() for p in parser.parameters()):,}")
    
    # Test forward pass
    batch_size, seq_len = 4, 20
    words = torch.randint(1, 10000, (batch_size, seq_len))
    pos_tags = torch.randint(1, 20, (batch_size, seq_len))
    lengths = torch.tensor([20, 15, 18, 12])
    heads = torch.randint(0, seq_len, (batch_size, seq_len))
    rels = torch.randint(1, 40, (batch_size, seq_len))
    
    # Forward
    outputs = parser(words, pos_tags, lengths)
    print(f"Arc scores shape: {outputs['arc_scores'].shape}")
    print(f"Label scores shape: {outputs['label_scores'].shape}")
    print(f"Tag logits shape: {outputs['tag_logits'].shape}")
    
    # Loss
    total_loss, loss_dict = parser.loss(
        outputs['arc_scores'],
        outputs['label_scores'],
        outputs['tag_logits'],
        heads, rels, pos_tags, lengths
    )
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {', '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())}")
    
    # Decode
    pred_heads, pred_rels = parser.decode(
        outputs['arc_scores'],
        outputs['label_scores'],
        lengths,
        algorithm='greedy'
    )
    print(f"Predicted heads shape: {pred_heads.shape}")
    print(f"Predicted rels shape: {pred_rels.shape}")
    
    print("\nAll tests passed!")
