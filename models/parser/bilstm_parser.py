import torch
import torch.nn as nn
import torch.nn.functional as F

from models.parser.base_parser import BaseParser
from models import BiaffineAttention

class BiLSTMParser(BaseParser):
    """BiLSTM + Biaffine Attention Parser"""
    def __init__(self, vocab_size, pos_size, embedding_dim=100, pos_dim=50,
                 hidden_dim=400, num_layers=3, arc_dim=500, label_dim=100,
                 num_labels=50, dropout=0.33, use_score_norm=True):
        super(BiLSTMParser, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        # Embeddings
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(pos_size, pos_dim, padding_idx=0)
        
        # BiLSTM encoder
        self.lstm = nn.LSTM(
            input_size=embedding_dim + pos_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # MLP cho arc (head prediction)
        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.arc_head_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, arc_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.arc_dep_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, arc_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # MLP cho label prediction
        self.label_head_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, label_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        self.label_dep_mlp = nn.Sequential(
            nn.Linear(lstm_output_dim, label_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )
        
        # Biaffine attention layers (with optional score normalization 1/√d)
        self.arc_attention = BiaffineAttention(arc_dim, 1, bias=(True, False), scale=use_score_norm)
        self.label_attention = BiaffineAttention(label_dim, num_labels, bias=(True, True), scale=use_score_norm)
        
    def forward(self, words, pos_tags, lengths):
        """
        words: [batch, seq_len]
        pos_tags: [batch, seq_len]
        lengths: [batch]
        """
        batch_size, seq_len = words.shape
        
        # Embeddings
        word_emb = self.word_embedding(words)  # [batch, seq_len, emb_dim]
        pos_emb = self.pos_embedding(pos_tags)  # [batch, seq_len, pos_dim]
        
        # Concatenate embeddings
        emb = torch.cat([word_emb, pos_emb], dim=-1)  # [batch, seq_len, emb_dim + pos_dim]
        emb = self.dropout(emb)
        
        # Pack sequence
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM
        packed_output, _ = self.lstm(packed_emb)
        
        # Unpack sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )  # [batch, seq_len, hidden_dim * 2]
        
        lstm_out = self.dropout(lstm_out)
        
        # Arc prediction
        arc_head = self.arc_head_mlp(lstm_out)  # [batch, seq_len, arc_dim]
        arc_dep = self.arc_dep_mlp(lstm_out)    # [batch, seq_len, arc_dim]
        
        # Arc scores: [batch, seq_len, seq_len, 1]
        arc_scores = self.arc_attention(arc_dep, arc_head)
        arc_scores = arc_scores.squeeze(-1)  # [batch, seq_len, seq_len]
        
        # Label prediction
        label_head = self.label_head_mlp(lstm_out)  # [batch, seq_len, label_dim]
        label_dep = self.label_dep_mlp(lstm_out)    # [batch, seq_len, label_dim]
        
        # Label scores: [batch, seq_len, seq_len, num_labels]
        label_scores = self.label_attention(label_dep, label_head)
        
        return arc_scores, label_scores
    
    def loss(self, arc_scores, label_scores, heads, rels, lengths):
        """
        Tính cross entropy loss cho arc và label prediction
        """
        batch_size, seq_len = heads.shape
        
        # Create mask
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=heads.device)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        
        # Arc loss
        arc_scores = arc_scores.masked_fill(~mask.unsqueeze(1), -1e9)
        arc_loss = F.cross_entropy(
            arc_scores.reshape(batch_size * seq_len, seq_len),
            heads.reshape(-1),
            ignore_index=0,
            reduction='sum'
        ) / mask.sum()
        
        # Label loss - sử dụng gold heads
        batch_indices = torch.arange(batch_size, device=label_scores.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=label_scores.device).unsqueeze(0).expand(batch_size, -1)
        
        # [batch, seq_len, num_labels]
        selected_label_scores = label_scores[batch_indices, seq_indices, heads]
        
        label_loss = F.cross_entropy(
            selected_label_scores.reshape(batch_size * seq_len, -1),
            rels.reshape(-1),
            ignore_index=0,
            reduction='sum'
        ) / mask.sum()
        
        return arc_loss, label_loss
    
    def decode(self, arc_scores, label_scores, lengths):
        """
        Greedy decoding
        arc_scores: [batch, seq_len, seq_len]
        label_scores: [batch, seq_len, seq_len, num_labels]
        """
        batch_size, seq_len, _ = arc_scores.shape
        
        # Predict heads (argmax)
        pred_heads = arc_scores.argmax(dim=-1)  # [batch, seq_len]
        
        # Predict labels based on predicted heads
        batch_indices = torch.arange(batch_size, device=label_scores.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=label_scores.device).unsqueeze(0).expand(batch_size, -1)
        
        selected_label_scores = label_scores[batch_indices, seq_indices, pred_heads]
        pred_rels = selected_label_scores.argmax(dim=-1)  # [batch, seq_len]
        
        # Mask padding
        for i, length in enumerate(lengths):
            pred_heads[i, length:] = 0
            pred_rels[i, length:] = 0
        
        return pred_heads, pred_rels
