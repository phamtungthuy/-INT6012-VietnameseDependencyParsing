"""
Score Decoder cho Dependency Parsing

Supports:
    - Greedy decoding (fast but may produce cycles)
    - Eisner's algorithm (O(n³), projective trees)
    - Chu-Liu/Edmonds algorithm (O(n²), non-projective trees) [TODO]

Reference:
    - Eisner (1996): Three New Probabilistic Models for Dependency Parsing
    - Dozat & Manning (2017): Deep Biaffine Attention for Neural Dependency Parsing
"""

import torch


class ScoreDecoder:
    """
    Score Decoder: converts scores to dependency tree
    """
    
    @staticmethod
    def greedy_decode(arc_scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Greedy decoding: for each token, pick the highest scoring head
        
        Args:
            arc_scores: [batch, seq_len, seq_len]
            lengths: [batch]
        
        Returns:
            heads: [batch, seq_len]
        """
        heads = arc_scores.argmax(dim=-1)
        
        # Mask padding positions
        batch_size, seq_len = heads.shape
        for i, length in enumerate(lengths):
            heads[i, length:] = 0
            # ROOT token (position 0) should have head 0
            heads[i, 0] = 0
        
        return heads
    
    @staticmethod
    def eisner_decode(arc_scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Eisner's algorithm for projective dependency parsing
        
        Time complexity: O(n³)
        
        Args:
            arc_scores: [batch, seq_len, seq_len]
            lengths: [batch]
        
        Returns:
            heads: [batch, seq_len]
        """
        batch_size, seq_len, _ = arc_scores.shape
        device = arc_scores.device
        
        heads = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            n = lengths[b].item()
            scores = arc_scores[b, :n, :n].cpu().numpy()
            
            # Initialize DP tables
            # complete[i][j][d]: best complete span from i to j, d=0 left, d=1 right
            # incomplete[i][j][d]: best incomplete span
            complete = [[[-float('inf'), -float('inf')] for _ in range(n)] for _ in range(n)]
            incomplete = [[[-float('inf'), -float('inf')] for _ in range(n)] for _ in range(n)]
            complete_bp = [[[None, None] for _ in range(n)] for _ in range(n)]
            incomplete_bp = [[[None, None] for _ in range(n)] for _ in range(n)]
            
            # Base case: single words
            for i in range(n):
                complete[i][i][0] = 0
                complete[i][i][1] = 0
            
            # Fill DP table
            for width in range(1, n):
                for i in range(n - width):
                    j = i + width
                    
                    # Incomplete spans (creating new arc)
                    for r in range(i, j):
                        # Left arc: j -> i
                        score = complete[i][r][1] + complete[r+1][j][0] + scores[i, j]
                        if score > incomplete[i][j][0]:
                            incomplete[i][j][0] = score
                            incomplete_bp[i][j][0] = (r, j)  # head is j
                        
                        # Right arc: i -> j
                        score = complete[i][r][1] + complete[r+1][j][0] + scores[j, i]
                        if score > incomplete[i][j][1]:
                            incomplete[i][j][1] = score
                            incomplete_bp[i][j][1] = (r, i)  # head is i
                    
                    # Complete spans
                    for r in range(i, j):
                        # Left complete
                        score = complete[i][r][0] + incomplete[r][j][0]
                        if score > complete[i][j][0]:
                            complete[i][j][0] = score
                            complete_bp[i][j][0] = r
                    
                    for r in range(i + 1, j + 1):
                        # Right complete
                        score = incomplete[i][r][1] + complete[r][j][1]
                        if score > complete[i][j][1]:
                            complete[i][j][1] = score
                            complete_bp[i][j][1] = r
            
            # Backtrack to find heads
            def backtrack_complete(i, j, d):
                if i == j:
                    return
                r = complete_bp[i][j][d]
                if r is None:
                    return
                if d == 0:
                    backtrack_complete(i, r, 0)
                    backtrack_incomplete(r, j, 0)
                else:
                    backtrack_incomplete(i, r, 1)
                    backtrack_complete(r, j, 1)
            
            def backtrack_incomplete(i, j, d):
                r, head = incomplete_bp[i][j][d]
                if d == 0:
                    heads[b, i] = head
                    backtrack_complete(i, r, 1)
                    backtrack_complete(r+1, j, 0)
                else:
                    heads[b, j] = head
                    backtrack_complete(i, r, 1)
                    backtrack_complete(r+1, j, 0)
            
            backtrack_complete(0, n-1, 1)
            heads[b, 0] = 0  # ROOT
        
        return heads
    
    @staticmethod
    def mst_decode(arc_scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Chu-Liu/Edmonds algorithm for non-projective dependency parsing
        
        Time complexity: O(n²)
        
        TODO: Implement this algorithm
        
        Args:
            arc_scores: [batch, seq_len, seq_len]
            lengths: [batch]
        
        Returns:
            heads: [batch, seq_len]
        """
        # For now, fall back to greedy decoding
        return ScoreDecoder.greedy_decode(arc_scores, lengths)
