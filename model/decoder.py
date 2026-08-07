"""
Pointer Attention Decoder (`decoder.py`)

WHY THIS DECODER DESIGN (Vinyals et al. 2015, Kool et al. 2019):
In sequence generation problems where the output elements must be selected directly from 
the input set (like picking piece indices), standard fixed-size softmax fails.

The Pointer Decoder uses an Attention Query (representing current problem context) to score 
compatibility against Key vectors (representing candidate piece embeddings).

Key Features:
1. Action Masking: Already placed pieces have their logits forced to -1e9, guaranteeing 
   softmax assigns them 0.0 probability.
2. Tanh Logit Clipping (C = 10.0): Prevents softmax saturation during early RL training steps, 
   ensuring smooth gradient flow (Kool et al., 2019).
"""

import math
import torch
import torch.nn as nn


class PointerDecoder(nn.Module):
    def __init__(self, d_model: int = 128, clip_const: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.clip_const = clip_const

        # Linear projections for Query, Key
        self.project_q_graph = nn.Linear(d_model, d_model, bias=False)
        self.project_q_last = nn.Linear(d_model, d_model, bias=False)
        self.project_k = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        h_mean: torch.Tensor,
        last_placed_embed: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        h : torch.Tensor of shape (batch_size, N, d_model)
            Piece contextual embeddings from encoder.
        h_mean : torch.Tensor of shape (batch_size, d_model)
            Graph-level mean pooling vector.
        last_placed_embed : torch.Tensor of shape (batch_size, d_model)
            Embedding of the piece placed in the previous step (or 0 vector at step 0).
        mask : torch.Tensor of shape (batch_size, N) bool
            True for available (unplaced) pieces, False for already placed pieces.

        Returns:
        --------
        probs : torch.Tensor of shape (batch_size, N)
            Probability distribution over pieces for the current step.
        """
        batch_size, num_pieces, _ = h.shape

        # 1. Compute Context Query Vector q: (batch_size, 1, d_model)
        # Combines overall instance summary (h_mean) and state summary (last_placed_embed)
        q = self.project_q_graph(h_mean) + self.project_q_last(last_placed_embed)
        q = q.unsqueeze(1)  # (batch_size, 1, d_model)

        # 2. Compute Key Vectors K: (batch_size, N, d_model)
        k = self.project_k(h)

        # 3. Compute Dot-Product Attention Compatibility Scores (Logits)
        # (batch_size, 1, d_model) x (batch_size, d_model, N) -> (batch_size, 1, N)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.d_model)
        logits = scores.squeeze(1)  # (batch_size, N)

        # 4. Tanh Logit Clipping (Kool et al. 2019 / Bello et al. 2016)
        # Clips logits to range [-C, +C] to prevent extreme confidence early in training
        if self.clip_const > 0:
            logits = self.clip_const * torch.tanh(logits)

        # 5. Action Masking
        # Set logits of already placed pieces (where mask == False) to -1e9
        # softmax(-1e9) yields 0.0, rendering invalid actions impossible
        logits = logits.masked_fill(~mask, -1e9)

        # 6. Compute Softmax Probabilities over available pieces
        probs = torch.softmax(logits, dim=-1)

        return probs
