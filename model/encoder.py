"""
Transformer Self-Attention Encoder (`encoder.py`)

WHY THIS ENCODER DESIGN (Kool et al., 2019):
In sheet metal nesting, an instance consists of an unordered set of N rectangular pieces.
Standard Recurrent Neural Networks (RNNs) are sensitive to input order, which is undesirable 
because piece lists have no inherent sequence order before cutting.

A Transformer Multi-Head Self-Attention (MHSA) encoder is permutation-equivariant. 
It allows every piece vector to attend to every other piece vector, embedding relational 
spatial features (e.g., "is this piece much larger than the average piece in this set?") 
into each piece's embedding vector.
"""

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 128, num_heads: int = 8, d_ff: int = 512):
        super().__init__()
        # Multi-Head Self-Attention layer
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-Forward Subnetwork (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Multi-Head Self-Attention with residual connection & normalization
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # 2. Feed-forward network with residual connection & normalization
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, d_model: int = 128, num_heads: int = 8, num_layers: int = 2, d_ff: int = 512):
        super().__init__()
        self.d_model = d_model

        # Linear projection from 2D raw features [normalized_width, normalized_height] to d_model
        self.init_embed = nn.Linear(input_dim, d_model)

        # Stack of Transformer Self-Attention encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        Parameters:
        -----------
        x : torch.Tensor of shape (batch_size, N, 2)
            Raw normalized piece dimensions [width/sheet_w, height/sheet_h].

        Returns:
        --------
        h : torch.Tensor of shape (batch_size, N, d_model)
            Contextual embeddings for each piece.
        h_mean : torch.Tensor of shape (batch_size, d_model)
            Graph-level mean pooling embedding vector.
        """
        # Linear embedding: (batch_size, N, 2) -> (batch_size, N, d_model)
        h = self.init_embed(x)

        # Pass through stack of self-attention layers
        for layer in self.layers:
            h = layer(h)

        # Compute graph-level mean embedding across all nodes
        h_mean = h.mean(dim=1)  # Shape: (batch_size, d_model)

        return h, h_mean
