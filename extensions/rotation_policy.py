"""
Rotation-Aware Policy Network (`rotation_policy.py`)

WHY THIS MODULE EXISTS:
Extends AttentionPolicy to handle 2N candidate action choices (N pieces x 2 orientations).

Key Neural Mechanics:
1. Rotated Key Projection: Computes dual key embeddings for both 0° [w, h] and 90° [h, w] 
   orientations for each piece.
2. Dual Action Masking: When action a_t is sampled (selecting piece k in either orientation), 
   both candidate indices k and k+N are masked out.
"""

import math
import torch
import torch.nn as nn
from model.encoder import TransformerEncoder


class RotationPointerDecoder(nn.Module):
    def __init__(self, d_model: int = 128, clip_const: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.clip_const = clip_const

        self.project_q_graph = nn.Linear(d_model, d_model, bias=False)
        self.project_q_last = nn.Linear(d_model, d_model, bias=False)

        # Separate projections for unrotated (0°) and rotated (90°) piece keys
        self.project_k_unrotated = nn.Linear(d_model, d_model, bias=False)
        self.project_k_rotated = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,
        h_mean: torch.Tensor,
        last_placed_embed: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        batch_size, num_pieces, _ = h.shape

        # Query vector q
        q = self.project_q_graph(h_mean) + self.project_q_last(last_placed_embed)
        q = q.unsqueeze(1)  # (batch_size, 1, d_model)

        # Dual Key projections for 0° and 90°
        k_0 = self.project_k_unrotated(h)  # (batch_size, N, d_model)
        k_90 = self.project_k_rotated(h)   # (batch_size, N, d_model)

        # Concatenate keys along node dimension: (batch_size, 2N, d_model)
        k_combined = torch.cat([k_0, k_90], dim=1)

        # Attention dot product scores: (batch_size, 1, 2N)
        scores = torch.matmul(q, k_combined.transpose(1, 2)) / math.sqrt(self.d_model)
        logits = scores.squeeze(1)  # (batch_size, 2N)

        if self.clip_const > 0:
            logits = self.clip_const * torch.tanh(logits)

        # Apply 2N action mask (-1e9 for unavailable choices)
        logits = logits.masked_fill(~mask, -1e9)

        probs = torch.softmax(logits, dim=-1)
        return probs


class RotationAttentionPolicy(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        sheet_width: float = 100.0,
        sheet_height: float = 100.0
    ):
        super().__init__()
        self.d_model = d_model
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height

        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )
        self.decoder = RotationPointerDecoder(d_model=d_model, clip_const=10.0)

    def forward(
        self,
        x: torch.Tensor,
        decode_type: str = "sample"
    ):
        batch_size, num_pieces, _ = x.shape

        normalized_x = x.clone()
        normalized_x[:, :, 0] /= self.sheet_width
        normalized_x[:, :, 1] /= self.sheet_height

        h, h_mean = self.encoder(normalized_x)

        # Mask over 2N candidate actions
        mask = torch.ones(batch_size, 2 * num_pieces, dtype=torch.bool, device=x.device)
        last_placed_embed = torch.zeros(batch_size, self.d_model, device=x.device)

        actions_list = []
        log_probs_list = []

        for step in range(num_pieces):
            probs = self.decoder(h, h_mean, last_placed_embed, mask)

            if decode_type == "sample":
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            elif decode_type == "greedy":
                action = torch.argmax(probs, dim=-1)
                action_prob = probs.gather(1, action.unsqueeze(1)).squeeze(1)
                log_prob = torch.log(action_prob + 1e-9)
            else:
                raise ValueError(f"Unknown decode_type: '{decode_type}'")

            actions_list.append(action)
            log_probs_list.append(log_prob)

            # Dual Action Masking: disable BOTH k and k+N for chosen piece
            piece_idx = action % num_pieces
            mask = mask.scatter(1, piece_idx.unsqueeze(1), False)
            mask = mask.scatter(1, (piece_idx + num_pieces).unsqueeze(1), False)

            # Update state embedding
            chosen_embed = h.gather(1, piece_idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)).squeeze(1)
            last_placed_embed = chosen_embed

        actions = torch.stack(actions_list, dim=1)
        log_probs_sum = torch.stack(log_probs_list, dim=1).sum(dim=1)

        return actions, log_probs_sum
