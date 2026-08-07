"""
Attention Policy Model (`policy.py`)

WHY THIS CLASS EXISTS:
Combines the Transformer Encoder and Pointer Decoder into an end-to-end policy network 
that takes a batch of raw rectangle instances and autoregressively generates complete piece 
placement sequences.

Key Functionality:
1. Autoregressive Rollout: Decodes piece choices one step at a time for N steps.
2. Trajectory Log Probability Tracking: Accumulates sum of log probs log(p(a_t | s_t)) 
   across the episode, which is required for computing REINFORCE policy gradients.
3. Dual Decoding Modes:
   - 'sample': Stochastic sampling for exploration during RL training.
   - 'greedy': Deterministic argmax selection for inference evaluation.
"""

from typing import Tuple
import torch
import torch.nn as nn
from model.encoder import TransformerEncoder
from model.decoder import PointerDecoder


class AttentionPolicy(nn.Module):
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
        self.decoder = PointerDecoder(d_model=d_model, clip_const=10.0)

    def forward(
        self,
        x: torch.Tensor,
        decode_type: str = "sample"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
        -----------
        x : torch.Tensor of shape (batch_size, N, 2)
            Raw piece dimensions [width, height].
        decode_type : str
            'sample' for categorical sampling (training), or 'greedy' for argmax (eval).

        Returns:
        --------
        actions : torch.Tensor of shape (batch_size, N)
            Sequence of selected piece indices for each instance.
        log_probs_sum : torch.Tensor of shape (batch_size,)
            Sum of log probabilities over all N steps for each trajectory log p(tau).
        step_probs : torch.Tensor of shape (batch_size, N, N)
            Step-by-step probability distributions across all steps (for visualization/debugging).
        """
        batch_size, num_pieces, _ = x.shape

        # 1. Feature Normalization: scale width/height to [0, 1] relative to sheet dimensions
        normalized_x = x.clone()
        normalized_x[:, :, 0] /= self.sheet_width
        normalized_x[:, :, 1] /= self.sheet_height

        # 2. Run Transformer Encoder: get node embeddings H and graph mean embedding h_mean
        h, h_mean = self.encoder(normalized_x)

        # Initialize rollout state
        mask = torch.ones(batch_size, num_pieces, dtype=torch.bool, device=x.device)
        last_placed_embed = torch.zeros(batch_size, self.d_model, device=x.device)

        actions_list = []
        log_probs_list = []
        step_probs_list = []

        # 3. Autoregressive Decoding Loop over N steps
        for step in range(num_pieces):
            # Compute action probability distribution over available pieces
            probs = self.decoder(
                h=h,
                h_mean=h_mean,
                last_placed_embed=last_placed_embed,
                mask=mask
            )
            step_probs_list.append(probs)

            # Action selection & log prob computation
            if decode_type == "sample":
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()  # Shape: (batch_size,)
                log_prob = dist.log_prob(action)
            elif decode_type == "greedy":
                action = torch.argmax(probs, dim=-1)  # Shape: (batch_size,)
                # Gather probability of chosen greedy action
                action_prob = probs.gather(1, action.unsqueeze(1)).squeeze(1)
                log_prob = torch.log(action_prob + 1e-9)
            else:
                raise ValueError(f"Unknown decode_type: '{decode_type}'. Choose 'sample' or 'greedy'.")

            actions_list.append(action)
            log_probs_list.append(log_prob)

            # Update Action Mask: set selected piece to False (unavailable)
            mask = mask.scatter(1, action.unsqueeze(1), False)

            # Update State Embedding: set last_placed_embed to embedding of selected piece
            # Gather chosen piece's embedding vector from H
            chosen_embed = h.gather(1, action.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.d_model)).squeeze(1)
            last_placed_embed = chosen_embed

        # Stack trajectory sequences
        actions = torch.stack(actions_list, dim=1)        # Shape: (batch_size, N)
        log_probs = torch.stack(log_probs_list, dim=1)    # Shape: (batch_size, N)
        step_probs = torch.stack(step_probs_list, dim=1)  # Shape: (batch_size, N, N)

        # Sum log probabilities across all N steps to get total trajectory log probability: log p(tau)
        log_probs_sum = log_probs.sum(dim=1)  # Shape: (batch_size,)

        return actions, log_probs_sum, step_probs
