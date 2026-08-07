"""
Nesting Environment Class (`NestingEnv`)

WHY THIS CLASS EXISTS:
In RL, the environment serves as the interface between the decision-making agent (policy)
and the physical world. It handles state tracking, action execution, and reward calculation.

Key RL Concepts implemented here:
1. Environment State: Keeps track of piece dimensions, which pieces are available (masking), 
   and the geometry of placed pieces on the sheet.
2. Action Execution (`step`): Takes a selected piece index, passes it to the placement 
   decoder, and updates the layout.
3. Delayed/Sparse Reward (`score`): Reward is 0 during intermediate placement steps, and 
   equals final sheet utilization % at the end of the episode.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from env.generator import generate_instance
from env.decoder import place_bottom_left


class NestingEnv:
    def __init__(
        self,
        sheet_width: float = 100.0,
        sheet_height: float = 100.0,
        num_pieces: int = 10
    ):
        """
        Initializes the nesting environment with fixed sheet dimensions.
        """
        self.sheet_width = float(sheet_width)
        self.sheet_height = float(sheet_height)
        self.sheet_area = self.sheet_width * self.sheet_height
        self.num_pieces = num_pieces

        # State attributes (populated on reset)
        self.pieces: Optional[np.ndarray] = None          # Shape (N, 2) -> [width, height]
        self.mask: Optional[np.ndarray] = None            # Shape (N,) bool -> True if available
        self.placed_rects: List[Tuple[float, float, float, float]] = []  # List of (x, y, w, h)
        self.placed_indices: List[int] = []               # Order in which piece indices were placed
        self.step_count: int = 0

    def reset(self, pieces: Optional[np.ndarray] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Resets the environment for a new episode.

        Parameters:
        -----------
        pieces : np.ndarray, optional
            If provided, uses these explicit pieces. Otherwise, generates a new random instance.
        seed : int, optional
            Random seed for instance generation.

        Returns:
        --------
        dict representing the initial environment state.
        """
        if pieces is not None:
            self.pieces = np.array(pieces, dtype=np.float32)
            self.num_pieces = len(pieces)
        else:
            self.pieces = generate_instance(
                num_pieces=self.num_pieces,
                sheet_width=self.sheet_width,
                sheet_height=self.sheet_height,
                seed=seed
            )

        # Initially, all pieces are unplaced (mask = True)
        self.mask = np.ones(self.num_pieces, dtype=bool)
        self.placed_rects = []
        self.placed_indices = []
        self.step_count = 0

        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        """
        Constructs state dictionary exposed to the RL agent.
        """
        return {
            "pieces": self.pieces.copy(),
            "mask": self.mask.copy(),
            "placed_rects": list(self.placed_rects),
            "step_count": self.step_count,
            "utilization": self.compute_utilization()
        }

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Executes an action: places the selected piece index on the sheet.

        WHY MASKING IS CRITICAL IN CO:
        In Pointer Networks for Combinatorial Optimization (Vinyals et al. 2015), 
        each item can only be picked once. Masking guarantees the agent cannot re-select 
        an already placed piece.

        Parameters:
        -----------
        action_idx : int
            Index of the piece selected by the policy.

        Returns:
        --------
        (next_state, reward, done, info)
        """
        if not self.mask[action_idx]:
            raise ValueError(f"Action error: Piece {action_idx} has already been placed or is invalid!")

        piece_w, piece_h = self.pieces[action_idx]

        # Use the deterministic bottom-left placement decoder (plumbing)
        placement = place_bottom_left(
            piece_w=piece_w,
            piece_h=piece_h,
            sheet_w=self.sheet_width,
            sheet_h=self.sheet_height,
            placed_rects=self.placed_rects
        )

        # Mark piece as no longer available (mask = False)
        self.mask[action_idx] = False
        self.step_count += 1

        placed_successfully = False
        if placement is not None:
            x, y = placement
            self.placed_rects.append((x, y, piece_w, piece_h))
            self.placed_indices.append(action_idx)
            placed_successfully = True

        # Episode finishes when all pieces have been processed
        done = (self.step_count == self.num_pieces)

        # Sparse reward: 0 for intermediate steps, final utilization % at episode end
        reward = self.score() if done else 0.0

        info = {
            "placed_successfully": placed_successfully,
            "action_idx": action_idx
        }

        return self._get_state(), reward, done, info

    def compute_utilization(self) -> float:
        """
        Calculates current utilization ratio = (total placed area) / (total sheet area).
        """
        total_placed_area = sum(w * h for _, _, w, h in self.placed_rects)
        return float(total_placed_area / self.sheet_area)

    def score(self) -> float:
        """
        Returns final utilization percentage (0.0 to 100.0%).
        """
        return self.compute_utilization() * 100.0
