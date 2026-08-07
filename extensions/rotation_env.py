"""
Rotation-Aware Nesting Environment (`rotation_env.py`)

WHY THIS CLASS EXISTS:
Extends the standard NestingEnv to support 90-degree piece rotations.

Action Space Formulation:
- Total Actions: 2N candidate choices.
- Actions 0 ... N-1: Place piece in 0-degree orientation (width=w, height=h).
- Actions N ... 2N-1: Place piece in 90-degree orientation (width=h, height=w).

Dual Action Masking:
When piece k is placed (either 0° or 90°), BOTH action k AND action k+N are masked out.
This ensures a piece cannot be placed twice in different orientations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from env.generator import generate_instance
from env.decoder import place_bottom_left


class RotationNestingEnv:
    def __init__(
        self,
        sheet_width: float = 100.0,
        sheet_height: float = 100.0,
        num_pieces: int = 10
    ):
        self.sheet_width = float(sheet_width)
        self.sheet_height = float(sheet_height)
        self.sheet_area = self.sheet_width * self.sheet_height
        self.num_pieces = num_pieces

        self.pieces: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None  # Shape: (2N,)
        self.placed_rects: List[Tuple[float, float, float, float]] = []
        self.placed_indices: List[int] = []
        self.placed_rotations: List[bool] = []
        self.step_count: int = 0

    def reset(self, pieces: Optional[np.ndarray] = None, seed: Optional[int] = None) -> Dict[str, Any]:
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

        # Action space size is 2N (N unrotated + N rotated)
        self.mask = np.ones(2 * self.num_pieces, dtype=bool)
        self.placed_rects = []
        self.placed_indices = []
        self.placed_rotations = []
        self.step_count = 0

        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        return {
            "pieces": self.pieces.copy(),
            "mask": self.mask.copy(),
            "placed_rects": list(self.placed_rects),
            "step_count": self.step_count,
            "utilization": self.compute_utilization()
        }

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self.mask[action_idx]:
            raise ValueError(f"Invalid Action: Action {action_idx} is already masked!")

        # Determine original piece index and rotation flag
        piece_idx = action_idx % self.num_pieces
        is_rotated = (action_idx >= self.num_pieces)

        orig_w, orig_h = self.pieces[piece_idx]
        w, h = (orig_h, orig_w) if is_rotated else (orig_w, orig_h)

        # Pass effective (w, h) to bottom-left decoder
        placement = place_bottom_left(
            piece_w=w,
            piece_h=h,
            sheet_w=self.sheet_width,
            sheet_h=self.sheet_height,
            placed_rects=self.placed_rects
        )

        # Dual Masking: disable BOTH 0° and 90° choices for this piece
        self.mask[piece_idx] = False
        self.mask[piece_idx + self.num_pieces] = False
        self.step_count += 1

        placed_successfully = False
        if placement is not None:
            x, y = placement
            self.placed_rects.append((x, y, w, h))
            self.placed_indices.append(piece_idx)
            self.placed_rotations.append(is_rotated)
            placed_successfully = True

        done = (self.step_count == self.num_pieces)
        reward = self.score() if done else 0.0

        info = {
            "placed_successfully": placed_successfully,
            "piece_idx": piece_idx,
            "is_rotated": is_rotated
        }

        return self._get_state(), reward, done, info

    def compute_utilization(self) -> float:
        total_placed_area = sum(w * h for _, _, w, h in self.placed_rects)
        return float(total_placed_area / self.sheet_area)

    def score(self) -> float:
        return self.compute_utilization() * 100.0
