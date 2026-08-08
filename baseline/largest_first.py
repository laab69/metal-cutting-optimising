"""
Largest-Piece-First Heuristic (`largest_first.py`)

WHY THIS HEURISTIC EXISTS:
In classical 2D bin packing and sheet metal nesting literature, "Largest-Area-First" 
is the single standard greedy baseline. 

By placing the largest, most bulky rectangular pieces first when the sheet has maximum 
unobstructed open space, we minimize the chance of fragmentation. Smaller pieces are then 
used as "filler" to occupy the residual gaps around the large pieces.

This provides the deterministic benchmark floor that our neural network policy must 
exceed in Phase D and E.
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from env.nesting_env import NestingEnv


def largest_piece_first_sequence(pieces: np.ndarray) -> np.ndarray:
    """Returns sorted piece indices in descending order of area."""
    areas = pieces[:, 0] * pieces[:, 1]
    return np.argsort(-areas)


def run_largest_first_heuristic(env: NestingEnv, pieces: np.ndarray) -> Tuple[float, List[Tuple[float, float, float, float]]]:
    state = env.reset(pieces=pieces)
    sorted_indices = largest_piece_first_sequence(pieces)

    done = False
    for action_idx in sorted_indices:
        state, reward, done, info = env.step(action_idx)

    final_utilization = env.score()
    return final_utilization, env.placed_rects
