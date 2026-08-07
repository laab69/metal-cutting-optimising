"""
Held-Out Test Instance Generator (`heldout_generator.py`)

WHY THIS MODULE EXISTS:
To rigorously test machine learning generalization, the evaluation test set MUST be strictly 
isolated from the data generated during training.

This script creates a deterministic batch of 200 held-out rectangular instances that the 
policy network has NEVER seen, allowing us to evaluate true zero-shot generalization.
"""

from typing import List
import numpy as np
from env.generator import generate_instance


def get_heldout_test_set(
    num_instances: int = 200,
    num_pieces: int = 10,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    seed: int = 5555
) -> List[np.ndarray]:
    """
    Generates a deterministic set of held-out test instances.

    Parameters:
    -----------
    num_instances : int
        Number of test instances to generate (default: 200).
    num_pieces : int
        Number of pieces per instance (default: 10).
    seed : int
        Random seed for reproducible evaluation.

    Returns:
    --------
    list of np.ndarray, each of shape (num_pieces, 2)
    """
    test_instances = [
        generate_instance(
            num_pieces=num_pieces,
            sheet_width=sheet_width,
            sheet_height=sheet_height,
            seed=seed + i
        )
        for i in range(num_instances)
    ]
    return test_instances
