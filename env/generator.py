import numpy as np

def generate_instance(
    num_pieces: int = 10,
    sheet_width: float = 100.0,
    sheet_height: float = 100.0,
    min_size: float = 10.0,
    max_size: float = 40.0,
    seed: int = None
) -> np.ndarray:
    """
    Generates a synthetic instance of 2D rectangular pieces for sheet metal nesting.

    WHY THIS DESIGN:
    In Neural Combinatorial Optimization (e.g., Kool et al. 2019, Bello et al. 2016),
    we don't train on a static fixed dataset. Instead, we sample problem instances 
    from a random distribution on the fly during training. This forces the neural network 
    to learn generic spatial geometric heuristics rather than memorizing specific piece sizes.

    Parameters:
    -----------
    num_pieces : int
        Number of rectangular pieces to generate for this problem instance.
    sheet_width : float
        Width of the target sheet metal plate.
    sheet_height : float
        Height of the target sheet metal plate.
    min_size : float
        Minimum width/height for generated rectangles (prevents tiny useless pieces).
    max_size : float
        Maximum width/height for generated rectangles (ensures pieces fit on sheet).
    seed : int, optional
        Random seed for reproducibility during testing/debugging.

    Returns:
    --------
    np.ndarray of shape (num_pieces, 2)
        Array where each row represents [width, height] of a piece.
    """
    if seed is not None:
        np.random.seed(seed)

    # Uniformly sample width and height for each piece independently
    widths = np.random.uniform(min_size, max_size, size=num_pieces)
    heights = np.random.uniform(min_size, max_size, size=num_pieces)

    # Combine into (num_pieces, 2) array
    pieces = np.column_stack([widths, heights])
    return pieces
