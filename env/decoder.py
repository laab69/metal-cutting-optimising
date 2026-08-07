"""
Placement Decoder (Plumbing)

WHY THIS EXISTS:
The neural network's role is to solve the sequence problem: deciding *which order* to hand 
pieces to the cutting machine. Once an order is chosen, we need a deterministic physical 
rule to translate piece selection into exact (x, y) coordinates on the sheet.

This decoder uses the classic "Bottom-Left" (Lowest-then-Leftmost) heuristic.
It is purely non-learned infrastructure (<30 lines) so an episode can produce a layout 
to compute scalar utilization reward. We do NOT tune or optimize this decoder.
"""

from typing import List, Tuple, Optional

def place_bottom_left(
    piece_w: float, 
    piece_h: float, 
    sheet_w: float, 
    sheet_h: float, 
    placed_rects: List[Tuple[float, float, float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Finds the lowest-then-leftmost valid (x, y) coordinate for a piece on the sheet.
    
    placed_rects: list of tuples (x, y, width, height) of pieces already placed.
    Returns (x, y) tuple if placement is possible, or None if it doesn't fit anywhere.
    """
    # Candidate coordinates are origin (0, 0) and top/right edges of existing placed pieces
    candidate_xs = sorted(set([0.0] + [r[0] + r[2] for r in placed_rects]))
    candidate_ys = sorted(set([0.0] + [r[1] + r[3] for r in placed_rects]))

    # Search candidates ordered by lowest y first, then leftmost x
    for y in candidate_ys:
        if y + piece_h > sheet_h:
            continue  # Exceeds top boundary of sheet
        for x in candidate_xs:
            if x + piece_w > sheet_w:
                continue  # Exceeds right boundary of sheet

            # Check for overlap with any existing rectangle
            overlap = False
            for px, py, pw, ph in placed_rects:
                if not (x + piece_w <= px or x >= px + pw or y + piece_h <= py or y >= py + ph):
                    overlap = True
                    break

            if not overlap:
                return (x, y)

    return None  # Could not place piece anywhere on the sheet
