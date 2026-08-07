"""
Multi-Angle Rotation Nesting Environment (`rotation_env.py`)

WHY THIS MODULE EXISTS:
Extends the Nesting Environment to support multi-angle rotations (0°, 45°, 90°, 135°).

Action Space Formulation:
- Total Actions: K * N candidate choices (where K = 4 angles: 0°, 45°, 90°, 135°).
- Multi-Angle Action Masking: Placing piece k in ANY angle masks out ALL K angle candidates 
  (k, k+N, k+2N, k+3N), ensuring a piece is placed exactly once.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import shapely
from shapely.geometry import Polygon, box
from shapely.affinity import translate, rotate
from env.generator import generate_instance


SUPPORTED_ANGLES = [0.0, 45.0, 90.0, 135.0]


class MultiAngleNestingEnv:
    def __init__(
        self,
        sheet_width: float = 100.0,
        sheet_height: float = 100.0,
        num_pieces: int = 10,
        angles: List[float] = None
    ):
        self.sheet_width = float(sheet_width)
        self.sheet_height = float(sheet_height)
        self.sheet_poly = box(0, 0, self.sheet_width, self.sheet_height)
        self.sheet_area = self.sheet_width * self.sheet_height
        self.num_pieces = num_pieces
        self.angles = angles if angles is not None else SUPPORTED_ANGLES
        self.num_angles = len(self.angles)

        self.pieces: Optional[np.ndarray] = None
        self.polygons: List[Polygon] = []
        self.mask: Optional[np.ndarray] = None  # Shape: (K * N,)
        self.placed_polygons: List[Polygon] = []
        self.placed_indices: List[int] = []
        self.placed_angles: List[float] = []
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

        # Convert rectangular dimensions to Shapely Polygon objects
        self.polygons = [Polygon([(0, 0), (w, 0), (w, h), (0, h)]) for w, h in self.pieces]

        # Action space size is K * N
        self.mask = np.ones(self.num_angles * self.num_pieces, dtype=bool)
        self.placed_polygons = []
        self.placed_indices = []
        self.placed_angles = []
        self.step_count = 0

        return self._get_state()

    def _get_state(self) -> Dict[str, Any]:
        return {
            "pieces": self.pieces.copy(),
            "mask": self.mask.copy(),
            "placed_polygons": list(self.placed_polygons),
            "step_count": self.step_count,
            "utilization": self.compute_utilization()
        }

    def place_polygon_at_angle(self, poly: Polygon, angle: float, grid_step: float = 1.0) -> Optional[Polygon]:
        """
        Rotates polygon by target angle and finds lowest-leftmost valid coordinate on sheet.
        """
        rotated_poly = rotate(poly, angle, origin=(0, 0)) if angle != 0.0 else poly

        minx, miny, maxx, maxy = rotated_poly.bounds
        poly_w = maxx - minx
        poly_h = maxy - miny

        xs = np.arange(0.0, self.sheet_width - poly_w + 0.1, grid_step)
        ys = np.arange(0.0, self.sheet_height - poly_h + 0.1, grid_step)

        for y in ys:
            for x in xs:
                shifted = translate(rotated_poly, xoff=x - minx, yoff=y - miny)
                if not self.sheet_poly.contains(shifted):
                    continue

                overlap = any(shifted.intersects(p) and not shifted.touches(p) for p in self.placed_polygons)
                if not overlap:
                    return shifted

        return None

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if not self.mask[action_idx]:
            raise ValueError(f"Action Error: Action {action_idx} is already masked!")

        piece_idx = action_idx % self.num_pieces
        angle_idx = action_idx // self.num_pieces
        target_angle = self.angles[angle_idx]

        poly = self.polygons[piece_idx]
        placed_poly = self.place_polygon_at_angle(poly, target_angle)

        # Multi-Angle Action Masking: mask out ALL angle candidates for piece_idx
        for a_i in range(self.num_angles):
            self.mask[piece_idx + a_i * self.num_pieces] = False

        self.step_count += 1

        placed_successfully = False
        if placed_poly is not None:
            self.placed_polygons.append(placed_poly)
            self.placed_indices.append(piece_idx)
            self.placed_angles.append(target_angle)
            placed_successfully = True

        done = (self.step_count == self.num_pieces)
        reward = self.score() if done else 0.0

        info = {
            "placed_successfully": placed_successfully,
            "piece_idx": piece_idx,
            "angle": target_angle
        }

        return self._get_state(), reward, done, info

    def compute_utilization(self) -> float:
        total_placed_area = sum(p.area for p in self.placed_polygons)
        return float(total_placed_area / self.sheet_area)

    def score(self) -> float:
        return self.compute_utilization() * 100.0


# Backward-compatible alias
RotationNestingEnv = MultiAngleNestingEnv

